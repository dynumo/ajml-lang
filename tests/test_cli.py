"""Tests for the AJML CLI commands."""

import os
import shutil
import tempfile

import pytest

from ajml.cli import cmd_build, cmd_init


class MockArgs:
    """Mock argparse namespace."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestInitCommand:
    """Test the init command."""

    def test_creates_project_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_name = os.path.join(tmpdir, "my_project")
            args = MockArgs(project_name=project_name)
            result = cmd_init(args)
            assert result == 0

            assert os.path.exists(os.path.join(project_name, "agents", "_project.ajml"))
            assert os.path.exists(os.path.join(project_name, "agents", "example_agent.ajml"))
            assert os.path.exists(os.path.join(project_name, "tools", "example_tool.py"))
            assert os.path.exists(os.path.join(project_name, ".env.example"))
            assert os.path.exists(os.path.join(project_name, ".gitignore"))
            assert os.path.isdir(os.path.join(project_name, "build"))

    def test_existing_directory_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = MockArgs(project_name=tmpdir)
            result = cmd_init(args)
            assert result == 2


class TestBuildCommand:
    """Test the build command."""

    def test_builds_simple_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create project structure
            agents_dir = os.path.join(tmpdir, "agents")
            os.makedirs(agents_dir)

            project_ajml = '''<?xml version="1.0" encoding="UTF-8"?>
<project name="test" ajml_version="2.0">
    <config>
        <llm provider="openai" model="gpt-4o" />
    </config>
</project>'''
            with open(os.path.join(agents_dir, "_project.ajml"), "w") as f:
                f.write(project_ajml)

            agent_ajml = '''<?xml version="1.0" encoding="UTF-8"?>
<agent name="simple_agent" version="1.0" description="A simple agent.">
    <state>
        <field name="input" type="string" required="true" />
        <field name="output" type="string" default="" />
    </state>
    <graph>
        <node id="respond" type="llm">
            <system_prompt>Respond to input.</system_prompt>
            <output_schema>
                <field name="output" type="string" description="Response" />
            </output_schema>
        </node>
        <edge source="__START__" target="respond" />
        <edge source="respond" target="__END__" />
    </graph>
</agent>'''
            with open(os.path.join(agents_dir, "simple_agent.ajml"), "w") as f:
                f.write(agent_ajml)

            args = MockArgs(
                project_dir=tmpdir,
                output="build",
                verbose=True,
                strict=False,
                dry_run=False,
            )
            result = cmd_build(args)
            assert result == 0

            build_dir = os.path.join(tmpdir, "build")
            assert os.path.exists(os.path.join(build_dir, "compiled_simple_agent.py"))
            assert os.path.exists(os.path.join(build_dir, "main.py"))
            assert os.path.exists(os.path.join(build_dir, "requirements.txt"))
            assert os.path.exists(os.path.join(build_dir, ".env.example"))

    def test_dry_run_no_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = os.path.join(tmpdir, "agents")
            os.makedirs(agents_dir)

            with open(os.path.join(agents_dir, "_project.ajml"), "w") as f:
                f.write('''<?xml version="1.0"?>
<project name="test" ajml_version="2.0">
    <config><llm provider="openai" model="gpt-4o" /></config>
</project>''')

            with open(os.path.join(agents_dir, "test.ajml"), "w") as f:
                f.write('''<?xml version="1.0"?>
<agent name="test">
    <state><field name="x" type="string" default="" /></state>
    <graph>
        <node id="a" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="__END__" />
    </graph>
</agent>''')

            args = MockArgs(
                project_dir=tmpdir,
                output="build",
                verbose=False,
                strict=False,
                dry_run=True,
            )
            result = cmd_build(args)
            assert result == 0

            build_dir = os.path.join(tmpdir, "build")
            assert not os.path.exists(build_dir)

    def test_missing_project_file_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "agents"))
            args = MockArgs(
                project_dir=tmpdir,
                output="build",
                verbose=False,
                strict=False,
                dry_run=False,
            )
            result = cmd_build(args)
            assert result == 1

    def test_builds_intent_classifier_example(self):
        """Build the §16.1 example from the spec."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = os.path.join(tmpdir, "agents")
            os.makedirs(agents_dir)

            with open(os.path.join(agents_dir, "_project.ajml"), "w") as f:
                f.write('''<?xml version="1.0" encoding="UTF-8"?>
<project name="classifier_demo" ajml_version="2.0">
    <config>
        <llm provider="openai" model="gpt-4o-mini" />
    </config>
</project>''')

            with open(os.path.join(agents_dir, "intent_classifier.ajml"), "w") as f:
                f.write('''<?xml version="1.0" encoding="UTF-8"?>
<agent name="intent_classifier" version="1.0"
       description="Classifies customer messages into intent categories.">

    <state>
        <field name="customer_message" type="string" required="true" />
        <field name="intent" type="string" default="" />
        <field name="confidence" type="float" default="0.0" />
    </state>

    <graph>
        <node id="classify" type="llm">
            <system_prompt>
                You are an intent classifier. Given the customer message below,
                classify it into exactly one of these categories:
                refund, complaint, question, praise, other.

                Also provide a confidence score between 0 and 1.
            </system_prompt>
            <output_schema>
                <field name="intent" type="string" description="The intent category" />
                <field name="confidence" type="float" description="Confidence score 0-1" />
            </output_schema>
        </node>

        <edge source="__START__" target="classify" />
        <edge source="classify" target="__END__" />
    </graph>
</agent>''')

            args = MockArgs(
                project_dir=tmpdir,
                output="build",
                verbose=True,
                strict=False,
                dry_run=False,
            )
            result = cmd_build(args)
            assert result == 0

            build_dir = os.path.join(tmpdir, "build")
            compiled_path = os.path.join(build_dir, "compiled_intent_classifier.py")
            assert os.path.exists(compiled_path)

            with open(compiled_path) as f:
                code = f.read()

            # Validate the compiled code is valid Python
            compile(code, compiled_path, "exec")

            assert "class AgentState(TypedDict):" in code
            assert "class ClassifyOutput(BaseModel):" in code
            assert "def classify(state: AgentState):" in code
            assert "graph = graph_builder.compile()" in code
