"""Tests for the AJML AST Validator (Phase 2)."""

import os
import tempfile

import pytest

from ajml.errors import AJMLCompilationError
from ajml.preprocessor import preprocess
from ajml.validator import (
    AgentAST,
    check_circular_dependencies,
    check_duplicate_agent_names,
    validate_agent,
    validate_project,
)


def _make_project(provider="openai", model="gpt-4o"):
    """Create a minimal ProjectAST for testing."""
    raw = f'''<?xml version="1.0" encoding="UTF-8"?>
<project name="test" ajml_version="2.0">
    <config>
        <llm provider="{provider}" model="{model}" />
    </config>
</project>'''
    root = preprocess(raw)
    return validate_project(root, "agents/_project.ajml")


def _parse_agent(ajml_text, project=None, project_dir=None):
    """Helper to preprocess and validate an agent."""
    if project is None:
        project = _make_project()
    root = preprocess(ajml_text)
    return validate_agent(
        root, "agents/test.ajml", project, project_dir=project_dir or "."
    )


class TestProjectValidation:
    """Test _project.ajml validation."""

    def test_valid_project(self):
        project = _make_project()
        assert project.name == "test"
        assert project.ajml_version == "2.0"
        assert project.llm_provider == "openai"

    def test_invalid_root_element(self):
        raw = '<?xml version="1.0"?><agent name="test" />'
        root = preprocess(raw)
        with pytest.raises(AJMLCompilationError, match="E001"):
            validate_project(root, "_project.ajml")

    def test_missing_name(self):
        raw = '<?xml version="1.0"?><project ajml_version="2.0" />'
        root = preprocess(raw)
        with pytest.raises(AJMLCompilationError, match="E001"):
            validate_project(root, "_project.ajml")

    def test_invalid_version(self):
        raw = '<?xml version="1.0"?><project name="t" ajml_version="1.0" />'
        root = preprocess(raw)
        with pytest.raises(AJMLCompilationError, match="E004"):
            validate_project(root, "_project.ajml")

    def test_invalid_provider(self):
        raw = '''<?xml version="1.0"?>
<project name="t" ajml_version="2.0">
    <config><llm provider="invalid_provider" model="x" /></config>
</project>'''
        root = preprocess(raw)
        with pytest.raises(AJMLCompilationError, match="E401"):
            validate_project(root, "_project.ajml")

    def test_server_config(self):
        raw = '''<?xml version="1.0"?>
<project name="t" ajml_version="2.0">
    <config>
        <llm provider="openai" model="gpt-4o" />
        <server cors_origins="https://example.com" auth_env="MY_KEY"
                docs_public="false" port="9000" host="127.0.0.1" />
    </config>
</project>'''
        root = preprocess(raw)
        project = validate_project(root, "_project.ajml")
        assert project.cors_origins == "https://example.com"
        assert project.auth_env == "MY_KEY"
        assert project.docs_public is False
        assert project.port == 9000
        assert project.host == "127.0.0.1"

    def test_env_vars(self):
        raw = '''<?xml version="1.0"?>
<project name="t" ajml_version="2.0">
    <config>
        <llm provider="openai" model="gpt-4o" />
        <env>
            <var name="API_KEY" required="true" />
            <var name="LOG_LEVEL" default="INFO" />
        </env>
    </config>
</project>'''
        root = preprocess(raw)
        project = validate_project(root, "_project.ajml")
        assert len(project.env_vars) == 2
        assert project.env_vars[0]["name"] == "API_KEY"
        assert project.env_vars[0]["required"] is True
        assert project.env_vars[1]["default"] == "INFO"


class TestAgentValidation:
    """Test agent-level validation."""

    def test_missing_state(self):
        raw = '''<?xml version="1.0"?>
<agent name="test">
    <graph>
        <node id="a" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="__END__" />
    </graph>
</agent>'''
        with pytest.raises(AJMLCompilationError, match="E002"):
            _parse_agent(raw)

    def test_missing_graph(self):
        raw = '''<?xml version="1.0"?>
<agent name="test">
    <state>
        <field name="x" type="string" default="" />
    </state>
</agent>'''
        with pytest.raises(AJMLCompilationError, match="E002"):
            _parse_agent(raw)

    def test_reserved_word_messages(self):
        raw = '''<?xml version="1.0"?>
<agent name="test">
    <state>
        <field name="messages" type="list" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="__END__" />
    </graph>
</agent>'''
        with pytest.raises(AJMLCompilationError, match="E201"):
            _parse_agent(raw)

    def test_duplicate_field_name(self):
        raw = '''<?xml version="1.0"?>
<agent name="test">
    <state>
        <field name="x" type="string" default="" />
        <field name="x" type="int" default="0" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="__END__" />
    </graph>
</agent>'''
        with pytest.raises(AJMLCompilationError, match="E103"):
            _parse_agent(raw)

    def test_invalid_type(self):
        raw = '''<?xml version="1.0"?>
<agent name="test">
    <state>
        <field name="x" type="invalid_type" default="" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="__END__" />
    </graph>
</agent>'''
        with pytest.raises(AJMLCompilationError, match="E107"):
            _parse_agent(raw)

    def test_required_with_default(self):
        raw = '''<?xml version="1.0"?>
<agent name="test">
    <state>
        <field name="x" type="string" required="true" default="hello" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="__END__" />
    </graph>
</agent>'''
        with pytest.raises(AJMLCompilationError, match="E202"):
            _parse_agent(raw)

    def test_invalid_reducer(self):
        raw = '''<?xml version="1.0"?>
<agent name="test">
    <state>
        <field name="x" type="string" default="" reducer="append" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="__END__" />
    </graph>
</agent>'''
        with pytest.raises(AJMLCompilationError, match="E108"):
            _parse_agent(raw)

    def test_enum_default_not_in_values(self):
        raw = '''<?xml version="1.0"?>
<agent name="test">
    <state>
        <field name="status" type="enum" values="a, b, c" default="d" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="__END__" />
    </graph>
</agent>'''
        with pytest.raises(AJMLCompilationError, match="E109"):
            _parse_agent(raw)

    def test_duplicate_node_id(self):
        raw = '''<?xml version="1.0"?>
<agent name="test">
    <state>
        <field name="x" type="string" default="" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <node id="a" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="__END__" />
    </graph>
</agent>'''
        with pytest.raises(AJMLCompilationError, match="E101"):
            _parse_agent(raw)

    def test_invalid_node_type(self):
        raw = '''<?xml version="1.0"?>
<agent name="test">
    <state>
        <field name="x" type="string" default="" />
    </state>
    <graph>
        <node id="a" type="invalid_type" />
        <edge source="__START__" target="a" />
        <edge source="a" target="__END__" />
    </graph>
</agent>'''
        with pytest.raises(AJMLCompilationError, match="E105"):
            _parse_agent(raw)

    def test_no_start_edge(self):
        raw = '''<?xml version="1.0"?>
<agent name="test">
    <state>
        <field name="x" type="string" default="" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <edge source="a" target="__END__" />
    </graph>
</agent>'''
        with pytest.raises(AJMLCompilationError, match="E301"):
            _parse_agent(raw)

    def test_edge_target_not_found(self):
        raw = '''<?xml version="1.0"?>
<agent name="test">
    <state>
        <field name="x" type="string" default="" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <edge source="__START__" target="nonexistent" />
        <edge source="a" target="__END__" />
    </graph>
</agent>'''
        with pytest.raises(AJMLCompilationError, match="E302"):
            _parse_agent(raw)

    def test_edge_source_not_found(self):
        raw = '''<?xml version="1.0"?>
<agent name="test">
    <state>
        <field name="x" type="string" default="" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="nonexistent" target="__END__" />
    </graph>
</agent>'''
        with pytest.raises(AJMLCompilationError, match="E303"):
            _parse_agent(raw)

    def test_conditional_group_missing_default(self):
        raw = '''<?xml version="1.0"?>
<agent name="test">
    <state>
        <field name="x" type="int" default="0" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <node id="b" type="llm" />
        <node id="c" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="b">
            <condition>state.get('x') == 1</condition>
        </edge>
        <edge source="a" target="c">
            <condition>state.get('x') == 2</condition>
        </edge>
        <edge source="b" target="__END__" />
        <edge source="c" target="__END__" />
    </graph>
</agent>'''
        with pytest.raises(AJMLCompilationError, match="E304"):
            _parse_agent(raw)

    def test_default_edge_with_condition(self):
        raw = '''<?xml version="1.0"?>
<agent name="test">
    <state>
        <field name="x" type="int" default="0" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <node id="b" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="b" default="true">
            <condition>state.get('x') == 1</condition>
        </edge>
        <edge source="b" target="__END__" />
    </graph>
</agent>'''
        with pytest.raises(AJMLCompilationError, match="E305"):
            _parse_agent(raw)

    def test_condition_invalid_name(self):
        raw = '''<?xml version="1.0"?>
<agent name="test">
    <state>
        <field name="x" type="int" default="0" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <node id="b" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="b">
            <condition>os.system('rm -rf /')</condition>
        </edge>
        <edge source="a" target="__END__" default="true" />
        <edge source="b" target="__END__" />
    </graph>
</agent>'''
        with pytest.raises(AJMLCompilationError, match="E501"):
            _parse_agent(raw)

    def test_valid_minimal_agent(self):
        raw = '''<?xml version="1.0"?>
<agent name="test_agent" version="1.0" description="Test agent">
    <state>
        <field name="input" type="string" required="true" />
        <field name="output" type="string" default="" />
    </state>
    <graph>
        <node id="process" type="llm">
            <system_prompt>Process the input.</system_prompt>
            <output_schema>
                <field name="output" type="string" description="The output" />
            </output_schema>
        </node>
        <edge source="__START__" target="process" />
        <edge source="process" target="__END__" />
    </graph>
</agent>'''
        agent, warnings = _parse_agent(raw)
        assert agent.name == "test_agent"
        assert len(agent.state_fields) == 2
        assert len(agent.nodes) == 1
        assert len(agent.edges) == 2

    def test_valid_enum_field(self):
        raw = '''<?xml version="1.0"?>
<agent name="test_agent">
    <state>
        <field name="status" type="enum" values="pending, active, done" default="pending" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="__END__" />
    </graph>
</agent>'''
        agent, warnings = _parse_agent(raw)
        assert agent.state_fields[0]["type"] == "enum"
        assert agent.state_fields[0]["values"] == "pending, active, done"

    def test_valid_parameterised_types(self):
        raw = '''<?xml version="1.0"?>
<agent name="test_agent">
    <state>
        <field name="tags" type="list[string]" default="[]" reducer="append" />
        <field name="scores" type="list[float]" default="[]" reducer="append" />
        <field name="settings" type="dict[string]" default="{}" reducer="merge" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="__END__" />
    </graph>
</agent>'''
        agent, warnings = _parse_agent(raw)
        assert len(agent.state_fields) == 3

    def test_unreachable_node(self):
        raw = '''<?xml version="1.0"?>
<agent name="test">
    <state>
        <field name="x" type="string" default="" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <node id="orphan" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="__END__" />
        <edge source="orphan" target="__END__" />
    </graph>
</agent>'''
        with pytest.raises(AJMLCompilationError, match="E312"):
            _parse_agent(raw)

    def test_node_no_outgoing_edges(self):
        raw = '''<?xml version="1.0"?>
<agent name="test">
    <state>
        <field name="x" type="string" default="" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <node id="b" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="b" />
    </graph>
</agent>'''
        with pytest.raises(AJMLCompilationError, match="E313"):
            _parse_agent(raw)


class TestCrossAgentValidation:
    """Test cross-agent checks."""

    def test_duplicate_agent_names(self):
        raw = '''<?xml version="1.0"?>
<agent name="same_name">
    <state><field name="x" type="string" default="" /></state>
    <graph>
        <node id="a" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="__END__" />
    </graph>
</agent>'''
        project = _make_project()
        root1 = preprocess(raw)
        root2 = preprocess(raw)
        agent1, _ = validate_agent(root1, "agents/a.ajml", project)
        agent2, _ = validate_agent(root2, "agents/b.ajml", project)
        agents = {"a": agent1, "b": agent2}
        with pytest.raises(AJMLCompilationError, match="E403"):
            check_duplicate_agent_names(agents)

    def test_circular_dependencies(self):
        raw_a = '''<?xml version="1.0"?>
<agent name="agent_a">
    <state><field name="x" type="string" default="" /></state>
    <graph>
        <node id="sub" type="subgraph" agent_ref="agent_b">
            <input_map /><output_map />
        </node>
        <edge source="__START__" target="sub" />
        <edge source="sub" target="__END__" />
    </graph>
</agent>'''
        raw_b = '''<?xml version="1.0"?>
<agent name="agent_b">
    <state><field name="x" type="string" default="" /></state>
    <graph>
        <node id="sub" type="subgraph" agent_ref="agent_a">
            <input_map /><output_map />
        </node>
        <edge source="__START__" target="sub" />
        <edge source="sub" target="__END__" />
    </graph>
</agent>'''
        project = _make_project()
        root_a = preprocess(raw_a)
        root_b = preprocess(raw_b)
        # Skip subgraph ref validation since we're testing circular deps
        agent_a, _ = validate_agent(root_a, "agents/a.ajml", project)
        agent_b, _ = validate_agent(root_b, "agents/b.ajml", project)
        agents = {"agent_a": agent_a, "agent_b": agent_b}
        with pytest.raises(AJMLCompilationError, match="E402"):
            check_circular_dependencies(agents)


class TestUnreferencedFieldWarnings:
    """Test W302 warnings for state fields never referenced in LLM prompts."""

    def test_unreferenced_field_warns(self):
        raw = '''<?xml version="1.0"?>
<agent name="test_agent">
    <state>
        <field name="user_input" type="string" required="true" />
        <field name="response" type="string" default="" />
    </state>
    <graph>
        <node id="respond" type="llm">
            <system_prompt>You are a helpful assistant.</system_prompt>
            <output_schema>
                <field name="response" type="string" description="Your response" />
            </output_schema>
        </node>
        <edge source="__START__" target="respond" />
        <edge source="respond" target="__END__" />
    </graph>
</agent>'''
        agent, warnings = _parse_agent(raw)
        w302 = [w for w in warnings if w.code == "W302"]
        assert len(w302) == 1
        assert "user_input" in w302[0].message

    def test_referenced_field_no_warning(self):
        raw = '''<?xml version="1.0"?>
<agent name="test_agent">
    <state>
        <field name="user_input" type="string" required="true" />
        <field name="response" type="string" default="" />
    </state>
    <graph>
        <node id="respond" type="llm">
            <system_prompt>Respond to: ${user_input}</system_prompt>
            <output_schema>
                <field name="response" type="string" description="Your response" />
            </output_schema>
        </node>
        <edge source="__START__" target="respond" />
        <edge source="respond" target="__END__" />
    </graph>
</agent>'''
        agent, warnings = _parse_agent(raw)
        w302 = [w for w in warnings if w.code == "W302"]
        assert len(w302) == 0
