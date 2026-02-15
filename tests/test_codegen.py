"""Tests for the AJML Code Generator (Phase 3)."""

import pytest

from ajml.codegen import generate_agent_code
from ajml.preprocessor import preprocess
from ajml.validator import validate_agent, validate_project


def _make_project(provider="openai", model="gpt-4o"):
    raw = f'''<?xml version="1.0" encoding="UTF-8"?>
<project name="test" ajml_version="2.0">
    <config>
        <llm provider="{provider}" model="{model}" />
    </config>
</project>'''
    root = preprocess(raw)
    return validate_project(root, "agents/_project.ajml")


class TestCodeGeneration:
    """Test Python code generation from AJML agents."""

    def test_generates_state_typeddict(self):
        raw = '''<?xml version="1.0"?>
<agent name="test_agent">
    <state>
        <field name="input_text" type="string" required="true" />
        <field name="count" type="int" default="0" />
        <field name="items" type="list" reducer="append" />
    </state>
    <graph>
        <node id="process" type="llm" />
        <edge source="__START__" target="process" />
        <edge source="process" target="__END__" />
    </graph>
</agent>'''
        project = _make_project()
        root = preprocess(raw)
        agent, _ = validate_agent(root, "test.ajml", project)
        code = generate_agent_code(agent, project)

        assert "class AgentState(TypedDict):" in code
        assert "messages: Annotated[list, add_messages]" in code
        assert "input_text: str" in code
        assert "count: int" in code
        assert "items: Annotated[list, operator.add]" in code

    def test_generates_llm_node_with_structured_output(self):
        raw = '''<?xml version="1.0"?>
<agent name="test_agent">
    <state>
        <field name="intent" type="string" default="" />
    </state>
    <graph>
        <node id="classify" type="llm">
            <system_prompt>Classify the intent.</system_prompt>
            <output_schema>
                <field name="intent" type="string" description="The intent" />
            </output_schema>
        </node>
        <edge source="__START__" target="classify" />
        <edge source="classify" target="__END__" />
    </graph>
</agent>'''
        project = _make_project()
        root = preprocess(raw)
        agent, _ = validate_agent(root, "test.ajml", project)
        code = generate_agent_code(agent, project)

        assert "class ClassifyOutput(BaseModel):" in code
        assert 'intent: str = Field(description="The intent")' in code
        assert "def classify(state: AgentState):" in code
        assert "structured_llm = llm.with_structured_output(ClassifyOutput)" in code

    def test_generates_plain_llm_node(self):
        raw = '''<?xml version="1.0"?>
<agent name="test_agent">
    <state>
        <field name="x" type="string" default="" />
    </state>
    <graph>
        <node id="chat" type="llm">
            <system_prompt>You are helpful.</system_prompt>
        </node>
        <edge source="__START__" target="chat" />
        <edge source="chat" target="__END__" />
    </graph>
</agent>'''
        project = _make_project()
        root = preprocess(raw)
        agent, _ = validate_agent(root, "test.ajml", project)
        code = generate_agent_code(agent, project)

        assert "def chat(state: AgentState):" in code
        assert "response = llm.invoke(messages)" in code

    def test_generates_conditional_routing(self):
        raw = '''<?xml version="1.0"?>
<agent name="test_agent">
    <state>
        <field name="x" type="int" default="0" />
    </state>
    <graph>
        <node id="check" type="llm" />
        <node id="yes" type="llm" />
        <node id="no" type="llm" />
        <edge source="__START__" target="check" />
        <edge source="check" target="yes">
            <condition>state.get('x') == 1</condition>
        </edge>
        <edge source="check" target="no" default="true" />
        <edge source="yes" target="__END__" />
        <edge source="no" target="__END__" />
    </graph>
</agent>'''
        project = _make_project()
        root = preprocess(raw)
        agent, _ = validate_agent(root, "test.ajml", project)
        code = generate_agent_code(agent, project)

        assert "def route_check(state: AgentState):" in code
        assert "state.get('x') == 1" in code
        assert 'return "yes"' in code
        assert 'return "no"' in code

    def test_generates_graph_assembly(self):
        raw = '''<?xml version="1.0"?>
<agent name="test_agent">
    <state>
        <field name="x" type="string" default="" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <node id="b" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="b" />
        <edge source="b" target="__END__" />
    </graph>
</agent>'''
        project = _make_project()
        root = preprocess(raw)
        agent, _ = validate_agent(root, "test.ajml", project)
        code = generate_agent_code(agent, project)

        assert "graph_builder = StateGraph(AgentState)" in code
        assert 'graph_builder.add_node("a", a)' in code
        assert 'graph_builder.add_node("b", b)' in code
        assert "graph_builder.add_edge(START, " in code
        assert "graph = graph_builder.compile()" in code

    def test_generates_llm_init(self):
        project = _make_project("openai", "gpt-4o")
        raw = '''<?xml version="1.0"?>
<agent name="test_agent">
    <state><field name="x" type="string" default="" /></state>
    <graph>
        <node id="a" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="__END__" />
    </graph>
</agent>'''
        root = preprocess(raw)
        agent, _ = validate_agent(root, "test.ajml", project)
        code = generate_agent_code(agent, project)

        assert 'llm = ChatOpenAI(model="gpt-4o", max_retries=2)' in code
        assert "from langchain_openai import ChatOpenAI" in code

    def test_agent_override_provider(self):
        project = _make_project("openai", "gpt-4o")
        raw = '''<?xml version="1.0"?>
<agent name="test_agent">
    <config>
        <llm provider="anthropic" model="claude-sonnet-4-20250514" max_retries="5" />
    </config>
    <state><field name="x" type="string" default="" /></state>
    <graph>
        <node id="a" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="__END__" />
    </graph>
</agent>'''
        root = preprocess(raw)
        agent, _ = validate_agent(root, "test.ajml", project)
        code = generate_agent_code(agent, project)

        assert 'llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_retries=5)' in code

    def test_generates_parallel_edges(self):
        raw = '''<?xml version="1.0"?>
<agent name="test_agent">
    <state>
        <field name="a_result" type="string" default="" />
        <field name="b_result" type="string" default="" />
    </state>
    <graph>
        <node id="task_a" type="llm" />
        <node id="task_b" type="llm" />
        <node id="merge" type="llm" />
        <edge source="__START__" target="task_a" />
        <edge source="__START__" target="task_b" />
        <edge source="task_a" target="merge" />
        <edge source="task_b" target="merge" />
        <edge source="merge" target="__END__" />
    </graph>
</agent>'''
        project = _make_project()
        root = preprocess(raw)
        agent, _ = validate_agent(root, "test.ajml", project)
        code = generate_agent_code(agent, project)

        assert 'graph_builder.add_edge(START, "task_a")' in code
        assert 'graph_builder.add_edge(START, "task_b")' in code

    def test_generates_merge_reducer(self):
        raw = '''<?xml version="1.0"?>
<agent name="test_agent">
    <state>
        <field name="data" type="dict" reducer="merge" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="__END__" />
    </graph>
</agent>'''
        project = _make_project()
        root = preprocess(raw)
        agent, _ = validate_agent(root, "test.ajml", project)
        code = generate_agent_code(agent, project)

        assert "data: Annotated[dict, lambda a, b: {**a, **b}]" in code

    def test_generates_concat_reducer(self):
        raw = '''<?xml version="1.0"?>
<agent name="test_agent">
    <state>
        <field name="log" type="string" reducer="concat" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="__END__" />
    </graph>
</agent>'''
        project = _make_project()
        root = preprocess(raw)
        agent, _ = validate_agent(root, "test.ajml", project)
        code = generate_agent_code(agent, project)

        assert "log: Annotated[str, operator.concat]" in code

    def test_generated_code_is_valid_python(self):
        """Verify generated code can be parsed as valid Python."""
        raw = '''<?xml version="1.0"?>
<agent name="test_agent">
    <state>
        <field name="input_text" type="string" required="true" />
        <field name="result" type="string" default="" />
        <field name="count" type="int" default="0" reducer="add" />
    </state>
    <graph>
        <node id="process" type="llm">
            <system_prompt>Process: ${input_text}</system_prompt>
            <output_schema>
                <field name="result" type="string" description="Result" />
            </output_schema>
        </node>
        <edge source="__START__" target="process" />
        <edge source="process" target="__END__" />
    </graph>
</agent>'''
        project = _make_project()
        root = preprocess(raw)
        agent, _ = validate_agent(root, "test.ajml", project)
        code = generate_agent_code(agent, project)

        # This should not raise
        compile(code, "<test>", "exec")
