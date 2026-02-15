"""Tests for the AJML lexical pre-processor (Phase 1)."""

import pytest

from ajml.preprocessor import count_sanitised_conditions, preprocess, unescape_content


class TestPreprocess:
    """Test the pre-processing of raw AJML text."""

    def test_basic_agent_parses(self):
        raw = '''<?xml version="1.0" encoding="UTF-8"?>
<agent name="test" version="1.0">
    <state>
        <field name="x" type="string" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="__END__" />
    </graph>
</agent>'''
        root = preprocess(raw)
        assert root.tag == "agent"
        assert root.get("name") == "test"

    def test_condition_with_comparison_operators(self):
        raw = '''<?xml version="1.0" encoding="UTF-8"?>
<agent name="test" version="1.0">
    <state>
        <field name="x" type="int" default="0" />
    </state>
    <graph>
        <node id="a" type="llm" />
        <node id="b" type="llm" />
        <edge source="a" target="b">
            <condition>state.get('x') > 5 and state.get('x') < 10</condition>
        </edge>
        <edge source="__START__" target="a" />
        <edge source="b" target="__END__" />
    </graph>
</agent>'''
        root = preprocess(raw)
        graph = root.find("graph")
        edges = graph.findall("edge")
        cond_edge = [e for e in edges if e.find("condition") is not None][0]
        cond_text = cond_edge.find("condition").text
        # ElementTree un-escapes entities on parse, so we get the original text back
        # The key test is that this parses at all — raw XML would choke on < and >
        assert ">" in cond_text and "<" in cond_text

    def test_system_prompt_with_angle_brackets(self):
        raw = '''<?xml version="1.0" encoding="UTF-8"?>
<agent name="test" version="1.0">
    <state>
        <field name="x" type="string" default="" />
    </state>
    <graph>
        <node id="a" type="llm">
            <system_prompt>If the refund amount > 100, set flag.</system_prompt>
        </node>
        <edge source="__START__" target="a" />
        <edge source="a" target="__END__" />
    </graph>
</agent>'''
        root = preprocess(raw)
        node = root.find(".//node")
        prompt = node.find("system_prompt")
        assert prompt is not None
        assert prompt.text is not None

    def test_project_file_parses(self):
        raw = '''<?xml version="1.0" encoding="UTF-8"?>
<project name="test_project" ajml_version="2.0">
    <config>
        <llm provider="openai" model="gpt-4o" />
    </config>
</project>'''
        root = preprocess(raw)
        assert root.tag == "project"
        assert root.get("name") == "test_project"


class TestUnescapeContent:
    """Test un-escaping of pre-processed content."""

    def test_unescape_lt(self):
        assert unescape_content("a &lt; b") == "a < b"

    def test_unescape_gt(self):
        assert unescape_content("a &gt; b") == "a > b"

    def test_unescape_amp(self):
        assert unescape_content("a &amp; b") == "a & b"

    def test_unescape_all(self):
        assert unescape_content("x &lt; 5 &amp;&amp; x &gt; 0") == "x < 5 && x > 0"

    def test_plain_text_unchanged(self):
        assert unescape_content("hello world") == "hello world"


class TestCountSanitisedConditions:
    """Test counting of conditions that needed sanitisation."""

    def test_no_conditions(self):
        assert count_sanitised_conditions("<agent></agent>") == 0

    def test_condition_without_operators(self):
        raw = '<condition>state.get("x") == 5</condition>'
        assert count_sanitised_conditions(raw) == 0

    def test_condition_with_lt(self):
        raw = "<condition>state.get('x') < 5</condition>"
        assert count_sanitised_conditions(raw) == 1

    def test_multiple_conditions(self):
        raw = """
        <condition>state.get('x') < 5</condition>
        <condition>state.get('y') > 10</condition>
        <condition>state.get('z') == 0</condition>
        """
        assert count_sanitised_conditions(raw) == 2
