"""Tests for the AJML Output Assembler (Phase 4)."""

import pytest

from ajml.assembler import (
    generate_env_example,
    generate_main_py,
    generate_requirements_txt,
)
from ajml.preprocessor import preprocess
from ajml.validator import validate_agent, validate_project


def _make_project_and_agent():
    """Create a project and a simple agent for testing."""
    proj_raw = '''<?xml version="1.0"?>
<project name="test_project" ajml_version="2.0">
    <config>
        <llm provider="openai" model="gpt-4o" />
        <server cors_origins="https://example.com" auth_env="MY_API_KEY" />
        <env>
            <var name="OPENAI_API_KEY" required="true" />
            <var name="LOG_LEVEL" default="INFO" />
        </env>
    </config>
</project>'''
    proj_root = preprocess(proj_raw)
    project = validate_project(proj_root, "_project.ajml")

    agent_raw = '''<?xml version="1.0"?>
<agent name="my_agent" version="1.0" description="Test agent.">
    <state>
        <field name="input_text" type="string" required="true" />
        <field name="result" type="string" default="" />
        <field name="internal" type="string" default="" expose="false" />
    </state>
    <graph>
        <node id="process" type="llm">
            <system_prompt>Process the input.</system_prompt>
            <output_schema>
                <field name="result" type="string" description="The result" />
            </output_schema>
        </node>
        <edge source="__START__" target="process" />
        <edge source="process" target="__END__" />
    </graph>
</agent>'''
    agent_root = preprocess(agent_raw)
    agent, _ = validate_agent(agent_root, "agents/my_agent.ajml", project)

    return project, {"my_agent": agent}


class TestGenerateMainPy:
    """Test main.py generation."""

    def test_contains_fastapi_app(self):
        project, agents = _make_project_and_agent()
        code = generate_main_py(project, agents)
        assert 'app = FastAPI(title="test_project")' in code

    def test_contains_health_endpoint(self):
        project, agents = _make_project_and_agent()
        code = generate_main_py(project, agents)
        assert '@app.get("/health")' in code
        assert '{"status": "ok"}' in code

    def test_contains_agent_endpoint(self):
        project, agents = _make_project_and_agent()
        code = generate_main_py(project, agents)
        assert '@app.post("/run/my_agent"' in code
        assert "async def run_my_agent" in code

    def test_contains_cors_middleware(self):
        project, agents = _make_project_and_agent()
        code = generate_main_py(project, agents)
        assert "CORSMiddleware" in code
        assert '"https://example.com"' in code

    def test_contains_auth(self):
        project, agents = _make_project_and_agent()
        code = generate_main_py(project, agents)
        assert "verify_api_key" in code
        assert "X-API-Key" in code
        assert 'API_KEY = os.getenv("MY_API_KEY"' in code

    def test_contains_request_model(self):
        project, agents = _make_project_and_agent()
        code = generate_main_py(project, agents)
        assert "class MyAgentRequest(BaseModel):" in code
        assert "input_text: str" in code

    def test_contains_response_model(self):
        project, agents = _make_project_and_agent()
        code = generate_main_py(project, agents)
        assert "class MyAgentResponse(BaseModel):" in code
        assert "result" in code
        # internal field with expose=false should NOT be in response
        # (it's filtered out in endpoint logic)

    def test_contains_settings(self):
        project, agents = _make_project_and_agent()
        code = generate_main_py(project, agents)
        assert "class Settings(BaseSettings):" in code
        assert "OPENAI_API_KEY: str" in code
        assert 'LOG_LEVEL: str = "INFO"' in code

    def test_no_auth_when_not_configured(self):
        proj_raw = '''<?xml version="1.0"?>
<project name="noauth" ajml_version="2.0">
    <config><llm provider="openai" model="gpt-4o" /></config>
</project>'''
        proj_root = preprocess(proj_raw)
        project = validate_project(proj_root, "_project.ajml")

        agent_raw = '''<?xml version="1.0"?>
<agent name="simple">
    <state><field name="x" type="string" default="" /></state>
    <graph>
        <node id="a" type="llm" />
        <edge source="__START__" target="a" />
        <edge source="a" target="__END__" />
    </graph>
</agent>'''
        agent_root = preprocess(agent_raw)
        agent, _ = validate_agent(agent_root, "test.ajml", project)

        code = generate_main_py(project, {"simple": agent})
        assert "verify_api_key" not in code


class TestGenerateRequirementsTxt:
    """Test requirements.txt generation."""

    def test_core_packages(self):
        project, agents = _make_project_and_agent()
        reqs = generate_requirements_txt(project, agents)
        assert "fastapi" in reqs
        assert "uvicorn" in reqs
        assert "pydantic>=" in reqs
        assert "langgraph" in reqs
        assert "langchain-core" in reqs

    def test_provider_package(self):
        project, agents = _make_project_and_agent()
        reqs = generate_requirements_txt(project, agents)
        assert "langchain-openai" in reqs


class TestGenerateEnvExample:
    """Test .env.example generation."""

    def test_contains_env_vars(self):
        project, agents = _make_project_and_agent()
        env = generate_env_example(project)
        assert "MY_API_KEY=your-api-key-here" in env
        assert "OPENAI_API_KEY=" in env
        assert "LOG_LEVEL=INFO" in env
