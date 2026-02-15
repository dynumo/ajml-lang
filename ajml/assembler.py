"""Phase 4: Output Assembly.

Produces the final build output: main.py, requirements.txt, .env.example,
and copies compiled agent files to the build directory.
"""

from .codegen import PROVIDER_MAP
from .validator import AgentAST, ProjectAST


def generate_main_py(project: ProjectAST, agents: dict[str, AgentAST]) -> str:
    """Generate the main.py FastAPI server file."""
    lines: list[str] = []

    # Imports
    lines.append("import logging")
    lines.append("import os")
    lines.append("import uuid")
    lines.append("")
    lines.append("from fastapi import Depends, FastAPI, HTTPException, Request")
    lines.append("from fastapi.middleware.cors import CORSMiddleware")
    lines.append("from pydantic import BaseModel, Field")
    lines.append("from pydantic_settings import BaseSettings")
    lines.append("from typing import Any, Literal, Optional")
    lines.append("")

    # Import compiled agents
    for name in sorted(agents.keys()):
        lines.append(f"from compiled_{name} import AgentState as {_to_pascal(name)}State")
        lines.append(f"from compiled_{name} import graph as {name}_graph")
    lines.append("")
    lines.append("")

    # Settings class
    lines.append("# --- Environment Settings ---")
    lines.append("class Settings(BaseSettings):")
    if project.env_vars:
        for var in project.env_vars:
            if var["required"] and var.get("default") is None:
                lines.append(f'    {var["name"]}: str')
            elif var.get("default") is not None:
                lines.append(f'    {var["name"]}: str = "{var["default"]}"')
            else:
                lines.append(f'    {var["name"]}: str = ""')
    else:
        lines.append("    pass")
    lines.append("")
    lines.append("    class Config:")
    lines.append("        env_file = '.env'")
    lines.append("")
    lines.append("")
    lines.append("settings = Settings()")
    lines.append("")
    lines.append("")

    # Logger
    lines.append('logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))')
    lines.append("logger = logging.getLogger(__name__)")
    lines.append("")
    lines.append("")

    # FastAPI app
    lines.append(f'app = FastAPI(title="{project.name}")')
    lines.append("")

    # CORS
    origins = [o.strip() for o in project.cors_origins.split(",")]
    origins_str = ", ".join(f'"{o}"' for o in origins)
    lines.append("app.add_middleware(")
    lines.append("    CORSMiddleware,")
    lines.append(f"    allow_origins=[{origins_str}],")
    lines.append('    allow_credentials=True,')
    lines.append('    allow_methods=["*"],')
    lines.append('    allow_headers=["*"],')
    lines.append(")")
    lines.append("")
    lines.append("")

    # Auth dependency
    if project.auth_env:
        lines.append("# --- Authentication ---")
        lines.append(f'API_KEY = os.getenv("{project.auth_env}", "")')
        lines.append("")
        lines.append("")
        lines.append("async def verify_api_key(request: Request):")
        lines.append('    api_key = request.headers.get("X-API-Key")')
        lines.append("    if not api_key or api_key != API_KEY:")
        lines.append('        raise HTTPException(status_code=403, detail="Invalid or missing API key")')
        lines.append("")
        lines.append("")

    # Documentation endpoint visibility
    if project.auth_env and not project.docs_public:
        lines.append("# Documentation requires auth")
        lines.append('app.docs_url = "/docs"')
        lines.append('app.redoc_url = "/redoc"')
        lines.append("")
        lines.append("")

    # Health endpoint
    lines.append("# --- Health Check ---")
    lines.append('@app.get("/health")')
    lines.append("async def health():")
    lines.append('    return {"status": "ok"}')
    lines.append("")
    lines.append("")

    # Request/Response models and endpoints for each agent
    for name, agent in sorted(agents.items()):
        lines.extend(_generate_agent_endpoint(name, agent, project))
        lines.append("")
        lines.append("")

    # Uvicorn runner
    lines.append('if __name__ == "__main__":')
    lines.append("    import uvicorn")
    lines.append(f'    uvicorn.run(app, host="{project.host}", port={project.port})')
    lines.append("")

    return "\n".join(lines)


def _generate_agent_endpoint(
    name: str, agent: AgentAST, project: ProjectAST
) -> list[str]:
    """Generate request/response models and endpoint for an agent."""
    lines: list[str] = []
    pascal = _to_pascal(name)

    # Request model
    lines.append(f"# --- Agent: {name} ---")
    lines.append(f"class {pascal}Request(BaseModel):")
    has_fields = False
    for field in agent.state_fields:
        if field["required"]:
            py_type = _get_api_type(field)
            lines.append(f'    {field["name"]}: {py_type}')
            has_fields = True
        elif field["default"] is not None:
            py_type = _get_api_type(field)
            default_val = _get_python_default(field)
            lines.append(f'    {field["name"]}: {py_type} = {default_val}')
            has_fields = True
    if not has_fields:
        lines.append("    pass")
    lines.append("")
    lines.append("")

    # Response model
    lines.append(f"class {pascal}Response(BaseModel):")
    has_response_fields = False
    for field in agent.state_fields:
        if field["expose"]:
            py_type = _get_api_type(field)
            lines.append(f'    {field["name"]}: Optional[{py_type}] = None')
            has_response_fields = True
    lines.append('    messages: list[dict[str, Any]] = []')
    lines.append("")
    lines.append("")

    # Endpoint
    deps = ""
    if project.auth_env:
        deps = ", dependencies=[Depends(verify_api_key)]"

    desc = agent.description or f"Execute the {name} agent."
    lines.append(f'@app.post("/run/{name}"{deps})')
    lines.append(f"async def run_{name}(request: {pascal}Request):")
    lines.append(f'    """{ desc }"""')
    lines.append(f"    request_id = str(uuid.uuid4())[:12]")
    lines.append(f"    try:")
    lines.append(f"        input_state = request.model_dump()")
    lines.append(f'        input_state["messages"] = []')
    lines.append(f"        result = {name}_graph.invoke(input_state)")
    lines.append(f"")
    lines.append(f"        # Serialize messages")
    lines.append(f"        serialized_messages = []")
    lines.append(f'        for msg in result.get("messages", []):')
    lines.append(f"            if hasattr(msg, 'type'):")
    lines.append(f"                role = msg.type")
    lines.append(f'                if role == "human":')
    lines.append(f'                    role = "human"')
    lines.append(f'                elif role == "ai":')
    lines.append(f'                    role = "assistant"')
    lines.append(f'                serialized_messages.append({{"role": role, "content": msg.content}})')
    lines.append(f"            elif isinstance(msg, dict):")
    lines.append(f"                serialized_messages.append(msg)")
    lines.append(f"")
    lines.append(f"        # Build response (only exposed fields)")
    lines.append(f"        response_data = {{}}")

    for field in agent.state_fields:
        if field["expose"]:
            lines.append(f'        response_data["{field["name"]}"] = result.get("{field["name"]}")')
    lines.append(f'        response_data["messages"] = serialized_messages')
    lines.append(f"")
    lines.append(f"        return response_data")
    lines.append(f"")
    lines.append(f"    except Exception as e:")
    lines.append(f"        logger.exception(f\"Agent {name} failed: {{e}}\")")
    lines.append(f'        error_detail = {{"detail": "Agent execution failed", "error_code": "R500", "agent": "{name}", "request_id": f"req_{{request_id}}"}}')
    lines.append(f'        if os.getenv("LOG_LEVEL", "INFO") == "DEBUG":')
    lines.append(f'            error_detail["error"] = str(e)')
    lines.append(f"        raise HTTPException(status_code=500, detail=error_detail)")

    return lines


def generate_requirements_txt(project: ProjectAST, agents: dict[str, AgentAST]) -> str:
    """Generate requirements.txt with all required packages."""
    packages = set()
    packages.add("fastapi>=0.104.0")
    packages.add("uvicorn[standard]>=0.24.0")
    packages.add("pydantic>=2.0.0")
    packages.add("pydantic-settings>=2.0.0")
    packages.add("langgraph>=0.2.0")
    packages.add("langchain-core>=0.2.0")

    # Provider-specific packages
    provider = project.llm_provider
    if provider in PROVIDER_MAP:
        _, pkg = PROVIDER_MAP[provider]
        packages.add(f"{pkg.replace('_', '-')}>=0.1.0")

    # Check agent-level overrides
    for agent in agents.values():
        if agent.config and agent.config.get("llm_provider"):
            p = agent.config["llm_provider"]
            if p in PROVIDER_MAP:
                _, pkg = PROVIDER_MAP[p]
                packages.add(f"{pkg.replace('_', '-')}>=0.1.0")

    # Check for tools needing httpx/tenacity
    for agent in agents.values():
        for tool in agent.tools:
            if tool["type"] == "api_call":
                packages.add("httpx>=0.25.0")
                if tool.get("max_retries", 0) > 0:
                    packages.add("tenacity>=8.2.0")

    return "\n".join(sorted(packages)) + "\n"


def generate_env_example(project: ProjectAST) -> str:
    """Generate .env.example file with all declared environment variables."""
    lines: list[str] = []

    if project.auth_env:
        lines.append(f"# API Authentication")
        lines.append(f"{project.auth_env}=your-api-key-here")
        lines.append("")

    for var in project.env_vars:
        if var.get("default") is not None:
            lines.append(f'{var["name"]}={var["default"]}')
        elif var["required"]:
            lines.append(f'{var["name"]}=  # Required')
        else:
            lines.append(f'{var["name"]}=')

    if not lines:
        lines.append("# No environment variables configured")

    return "\n".join(lines) + "\n"


def _to_pascal(name: str) -> str:
    """Convert snake_case to PascalCase."""
    return "".join(word.capitalize() for word in name.split("_"))


def _get_api_type(field: dict) -> str:
    """Get the Python type for API models (with Literal for enums)."""
    type_map = {
        "string": "str",
        "int": "int",
        "float": "float",
        "bool": "bool",
        "list": "list",
        "dict": "dict",
        "list[string]": "list[str]",
        "list[int]": "list[int]",
        "list[float]": "list[float]",
        "list[dict]": "list[dict]",
        "dict[string]": "dict[str, str]",
        "dict[int]": "dict[str, int]",
        "dict[any]": "dict",
    }

    if field["type"] == "enum" and field.get("values"):
        enum_values = [v.strip() for v in field["values"].split(",")]
        literal_values = ", ".join(f'"{v}"' for v in enum_values)
        return f"Literal[{literal_values}]"

    return type_map.get(field["type"], "str")


def _get_python_default(field: dict) -> str:
    """Get the Python default value for a field."""
    default = field["default"]
    if default is None:
        return "None"

    field_type = field["type"]
    if field_type in ("string", "enum"):
        return f'"{default}"'
    elif field_type in ("int",):
        return str(int(default))
    elif field_type in ("float",):
        return str(float(default))
    elif field_type in ("bool",):
        return "True" if default.lower() == "true" else "False"
    elif field_type in ("list",) or field_type.startswith("list["):
        if default == "[]":
            return "[]"
        return default
    elif field_type in ("dict",) or field_type.startswith("dict["):
        if default == "{}":
            return "{}"
        return default
    return f'"{default}"'
