"""Phase 3: Code Generator.

Emits Python source code from validated AJML ASTs.
"""

import re
from typing import Any

from .validator import AgentAST, ProjectAST

# --- Type mapping ---
TYPE_MAP = {
    "string": "str",
    "int": "int",
    "float": "float",
    "bool": "bool",
    "list": "list",
    "dict": "dict",
    "enum": "str",
    "list[string]": "list[str]",
    "list[int]": "list[int]",
    "list[float]": "list[float]",
    "list[dict]": "list[dict]",
    "dict[string]": "dict[str, str]",
    "dict[int]": "dict[str, int]",
    "dict[any]": "dict",
}

# Provider to LangChain class mapping
PROVIDER_MAP = {
    "openai": ("ChatOpenAI", "langchain_openai"),
    "anthropic": ("ChatAnthropic", "langchain_anthropic"),
    "google": ("ChatGoogleGenerativeAI", "langchain_google_genai"),
    "mistral": ("ChatMistralAI", "langchain_mistralai"),
    "groq": ("ChatGroq", "langchain_groq"),
    "ollama": ("ChatOllama", "langchain_ollama"),
    "azure_openai": ("AzureChatOpenAI", "langchain_openai"),
    "bedrock": ("ChatBedrock", "langchain_aws"),
}


def _to_pascal_case(snake_str: str) -> str:
    """Convert snake_case to PascalCase."""
    return "".join(word.capitalize() for word in snake_str.split("_"))


def _interpolate_prompt(prompt: str) -> str:
    """Convert ${field_name} to Python f-string interpolation for state fields.

    Returns a list of f-string lines.
    """
    if not prompt:
        return '""'

    lines = prompt.split("\n")
    result_parts = []
    for line in lines:
        # Replace ${field_name} with {state.get('field_name', '')}
        converted = re.sub(
            r"\$\{(\w+)\}",
            lambda m: "{state.get('" + m.group(1) + "', '')}",
            line,
        )
        result_parts.append(converted)

    joined = "\\n".join(result_parts)
    return f'f"{joined}"'


def _interpolate_prompt_multiline(prompt: str) -> str:
    """Convert a system prompt to Python code that builds the string."""
    if not prompt:
        return '    system_content = ""\n'

    lines = prompt.split("\n")
    parts = []
    for i, line in enumerate(lines):
        converted = re.sub(
            r"\$\{(\w+)\}",
            lambda m: "{state.get('" + m.group(1) + "', '')}",
            line,
        )
        # Escape any double quotes in the line
        converted = converted.replace('"', '\\"')
        parts.append(converted)

    if len(parts) == 1:
        return f'    system_content = f"{parts[0]}"\n'

    result = "    system_content = (\n"
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            result += f'        f"{part}\\n"\n'
        else:
            result += f'        f"{part}"\n'
    result += "    )\n"
    return result


def generate_agent_code(
    agent: AgentAST,
    project: ProjectAST,
    all_agents: dict[str, AgentAST] | None = None,
) -> str:
    """Generate Python code for a single AJML agent.

    Returns:
        Complete Python source code as a string.
    """
    all_agents = all_agents or {}
    lines: list[str] = []

    # Determine LLM config (agent override or project default)
    provider = project.llm_provider
    model = project.llm_model
    max_retries = project.llm_max_retries
    if agent.config:
        provider = agent.config.get("llm_provider", provider)
        model = agent.config.get("llm_model", model)
        max_retries = agent.config.get("llm_max_retries", max_retries)

    # Collect what we need to import
    needs_operator = any(
        f["reducer"] in ("append", "add") for f in agent.state_fields
    )
    needs_concat = any(f["reducer"] == "concat" for f in agent.state_fields)
    needs_structured_output = any(
        n["type"] == "llm" and n.get("output_schema") for n in agent.nodes
    )
    needs_tool_node = any(
        n["type"] == "llm" and n.get("tool_binds") for n in agent.nodes
    )
    needs_httpx = any(t["type"] == "api_call" for t in agent.tools)
    needs_tenacity = any(
        t["type"] == "api_call" and t.get("max_retries", 0) > 0
        for t in agent.tools
    )
    needs_send = any(e["type"] == "map" for e in agent.edges)
    has_api_tools = any(t["type"] == "api_call" for t in agent.tools)
    has_any_tools = len(agent.tools) > 0

    # --- Imports ---
    imports = []
    if needs_operator or needs_concat:
        imports.append("import operator")
    if needs_httpx or has_api_tools:
        imports.append("import logging")
        imports.append("import os")
    if needs_httpx:
        imports.append("")
        imports.append("import httpx")

    imports.append("")
    imports.append("from typing import Annotated, TypedDict")

    # LangChain/LangGraph imports
    lc_imports = set()
    lc_imports.add("from langchain_core.messages import AIMessage, SystemMessage")
    if has_any_tools:
        lc_imports.add("from langchain_core.tools import tool")
    if needs_structured_output:
        lc_imports.add("from pydantic import BaseModel, Field")
    elif has_api_tools:
        lc_imports.add("from pydantic import BaseModel, Field")
    if needs_tool_node:
        lc_imports.add("from langgraph.prebuilt import ToolNode")
    if needs_send:
        lc_imports.add("from langgraph.constants import Send")
    if needs_tenacity:
        tenacity_parts = ["retry", "stop_after_attempt"]
        # Check backoff strategies
        for t in agent.tools:
            if t["type"] == "api_call" and t.get("max_retries", 0) > 0:
                if t.get("backoff", "exponential") == "exponential":
                    if "wait_exponential" not in tenacity_parts:
                        tenacity_parts.append("wait_exponential")
                else:
                    if "wait_fixed" not in tenacity_parts:
                        tenacity_parts.append("wait_fixed")
                tenacity_parts.append("retry_if_exception_type")
        tenacity_parts = sorted(set(tenacity_parts))
        lc_imports.add(
            f"from tenacity import {', '.join(tenacity_parts)}"
        )

    lc_imports.add("from langgraph.graph import END, START, StateGraph")
    lc_imports.add("from langgraph.graph.message import add_messages")

    # Provider import
    if provider in PROVIDER_MAP:
        cls_name, pkg_name = PROVIDER_MAP[provider]
        lc_imports.add(f"from {pkg_name} import {cls_name}")

    imports.extend(sorted(lc_imports))

    lines.extend(imports)
    lines.append("")

    if has_api_tools:
        lines.append("")
        lines.append("logger = logging.getLogger(__name__)")

    lines.append("")
    lines.append("")

    # --- ToolExecutionError ---
    if has_api_tools:
        lines.append("class ToolExecutionError(Exception):")
        lines.append("    pass")
        lines.append("")
        lines.append("")

    # --- State TypedDict ---
    lines.append("class AgentState(TypedDict):")
    lines.append("    messages: Annotated[list, add_messages]")
    for field in agent.state_fields:
        py_type = TYPE_MAP.get(field["type"], "str")
        reducer = field["reducer"]
        if reducer == "overwrite":
            lines.append(f"    {field['name']}: {py_type}")
        elif reducer == "append":
            lines.append(f"    {field['name']}: Annotated[{py_type}, operator.add]")
        elif reducer == "add":
            lines.append(f"    {field['name']}: Annotated[{py_type}, operator.add]")
        elif reducer == "merge":
            lines.append(f"    {field['name']}: Annotated[{py_type}, lambda a, b: {{**a, **b}}]")
        elif reducer == "concat":
            lines.append(f"    {field['name']}: Annotated[{py_type}, operator.concat]")
    lines.append("")
    lines.append("")

    # --- LLM initialization ---
    if provider in PROVIDER_MAP:
        cls_name, _ = PROVIDER_MAP[provider]
        lines.append(f'llm = {cls_name}(model="{model}", max_retries={max_retries})')
    else:
        lines.append(f'# LLM provider "{provider}" — configure manually')
        lines.append("llm = None")
    lines.append("")
    lines.append("")

    # --- Tool definitions ---
    for tool_def in agent.tools:
        lines.extend(_generate_tool(tool_def))
        lines.append("")
        lines.append("")

    # --- Node functions ---
    for node in agent.nodes:
        lines.extend(_generate_node(node, agent))
        lines.append("")
        lines.append("")

    # --- Routing functions ---
    lines.extend(_generate_routing_functions(agent))

    # --- Graph assembly ---
    lines.append("graph_builder = StateGraph(AgentState)")
    lines.append("")

    # Add nodes
    lines.append("# Add all nodes")
    for node in agent.nodes:
        node_id = node["id"]
        lines.append(f'graph_builder.add_node("{node_id}", {node_id})')
        # If LLM node with tool binds, also add tool node
        if node["type"] == "llm" and node.get("tool_binds"):
            lines.append(f'graph_builder.add_node("{node_id}_tools", {node_id}_tools)')
    lines.append("")

    # Add edges
    lines.append("# Add edges")
    edge_groups: dict[str, list[dict]] = {}
    for edge in agent.edges:
        source = edge["source"]
        if source not in edge_groups:
            edge_groups[source] = []
        edge_groups[source].append(edge)

    for source, edges in edge_groups.items():
        source_py = "START" if source == "__START__" else f'"{source}"'

        # Check if this is a conditional group
        has_conditions = any(e["condition"] for e in edges)
        has_map = any(e["type"] == "map" for e in edges)

        if has_conditions:
            lines.append(f"graph_builder.add_conditional_edges({source_py}, route_{source.replace('__START__', 'start')})")
        elif has_map:
            map_edge = edges[0]
            target = map_edge["target"]
            lines.append(
                f'graph_builder.add_conditional_edges({source_py}, route_{source}, ["{target}"])'
            )
        else:
            for edge in edges:
                target = edge["target"]
                target_py = "END" if target == "__END__" else f'"{target}"'
                lines.append(f"graph_builder.add_edge({source_py}, {target_py})")

    # Add tool-calling loop edges for LLM nodes with tool binds
    for node in agent.nodes:
        if node["type"] == "llm" and node.get("tool_binds"):
            node_id = node["id"]
            lines.append(
                f'graph_builder.add_conditional_edges("{node_id}", route_{node_id})'
            )
            lines.append(
                f'graph_builder.add_edge("{node_id}_tools", "{node_id}")'
            )

    lines.append("")
    lines.append("# Compile")
    lines.append("graph = graph_builder.compile()")
    lines.append("")

    return "\n".join(lines)


def _generate_tool(tool_def: dict[str, Any]) -> list[str]:
    """Generate code for a tool definition."""
    lines: list[str] = []
    tool_type = tool_def["type"]

    if tool_type == "api_call":
        lines.extend(_generate_api_tool(tool_def))
    elif tool_type == "local_script":
        lines.extend(_generate_local_script_tool(tool_def))
    elif tool_type == "script_tool":
        lines.extend(_generate_script_tool(tool_def))

    return lines


def _generate_api_tool(tool_def: dict[str, Any]) -> list[str]:
    """Generate code for an api_call tool."""
    lines: list[str] = []
    tool_id = tool_def["id"]
    pascal_name = _to_pascal_case(tool_id)

    # Input model
    lines.append(f"class {pascal_name}Input(BaseModel):")
    params = tool_def.get("parameters", [])
    if params:
        for param in params:
            py_type = TYPE_MAP.get(param["type"], "str")
            desc = param.get("description", "")
            if desc:
                lines.append(f'    {param["name"]}: {py_type} = Field(description="{desc}")')
            else:
                lines.append(f'    {param["name"]}: {py_type}')
    else:
        lines.append("    pass")
    lines.append("")
    lines.append("")

    # Tool function
    max_retries = tool_def.get("max_retries", 0)
    timeout = tool_def.get("timeout", 30.0)
    retry_codes_str = tool_def.get("retry_status_codes", "429, 500, 502, 503, 504")
    retry_codes = [int(c.strip()) for c in retry_codes_str.split(",")]
    backoff = tool_def.get("backoff", "exponential")
    backoff_base = tool_def.get("backoff_base", 1.0)

    if max_retries > 0:
        lines.append(f'@tool("{tool_id}", args_schema={pascal_name}Input)')
        lines.append("@retry(")
        lines.append(f"    stop=stop_after_attempt({max_retries}),")
        if backoff == "exponential":
            lines.append(f"    wait=wait_exponential(multiplier={backoff_base}, min={backoff_base}, max=60.0),")
        else:
            lines.append(f"    wait=wait_fixed({backoff_base}),")
        lines.append("    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),")
        lines.append(")")
    else:
        lines.append(f'@tool("{tool_id}", args_schema={pascal_name}Input)')

    # Function signature
    param_strs = []
    for param in params:
        py_type = TYPE_MAP.get(param["type"], "str")
        param_strs.append(f'{param["name"]}: {py_type}')
    lines.append(f'def {tool_id}({", ".join(param_strs)}) -> dict:')

    # Docstring
    desc = tool_def.get("description", f"Calls the {tool_id} API.")
    lines.append(f'    """{desc}"""')

    # Build URL
    url = tool_def.get("url", "")
    # Replace ${param} with f-string
    url_py = re.sub(r"\$\{env:(\w+)\}", lambda m: "{os.getenv('" + m.group(1) + "')}", url)
    # Map path parameters
    for param in params:
        if param.get("in") == "path":
            map_to = param.get("map_to", param["name"])
            url_py = url_py.replace(f"${{{map_to}}}", f"{{{param['name']}}}")
    lines.append(f'    url = f"{url_py}"')

    # Headers
    headers = tool_def.get("headers", [])
    if headers:
        lines.append("    headers = {")
        for h in headers:
            val = h["value"]
            # Replace ${env:VAR} in header values
            val_py = re.sub(
                r"\$\{env:(\w+)\}",
                lambda m: "{os.getenv('" + m.group(1) + "')}",
                val,
            )
            lines.append(f'        "{h["name"]}": f"{val_py}",')
        lines.append("    }")

    # Query params
    query_params = [p for p in params if p.get("in") == "query"]
    if query_params:
        lines.append("    params = {")
        for qp in query_params:
            lines.append(f'        "{qp.get("map_to", qp["name"])}": {qp["name"]},')
        lines.append("    }")

    # Body
    body = tool_def.get("body")
    if body and body["fields"]:
        lines.append("    json_body = {")
        for bf in body["fields"]:
            lines.append(f'        "{bf["name"]}": {bf["from_state"]},')
        lines.append("    }")

    # Make the request
    method = tool_def.get("method", "GET").lower()
    call_args = ["url"]
    if headers:
        call_args.append("headers=headers")
    if query_params:
        call_args.append("params=params")
    if body and body["fields"]:
        call_args.append("json=json_body")
    call_args.append(f"timeout={timeout}")

    lines.append(f'    response = httpx.{method}({", ".join(call_args)})')
    lines.append("")

    # Retry status codes check
    if max_retries > 0:
        codes_tuple = ", ".join(str(c) for c in retry_codes)
        lines.append(f"    # Raise for retryable status codes")
        lines.append(f"    if response.status_code in ({codes_tuple}):")
        lines.append("        response.raise_for_status()")

    lines.append("    # Raise for other non-2xx codes")
    lines.append("    response.raise_for_status()")
    lines.append("")

    # Validate JSON
    lines.append('    content_type = response.headers.get("content-type", "")')
    lines.append('    if "application/json" not in content_type:')
    lines.append("        raise ToolExecutionError(")
    lines.append('            f"Expected JSON response, got {content_type}: "')
    lines.append('            f"{response.text[:200]}"')
    lines.append("        )")
    lines.append("")
    lines.append("    data = response.json()")
    lines.append("")

    # Map returns
    returns = tool_def.get("returns", [])
    if returns:
        lines.append("    # Map response fields to state")
        for ret in returns:
            api_field = ret["api_field"]
            state_field = ret["state_field"]
            # Handle dot-notation paths
            parts = api_field.split(".")
            if len(parts) == 1:
                lines.append(f'    {state_field} = data.get("{api_field}")')
                lines.append(f'    if {state_field} is None:')
                lines.append(f'        logger.warning("{tool_id}: API field \'{api_field}\' not found in response")')
            else:
                # Nested access
                var_name = state_field
                access_code = "data"
                for part in parts[:-1]:
                    access_code += f'.get("{part}", {{}})'
                access_code += f'.get("{parts[-1]}")'
                lines.append(f'    {var_name} = {access_code}')
                lines.append(f'    if {var_name} is None:')
                lines.append(f'        logger.warning("{tool_id}: API field \'{api_field}\' not found in response")')
        lines.append("")
        result_dict = ", ".join(f'"{r["state_field"]}": {r["state_field"]}' for r in returns)
        lines.append(f"    return {{{result_dict}}}")
    else:
        lines.append("    return data")

    return lines


def _generate_local_script_tool(tool_def: dict[str, Any]) -> list[str]:
    """Generate code for a local_script tool."""
    lines: list[str] = []
    tool_id = tool_def["id"]
    path = tool_def["path"]
    module_name = path.replace(".py", "").replace("/", ".")
    desc = tool_def.get("description", f"Runs {path}")

    lines.append(f"# Tool: {tool_id} (local_script: {path})")
    lines.append(f"import importlib")
    lines.append(f'_{tool_id}_module = importlib.import_module("tools.{module_name}")')
    lines.append(f'_{tool_id}_schema = getattr(_{tool_id}_module, "SCHEMA", {{}})')
    lines.append("")
    lines.append("")

    # Build input model from SCHEMA
    pascal_name = _to_pascal_case(tool_id)
    lines.append(f"class {pascal_name}Input(BaseModel):")
    lines.append(f"    class Config:")
    lines.append(f"        extra = 'allow'")
    lines.append("")
    lines.append("")

    lines.append(f'@tool("{tool_id}", args_schema={pascal_name}Input)')
    lines.append(f"def {tool_id}(**kwargs) -> dict:")
    lines.append(f'    """{desc}"""')
    lines.append(f"    return _{tool_id}_module.run(**kwargs)")

    return lines


def _generate_script_tool(tool_def: dict[str, Any]) -> list[str]:
    """Generate code for a script_tool."""
    lines: list[str] = []
    tool_id = tool_def["id"]
    src = tool_def["src"]
    module_name = src.replace(".py", "").replace("/", ".")
    desc = tool_def.get("description", f"Runs {src}")
    params = tool_def.get("parameters", [])

    lines.append(f"# Tool: {tool_id} (script_tool: {src})")
    lines.append(f"import importlib")
    lines.append(f'_{tool_id}_module = importlib.import_module("tools.{module_name}")')
    lines.append("")

    # Build input model from AJML parameters
    pascal_name = _to_pascal_case(tool_id)
    lines.append(f"class {pascal_name}Input(BaseModel):")
    if params:
        for param in params:
            py_type = TYPE_MAP.get(param["type"], "str")
            p_desc = param.get("description", "")
            if p_desc:
                lines.append(f'    {param["name"]}: {py_type} = Field(description="{p_desc}")')
            else:
                lines.append(f'    {param["name"]}: {py_type}')
    else:
        lines.append("    pass")
    lines.append("")
    lines.append("")

    # Function
    param_strs = []
    for param in params:
        py_type = TYPE_MAP.get(param["type"], "str")
        param_strs.append(f'{param["name"]}: {py_type}')
    sig = ", ".join(param_strs)

    lines.append(f'@tool("{tool_id}", args_schema={pascal_name}Input)')
    lines.append(f"def {tool_id}({sig}) -> dict:")
    lines.append(f'    """{desc}"""')
    if param_strs:
        kw_args = ", ".join(f'{p["name"]}={p["name"]}' for p in params)
        lines.append(f"    return _{tool_id}_module.run({kw_args})")
    else:
        lines.append(f"    return _{tool_id}_module.run()")

    return lines


def _generate_node(node: dict[str, Any], agent: AgentAST) -> list[str]:
    """Generate code for a node function."""
    node_type = node["type"]

    if node_type == "llm":
        return _generate_llm_node(node, agent)
    elif node_type == "action":
        return _generate_action_node(node, agent)
    elif node_type == "script":
        return _generate_script_node(node, agent)
    elif node_type == "subgraph":
        return _generate_subgraph_node(node, agent)
    return []


def _generate_llm_node(node: dict[str, Any], agent: AgentAST) -> list[str]:
    """Generate code for an LLM node."""
    lines: list[str] = []
    node_id = node["id"]
    prompt = node.get("system_prompt", "")
    output_schema = node.get("output_schema", [])
    tool_binds = node.get("tool_binds", [])

    # Output schema model
    if output_schema:
        pascal_name = _to_pascal_case(node_id)
        lines.append(f"class {pascal_name}Output(BaseModel):")
        for field in output_schema:
            field_type = field.get("type", "string")
            py_type = TYPE_MAP.get(field_type, "str")
            # Handle enum type in output_schema
            if field_type == "enum" and field.get("values"):
                enum_values = [v.strip() for v in field["values"].split(",")]
                literal_values = ", ".join(f'"{v}"' for v in enum_values)
                py_type = f"Literal[{literal_values}]"
            desc = field.get("description", "")
            if desc:
                lines.append(f'    {field["name"]}: {py_type} = Field(description="{desc}")')
            else:
                lines.append(f'    {field["name"]}: {py_type}')
        lines.append("")
        lines.append("")

    # Node function
    lines.append(f"def {node_id}(state: AgentState):")

    # Build system prompt
    if prompt:
        lines.append(_interpolate_prompt_multiline(prompt).rstrip("\n"))
    else:
        lines.append('    system_content = ""')

    lines.append('    messages = [SystemMessage(content=system_content)] + state["messages"]')
    lines.append("")

    if output_schema and not tool_binds:
        # Structured output
        pascal_name = _to_pascal_case(node_id)
        lines.append(f"    structured_llm = llm.with_structured_output({pascal_name}Output)")
        lines.append("    result = structured_llm.invoke(messages)")
        lines.append("")
        lines.append("    updates = {}")
        lines.append("    result_dict = result.model_dump()")
        lines.append("    for key, value in result_dict.items():")
        lines.append("        if key in AgentState.__annotations__:")
        lines.append("            updates[key] = value")
        lines.append("")
        lines.append('    updates["messages"] = [AIMessage(content=str(result_dict))]')
        lines.append("    return updates")
    elif tool_binds:
        # Tool-bound LLM
        tools_list = ", ".join(tool_binds)
        lines.append(f"    bound_llm = llm.bind_tools([{tools_list}])")
        lines.append("    response = bound_llm.invoke(messages)")
        lines.append("")
        lines.append('    return {"messages": [response]}')
        lines.append("")
        lines.append("")
        # Tool node
        lines.append(f"{node_id}_tools = ToolNode(")
        lines.append(f"    [{tools_list}],")
        lines.append("    handle_tool_errors=True")
        lines.append(")")
    else:
        # Plain LLM call
        lines.append("    response = llm.invoke(messages)")
        lines.append("")
        lines.append('    return {"messages": [response]}')

    return lines


def _generate_action_node(node: dict[str, Any], agent: AgentAST) -> list[str]:
    """Generate code for an action node."""
    lines: list[str] = []
    node_id = node["id"]
    tool_ref = node["tool_ref"]

    # Find the tool definition to get its parameters
    tool_def = None
    for t in agent.tools:
        if t["id"] == tool_ref:
            tool_def = t
            break

    lines.append(f"def {node_id}(state: AgentState):")

    if tool_def and tool_def.get("parameters"):
        params = tool_def["parameters"]
        invoke_dict = {}
        for param in params:
            state_key = param["name"]
            invoke_dict[state_key] = f"state.get(\"{state_key}\")"
        dict_parts = ", ".join(f'"{k}": {v}' for k, v in invoke_dict.items())
        lines.append(f"    result = {tool_ref}.invoke({{{dict_parts}}})")
    else:
        lines.append(f"    result = {tool_ref}.invoke({{}})")

    lines.append("    # Filter to known state fields only")
    lines.append("    updates = {}")
    lines.append("    for key, value in result.items():")
    lines.append("        if key in AgentState.__annotations__:")
    lines.append("            updates[key] = value")
    lines.append("    return updates")

    return lines


def _generate_script_node(node: dict[str, Any], agent: AgentAST) -> list[str]:
    """Generate code for a script node."""
    lines: list[str] = []
    node_id = node["id"]
    path = node["path"]
    module_name = path.replace(".py", "").replace("/", ".")

    lines.append(f"# Script node: {node_id} ({path})")
    lines.append(f"import importlib")
    lines.append(f'_{node_id}_module = importlib.import_module("tools.{module_name}")')
    lines.append("")
    lines.append("")
    lines.append(f"def {node_id}(state: AgentState):")
    lines.append(f"    result = _{node_id}_module.run(dict(state))")
    lines.append("    # Filter to known state fields only")
    lines.append("    updates = {}")
    lines.append("    for key, value in result.items():")
    lines.append("        if key in AgentState.__annotations__:")
    lines.append("            updates[key] = value")
    lines.append("    return updates")

    return lines


def _generate_subgraph_node(node: dict[str, Any], agent: AgentAST) -> list[str]:
    """Generate code for a subgraph node."""
    lines: list[str] = []
    node_id = node["id"]
    agent_ref = node["agent_ref"]
    input_map = node.get("input_map", [])
    output_map = node.get("output_map", [])

    lines.append(f"# Subgraph node: {node_id} -> {agent_ref}")
    lines.append(f"from compiled_{agent_ref} import graph as {agent_ref}_graph")
    lines.append("")
    lines.append("")
    lines.append(f"def {node_id}(state: AgentState):")

    # Build input state
    lines.append("    child_input = {")
    lines.append('        "messages": state["messages"],')
    for mapping in input_map:
        lines.append(f'        "{mapping["target"]}": state.get("{mapping["source"]}"),')
    lines.append("    }")
    lines.append(f"    child_result = {agent_ref}_graph.invoke(child_input)")
    lines.append("")

    # Map outputs
    lines.append("    updates = {}")
    for mapping in output_map:
        lines.append(f'    updates["{mapping["target"]}"] = child_result.get("{mapping["source"]}")')
    lines.append("    return updates")

    return lines


def _generate_routing_functions(agent: AgentAST) -> list[str]:
    """Generate routing functions for conditional edges and tool-calling loops."""
    lines: list[str] = []

    # Group edges by source
    edge_groups: dict[str, list[dict]] = {}
    for edge in agent.edges:
        source = edge["source"]
        if source not in edge_groups:
            edge_groups[source] = []
        edge_groups[source].append(edge)

    # Conditional routing functions
    for source, edges in edge_groups.items():
        has_conditions = any(e["condition"] for e in edges)
        has_map = any(e["type"] == "map" for e in edges)

        if has_conditions:
            func_name = f"route_{source.replace('__START__', 'start')}"
            lines.append(f"def {func_name}(state: AgentState):")
            for edge in edges:
                if edge["condition"]:
                    lines.append(f"    if {edge['condition']}:")
                    lines.append(f'        return "{edge["target"]}"')
                elif edge["default"]:
                    lines.append(f'    return "{edge["target"]}"')
            lines.append("")
            lines.append("")

        elif has_map:
            map_edge = edges[0]
            mc = map_edge["map_config"]
            items_field = mc["items_field"]
            item_var = mc["item_var"]
            target = map_edge["target"]

            lines.append(f"def route_{source}(state: AgentState):")
            lines.append(f'    items = state.get("{items_field}", [])')
            lines.append("    if not items:")
            lines.append("        return []  # Skip — empty list")
            lines.append("    return [")
            lines.append(f'        Send("{target}", {{**state, "{item_var}": item}})')
            lines.append("        for item in items")
            lines.append("    ]")
            lines.append("")
            lines.append("")

    # Tool-calling loop routing functions
    for node in agent.nodes:
        if node["type"] == "llm" and node.get("tool_binds"):
            node_id = node["id"]
            lines.append(f"def route_{node_id}(state: AgentState):")
            lines.append('    last_message = state["messages"][-1]')
            lines.append("    if hasattr(last_message, \"tool_calls\") and last_message.tool_calls:")
            lines.append(f'        return "{node_id}_tools"')

            # Find where this node's outgoing edges go
            next_targets = []
            for edge in agent.edges:
                if edge["source"] == node_id:
                    next_targets.append(edge["target"])

            if len(next_targets) == 1:
                target = next_targets[0]
                target_py = "__end__" if target == "__END__" else target
                lines.append(f'    return "{target_py}"')
            else:
                lines.append(f'    return "__end__"')
            lines.append("")
            lines.append("")

    return lines
