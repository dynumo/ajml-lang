"""Phase 2: AST Validator.

Validates parsed AJML ElementTree against structural and semantic rules.
"""

import ast
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any

from .errors import (
    CONDITION_ALLOWED_NAMES,
    PARAM_TYPE_BASE,
    RESERVED_WORDS,
    SUPPORTED_PROVIDERS,
    VALID_NODE_TYPES,
    VALID_REDUCERS,
    VALID_TOOL_TYPES,
    VALID_TYPES,
    AJMLCompilationError,
    AJMLWarning,
    E001, E002, E003, E004,
    E101, E102, E103, E104, E105, E106, E107, E108, E109,
    E201, E202,
    E301, E302, E303, E304, E305, E306, E307, E308,
    E309, E310, E311, E312, E313, E314, E315, E316,
    E401, E402, E403,
    E501, E502,
    W301,
)
from .preprocessor import unescape_content


def _base_type(type_str: str) -> str:
    """Get the base type for reducer validation."""
    return PARAM_TYPE_BASE.get(type_str, type_str)


def _is_list_type(type_str: str) -> bool:
    """Check if a type is a list type."""
    return type_str == "list" or type_str.startswith("list[")


def _is_valid_python_identifier(name: str) -> bool:
    """Check if a name is a valid Python identifier matching [a-z_][a-z0-9_]*."""
    return bool(re.match(r"^[a-z_][a-z0-9_]*$", name))


class AgentAST:
    """Parsed representation of a single AJML agent file."""

    def __init__(self, root: ET.Element, filename: str):
        self.root = root
        self.filename = filename
        self.name = root.get("name", "")
        self.version = root.get("version", "1.0")
        self.description = root.get("description", "")

        # Parsed sections
        self.config = None
        self.state_fields: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []


class ProjectAST:
    """Parsed representation of the _project.ajml file."""

    def __init__(self, root: ET.Element, filename: str):
        self.root = root
        self.filename = filename
        self.name = root.get("name", "")
        self.ajml_version = root.get("ajml_version", "")

        # Parsed config
        self.llm_provider = ""
        self.llm_model = ""
        self.llm_max_retries = 2
        self.cors_origins = "*"
        self.auth_env = ""
        self.docs_public = None  # Will be derived
        self.port = 8000
        self.host = "0.0.0.0"
        self.env_vars: list[dict[str, Any]] = []


def validate_project(root: ET.Element, filename: str) -> ProjectAST:
    """Validate and parse a _project.ajml file."""
    if root.tag != "project":
        raise AJMLCompilationError(
            E001,
            "Root element must be `<project>` for _project.ajml.",
            filename,
        )

    project = ProjectAST(root, filename)

    if not project.name:
        raise AJMLCompilationError(
            E001,
            "Root element `<project>` must have a `name` attribute.",
            filename,
        )

    if not project.ajml_version:
        raise AJMLCompilationError(
            E004,
            "Missing `ajml_version` in _project.ajml.",
            filename,
        )

    if project.ajml_version != "2.0":
        raise AJMLCompilationError(
            E004,
            f"Invalid `ajml_version` '{project.ajml_version}' in _project.ajml. Supported versions: 2.0.",
            filename,
        )

    config_elem = root.find("config")
    if config_elem is not None:
        # LLM config
        llm_elem = config_elem.find("llm")
        if llm_elem is not None:
            project.llm_provider = llm_elem.get("provider", "")
            project.llm_model = llm_elem.get("model", "")
            project.llm_max_retries = int(llm_elem.get("max_retries", "2"))

            if project.llm_provider and project.llm_provider not in SUPPORTED_PROVIDERS:
                raise AJMLCompilationError(
                    E401,
                    f"Unknown LLM provider `{project.llm_provider}`. "
                    f"Must be one of: {', '.join(sorted(SUPPORTED_PROVIDERS))}.",
                    filename,
                )

        # Server config
        server_elem = config_elem.find("server")
        if server_elem is not None:
            project.cors_origins = server_elem.get("cors_origins", "*")
            project.auth_env = server_elem.get("auth_env", "")
            docs_public_str = server_elem.get("docs_public")
            if docs_public_str is not None:
                project.docs_public = docs_public_str.lower() == "true"
            else:
                project.docs_public = not bool(project.auth_env)
            project.port = int(server_elem.get("port", "8000"))
            project.host = server_elem.get("host", "0.0.0.0")
        else:
            project.docs_public = True

        # Env vars
        env_elem = config_elem.find("env")
        if env_elem is not None:
            for var_elem in env_elem.findall("var"):
                project.env_vars.append({
                    "name": var_elem.get("name", ""),
                    "required": var_elem.get("required", "false").lower() == "true",
                    "default": var_elem.get("default"),
                })

    return project


def validate_agent(
    root: ET.Element,
    filename: str,
    project: ProjectAST,
    all_agents: dict[str, "AgentAST"] | None = None,
    project_dir: str = ".",
) -> tuple[AgentAST, list[AJMLWarning]]:
    """Validate and parse a single agent .ajml file.

    Returns:
        Tuple of (AgentAST, list of warnings).

    Raises:
        AJMLCompilationError on validation failure.
    """
    warnings: list[AJMLWarning] = []
    all_agents = all_agents or {}

    if root.tag != "agent":
        raise AJMLCompilationError(
            E001,
            "Root element must be `<agent>` with a `name` attribute.",
            filename,
        )

    agent = AgentAST(root, filename)

    if not agent.name:
        raise AJMLCompilationError(
            E001,
            "Root element `<agent>` must have a `name` attribute.",
            filename,
        )

    # --- Validate structure ---
    state_elem = root.find("state")
    graph_elem = root.find("graph")

    if state_elem is None:
        raise AJMLCompilationError(E002, "Required block `<state>` is missing.", filename)
    if graph_elem is None:
        raise AJMLCompilationError(E002, "Required block `<graph>` is missing.", filename)

    # --- Parse agent config ---
    config_elem = root.find("config")
    if config_elem is not None:
        agent.config = {}
        llm_elem = config_elem.find("llm")
        if llm_elem is not None:
            provider = llm_elem.get("provider", "")
            if provider and provider not in SUPPORTED_PROVIDERS:
                raise AJMLCompilationError(
                    E401,
                    f"Unknown LLM provider `{provider}`. "
                    f"Must be one of: {', '.join(sorted(SUPPORTED_PROVIDERS))}.",
                    filename,
                )
            agent.config["llm_provider"] = provider
            agent.config["llm_model"] = llm_elem.get("model", "")
            agent.config["llm_max_retries"] = int(llm_elem.get("max_retries", "2"))

    # --- Validate state fields ---
    field_names: set[str] = set()
    for field_elem in state_elem.findall("field"):
        name = field_elem.get("name", "")
        type_str = field_elem.get("type", "")
        required = field_elem.get("required", "false").lower() == "true"
        default = field_elem.get("default")
        reducer = field_elem.get("reducer", "overwrite")
        expose = field_elem.get("expose", "true").lower() == "true"
        values = field_elem.get("values", "")

        # Reserved word check
        if name in RESERVED_WORDS:
            if name == "messages":
                raise AJMLCompilationError(
                    E201,
                    "Cannot declare state field `messages`. "
                    "This field is implicitly managed by the framework.",
                    filename,
                )
            raise AJMLCompilationError(
                E104,
                f"`{name}` is a reserved word and cannot be used as a field name.",
                filename,
            )

        # Duplicate check
        if name in field_names:
            raise AJMLCompilationError(
                E103,
                f"Duplicate state field name `{name}`.",
                filename,
            )
        field_names.add(name)

        # Type validity
        if type_str not in VALID_TYPES:
            raise AJMLCompilationError(
                E107,
                f"Invalid state field type `{type_str}`. "
                "Must be a valid simple type, parameterised type, or enum. See §3.8.",
                filename,
            )

        # Required + default conflict
        if required and default is not None:
            raise AJMLCompilationError(
                E202,
                f"Required field `{name}` cannot have a default value.",
                filename,
            )

        # Reducer validity
        base = _base_type(type_str)
        valid_reducers = VALID_REDUCERS.get(base, {"overwrite"})
        if reducer not in valid_reducers:
            raise AJMLCompilationError(
                E108,
                f"Reducer `{reducer}` is not valid for type `{type_str}`. "
                f"Valid reducers: {', '.join(sorted(valid_reducers))}.",
                filename,
            )

        # Enum validation
        if type_str == "enum":
            if not values:
                raise AJMLCompilationError(
                    E107,
                    f"Enum field `{name}` must have a `values` attribute.",
                    filename,
                )
            enum_values = [v.strip() for v in values.split(",")]
            # Allow empty string as a special "unset" default for enums
            if default is not None and default != "" and default not in enum_values:
                raise AJMLCompilationError(
                    E109,
                    f"Enum default value `{default}` is not in the declared "
                    f"values list: {', '.join(enum_values)}.",
                    filename,
                )

        agent.state_fields.append({
            "name": name,
            "type": type_str,
            "required": required,
            "default": default,
            "reducer": reducer,
            "expose": expose,
            "values": values,
        })

    # --- Validate tools ---
    tools_elem = root.find("tools")
    tool_ids: set[str] = set()
    if tools_elem is not None:
        for tool_elem in tools_elem.findall("tool"):
            tool_id = tool_elem.get("id", "")
            tool_type = tool_elem.get("type", "")

            if tool_id in RESERVED_WORDS:
                raise AJMLCompilationError(
                    E104,
                    f"`{tool_id}` is a reserved word and cannot be used as a tool ID.",
                    filename,
                )

            if tool_id in tool_ids:
                raise AJMLCompilationError(
                    E102,
                    f"Duplicate tool ID `{tool_id}`.",
                    filename,
                )
            tool_ids.add(tool_id)

            if tool_type not in VALID_TOOL_TYPES:
                raise AJMLCompilationError(
                    E106,
                    f"Invalid tool type `{tool_type}`. "
                    f"Must be one of: {', '.join(sorted(VALID_TOOL_TYPES))}.",
                    filename,
                )

            tool_data: dict[str, Any] = {
                "id": tool_id,
                "type": tool_type,
            }

            if tool_type == "api_call":
                tool_data.update(_parse_api_call_tool(tool_elem, filename))
            elif tool_type == "local_script":
                path = tool_elem.get("path", "")
                tool_data["path"] = path
                tool_data["description"] = tool_elem.get("description", "")
                script_path = os.path.join(project_dir, "tools", path)
                if not os.path.exists(script_path):
                    raise AJMLCompilationError(
                        E308,
                        f"Script file `tools/{path}` does not exist.",
                        filename,
                    )
            elif tool_type == "script_tool":
                src = tool_elem.get("src", "")
                tool_data["src"] = src
                tool_data["description"] = tool_elem.get("description", "")
                script_path = os.path.join(project_dir, "tools", src)
                if not os.path.exists(script_path):
                    raise AJMLCompilationError(
                        E308,
                        f"Script file `tools/{src}` does not exist.",
                        filename,
                    )
                # Parse parameters from AJML
                params_elem = tool_elem.find("parameters")
                tool_data["parameters"] = []
                if params_elem is not None:
                    for param_elem in params_elem.findall("param"):
                        tool_data["parameters"].append({
                            "name": param_elem.get("name", ""),
                            "type": param_elem.get("type", ""),
                            "description": param_elem.get("description", ""),
                        })

            agent.tools.append(tool_data)

    # --- Validate nodes ---
    node_ids: set[str] = set()
    for node_elem in graph_elem.findall("node"):
        node_id = node_elem.get("id", "")
        node_type = node_elem.get("type", "")

        if node_id in RESERVED_WORDS:
            raise AJMLCompilationError(
                E104,
                f"`{node_id}` is a reserved word and cannot be used as a node ID.",
                filename,
            )

        if node_id in node_ids:
            raise AJMLCompilationError(
                E101,
                f"Duplicate node ID `{node_id}`.",
                filename,
            )
        node_ids.add(node_id)

        if node_type not in VALID_NODE_TYPES:
            raise AJMLCompilationError(
                E105,
                f"Invalid node type `{node_type}`. "
                f"Must be one of: {', '.join(sorted(VALID_NODE_TYPES))}.",
                filename,
            )

        node_data: dict[str, Any] = {
            "id": node_id,
            "type": node_type,
        }

        if node_type == "llm":
            node_data.update(_parse_llm_node(node_elem, tool_ids, filename))
        elif node_type == "action":
            tool_ref = node_elem.get("tool_ref", "")
            if tool_ref not in tool_ids:
                raise AJMLCompilationError(
                    E307,
                    f"Tool reference `{tool_ref}` does not match any declared tool ID.",
                    filename,
                )
            node_data["tool_ref"] = tool_ref
        elif node_type == "script":
            path = node_elem.get("path", "")
            node_data["path"] = path
            script_path = os.path.join(project_dir, "tools", path)
            if not os.path.exists(script_path):
                raise AJMLCompilationError(
                    E308,
                    f"Script file `tools/{path}` does not exist.",
                    filename,
                )
        elif node_type == "subgraph":
            agent_ref = node_elem.get("agent_ref", "")
            node_data["agent_ref"] = agent_ref
            node_data["input_map"] = []
            node_data["output_map"] = []

            input_map_elem = node_elem.find("input_map")
            if input_map_elem is not None:
                for map_elem in input_map_elem.findall("map"):
                    node_data["input_map"].append({
                        "source": map_elem.get("source", ""),
                        "target": map_elem.get("target", ""),
                    })

            output_map_elem = node_elem.find("output_map")
            if output_map_elem is not None:
                for map_elem in output_map_elem.findall("map"):
                    node_data["output_map"].append({
                        "source": map_elem.get("source", ""),
                        "target": map_elem.get("target", ""),
                    })

        agent.nodes.append(node_data)

    # --- Validate edges ---
    edge_sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    has_start_edge = False

    for edge_elem in graph_elem.findall("edge"):
        source = edge_elem.get("source", "")
        target = edge_elem.get("target", "")
        default = edge_elem.get("default", "false").lower() == "true"
        edge_type = edge_elem.get("type", "")

        # Validate source
        if source == "__START__":
            has_start_edge = True
        elif source not in node_ids:
            raise AJMLCompilationError(
                E303,
                f"Edge source `{source}` does not match any declared node ID or `__START__`.",
                filename,
            )

        # Validate target
        if target != "__END__" and target not in node_ids:
            raise AJMLCompilationError(
                E302,
                f"Edge target `{target}` does not match any declared node ID or `__END__`.",
                filename,
            )

        condition_elem = edge_elem.find("condition")
        condition_text = ""
        if condition_elem is not None and condition_elem.text:
            condition_text = unescape_content(condition_elem.text.strip())

        # Default edge with condition check
        if default and condition_text:
            raise AJMLCompilationError(
                E305,
                "Edge with `default=\"true\"` must not also contain a `<condition>`.",
                filename,
            )

        map_config = None
        if edge_type == "map":
            mc_elem = edge_elem.find("map_config")
            if mc_elem is not None:
                items_field = mc_elem.get("items_field", "")
                item_var = mc_elem.get("item_var", "")
                # Validate items_field references a list type
                field_type = None
                for f in agent.state_fields:
                    if f["name"] == items_field:
                        field_type = f["type"]
                        break
                if field_type is not None and not _is_list_type(field_type):
                    raise AJMLCompilationError(
                        E315,
                        f"Map edge `items_field` `{items_field}` must reference a list-type state field.",
                        filename,
                    )
                map_config = {"items_field": items_field, "item_var": item_var}

        edge_data = {
            "source": source,
            "target": target,
            "default": default,
            "type": edge_type,
            "condition": condition_text,
            "map_config": map_config,
        }

        edge_sources[source].append(edge_data)
        agent.edges.append(edge_data)

    # Must have __START__ edge
    if not has_start_edge:
        raise AJMLCompilationError(
            E301,
            "No entry point found. At least one edge must have `source=\"__START__\"`.",
            filename,
        )

    # Validate edge groups (mixing rules)
    for source, edges in edge_sources.items():
        has_conditions = any(e["condition"] for e in edges)
        has_defaults = any(e["default"] for e in edges)
        has_map = any(e["type"] == "map" for e in edges)
        has_plain = any(
            not e["condition"] and not e["default"] and not e["type"]
            for e in edges
        )

        categories = []
        if has_conditions or has_defaults:
            categories.append("conditional")
        if has_map:
            categories.append("map")
        if has_plain and not has_conditions and not has_defaults:
            categories.append("unconditional")

        if len(categories) > 1:
            raise AJMLCompilationError(
                E314,
                f"Mixed edge types from source `{source}`. "
                "A node's outgoing edges must be all unconditional, "
                "all conditional, or a single map edge.",
                filename,
            )

        # Conditional group must have exactly one default
        if has_conditions:
            defaults = [e for e in edges if e["default"]]
            if len(defaults) == 0:
                raise AJMLCompilationError(
                    E304,
                    f"Conditional edge group from `{source}` is missing a `default=\"true\"` edge.",
                    filename,
                )
            if len(defaults) > 1:
                raise AJMLCompilationError(
                    E306,
                    f"Multiple `default=\"true\"` edges from source `{source}`.",
                    filename,
                )

    # Validate condition expressions
    for edge in agent.edges:
        if edge["condition"]:
            _validate_condition_expression(edge["condition"], filename)

    # Check node reachability
    _check_reachability(agent, filename)

    # Check for nodes without outgoing edges (except __END__ targets)
    _check_outgoing_edges(agent, filename)

    # Check parallel write warnings
    warnings.extend(_check_parallel_writes(agent, filename))

    # Validate subgraph references (if all_agents available)
    if all_agents:
        _validate_subgraph_refs(agent, all_agents, filename)

    return agent, warnings


def _parse_api_call_tool(tool_elem: ET.Element, filename: str) -> dict[str, Any]:
    """Parse an api_call tool element."""
    data: dict[str, Any] = {
        "max_retries": int(tool_elem.get("max_retries", "0")),
        "timeout": float(tool_elem.get("timeout", "30.0")),
        "retry_status_codes": tool_elem.get("retry_status_codes", "429, 500, 502, 503, 504"),
        "backoff": tool_elem.get("backoff", "exponential"),
        "backoff_base": float(tool_elem.get("backoff_base", "1.0")),
        "description": tool_elem.get("description", ""),
    }

    # Endpoint
    endpoint_elem = tool_elem.find("endpoint")
    if endpoint_elem is not None:
        data["url"] = endpoint_elem.get("url", "")
        data["method"] = endpoint_elem.get("method", "GET")

    # Headers
    data["headers"] = []
    headers_elem = tool_elem.find("headers")
    if headers_elem is not None:
        for header_elem in headers_elem.findall("header"):
            data["headers"].append({
                "name": header_elem.get("name", ""),
                "value": header_elem.get("value", ""),
            })

    # Parameters
    data["parameters"] = []
    params_elem = tool_elem.find("parameters")
    if params_elem is not None:
        for param_elem in params_elem.findall("param"):
            data["parameters"].append({
                "name": param_elem.get("name", ""),
                "type": param_elem.get("type", ""),
                "map_to": param_elem.get("map_to", ""),
                "in": param_elem.get("in", ""),
                "description": param_elem.get("description", ""),
            })

    # Body
    data["body"] = None
    body_elem = tool_elem.find("body")
    if body_elem is not None:
        data["body"] = {
            "format": body_elem.get("format", "json"),
            "fields": [],
        }
        for field_elem in body_elem.findall("field"):
            data["body"]["fields"].append({
                "name": field_elem.get("name", ""),
                "type": field_elem.get("type", ""),
                "from_state": field_elem.get("from_state", ""),
            })

    # Returns
    data["returns"] = []
    returns_elem = tool_elem.find("returns")
    if returns_elem is not None:
        for map_elem in returns_elem.findall("map"):
            data["returns"].append({
                "api_field": map_elem.get("api_field", ""),
                "state_field": map_elem.get("state_field", ""),
            })

    return data


def _parse_llm_node(
    node_elem: ET.Element, tool_ids: set[str], filename: str
) -> dict[str, Any]:
    """Parse an LLM node element."""
    data: dict[str, Any] = {}

    # System prompt
    prompt_elem = node_elem.find("system_prompt")
    if prompt_elem is not None and prompt_elem.text:
        data["system_prompt"] = unescape_content(prompt_elem.text.strip())
    else:
        data["system_prompt"] = ""

    # Output schema
    output_schema_elem = node_elem.find("output_schema")
    data["output_schema"] = []
    if output_schema_elem is not None:
        for field_elem in output_schema_elem.findall("field"):
            data["output_schema"].append({
                "name": field_elem.get("name", ""),
                "type": field_elem.get("type", ""),
                "description": field_elem.get("description", ""),
                "values": field_elem.get("values", ""),
            })

    # Tool bindings
    data["tool_binds"] = []
    for bind_elem in node_elem.findall("tool_bind"):
        ref = bind_elem.get("ref", "")
        if ref not in tool_ids:
            raise AJMLCompilationError(
                E307,
                f"Tool reference `{ref}` does not match any declared tool ID.",
                filename,
            )
        data["tool_binds"].append(ref)

    return data


def _validate_condition_expression(expr: str, filename: str):
    """Validate a condition expression only uses allowed names."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise AJMLCompilationError(
            E501,
            f"Condition expression is not valid Python: `{expr}`.",
            filename,
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id not in CONDITION_ALLOWED_NAMES:
                raise AJMLCompilationError(
                    E501,
                    f"Condition expression references undefined name `{node.id}`. "
                    "Only `state` and allowed builtins are available.",
                    filename,
                )


def _check_reachability(agent: AgentAST, filename: str):
    """Check all nodes are reachable from __START__."""
    reachable: set[str] = set()
    queue = ["__START__"]

    while queue:
        current = queue.pop(0)
        for edge in agent.edges:
            if edge["source"] == current and edge["target"] not in reachable:
                if edge["target"] != "__END__":
                    reachable.add(edge["target"])
                    queue.append(edge["target"])

    node_ids = {n["id"] for n in agent.nodes}
    unreachable = node_ids - reachable
    for node_id in unreachable:
        raise AJMLCompilationError(
            E312,
            f"Node `{node_id}` is unreachable from `__START__`.",
            filename,
        )


def _check_outgoing_edges(agent: AgentAST, filename: str):
    """Check all nodes have at least one outgoing edge."""
    sources = {e["source"] for e in agent.edges}
    for node in agent.nodes:
        if node["id"] not in sources:
            raise AJMLCompilationError(
                E313,
                f"Node `{node['id']}` has no outgoing edges.",
                filename,
            )


def _check_parallel_writes(agent: AgentAST, filename: str) -> list[AJMLWarning]:
    """Check for parallel branches that may write to the same overwrite field."""
    warnings = []
    edge_groups: dict[str, list[dict]] = defaultdict(list)
    for e in agent.edges:
        edge_groups[e["source"]].append(e)

    # Find parallel fan-out sources (multiple unconditional edges from same source)
    for source, edges in edge_groups.items():
        if len(edges) > 1 and all(
            not e["condition"] and not e["default"] and not e["type"]
            for e in edges
        ):
            # These are parallel branches - check if they write to same overwrite fields
            # Simplified: warn about any overwrite fields in state
            overwrite_fields = [
                f["name"] for f in agent.state_fields
                if f["reducer"] == "overwrite"
            ]
            if overwrite_fields:
                for field_name in overwrite_fields:
                    warnings.append(AJMLWarning(
                        W301,
                        f"Parallel branches from `{source}` may both write to "
                        f"field `{field_name}` which uses the `overwrite` reducer. "
                        "Result may be non-deterministic.",
                        filename,
                    ))

    return warnings


def _validate_subgraph_refs(
    agent: AgentAST,
    all_agents: dict[str, AgentAST],
    filename: str,
):
    """Validate subgraph node references point to existing agents."""
    for node in agent.nodes:
        if node["type"] == "subgraph":
            agent_ref = node["agent_ref"]
            if agent_ref not in all_agents:
                raise AJMLCompilationError(
                    E309,
                    f"Agent reference `{agent_ref}` does not match any .ajml file in the project.",
                    filename,
                )

            child = all_agents[agent_ref]
            child_fields = {f["name"] for f in child.state_fields}

            for mapping in node.get("input_map", []):
                if mapping["target"] not in child_fields:
                    raise AJMLCompilationError(
                        E310,
                        f"Input map target `{mapping['target']}` does not exist "
                        f"in child agent `{agent_ref}` state.",
                        filename,
                    )

            for mapping in node.get("output_map", []):
                if mapping["source"] not in child_fields:
                    raise AJMLCompilationError(
                        E311,
                        f"Output map source `{mapping['source']}` does not exist "
                        f"in child agent `{agent_ref}` state.",
                        filename,
                    )


def check_circular_dependencies(
    agents: dict[str, AgentAST],
    filename: str = "",
):
    """Check for circular subgraph dependencies across all agents."""
    # Build dependency graph
    deps: dict[str, set[str]] = {}
    for name, agent in agents.items():
        deps[name] = set()
        for node in agent.nodes:
            if node["type"] == "subgraph":
                deps[name].add(node["agent_ref"])

    # DFS cycle detection
    visited: set[str] = set()
    in_stack: set[str] = set()
    path: list[str] = []

    def dfs(node: str) -> bool:
        visited.add(node)
        in_stack.add(node)
        path.append(node)

        for dep in deps.get(node, set()):
            if dep in in_stack:
                cycle = path[path.index(dep):] + [dep]
                raise AJMLCompilationError(
                    E402,
                    f"Circular subgraph dependency detected: {' → '.join(cycle)}.",
                    filename,
                )
            if dep not in visited:
                dfs(dep)

        path.pop()
        in_stack.remove(node)
        return False

    for name in agents:
        if name not in visited:
            dfs(name)


def check_duplicate_agent_names(agents: dict[str, AgentAST]):
    """Check that no two agents share the same name."""
    seen: dict[str, str] = {}
    for name, agent in agents.items():
        if agent.name in seen:
            raise AJMLCompilationError(
                E403,
                f"Duplicate agent name `{agent.name}` across project.",
                agent.filename,
            )
        seen[agent.name] = agent.filename
