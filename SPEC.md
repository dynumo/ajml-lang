# AJML — Agent Job Markup Language

## Specification

---

## Table of Contents

1. [Introduction & Philosophy](#1-introduction--philosophy)
2. [Language Reference](#2-language-reference)
3. [State Management](#3-state-management)
4. [Nodes](#4-nodes)
5. [Edges & Control Flow](#5-edges--control-flow)
6. [Tools](#6-tools)
7. [Multi-Agent Orchestration](#7-multi-agent-orchestration)
8. [Project Configuration](#8-project-configuration)
9. [Agent Configuration](#9-agent-configuration)
10. [Transpiler Architecture](#10-transpiler-architecture)
11. [Generated Code Specification](#11-generated-code-specification)
12. [Auto-Generated API & Documentation](#12-auto-generated-api--documentation)
13. [CLI Reference](#13-cli-reference)
14. [Directory Structure & Conventions](#14-directory-structure--conventions)
15. [Error Catalogue](#15-error-catalogue)
16. [Examples](#16-examples)
17. [Future Roadmap](#17-future-roadmap)

---

## 1. Introduction & Philosophy

### 1.1 What is AJML?

AJML (Agent Job Markup Language) is a declarative language with XML-like syntax for defining specialised AI agent workflows. AJML files are parsed by AJML tooling (not by general-purpose XML parsers) and compile to production-ready Python code backed by [LangGraph](https://www.langchain.com/langgraph), wrapped in a [FastAPI](https://fastapi.tiangolo.com/) REST server with authentication, CORS, and auto-generated OpenAPI documentation.

**Note on syntax:** AJML uses XML-like syntax for familiarity and readability, but is not intended to be valid XML. The AJML transpiler includes a lexical pre-processor that handles constructs (such as comparison operators in conditions) that would be invalid in strict XML. Developers should not attempt to parse `.ajml` files with generic XML tools; always use the AJML CLI. See §2.4 for the precise syntax rules.

### 1.2 Why AJML?

The current landscape of AI agent development has a problem: **the barrier to entry is too high for the value delivered.**

Building a deployment-ready AI agent today requires a developer to understand LangGraph's state machine primitives, LangChain's tool abstractions, FastAPI's dependency injection, Pydantic validation, retry logic, and a dozen other moving parts. This is fine for experienced engineers, but it locks out the vast majority of developers who simply need a specialised agent that does a well-defined job.

Meanwhile, the industry is trending towards generalised agents with open-ended "skills" — systems that are powerful but unpredictable. AJML takes the opposite stance: **agents should be specialists with well-defined workflows**, not generalists that might do what you want.

### 1.3 Design Principles

1. **Declarative over imperative.** The developer describes *what* the agent does, not *how* the underlying framework executes it. If a developer has to edit compiled Python code, AJML has failed.
2. **Convention over configuration.** Sensible defaults everywhere. A minimal `.ajml` file should produce a working agent with zero boilerplate.
3. **Deterministic orchestration by default.** The graph structure — which nodes run, in what order, under what conditions — is explicit and predictable. The LLM operates within the boundaries the developer sets, not outside them. (LLM outputs themselves are inherently non-deterministic; AJML controls the workflow around them.)
4. **Deployment-ready output.** Every build produces a self-contained API server with authentication, documentation, and health checks. No post-compilation wiring required.
5. **Provider-agnostic.** AJML supports any LLM provider in the supported provider list. Switching providers is a single attribute change.

### 1.4 Target Audience

AJML is designed for junior-to-mid-level developers who understand basic XML, REST APIs, and the concept of a state machine, but who should not need to learn LangGraph, LangChain, or FastAPI internals to ship a working AI agent.

### 1.5 Trust Model

AJML assumes a **trusted authoring environment**. The developer writing `.ajml` files and Python scripts in the `tools/` directory is trusted. Specifically:

- **Condition expressions** (§5.4) are evaluated using a restricted expression context that limits available names to `state` and a small set of Python builtins. This is a correctness and portability measure, not a security sandbox. The restriction is enforced by the transpiler via static analysis during compilation.
- **Script nodes and tool scripts** (§4.5, §6.4, §6.5) execute arbitrary Python code with no restrictions.
- **System prompts** (§4.3) can interpolate any state value into LLM context.

AJML is not designed for scenarios where untrusted users author or upload `.ajml` files. If your use case involves untrusted AJML input, you must implement your own validation and sandboxing layer around the transpiler and generated code.

---

## 2. Language Reference

### 2.1 Document Structure — Agent Files

Every `.ajml` file in `agents/` represents a single agent and must conform to this structure:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<agent name="agent_name" version="1.0" description="What this agent does.">
    <config>...</config>
    <state>...</state>
    <tools>...</tools>
    <graph>...</graph>
</agent>
```

**Root Element: `<agent>`**

| Attribute     | Required | Type   | Description                                                         |
|---------------|----------|--------|---------------------------------------------------------------------|
| `name`        | Yes      | string | Unique identifier. Must be a valid Python identifier (`[a-z_][a-z0-9_]*`). Used as the API endpoint name. |
| `version`     | No       | string | Semantic version of this agent definition. Informational only. Defaults to `"1.0"`. |
| `description` | No       | string | Human-readable description. Used in generated OpenAPI docs.         |

**Child elements must appear in the order shown above.** `<config>` and `<tools>` are optional. `<state>` and `<graph>` are required.

### 2.2 Document Structure — Project File

Every AJML project must contain a `_project.ajml` file in the `agents/` directory. This file defines project-level configuration (server, auth, shared environment variables) and is not an agent.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project name="my_project" ajml_version="2.0">
    <config>
        <llm provider="openai" model="gpt-4o" max_retries="3" />
        <server cors_origins="https://frontend.com" auth_env="AJML_API_KEY"
                docs_public="false" />
        <env>
            <var name="OPENAI_API_KEY" required="true" />
        </env>
    </config>
</project>
```

**Root Element: `<project>`**

| Attribute      | Required | Type   | Description                                                          |
|----------------|----------|--------|----------------------------------------------------------------------|
| `name`         | Yes      | string | Project name. Used in the generated FastAPI title.                   |
| `ajml_version` | Yes      | string | AJML specification version this project targets (e.g., `"2.0"`).    |

See §8 for full project configuration details.

### 2.3 Reserved Words

The following identifiers are reserved and must not be used as node IDs, tool IDs, or state field names:

`__START__`, `__END__`, `messages`, `__root__`, `__config__`, `__state__`

`messages` is reserved because it is implicitly added to every agent's state (see §3.4).

### 2.4 Syntax Rules

AJML uses XML-like syntax with the following specific rules:

- **Attributes** must always be quoted (double quotes).
- **Self-closing tags** are allowed (e.g., `<field name="x" type="string" />`).
- **XML comments** are allowed (`<!-- comment -->`).
- **CDATA sections** are not supported. Use external Python files for code.
- **Whitespace** in `<system_prompt>` content is preserved as-is, including indentation and newlines. Leading/trailing whitespace on the entire prompt block is trimmed.
- **Nested tags** inside `<system_prompt>` and `<condition>` elements are not permitted. These elements contain plain text only.
- **Encoding** must be UTF-8.

### 2.5 Interpolation Syntax

AJML uses `${...}` syntax for inserting dynamic values into text content. This applies to system prompts, URL templates, and header values.

**State interpolation:** `${field_name}` — replaced at runtime with the current value of the named state field.

```xml
<system_prompt>The customer's ID is ${customer_id}.</system_prompt>
```

**Environment variable interpolation:** `${env:VAR_NAME}` — replaced at runtime with the value of the named environment variable.

```xml
<header name="Authorization" value="Bearer ${env:API_TOKEN}" />
```

**Literal text:** All other text, including curly braces `{` and `}`, is treated as literal. No escaping is needed for JSON examples, code snippets, or any other content in prompts:

```xml
<system_prompt>
    The customer's ID is ${customer_id}.
    Return your answer as JSON like this:
    {"intent": "refund", "confidence": 0.95}
</system_prompt>
```

The transpiler only recognises the `${...}` pattern. Bare `{` and `}` are always passed through unchanged.

**Interpolation failure modes:**

| Scenario                                     | Behaviour                                                     |
|----------------------------------------------|---------------------------------------------------------------|
| `${field_name}` where field exists in state  | Replaced with the field's current value (converted to string). |
| `${field_name}` where field is missing       | Replaced with empty string `""`. A runtime warning is logged. |
| `${env:VAR}` where env var is set            | Replaced with the variable's value.                           |
| `${env:VAR}` where env var is not set and no default declared | Server fails to start (caught by env validation at startup). |
| `${unknown_syntax}`                          | Compilation error `E501`.                                     |

**Where interpolation is supported:**

| Context              | `${field}` (state) | `${env:VAR}` (env) |
|----------------------|--------------------|---------------------|
| `<system_prompt>`    | Yes                | No                  |
| `<endpoint url="…">` | Yes (via params)   | Yes                 |
| `<header value="…">` | No                 | Yes                 |
| `<condition>`        | No (use `state.get()`) | No              |

---

## 3. State Management

### 3.1 Overview

State is the shared data structure that flows through the agent's graph. Every node reads from and writes to this state. In AJML, state is declared explicitly in a `<state>` block, which compiles to a LangGraph `TypedDict` with annotated reducer functions.

### 3.2 The `<state>` Block

```xml
<state>
    <field name="customer_id" type="string" required="true" />
    <field name="refund_amount" type="float" default="0.0" />
    <field name="search_results" type="list[dict]" reducer="append" />
    <field name="metadata" type="dict" reducer="merge" />
    <field name="retry_count" type="int" reducer="add" default="0" />
    <field name="status" type="enum" values="pending, active, closed" default="pending" />
    <field name="internal_trace" type="string" expose="false" />
</state>
```

**Element: `<field>`**

| Attribute  | Required | Type    | Description                                                                                              |
|------------|----------|---------|----------------------------------------------------------------------------------------------------------|
| `name`     | Yes      | string  | Field identifier. Must be a valid Python identifier.                                                     |
| `type`     | Yes      | string  | Field type. See §3.8 for the full type system.                                                           |
| `required` | No       | boolean | If `true`, this field must be provided in the API request body. Defaults to `false`.                     |
| `default`  | No       | any     | Default value if not provided. Must be valid for the declared type. Required fields cannot have defaults. |
| `reducer`  | No       | enum    | Determines how concurrent or sequential updates to this field are merged. See §3.3.                      |
| `expose`   | No       | boolean | Whether this field is included in the API response. Defaults to `true`. See §3.6.                        |
| `values`   | No       | string  | Comma-separated list of allowed values. Only valid when `type="enum"`. See §3.8.                         |

### 3.3 Reducers

Reducers control how state updates are applied. When a node returns a value for a field, the reducer determines whether it overwrites the existing value or merges with it.

| Reducer     | Valid Types    | Behaviour                                                              | LangGraph Equivalent               |
|-------------|----------------|------------------------------------------------------------------------|-------------------------------------|
| `overwrite` | All            | New value replaces old value. **This is the default if no reducer is specified.** | Default (no annotation)             |
| `append`    | `list`         | New list items are appended to the existing list.                      | `Annotated[list, operator.add]`     |
| `add`       | `int`, `float` | New value is added to the existing value.                              | `Annotated[int, operator.add]`      |
| `merge`     | `dict`         | New dictionary is shallow-merged into the existing dictionary (right-hand wins on key conflicts, i.e., `{**old, **new}`). | `Annotated[dict, lambda a, b: {**a, **b}]` |
| `concat`    | `string`       | New string is concatenated to the existing string.                     | `Annotated[str, operator.concat]`   |

**Reducer type safety at runtime:**

If a node returns a value whose type does not match the reducer's expectation (e.g., returning a scalar for an `append` field that expects a list), the generated code raises a `TypeError` with a descriptive message including the field name, expected type, and received type. This is a runtime error, not a compilation error, because the transpiler cannot predict what values nodes will return.

**Parallel writes to `overwrite` fields:**

If two parallel nodes both write to the same field that uses the `overwrite` reducer (the default), the result is **non-deterministic** — whichever node completes last wins. This is inherent to LangGraph's super-step model. The transpiler emits a **compilation warning** (`W301`) when it detects that parallel branches may write to the same non-reduced field. Use an explicit reducer if you need deterministic parallel writes.

### 3.4 Implicit State Fields

Every agent automatically receives a `messages` field that tracks the LLM conversation history. This field is managed by the framework and should not be declared in the `<state>` block. Declaring it is a compilation error (`E201`).

**Internal representation (generated code):** Within the compiled LangGraph code, `messages` is a list of LangChain message objects (`HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`) annotated with the `add_messages` reducer. All node functions that interact with messages must use LangChain message objects. The generated code handles this automatically.

**API representation (REST responses):** When the final state is returned via the REST API, `messages` is serialised to a simplified list of dictionaries:

```json
[
    {"role": "system", "content": "You are an intent classifier..."},
    {"role": "human", "content": "I want a refund."},
    {"role": "assistant", "content": "refund"},
    {"role": "tool", "name": "fetch_user", "content": "{...}"}
]
```

This serialisation is handled automatically by the generated server code. Developers never need to manage the conversion. The `messages` field in API responses is always present unless explicitly hidden with `expose="false"` in a future version.

### 3.5 State as API Contract

The `<state>` block doubles as the API request/response schema:

- **Request body:** All fields marked `required="true"` become required properties in the generated Pydantic request model. Fields with defaults become optional. If a required field is missing from a request, FastAPI returns a `422 Unprocessable Entity` response with a validation error describing the missing field(s). Required fields cannot have defaults — this is enforced at compile time (`E202`).
- **Response body:** All fields with `expose="true"` (the default) are included in the API response, plus `messages` (serialised as §3.4 describes).

### 3.6 Response Filtering

The `expose` attribute controls whether a state field appears in the API response:

```xml
<field name="api_token_cache" type="string" expose="false" />
<field name="internal_score" type="float" expose="false" />
<field name="customer_name" type="string" expose="true" />  <!-- default -->
```

Fields with `expose="false"` are:
- Used normally within the graph (nodes can read/write them).
- Excluded from the generated response Pydantic model.
- Never sent to the API caller.

This prevents accidental leakage of secrets, internal debug data, or large intermediate artefacts through the API.

### 3.7 Unknown Keys Policy

When a node, tool, or script returns a dictionary containing keys that do not match any declared state field, the behaviour is:

- **Unknown keys are silently ignored.** They are not written to state.
- **A runtime warning is logged** with the node/tool name and the unknown key names.

This policy applies uniformly across all output sources: LLM structured output, tool return values, script return values, and action node results.

### 3.8 Type System

AJML supports both simple types and parameterised (generic) types for state fields.

**Simple types:**

| Type     | Python Equivalent | Description                          |
|----------|-------------------|--------------------------------------|
| `string` | `str`             | Text value.                          |
| `int`    | `int`             | Integer number.                      |
| `float`  | `float`           | Floating-point number.               |
| `bool`   | `bool`            | Boolean (`true` / `false`).          |
| `list`   | `list`            | Untyped list (any items).            |
| `dict`   | `dict`            | Untyped dictionary (any keys/values).|

**Parameterised types:**

For richer API documentation and optional runtime validation, AJML supports generic type syntax on `list` and `dict`:

| Type               | Python Equivalent     | Description                                       |
|--------------------|-----------------------|---------------------------------------------------|
| `list[string]`     | `list[str]`           | List where each item is a string.                 |
| `list[int]`        | `list[int]`           | List where each item is an integer.               |
| `list[float]`      | `list[float]`         | List where each item is a float.                  |
| `list[dict]`       | `list[dict]`          | List where each item is a dictionary.             |
| `dict[string]`     | `dict[str, str]`      | Dictionary with string keys and string values.    |
| `dict[int]`        | `dict[str, int]`      | Dictionary with string keys and integer values.   |
| `dict[any]`        | `dict`                | Same as bare `dict`.                              |

Parameterised types are used in two ways:

1. **OpenAPI documentation:** The generated request/response models use the parameterised types to produce more descriptive JSON schemas. For example, `list[string]` generates a JSON schema with `"type": "array", "items": {"type": "string"}` rather than just `"type": "array"`.
2. **Structured output schemas:** When a parameterised type appears in an `<output_schema>`, the generated Pydantic model uses the typed version, giving the LLM stronger guidance.

Parameterised types do **not** add runtime type-checking of individual items by default. The generated `TypedDict` uses the typed annotation for static analysis tools, but LangGraph does not enforce item types within lists or dicts at runtime.

```xml
<state>
    <field name="tags" type="list[string]" default="[]" reducer="append" />
    <field name="scores" type="list[float]" default="[]" reducer="append" />
    <field name="search_results" type="list[dict]" default="[]" reducer="append" />
    <field name="settings" type="dict[string]" default="{}" reducer="merge" />
</state>
```

**Enum type:**

The `enum` type defines a string field that only accepts a fixed set of values:

```xml
<field name="status" type="enum" values="pending, approved, rejected" default="pending" />
<field name="priority" type="enum" values="low, medium, high, critical" required="true" />
```

| Attribute | Required | Type   | Description                                                            |
|-----------|----------|--------|------------------------------------------------------------------------|
| `values`  | Yes (when `type="enum"`) | string | Comma-separated list of allowed values (whitespace around commas is trimmed). |

Enum behaviour:

- **API validation:** The generated Pydantic request model uses a `Literal[...]` type, so FastAPI rejects requests with values not in the allowed set (422 error).
- **Structured output:** When used in `<output_schema>`, the generated Pydantic model constrains the LLM to one of the allowed values.
- **Runtime:** Internally stored as a `str`. No runtime validation within the graph beyond what Pydantic enforces at the API boundary.
- **Compilation:** If `default` is provided, it must be one of the declared `values` — otherwise compilation error `E109`.

**Compiled output for enums:**

```xml
<field name="status" type="enum" values="pending, approved, rejected" default="pending" />
```

Generates:

```python
from typing import Literal

# In the AgentState TypedDict:
status: str  # Internal storage is plain str

# In the API request model:
status: Literal["pending", "approved", "rejected"] = "pending"
```

**Reducer compatibility:**

Reducers work with parameterised types the same way as their base types:

| Parameterised Type | Valid Reducers                 |
|--------------------|--------------------------------|
| `list[*]`          | `overwrite`, `append`          |
| `dict[*]`          | `overwrite`, `merge`           |
| `enum`             | `overwrite` only               |

---

## 4. Nodes

### 4.1 Overview

Nodes are the units of work in an AJML graph. Each node performs a single operation: invoking an LLM, executing a tool, running a local script, or delegating to another agent.

### 4.2 The `<node>` Element

All nodes share these common attributes:

| Attribute | Required | Type   | Description                                                              |
|-----------|----------|--------|--------------------------------------------------------------------------|
| `id`      | Yes      | string | Unique identifier within this agent. Must be a valid Python identifier.  |
| `type`    | Yes      | enum   | One of: `llm`, `action`, `script`, `subgraph`. See below.               |

### 4.3 Node Type: `llm`

An LLM node invokes the configured language model. The LLM receives the current `messages` state and any bound tools, then returns an AI message (and potentially tool calls) that is appended to `messages`.

```xml
<node id="classify_intent" type="llm">
    <system_prompt>
        You are an intent classifier. Given a customer message,
        classify it as one of: refund, complaint, question, other.
        Respond with ONLY the classification word.
    </system_prompt>
    <output_schema>
        <field name="intent" type="string" description="The classified intent" />
    </output_schema>
</node>
```

**Child Elements:**

| Element          | Required | Description                                                                                             |
|------------------|----------|---------------------------------------------------------------------------------------------------------|
| `<system_prompt>`| No       | System prompt injected before invocation. Supports `${field}` interpolation (see §2.5).                 |
| `<output_schema>`| No       | If present, forces structured output via `with_structured_output()`. Contains `<field>` children.       |
| `<tool_bind>`    | No       | Zero or more. Binds specific tools to this node's LLM call. If omitted, no tools are bound.             |

**`<system_prompt>` Content:**

System prompts are plain text with `${field}` interpolation. Because AJML uses the `${...}` delimiter (see §2.5), prompts can freely contain JSON examples, code snippets, curly braces, angle brackets, and any other text without escaping:

```xml
<system_prompt>
    The customer's ID is ${customer_id}. Their account tier is ${account_tier}.
    
    Respond as JSON:
    {"action": "refund", "reason": "your reason here"}
    
    If the refund amount > 100, set "needs_approval": true.
</system_prompt>
```

**`<output_schema>` Element:**

When present, the LLM is forced to return structured JSON matching the schema. Each `<field>` child defines a property:

| Attribute     | Required | Type   | Description                                        |
|---------------|----------|--------|----------------------------------------------------|
| `name`        | Yes      | string | Property name in the output JSON.                  |
| `type`        | Yes      | enum   | Any valid AJML type (see §3.8), including parameterised types and `enum`. |
| `description` | No       | string | Description provided to the LLM for this field.    |

The structured output is parsed and its fields are written to state. Unknown keys (field names not matching any declared state field) are handled per the unknown keys policy (§3.7).

**`<tool_bind>` Element and the Tool-Calling Loop:**

```xml
<node id="research" type="llm">
    <system_prompt>Research the customer's issue using available tools.</system_prompt>
    <tool_bind ref="fetch_user" />
    <tool_bind ref="search_knowledge_base" />
</node>
```

| Attribute | Required | Type   | Description                                                |
|-----------|----------|--------|------------------------------------------------------------|
| `ref`     | Yes      | string | References a `<tool id="...">` defined in the `<tools>` block. |

When tools are bound to an LLM node, the transpiler generates a **tool-calling loop** (the ReAct agent pattern):

1. The LLM is invoked with the bound tools available.
2. If the LLM responds with tool calls, each tool is executed via a `ToolNode`.
3. Tool results are appended to `messages` as `ToolMessage` objects.
4. The LLM is invoked again with the updated messages.
5. This repeats until the LLM responds without tool calls (a final text/structured response).

**Tool-loop limits:** The tool-calling loop is subject to the graph's overall recursion limit (default: 50 super-steps). There is no separate per-node limit. If a tool-calling loop exceeds the recursion limit, the graph raises a `GraphRecursionError` which surfaces as a 500 error via the API.

**Tool errors within the loop:** Tool errors are caught and returned to the LLM as a `ToolMessage` with the error description. This gives the LLM a chance to retry with different arguments or respond gracefully. This uses LangGraph's `ToolNode(tools, handle_tool_errors=True)` behaviour.

**Tool results and state:** Tools bound to an LLM node operate in **message-only mode** within the loop. Tool return values are appended to `messages` as `ToolMessage` content — they do **not** write to state fields via `<returns>` mappings during the tool-calling loop. State field updates from tool outputs only occur when a tool is invoked via an `action` node (§4.4). This distinction is important: within an LLM node's tool loop, tools are conversational; within an action node, tools are stateful.

### 4.4 Node Type: `action`

An action node executes a specific tool **without LLM involvement**. This is for deterministic operations where you always want a tool to run, regardless of what the LLM thinks.

```xml
<node id="execute_refund" type="action" tool_ref="process_refund" />
```

| Attribute  | Required | Type   | Description                                                     |
|------------|----------|--------|-----------------------------------------------------------------|
| `tool_ref` | Yes      | string | References a `<tool id="...">` defined in the `<tools>` block. |

The tool receives its inputs by mapping from the current state (as defined in the tool's `<parameters>` block) and writes its outputs back to state (as defined in the tool's `<returns>` block). In action nodes, `<returns>` mappings are applied: the tool's return value is parsed and mapped fields are written to state. Unknown keys in the return value are handled per the unknown keys policy (§3.7).

### 4.5 Node Type: `script`

A script node executes a local Python file from the `tools/` directory. This is for custom logic that doesn't fit the tool abstraction — data transformations, complex validations, routing synchronisation points, or anything that's easier to write in plain Python.

Script nodes operate on the **full agent state**: they receive the entire state dictionary and return a dictionary of state updates. This distinguishes them from tool scripts, which accept individual keyword arguments (see §6.4).

```xml
<node id="calculate_discount" type="script" path="calculate_discount.py" />
```

| Attribute | Required | Type   | Description                                                                      |
|-----------|----------|--------|----------------------------------------------------------------------------------|
| `path`    | Yes      | string | Relative path to a `.py` file in the `tools/` directory.                         |

**Script contract:** The Python file must define a function `run(state: dict) -> dict` that receives the full current state and returns a dictionary of state updates. Unknown keys in the return value are handled per the unknown keys policy (§3.7).

```python
# tools/calculate_discount.py
def run(state: dict) -> dict:
    amount = state.get("order_total", 0)
    tier = state.get("customer_tier", "standard")
    
    discount = 0.1 if tier == "premium" else 0.0
    
    return {
        "discount_amount": amount * discount,
        "final_total": amount * (1 - discount)
    }
```

### 4.6 Node Type: `subgraph`

A subgraph node delegates execution to another AJML agent defined in the same project. See §7 (Multi-Agent Orchestration) for full details.

```xml
<node id="run_verification" type="subgraph" agent_ref="verification_agent">
    <input_map>
        <map source="customer_id" target="user_id" />
        <map source="order_id" target="order_id" />
    </input_map>
    <output_map>
        <map source="is_verified" target="verification_result" />
    </output_map>
</node>
```

---

## 5. Edges & Control Flow

### 5.1 Overview

Edges define how execution flows between nodes. AJML supports unconditional edges, conditional edges, parallel fan-out, and map-reduce. A single source node's outgoing edges must be **one** of these categories — mixing is not permitted (see §5.8).

### 5.2 The `<graph>` Block

All nodes and edges are declared inside the `<graph>` block:

```xml
<graph>
    <node id="..." type="..." />
    <node id="..." type="..." />
    
    <edge source="__START__" target="classify_intent" />
    <edge source="classify_intent" target="handle_refund">
        <condition>state.get('intent') == 'refund'</condition>
    </edge>
    <edge source="classify_intent" target="handle_general" default="true" />
    <edge source="handle_refund" target="__END__" />
    <edge source="handle_general" target="__END__" />
</graph>
```

### 5.3 Unconditional Edges

An edge with no `<condition>` child, no `default` attribute, and no `type` attribute is unconditional. Execution always proceeds from `source` to `target`.

```xml
<edge source="fetch_data" target="process_data" />
```

**Element: `<edge>`**

| Attribute | Required | Type   | Description                                                                                    |
|-----------|----------|--------|------------------------------------------------------------------------------------------------|
| `source`  | Yes      | string | The origin node ID, or `__START__` for the entry point.                                        |
| `target`  | Yes      | string | The destination node ID, or `__END__` to terminate the graph.                                  |
| `default` | No       | boolean| If `true`, this edge is the fallback when no sibling conditions match. See §5.4.               |
| `type`    | No       | enum   | If set to `map`, enables map-reduce fan-out. See §5.7.                                         |

### 5.4 Conditional Edges

When multiple edges share the same `source`, and at least one contains a `<condition>`, they form a **conditional routing group**. Each edge (except the default) must contain a `<condition>` child. Conditions are evaluated in document order. The first condition that evaluates to `true` determines the next node.

```xml
<edge source="check_eligibility" target="approve">
    <condition>state.get('score') >= 80</condition>
</edge>
<edge source="check_eligibility" target="manual_review">
    <condition>state.get('score') >= 50</condition>
</edge>
<edge source="check_eligibility" target="reject" default="true" />
```

**The `<condition>` Element:**

Contains a Python expression that is evaluated at runtime. The expression must return a truthy or falsy value.

**Condition evaluation context:** The following — and **only** the following — names are available inside condition expressions:

| Name    | Type | Description                                   |
|---------|------|-----------------------------------------------|
| `state` | dict | The full current agent state dictionary.      |
| `len`   | func | Python built-in `len()`.                      |
| `any`   | func | Python built-in `any()`.                      |
| `all`   | func | Python built-in `all()`.                      |
| `abs`   | func | Python built-in `abs()`.                      |
| `min`   | func | Python built-in `min()`.                      |
| `max`   | func | Python built-in `max()`.                      |
| `str`   | type | Python built-in `str` type.                   |
| `int`   | type | Python built-in `int` type.                   |
| `float` | type | Python built-in `float` type.                 |
| `bool`  | type | Python built-in `bool` type.                  |
| `True`  |      | Python `True`.                                |
| `False` |      | Python `False`.                               |
| `None`  |      | Python `None`.                                |

No other names, modules, or builtins are available. The transpiler statically analyses condition expressions during Phase 2 compilation and raises `E501` if an expression references any name outside this list. Standard Python operators (`and`, `or`, `not`, `in`, `is`, `==`, `!=`, `<`, `>`, `<=`, `>=`, `+`, `-`, `*`, `/`, `%`) are permitted.

**Note:** Comparison operators (`<`, `>`) are valid in conditions. The AJML pre-processor handles the XML escaping automatically (see §10.1), so developers can write conditions naturally:

```xml
<condition>state.get('refund_amount') > 0 and state.get('retry_count') < 3</condition>
```

**Rules for conditional edge groups:**

- Every conditional group **must** include exactly one edge with `default="true"`. Omitting the default is `E304`.
- An edge with `default="true"` must **not** also contain a `<condition>`. This is `E305`.
- Multiple `default="true"` edges from the same source is `E306`.

### 5.5 Parallel Edges (Fan-Out)

When a single source node has multiple **unconditional** edges (no conditions, no defaults, no `type`), all targets execute in parallel within the same LangGraph super-step. Their state updates are merged using the declared reducers.

```xml
<edge source="__START__" target="fetch_user_profile" />
<edge source="__START__" target="fetch_order_history" />
<edge source="__START__" target="fetch_support_tickets" />
```

In this example, all three fetch nodes run concurrently.

**Fan-in (synchronisation):** Parallel branches implicitly synchronise when they all have edges pointing to the same downstream node. The downstream node only executes once all parallel branches have completed.

```xml
<!-- Fan-out -->
<edge source="start_research" target="search_web" />
<edge source="start_research" target="search_database" />

<!-- Fan-in: both must complete before summarise runs -->
<edge source="search_web" target="summarise" />
<edge source="search_database" target="summarise" />
```

### 5.6 Loops

AJML supports loops through edges that point back to a previous node. The developer is responsible for ensuring a termination condition exists via conditional edges.

```xml
<node id="attempt_fix" type="llm">
    <system_prompt>Attempt to fix the code. Current attempt: ${retry_count}.</system_prompt>
</node>

<edge source="attempt_fix" target="validate_fix" />
<edge source="validate_fix" target="attempt_fix">
    <condition>state.get('is_valid') == False and state.get('retry_count') < 3</condition>
</edge>
<edge source="validate_fix" target="__END__" default="true" />
```

### 5.7 Map-Reduce (Dynamic Fan-Out)

For cases where the number of parallel branches is determined at runtime (e.g., processing each item in a list), AJML provides a map edge. This compiles to LangGraph's `Send` API.

```xml
<edge source="generate_topics" target="write_section" type="map">
    <map_config
        items_field="topics"
        item_var="current_topic"
    />
</edge>

<edge source="write_section" target="combine_sections" />
```

**`<map_config>` Attributes:**

| Attribute     | Required | Type   | Description                                                                  |
|---------------|----------|--------|------------------------------------------------------------------------------|
| `items_field` | Yes      | string | State field (must be a `list`) whose items to iterate over.                  |
| `item_var`    | Yes      | string | State field name the individual item is assigned to in each parallel branch. |

**Map-reduce semantics:**

- Each parallel branch receives a **copy** of the full current state with `item_var` set to the individual item. Branches do **not** share memory.
- Results are merged back into the parent state using the declared reducers.
- If `items_field` is an empty list at runtime, the target node is skipped entirely and execution proceeds to the next edge from the target node (if any). A runtime warning is logged.
- Fan-in after a map works the same as regular parallel fan-in: the downstream node executes once all map branches have completed.

### 5.8 Edge Mixing Rules

A single source node's outgoing edges must fall into exactly one of these categories:

| Category       | Characteristics                                         | Allowed |
|----------------|--------------------------------------------------------|---------|
| Unconditional  | One or more edges, no conditions, no defaults, no type | Yes — single = sequential, multiple = parallel |
| Conditional    | Two or more edges, at least one condition, one default | Yes     |
| Map            | Exactly one edge with `type="map"`                     | Yes     |

Mixing categories from the same source node is a compilation error (`E314`). For example, you cannot have both unconditional and conditional edges from the same node.

---

## 6. Tools

### 6.1 Overview

Tools are the actions an agent can perform: calling APIs or executing custom logic. AJML supports three tool types, all declared within the `<tools>` block.

### 6.2 The `<tools>` Block

```xml
<tools>
    <tool id="fetch_user" type="api_call" max_retries="2">...</tool>
    <tool id="calculate_tax" type="local_script" path="tax_calc.py" />
    <tool id="format_currency" type="script_tool" src="format_currency.py"
          description="Formats a number as currency.">...</tool>
</tools>
```

### 6.3 Tool Type: `api_call`

Declares an HTTP API call. The transpiler generates a fully typed LangChain `@tool` with `httpx` and `tenacity` retry logic.

```xml
<tool id="fetch_user" type="api_call" max_retries="2" timeout="15"
      retry_status_codes="429, 500, 502" backoff="exponential">
    <endpoint url="https://api.example.com/users/${id}" method="GET" />
    <headers>
        <header name="Authorization" value="Bearer ${env:USER_API_TOKEN}" />
        <header name="Content-Type" value="application/json" />
    </headers>
    <parameters>
        <param name="customer_id" type="string" map_to="id" in="path"
               description="The customer's unique ID" />
    </parameters>
    <body format="json">
        <field name="amount" type="float" from_state="refund_amount" />
    </body>
    <returns>
        <map api_field="data.user" state_field="user_data" />
        <map api_field="data.email" state_field="user_email" />
    </returns>
</tool>
```

**`<tool>` Attributes (type=`api_call`):**

| Attribute            | Required | Type   | Description                                                                                           |
|----------------------|----------|--------|-------------------------------------------------------------------------------------------------------|
| `id`                 | Yes      | string | Unique tool identifier.                                                                               |
| `type`               | Yes      | enum   | Must be `api_call`.                                                                                   |
| `max_retries`        | No       | int    | Number of retry attempts on failure. Default: `0` (no retries).                                       |
| `timeout`            | No       | float  | Request timeout in seconds. Default: `30.0`.                                                          |
| `retry_status_codes` | No       | string | Comma-separated HTTP status codes that trigger a retry (e.g., `"429, 500, 502, 503, 504"`). Default: `"429, 500, 502, 503, 504"`. Only relevant when `max_retries > 0`. |
| `backoff`            | No       | enum   | Retry backoff strategy. One of: `fixed`, `exponential`. Default: `exponential`.                       |
| `backoff_base`       | No       | float  | Base delay in seconds for backoff. For `fixed`, this is the constant delay. For `exponential`, delay is `backoff_base * 2^(attempt-1)`. Default: `1.0`. |

**`<endpoint>` Attributes:**

| Attribute | Required | Type   | Description                                                                  |
|-----------|----------|--------|------------------------------------------------------------------------------|
| `url`     | Yes      | string | URL template. Supports `${param}` for path params and `${env:VAR}` for base URLs. |
| `method`  | Yes      | enum   | One of: `GET`, `POST`, `PUT`, `PATCH`, `DELETE`.                             |

**`<headers>` / `<header>` Attributes:**

| Attribute | Required | Type   | Description                                                               |
|-----------|----------|--------|---------------------------------------------------------------------------|
| `name`    | Yes      | string | HTTP header name.                                                         |
| `value`   | Yes      | string | Header value. Supports `${env:VAR_NAME}` for environment variable lookup. |

**`<parameters>` / `<param>` Attributes:**

| Attribute     | Required | Type   | Description                                                         |
|---------------|----------|--------|---------------------------------------------------------------------|
| `name`        | Yes      | string | The state field name this parameter reads from.                     |
| `type`        | Yes      | enum   | One of: `string`, `int`, `float`, `bool`.                           |
| `map_to`      | Yes      | string | The API parameter name (in the URL path or query string).           |
| `in`          | Yes      | enum   | Where the parameter goes: `path` or `query`. Body parameters use `<body>`. |
| `description` | No       | string | Description for the generated tool schema.                          |

**`<body>` Element:**

Only valid for `POST`, `PUT`, and `PATCH` methods.

| Attribute    | Required | Type   | Description                                     |
|--------------|----------|--------|-------------------------------------------------|
| `format`     | Yes      | enum   | Must be `json`.                                 |

**`<body>` / `<field>` Attributes:**

| Attribute    | Required | Type   | Description                                     |
|--------------|----------|--------|-------------------------------------------------|
| `name`       | Yes      | string | Key name in the JSON body.                      |
| `type`       | Yes      | enum   | One of: `string`, `int`, `float`, `bool`.       |
| `from_state` | Yes      | string | State field to read the value from.             |

**`<returns>` / `<map>` Attributes:**

| Attribute    | Required | Type   | Description                                                                  |
|--------------|----------|--------|------------------------------------------------------------------------------|
| `api_field`  | Yes      | string | Dot-notation path into the API response JSON (e.g., `data.user.email`).     |
| `state_field`| Yes      | string | State field to write the extracted value to.                                 |

**Note on `<returns>` and node context:** `<returns>` mappings are only applied when the tool is invoked via an `action` node (§4.4). When the same tool is bound to an LLM node via `<tool_bind>`, its return value flows through the message system instead (§4.3).

**Response validation:**

The generated tool code includes automatic response validation:

1. **Non-2xx status codes:** If the HTTP response status is not in the 2xx range and is not in `retry_status_codes` (or retries are exhausted), `httpx` raises an `HTTPStatusError`. This propagates as a tool error — either caught by the tool-calling loop (§4.3) or raised as a runtime error in an action node.
2. **Non-JSON responses:** If the response body cannot be decoded as JSON, a `ToolExecutionError` is raised with the status code, content type, and first 200 characters of the response body.
3. **Missing API fields:** If a dot-notation path in `<returns api_field="...">` does not resolve (e.g., the JSON lacks the expected key), the mapped `state_field` is set to `None` and a runtime warning is logged.
4. **Timeout:** If the request exceeds the configured `timeout`, `httpx` raises a `TimeoutException`. This is retryable if retries are configured.

### 6.4 Tool Type: `local_script`

Wraps a Python file from the `tools/` directory as a LangChain tool. Unlike script nodes (§4.5) which operate on the full state, `local_script` tools accept individual keyword arguments defined by a `SCHEMA` dictionary in the script. This makes them suitable for binding to LLM nodes, where the LLM needs to know what arguments the tool accepts.

```xml
<tool id="calculate_tax" type="local_script" path="tax_calc.py"
      description="Calculates tax for a given amount and region." />
```

| Attribute     | Required | Type   | Description                                                                |
|---------------|----------|--------|----------------------------------------------------------------------------|
| `id`          | Yes      | string | Unique tool identifier.                                                    |
| `type`        | Yes      | enum   | Must be `local_script`.                                                    |
| `path`        | Yes      | string | Relative path to a `.py` file in the `tools/` directory.                   |
| `description` | No       | string | Human-readable description for the LLM when this tool is bound to a node.  |

**Script contract (local_script tools):** The Python file must define:
- A dictionary `SCHEMA` that defines the tool's input parameters for the LLM.
- A function `run(**kwargs) -> dict` that accepts keyword arguments matching the schema and returns a dictionary.

```python
# tools/tax_calc.py

SCHEMA = {
    "amount": {"type": "float", "description": "The amount to calculate tax on"},
    "region": {"type": "string", "description": "The tax region code"}
}

def run(amount: float, region: str) -> dict:
    rates = {"UK": 0.20, "US": 0.08, "EU": 0.21}
    rate = rates.get(region, 0.20)
    return {
        "tax_amount": amount * rate,
        "total_with_tax": amount * (1 + rate)
    }
```

### 6.5 Tool Type: `script_tool`

A lightweight tool backed by a Python file, with its parameters declared in the AJML file rather than in a `SCHEMA` constant. This is useful for simple utility functions where you want the tool definition to be self-contained in the AJML.

```xml
<tool id="format_currency" type="script_tool" src="format_currency.py"
      description="Formats a number as currency.">
    <parameters>
        <param name="amount" type="float" description="The amount to format" />
        <param name="currency" type="string" description="Currency code (e.g., GBP, USD)" />
    </parameters>
</tool>
```

| Attribute     | Required | Type   | Description                                                                |
|---------------|----------|--------|----------------------------------------------------------------------------|
| `id`          | Yes      | string | Unique tool identifier.                                                    |
| `type`        | Yes      | enum   | Must be `script_tool`.                                                     |
| `src`         | Yes      | string | Relative path to a `.py` file in the `tools/` directory.                   |
| `description` | No       | string | Human-readable description for the LLM when this tool is bound to a node.  |

**Script contract (script_tool):** The Python file must define a function `run(**kwargs) -> dict`. No `SCHEMA` constant is needed — the AJML declaration is the schema.

```python
# tools/format_currency.py

def run(amount: float, currency: str) -> dict:
    symbols = {"GBP": "£", "USD": "$", "EUR": "€"}
    symbol = symbols.get(currency, currency)
    return {"formatted": f"{symbol}{amount:,.2f}"}
```

---

## 7. Multi-Agent Orchestration

### 7.1 Overview

AJML supports multi-agent workflows where one agent delegates to another. This is implemented through the `subgraph` node type, which compiles to LangGraph's subgraph mechanism. Each agent remains its own compiled unit with its own state schema; orchestration is handled through explicit input/output mapping.

### 7.2 The `subgraph` Node

```xml
<node id="run_verification" type="subgraph" agent_ref="verification_agent">
    <input_map>
        <map source="customer_id" target="user_id" />
        <map source="order_data" target="order_data" />
    </input_map>
    <output_map>
        <map source="is_verified" target="verification_result" />
        <map source="confidence_score" target="verification_confidence" />
    </output_map>
</node>
```

**`<node>` Attributes (type=`subgraph`):**

| Attribute   | Required | Type   | Description                                                                                |
|-------------|----------|--------|--------------------------------------------------------------------------------------------|
| `id`        | Yes      | string | Unique node identifier.                                                                    |
| `type`      | Yes      | enum   | Must be `subgraph`.                                                                        |
| `agent_ref` | Yes      | string | The `name` attribute of another `<agent>` in the project (i.e., another `.ajml` file).     |

**`<input_map>` / `<map>` Attributes:**

| Attribute | Required | Type   | Description                                                           |
|-----------|----------|--------|-----------------------------------------------------------------------|
| `source`  | Yes      | string | State field in the **parent** agent to read from.                     |
| `target`  | Yes      | string | State field in the **child** agent to write to.                       |

**`<output_map>` / `<map>` Attributes:**

| Attribute | Required | Type   | Description                                                           |
|-----------|----------|--------|-----------------------------------------------------------------------|
| `source`  | Yes      | string | State field in the **child** agent to read from on completion.        |
| `target`  | Yes      | string | State field in the **parent** agent to write the result to.           |

### 7.3 Compilation Behaviour

When the transpiler encounters a `subgraph` node, it:

1. Validates that `agent_ref` matches an existing `.ajml` file in the project.
2. Validates that all `<input_map>` targets match declared fields in the child agent's `<state>`.
3. Validates that all `<output_map>` sources match declared fields in the child agent's `<state>`.
4. Generates code that compiles the child agent's graph and adds it as a node in the parent graph.

### 7.4 State Isolation

Each sub-agent operates on its own state. The parent agent's state is **not** shared. Only the explicitly mapped fields are passed in and out. This prevents accidental coupling and makes agents independently testable.

### 7.5 Multi-Agent API Exposure

Every agent in the project gets its own API endpoint, regardless of whether it is also used as a subgraph. This means a `verification_agent` can be invoked both as a subgraph within another agent and as a standalone API endpoint.

---

## 8. Project Configuration

### 8.1 The `_project.ajml` File

Every AJML project must contain `agents/_project.ajml`. This file is not an agent — it defines project-wide settings that apply to all agents and the generated server.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project name="customer_support_system" ajml_version="2.0">
    <config>
        <llm provider="openai" model="gpt-4o" max_retries="3" />
        <server
            cors_origins="https://frontend.com, https://admin.com"
            auth_env="AJML_API_KEY"
            docs_public="false"
            port="8000"
            host="0.0.0.0"
        />
        <env>
            <var name="OPENAI_API_KEY" required="true" />
            <var name="CUSTOMER_API_TOKEN" required="true" />
            <var name="LOG_LEVEL" default="INFO" />
        </env>
    </config>
</project>
```

### 8.2 LLM Configuration (Project-Level Defaults)

**Element: `<llm>`**

| Attribute     | Required | Type   | Description                                                                                           |
|---------------|----------|--------|-------------------------------------------------------------------------------------------------------|
| `provider`    | Yes      | string | LLM provider identifier. Must be one of the supported providers listed in §8.4.                       |
| `model`       | Yes      | string | Model identifier as expected by the provider (e.g., `gpt-4o`, `claude-sonnet-4-20250514`).        |
| `max_retries` | No       | int    | Number of retries on transient LLM failures. Default: `2`.                                            |

### 8.3 Server Configuration

**Element: `<server>`**

| Attribute      | Required | Type    | Description                                                                                    |
|----------------|----------|---------|------------------------------------------------------------------------------------------------|
| `cors_origins` | No       | string  | Comma-separated list of allowed CORS origins. Defaults to `*` (all origins).                   |
| `auth_env`     | No       | string  | Environment variable name containing the expected API key. If omitted, no auth is required.    |
| `docs_public`  | No       | boolean | Whether `/docs`, `/redoc`, and `/openapi.json` are accessible without auth. Defaults to `true` when no auth is configured, `false` when `auth_env` is set. |
| `port`         | No       | int     | Server port. Default: `8000`.                                                                  |
| `host`         | No       | string  | Server host binding. Default: `0.0.0.0`.                                                      |

When `auth_env` is set, all agent endpoints require an `X-API-Key` header. The `/health` endpoint is always unauthenticated. Documentation endpoints (`/docs`, `/redoc`, `/openapi.json`) respect the `docs_public` setting.

### 8.4 Supported LLM Providers

AJML supports the following LLM providers. The `provider` attribute must match one of these identifiers exactly. Using an unlisted provider is a compilation error (`E401`).

| Provider        | LangChain Class          | Package                    | API Key Env Var (Convention)   |
|-----------------|--------------------------|----------------------------|-------------------------------|
| `openai`        | `ChatOpenAI`             | `langchain_openai`         | `OPENAI_API_KEY`              |
| `anthropic`     | `ChatAnthropic`          | `langchain_anthropic`      | `ANTHROPIC_API_KEY`           |
| `google`        | `ChatGoogleGenerativeAI` | `langchain_google_genai`   | `GOOGLE_API_KEY`              |
| `mistral`       | `ChatMistralAI`          | `langchain_mistralai`      | `MISTRAL_API_KEY`             |
| `groq`          | `ChatGroq`               | `langchain_groq`           | `GROQ_API_KEY`                |
| `ollama`        | `ChatOllama`             | `langchain_ollama`         | (none — local)                |
| `azure_openai`  | `AzureChatOpenAI`        | `langchain_openai`         | `AZURE_OPENAI_API_KEY`        |
| `bedrock`       | `ChatBedrock`            | `langchain_aws`            | (AWS credentials)             |

To request support for additional providers, open an issue on the AJML repository.

### 8.5 Environment Variables

**Element: `<env>` / `<var>`**

| Attribute  | Required | Type    | Description                                                                  |
|------------|----------|---------|------------------------------------------------------------------------------|
| `name`     | Yes      | string  | Environment variable name.                                                   |
| `required` | No       | boolean | If `true`, the server fails to start if this variable is not set.            |
| `default`  | No       | string  | Default value if the environment variable is not set.                        |

Environment variables declared here are validated at server startup. The transpiler generates a Pydantic `Settings` class using `pydantic-settings` for this purpose.

---

## 9. Agent Configuration

### 9.1 Per-Agent `<config>` Overrides

Individual agents can override the project-level LLM configuration by including their own `<config>` block. Only `<llm>` settings can be overridden per-agent. Server and environment settings are always project-level only.

```xml
<agent name="premium_support" version="1.0" description="Support using a more capable model.">
    <config>
        <llm provider="anthropic" model="claude-sonnet-4-20250514" max_retries="5" />
    </config>
    <state>...</state>
    <graph>...</graph>
</agent>
```

If an agent omits `<config>`, it inherits the project-level LLM settings entirely.

---

## 10. Transpiler Architecture

The transpiler processes `.ajml` files in four sequential phases. If any phase fails, compilation halts with a descriptive error.

### 10.1 Phase 1: Lexical Pre-processor

**Purpose:** Sanitise the `.ajml` file so its structure can be parsed.

**Context:** AJML uses XML-like syntax but is not strict XML. Specifically, `<condition>` elements may contain Python comparison operators (`<`, `>`, `<=`, `>=`) and `<system_prompt>` elements may contain arbitrary text including angle brackets. These would break a standard XML parser.

**Logic:**

1. Read the raw `.ajml` file as a string.
2. Identify all content within `<condition>...</condition>` and `<system_prompt>...</system_prompt>` tags using a regex-based scanner.
3. Constraints on the scanner: these elements must not contain nested XML-like tags or their own closing tag sequence. Content is plain text only.
4. Within matched regions:
   - Replace `<` → `&lt;` (but not `</` which signals a closing tag)
   - Replace `>` → `&gt;`
   - Replace bare `&` → `&amp;` (not already-escaped entities)
5. Pass the sanitised string to `xml.etree.ElementTree.fromstring()`.
6. During code generation (Phase 3), all `&lt;`, `&gt;`, and `&amp;` within condition and prompt content are un-escaped back to their original characters.

**Whitespace handling in prompts:** Leading and trailing whitespace on the entire `<system_prompt>` content is trimmed. Internal whitespace (newlines, indentation) is preserved as-is. This means the prompt the LLM receives matches the indentation in the AJML file, minus the outer margins.

**Output:** An `ElementTree` root element with all structural elements parseable.

### 10.2 Phase 2: AST Validator

**Purpose:** Validate the parsed structure against AJML's structural and semantic rules.

The validator performs all checks listed in the Error Catalogue (§15). Key validation groups:

- **Structure:** Valid root element, required blocks present.
- **Uniqueness:** Node IDs, tool IDs, state field names, agent names.
- **Type validity:** Node types, tool types, state field types, reducer compatibility.
- **Graph integrity:** Start node, edge targets/sources, conditional group rules, reachability, tool references, script file existence.
- **Cross-agent:** Subgraph references, circular dependencies, project-wide agent name uniqueness.
- **Condition analysis:** Static analysis of condition expressions to verify they only reference names in the restricted context (§5.4).
- **Parallel write detection:** Warning when parallel branches may write to the same non-reduced field.

### 10.3 Phase 3: Code Generator

**Purpose:** Emit Python source code from the validated AST.

The code generator uses string templates to produce readable, deterministic Python code. See §11 for the complete template specifications.

**Determinism guarantees:**
- Imports are ordered alphabetically by module name.
- Nodes and tools are emitted in document order (order of appearance in the `.ajml` file).
- Generated class names follow the pattern `{NodeId}Output` (PascalCase) and `{ToolId}Input` (PascalCase).
- Consistent newline policy: single blank line between functions, double blank line between classes.

**Interpolation handling:** During code generation, `${field_name}` references in system prompts are converted to Python string formatting: `state.get('field_name', '')`. The `${env:VAR}` references are converted to `os.getenv('VAR')` calls. All other text (including bare `{` and `}`) is preserved as-is.

### 10.4 Phase 4: Output Assembly

**Purpose:** Produce the final build output.

1. For each `.ajml` agent file, emit `build/compiled_{agent_name}.py`.
2. Emit `build/main.py` which:
   - Imports all compiled agents.
   - Configures FastAPI with CORS and auth (from `_project.ajml`).
   - Registers POST endpoints for each agent.
   - Registers a GET `/health` endpoint.
   - Configures documentation endpoint visibility.
3. Emit `build/requirements.txt` with all required Python packages.
4. Emit `build/.env.example` with all declared environment variables.

---

## 11. Generated Code Specification

This section defines the exact Python code the transpiler must produce for each AJML construct.

### 11.1 State → TypedDict

**AJML:**
```xml
<state>
    <field name="customer_id" type="string" required="true" />
    <field name="refund_amount" type="float" default="0.0" />
    <field name="results" type="list" reducer="append" />
    <field name="retry_count" type="int" reducer="add" default="0" />
    <field name="internal_trace" type="string" expose="false" default="" />
</state>
```

**Python:**
```python
import operator
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    customer_id: str
    refund_amount: float
    results: Annotated[list, operator.add]
    retry_count: Annotated[int, operator.add]
    internal_trace: str
```

**Type mapping:**

| AJML Type        | Python Type       |
|------------------|-------------------|
| `string`         | `str`             |
| `int`            | `int`             |
| `float`          | `float`           |
| `bool`           | `bool`            |
| `list`           | `list`            |
| `list[string]`   | `list[str]`       |
| `list[int]`      | `list[int]`       |
| `list[float]`    | `list[float]`     |
| `list[dict]`     | `list[dict]`      |
| `dict`           | `dict`            |
| `dict[string]`   | `dict[str, str]`  |
| `dict[int]`      | `dict[str, int]`  |
| `enum`           | `str` (internal) / `Literal[...]` (API models) |

**Reducer mapping:**

| AJML Reducer | Python Annotation                                    |
|--------------|------------------------------------------------------|
| `overwrite`  | (no annotation — bare type)                          |
| `append`     | `Annotated[list, operator.add]`                      |
| `add`        | `Annotated[int, operator.add]`                       |
| `merge`      | `Annotated[dict, lambda a, b: {**a, **b}]`          |
| `concat`     | `Annotated[str, operator.concat]`                    |

### 11.2 LLM Node with Structured Output

**AJML:**
```xml
<node id="classify_intent" type="llm">
    <system_prompt>
        You are an intent classifier for customer ${customer_id}.
        Classify as: refund, complaint, question, other.
    </system_prompt>
    <output_schema>
        <field name="intent" type="string" description="The classified intent" />
    </output_schema>
</node>
```

**Python:**
```python
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel, Field


class ClassifyIntentOutput(BaseModel):
    intent: str = Field(description="The classified intent")


def classify_intent(state: AgentState):
    system_content = (
        f"You are an intent classifier for customer {state.get('customer_id', '')}.\n"
        f"Classify as: refund, complaint, question, other."
    )
    messages = [SystemMessage(content=system_content)] + state["messages"]
    
    structured_llm = llm.with_structured_output(ClassifyIntentOutput)
    result = structured_llm.invoke(messages)
    
    updates = {}
    result_dict = result.model_dump()
    for key, value in result_dict.items():
        if key in AgentState.__annotations__:
            updates[key] = value
    
    updates["messages"] = [AIMessage(content=str(result_dict))]
    return updates
```

### 11.3 LLM Node with Tool Binding → Agent Loop

**AJML:**
```xml
<node id="research" type="llm">
    <system_prompt>Research the customer's issue using available tools.</system_prompt>
    <tool_bind ref="fetch_user" />
    <tool_bind ref="search_knowledge_base" />
</node>
```

**Python:**
```python
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition


def research(state: AgentState):
    system_content = "Research the customer's issue using available tools."
    messages = [SystemMessage(content=system_content)] + state["messages"]
    
    bound_llm = llm.bind_tools([fetch_user, search_knowledge_base])
    response = bound_llm.invoke(messages)
    
    return {"messages": [response]}


research_tools = ToolNode(
    [fetch_user, search_knowledge_base],
    handle_tool_errors=True
)


def route_research(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "research_tools"
    return "__next__"
```

The transpiler generates the tool-calling loop as an internal sub-cycle: `research` → `research_tools` → `research`, breaking when the LLM stops issuing tool calls.

### 11.4 Action Node → Direct Tool Call with State Writeback

**AJML:**
```xml
<node id="execute_refund" type="action" tool_ref="process_refund" />
```

**Python:**
```python
def execute_refund(state: AgentState):
    result = process_refund.invoke({
        "customer_id": state.get("customer_id"),
        "amount": state.get("refund_amount")
    })
    # Filter to known state fields only
    updates = {}
    for key, value in result.items():
        if key in AgentState.__annotations__:
            updates[key] = value
    return updates
```

### 11.5 API Tool → LangChain @tool

**AJML:**
```xml
<tool id="fetch_user" type="api_call" max_retries="2" timeout="15"
      retry_status_codes="429, 500, 502" backoff="exponential" backoff_base="1.0">
    <endpoint url="https://api.example.com/users/${id}" method="GET" />
    <headers>
        <header name="Authorization" value="Bearer ${env:USER_API_TOKEN}" />
    </headers>
    <parameters>
        <param name="customer_id" type="string" map_to="id" in="path"
               description="The customer's unique ID" />
    </parameters>
    <returns>
        <map api_field="data" state_field="user_data" />
    </returns>
</tool>
```

**Python:**
```python
import logging
import os

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class FetchUserInput(BaseModel):
    customer_id: str = Field(description="The customer's unique ID")


class ToolExecutionError(Exception):
    pass


@tool("fetch_user", args_schema=FetchUserInput)
@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1.0, min=1.0, max=60.0),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
)
def fetch_user(customer_id: str) -> dict:
    """Fetches user data from the API."""
    url = f"https://api.example.com/users/{customer_id}"
    headers = {
        "Authorization": f"Bearer {os.getenv('USER_API_TOKEN')}"
    }
    response = httpx.get(url, headers=headers, timeout=15.0)
    
    # Raise for retryable status codes
    if response.status_code in (429, 500, 502):
        response.raise_for_status()
    # Raise for other non-2xx codes (non-retryable)
    response.raise_for_status()
    
    # Validate JSON response
    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type:
        raise ToolExecutionError(
            f"Expected JSON response, got {content_type}: "
            f"{response.text[:200]}"
        )
    
    data = response.json()
    
    # Map response fields to state, with None for missing paths
    user_data = data.get("data")
    if user_data is None:
        logger.warning("fetch_user: API field 'data' not found in response")
    
    return {"user_data": user_data}
```

**Backoff strategies:**

| Strategy      | Tenacity Equivalent                                    | Behaviour                              |
|---------------|--------------------------------------------------------|----------------------------------------|
| `fixed`       | `wait_fixed(backoff_base)`                             | Constant delay between retries.        |
| `exponential` | `wait_exponential(multiplier=backoff_base, min=backoff_base, max=60.0)` | Delay doubles each attempt, capped at 60s. |

### 11.6 Conditional Edges → Routing Function

**AJML:**
```xml
<edge source="check_policy" target="execute_refund">
    <condition>state.get('refund_amount') > 0 and state.get('is_eligible') == True</condition>
</edge>
<edge source="check_policy" target="escalate">
    <condition>state.get('refund_amount') > 1000</condition>
</edge>
<edge source="check_policy" target="deny_refund" default="true" />
```

**Python:**
```python
def route_check_policy(state: AgentState):
    if state.get('refund_amount') > 0 and state.get('is_eligible') == True:
        return "execute_refund"
    if state.get('refund_amount') > 1000:
        return "escalate"
    return "deny_refund"


graph.add_conditional_edges("check_policy", route_check_policy)
```

### 11.7 Parallel Edges → Multiple add_edge Calls

**AJML:**
```xml
<edge source="__START__" target="fetch_user_profile" />
<edge source="__START__" target="fetch_order_history" />
```

**Python:**
```python
graph.add_edge(START, "fetch_user_profile")
graph.add_edge(START, "fetch_order_history")
```

### 11.8 Map-Reduce → Send API

**AJML:**
```xml
<edge source="generate_topics" target="write_section" type="map">
    <map_config items_field="topics" item_var="current_topic" />
</edge>
```

**Python:**
```python
from langgraph.constants import Send


def route_generate_topics(state: AgentState):
    items = state.get("topics", [])
    if not items:
        return []  # Skip — empty list
    return [
        Send("write_section", {**state, "current_topic": item})
        for item in items
    ]


graph.add_conditional_edges("generate_topics", route_generate_topics, ["write_section"])
```

### 11.9 Server → FastAPI Main

See §12 for the full generated server specification.

### 11.10 Graph Assembly

Each compiled agent file ends with the graph assembly:

```python
from langgraph.graph import END, START, StateGraph

graph_builder = StateGraph(AgentState)

# Add all nodes
graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("handle_refund", handle_refund)
graph_builder.add_node("handle_general", handle_general)

# Add edges
graph_builder.add_edge(START, "classify_intent")
graph_builder.add_conditional_edges("classify_intent", route_classify_intent)
graph_builder.add_edge("handle_refund", END)
graph_builder.add_edge("handle_general", END)

# Compile
graph = graph_builder.compile()
```

---

## 12. Auto-Generated API & Documentation

### 12.1 OpenAPI Schema Generation

The transpiler generates a full OpenAPI 3.1 specification from the AJML definitions. This is served by FastAPI at `/docs` (Swagger UI) and `/redoc` (ReDoc), subject to the `docs_public` setting (§8.3).

### 12.2 Schema Derivation

For each agent, the transpiler generates:

- **Request model:** A Pydantic `BaseModel` with all `required="true"` state fields as required properties and all other fields with defaults as optional properties.
- **Response model:** A Pydantic `BaseModel` with all state fields that have `expose="true"` (the default), plus `messages` serialised as a list of role/content dictionaries.

### 12.3 Standard Endpoints

Every generated server includes:

| Method | Path               | Auth     | Description                       |
|--------|-------------------|----------|-----------------------------------|
| GET    | `/health`          | Never    | Returns `{"status": "ok"}`.       |
| GET    | `/docs`            | Depends* | Swagger UI.                       |
| GET    | `/redoc`           | Depends* | ReDoc documentation.              |
| GET    | `/openapi.json`    | Depends* | Raw OpenAPI schema.               |
| POST   | `/run/{agent_name}`| Yes**    | Execute the named agent.          |

*Controlled by `docs_public` setting. When `false`, these endpoints require the same `X-API-Key` header.

**Auth required when `auth_env` is configured.

### 12.4 Request/Response Format

**Request:**
```json
POST /run/refund_agent
Content-Type: application/json
X-API-Key: your-api-key

{
    "customer_id": "CUST-123",
    "complaint_text": "I want a refund for my broken item."
}
```

**Response (success):**
```json
HTTP/1.1 200 OK

{
    "customer_id": "CUST-123",
    "complaint_text": "I want a refund for my broken item.",
    "intent": "refund",
    "refund_amount": 49.99,
    "refund_approved": true,
    "messages": [
        {"role": "system", "content": "You are an intent classifier..."},
        {"role": "human", "content": "I want a refund for my broken item."},
        {"role": "assistant", "content": "{\"intent\": \"refund\", \"refund_amount\": 49.99}"}
    ]
}
```

Note: Fields with `expose="false"` (e.g., `internal_trace`, `api_token_cache`) are absent from the response.

### 12.5 Error Response Format

All runtime errors use a standardised error envelope:

```json
HTTP/1.1 500 Internal Server Error

{
    "detail": "Agent execution failed",
    "error_code": "R500",
    "error": "ToolExecutionError: httpx.HTTPStatusError: 404 Not Found",
    "agent": "refund_agent",
    "node": "fetch_customer_data",
    "request_id": "req_a1b2c3d4"
}
```

| Field        | Always Present | Description                                                    |
|--------------|----------------|----------------------------------------------------------------|
| `detail`     | Yes            | Human-readable summary.                                       |
| `error_code` | Yes            | Runtime error code (see §15.2).                                |
| `error`      | Depends        | Error details. Only included when `LOG_LEVEL` is `DEBUG`.      |
| `agent`      | Yes            | Agent name where the error occurred.                           |
| `node`       | When available | Node ID where the error occurred (if identifiable).            |
| `request_id` | Yes            | Unique request identifier for tracing.                         |

---

## 13. CLI Reference

### 13.1 Build Command

```bash
python -m ajml build [project_dir] [options]
```

| Argument/Flag     | Required | Default | Description                                              |
|-------------------|----------|---------|----------------------------------------------------------|
| `project_dir`     | No       | `.`     | Path to the project root (containing `agents/`).         |
| `--output`, `-o`  | No       | `build` | Output directory name (relative to project root).        |
| `--verbose`, `-v` | No       | `false` | Enable verbose logging of each compilation phase.        |
| `--strict`        | No       | `false` | Treat warnings as errors.                                |
| `--dry-run`       | No       | `false` | Validate only — don't emit any files.                    |

**Process:**

1. Load `agents/_project.ajml` and validate project config.
2. Scan `agents/` for all other `.ajml` files.
3. Phase 1: Pre-process each file.
4. Phase 2: Validate all ASTs (including cross-agent checks).
5. Phase 3: Generate code for each agent.
6. Phase 4: Assemble `build/main.py`, `build/requirements.txt`, and `build/.env.example`.
7. Print a summary of agents compiled and endpoints registered.

**Example output:**
```
$ python -m ajml build .

AJML Transpiler v2.0
====================
Project: customer_support_system (ajml v2.0)
Scanning agents/ ... found 3 agent files + _project.ajml.

[1/3] refund_agent.ajml
  ✓ Pre-processed (2 conditions sanitised)
  ✓ Validated (4 nodes, 6 edges, 2 tools)
  ⚠ W301: Parallel branches from __START__ may both write to 'status' (overwrite reducer)
  ✓ Generated build/compiled_refund_agent.py

[2/3] verification_agent.ajml
  ✓ Pre-processed
  ✓ Validated (2 nodes, 3 edges, 1 tool)
  ✓ Generated build/compiled_verification_agent.py

[3/3] support_router.ajml
  ✓ Pre-processed (1 condition sanitised)
  ✓ Validated (3 nodes, 5 edges, 0 tools, 1 subgraph)
  ✓ Generated build/compiled_support_router.py

Assembling server...
  ✓ build/main.py (3 endpoints)
  ✓ build/requirements.txt
  ✓ build/.env.example

Build complete (1 warning). Run with:
  cd build && uvicorn main:app --reload
```

### 13.2 Validate Command

```bash
python -m ajml validate [project_dir]
```

Runs Phases 1 and 2 only. Useful for CI/CD pipelines and pre-commit hooks.

### 13.3 Init Command

```bash
python -m ajml init [project_name]
```

Scaffolds a new AJML project:

```
project_name/
├── agents/
│   ├── _project.ajml
│   └── example_agent.ajml
├── tools/
│   └── example_tool.py
├── build/           (empty, .gitignored)
├── .env.example
└── .gitignore
```

### 13.4 Exit Codes

| Code | Meaning                              |
|------|--------------------------------------|
| `0`  | Success.                             |
| `1`  | Compilation error (validation failed). |
| `2`  | File system error (missing directories, permission denied). |
| `3`  | Internal transpiler error (bug).     |

---

## 14. Directory Structure & Conventions

### 14.1 Standard Layout

```
my_project/
├── agents/                     # Source files
│   ├── _project.ajml           # Project config (required)
│   ├── refund_agent.ajml
│   ├── verification_agent.ajml
│   └── support_router.ajml
├── tools/                      # Local Python scripts
│   ├── tax_calc.py
│   ├── format_currency.py
│   └── noop.py
├── build/                      # Generated (do not edit)
│   ├── compiled_refund_agent.py
│   ├── compiled_verification_agent.py
│   ├── compiled_support_router.py
│   ├── main.py
│   ├── requirements.txt
│   └── .env.example
├── .env                        # Local environment variables (gitignored)
└── .gitignore
```

### 14.2 Naming Conventions

- **Project file:** Always `_project.ajml`. The underscore prefix ensures it sorts first and is visually distinct from agent files.
- **Agent files:** `snake_case.ajml`. The filename (without extension) should match the `<agent name="...">` attribute.
- **Tool scripts:** `snake_case.py`. Referenced by filename in AJML.
- **Build output:** `compiled_{agent_name}.py`. Generated by the transpiler; never edited manually.

### 14.3 The `build/` Directory

This directory is entirely generated and should be:
- Listed in `.gitignore`.
- Never edited manually.
- Regenerated on every `ajml build`.
- Treated as an ephemeral artefact.

---

## 15. Error Catalogue

### 15.1 Compilation Errors

All compilation errors follow the format:

```
AJMLCompilationError [E{code}]: {message}
  → File: agents/{filename}.ajml, Line: {line}
```

| Code   | Category          | Message Template                                                                                |
|--------|-------------------|-------------------------------------------------------------------------------------------------|
| `E001` | Structure         | Root element must be `<agent>` with a `name` attribute (or `<project>` for _project.ajml).     |
| `E002` | Structure         | Required block `<{block}>` is missing.                                                          |
| `E003` | Structure         | Missing `_project.ajml` file. Every AJML project requires a project configuration file.        |
| `E004` | Structure         | Invalid `ajml_version` in _project.ajml. Supported versions: 2.0.                              |
| `E101` | Uniqueness        | Duplicate node ID `{id}`. Node IDs must be unique within an agent.                              |
| `E102` | Uniqueness        | Duplicate tool ID `{id}`. Tool IDs must be unique within an agent.                              |
| `E103` | Uniqueness        | Duplicate state field name `{name}`.                                                            |
| `E104` | Uniqueness        | `{identifier}` is a reserved word and cannot be used as a node/tool/field name.                 |
| `E105` | Type Validity     | Invalid node type `{type}`. Must be one of: llm, action, script, subgraph.                     |
| `E106` | Type Validity     | Invalid tool type `{type}`. Must be one of: api_call, local_script, script_tool.                |
| `E107` | Type Validity     | Invalid state field type `{type}`. Must be a valid simple type, parameterised type, or enum. See §3.8. |
| `E108` | Type Validity     | Reducer `{reducer}` is not valid for type `{type}`. Valid reducers: {valid_list}.               |
| `E109` | Type Validity     | Enum default value `{default}` is not in the declared values list: {values}.                    |
| `E201` | State             | Cannot declare state field `messages`. This field is implicitly managed by the framework.       |
| `E202` | State             | Required field `{name}` cannot have a default value.                                            |
| `E301` | Graph Integrity   | No entry point found. At least one edge must have `source="__START__"`.                         |
| `E302` | Graph Integrity   | Edge target `{target}` does not match any declared node ID or `__END__`.                        |
| `E303` | Graph Integrity   | Edge source `{source}` does not match any declared node ID or `__START__`.                      |
| `E304` | Graph Integrity   | Conditional edge group from `{source}` is missing a `default="true"` edge.                     |
| `E305` | Graph Integrity   | Edge with `default="true"` must not also contain a `<condition>`.                               |
| `E306` | Graph Integrity   | Multiple `default="true"` edges from source `{source}`.                                        |
| `E307` | Graph Integrity   | Tool reference `{ref}` does not match any declared tool ID.                                     |
| `E308` | Graph Integrity   | Script file `tools/{path}` does not exist.                                                      |
| `E309` | Graph Integrity   | Agent reference `{agent_ref}` does not match any .ajml file in the project.                     |
| `E310` | Graph Integrity   | Input map target `{target}` does not exist in child agent `{agent_ref}` state.                  |
| `E311` | Graph Integrity   | Output map source `{source}` does not exist in child agent `{agent_ref}` state.                 |
| `E312` | Graph Integrity   | Node `{id}` is unreachable from `__START__`.                                                    |
| `E313` | Graph Integrity   | Node `{id}` has no outgoing edges.                                                              |
| `E314` | Graph Integrity   | Mixed edge types from source `{source}`. A node's outgoing edges must be all unconditional, all conditional, or a single map edge. |
| `E315` | Graph Integrity   | Map edge `items_field` `{field}` must reference a list-type state field.                        |
| `E316` | Graph Integrity   | Action node `{id}` references tool `{tool_ref}` but required parameters cannot be resolved from state. |
| `E401` | Configuration     | Unknown LLM provider `{provider}`. Must be one of: openai, anthropic, google, mistral, groq, ollama, azure_openai, bedrock. |
| `E402` | Configuration     | Circular subgraph dependency detected: {cycle_path}.                                            |
| `E403` | Configuration     | Duplicate agent name `{name}` across project.                                                   |
| `E501` | Expression        | Condition expression references undefined name `{name}`. Only `state` and allowed builtins are available. |
| `E502` | Interpolation     | Invalid interpolation syntax `{token}`. Use `${field}` for state or `${env:VAR}` for environment variables. |

### 15.2 Compilation Warnings

| Code   | Description                                                                                      |
|--------|--------------------------------------------------------------------------------------------------|
| `W301` | Parallel branches from `{source}` may both write to field `{field}` which uses the `overwrite` reducer. Result may be non-deterministic. |

### 15.3 Runtime Errors

| Error Code | HTTP Status | Description                                                       |
|------------|-------------|-------------------------------------------------------------------|
| `R422`     | 422         | Request body missing a required state field.                      |
| `R403`     | 403         | `X-API-Key` header missing or incorrect.                          |
| `R500`     | 500         | Agent execution failed (tool error, script error, unexpected).    |
| `R502`     | 502         | Upstream LLM API returned an error after all retries.             |
| `R504`     | 504         | Agent graph exceeded maximum execution time.                      |
| `R529`     | 500         | Graph exceeded the maximum number of super-steps (recursion limit). |

---

## 16. Examples

### 16.1 Simple Example: Intent Classifier

A minimal agent that classifies customer messages into categories.

**File: `agents/_project.ajml`**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<project name="classifier_demo" ajml_version="2.0">
    <config>
        <llm provider="openai" model="gpt-4o-mini" />
    </config>
</project>
```

**File: `agents/intent_classifier.ajml`**
```xml
<?xml version="1.0" encoding="UTF-8"?>
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
</agent>
```

### 16.2 Intermediate Example: Customer Support Agent with Tools

An agent that classifies intent, fetches customer data in parallel, and either processes a refund or provides general help.

**File: `agents/_project.ajml`**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<project name="customer_support" ajml_version="2.0">
    <config>
        <llm provider="anthropic" model="claude-sonnet-4-20250514" max_retries="3" />
        <server cors_origins="https://support.example.com" auth_env="SUPPORT_API_KEY" />
        <env>
            <var name="ANTHROPIC_API_KEY" required="true" />
            <var name="CUSTOMER_API_TOKEN" required="true" />
            <var name="REFUND_SERVICE_URL" required="true" />
        </env>
    </config>
</project>
```

**File: `agents/support_agent.ajml`**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<agent name="support_agent" version="1.0"
       description="Handles customer support requests with automatic refund processing.">
    
    <state>
        <field name="customer_id" type="string" required="true" />
        <field name="message" type="string" required="true" />
        <field name="intent" type="enum" values="refund, complaint, question, other" default="" />
        <field name="customer_data" type="dict" default="{}" expose="false" />
        <field name="refund_amount" type="float" default="0.0" />
        <field name="refund_processed" type="bool" default="false" />
        <field name="response_text" type="string" default="" />
    </state>
    
    <tools>
        <tool id="fetch_customer" type="api_call" max_retries="2" timeout="10">
            <endpoint url="https://api.example.com/customers/${id}" method="GET" />
            <headers>
                <header name="Authorization" value="Bearer ${env:CUSTOMER_API_TOKEN}" />
            </headers>
            <parameters>
                <param name="customer_id" type="string" map_to="id" in="path"
                       description="Customer ID to look up" />
            </parameters>
            <returns>
                <map api_field="data" state_field="customer_data" />
            </returns>
        </tool>
        
        <tool id="process_refund" type="api_call" max_retries="1">
            <endpoint url="${env:REFUND_SERVICE_URL}/refunds" method="POST" />
            <headers>
                <header name="Authorization" value="Bearer ${env:CUSTOMER_API_TOKEN}" />
            </headers>
            <parameters>
                <param name="customer_id" type="string" map_to="customer_id" in="query"
                       description="Customer to refund" />
            </parameters>
            <body format="json">
                <field name="amount" type="float" from_state="refund_amount" />
            </body>
            <returns>
                <map api_field="success" state_field="refund_processed" />
            </returns>
        </tool>
    </tools>
    
    <graph>
        <!-- Step 1: Classify intent and fetch customer data in parallel -->
        <node id="classify_intent" type="llm">
            <system_prompt>
                Classify the customer's message as one of: refund, complaint, question, other.
                If it's a refund, also determine the refund amount if mentioned.
            </system_prompt>
            <output_schema>
                <field name="intent" type="string" description="Intent category" />
                <field name="refund_amount" type="float"
                       description="Refund amount, 0 if not applicable" />
            </output_schema>
        </node>
        
        <node id="fetch_customer_data" type="action" tool_ref="fetch_customer" />
        
        <!-- Synchronisation point for parallel branches -->
        <node id="sync" type="script" path="noop.py" />
        
        <!-- Step 2: Route based on intent -->
        <node id="handle_refund" type="action" tool_ref="process_refund" />
        
        <node id="generate_response" type="llm">
            <system_prompt>
                You are a friendly customer support agent.
                Customer data: ${customer_data}
                Intent: ${intent}
                Refund processed: ${refund_processed}
                
                Generate a helpful, empathetic response to the customer.
            </system_prompt>
            <output_schema>
                <field name="response_text" type="string"
                       description="The response to send to the customer" />
            </output_schema>
        </node>
        
        <!-- Edges: parallel fan-out from start -->
        <edge source="__START__" target="classify_intent" />
        <edge source="__START__" target="fetch_customer_data" />
        
        <!-- Fan-in at sync point -->
        <edge source="classify_intent" target="sync" />
        <edge source="fetch_customer_data" target="sync" />
        
        <!-- Conditional routing -->
        <edge source="sync" target="handle_refund">
            <condition>state.get('intent') == 'refund' and state.get('refund_amount') > 0</condition>
        </edge>
        <edge source="sync" target="generate_response" default="true" />
        
        <edge source="handle_refund" target="generate_response" />
        <edge source="generate_response" target="__END__" />
    </graph>
</agent>
```

**File: `tools/noop.py`**
```python
# No-op script used as a synchronisation point for parallel branches.
def run(state: dict) -> dict:
    return {}
```

### 16.3 Complex Example: Multi-Agent Research Pipeline

A project with three agents: a router, a researcher, and a summariser. Full source in separate files.

**File: `agents/_project.ajml`**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<project name="research_pipeline" ajml_version="2.0">
    <config>
        <llm provider="openai" model="gpt-4o" max_retries="3" />
        <server cors_origins="*" auth_env="RESEARCH_API_KEY" />
        <env>
            <var name="OPENAI_API_KEY" required="true" />
            <var name="SEARCH_API_KEY" required="true" />
        </env>
    </config>
</project>
```

**File: `agents/research_router.ajml`**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<agent name="research_router" version="1.0"
       description="Routes research queries to specialised sub-agents.">
    
    <state>
        <field name="query" type="string" required="true" />
        <field name="research_type" type="enum" values="deep_dive, quick_lookup" default="" />
        <field name="research_results" type="list[string]" reducer="append" />
        <field name="final_summary" type="string" default="" />
    </state>
    
    <graph>
        <node id="classify_query" type="llm">
            <system_prompt>
                Classify this research query into a type:
                - "deep_dive" for questions requiring thorough multi-source research
                - "quick_lookup" for simple factual questions
            </system_prompt>
            <output_schema>
                <field name="research_type" type="string"
                       description="Type of research needed" />
            </output_schema>
        </node>
        
        <node id="run_deep_research" type="subgraph" agent_ref="deep_researcher">
            <input_map>
                <map source="query" target="research_query" />
            </input_map>
            <output_map>
                <map source="findings" target="research_results" />
            </output_map>
        </node>
        
        <node id="run_quick_lookup" type="llm">
            <system_prompt>Provide a concise, factual answer to: ${query}</system_prompt>
            <output_schema>
                <field name="final_summary" type="string" description="The answer" />
            </output_schema>
        </node>
        
        <node id="run_summariser" type="subgraph" agent_ref="summariser">
            <input_map>
                <map source="research_results" target="source_material" />
                <map source="query" target="original_query" />
            </input_map>
            <output_map>
                <map source="summary" target="final_summary" />
            </output_map>
        </node>
        
        <edge source="__START__" target="classify_query" />
        
        <edge source="classify_query" target="run_deep_research">
            <condition>state.get('research_type') == 'deep_dive'</condition>
        </edge>
        <edge source="classify_query" target="run_quick_lookup" default="true" />
        
        <edge source="run_deep_research" target="run_summariser" />
        <edge source="run_quick_lookup" target="__END__" />
        <edge source="run_summariser" target="__END__" />
    </graph>
</agent>
```

**File: `agents/deep_researcher.ajml`**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<agent name="deep_researcher" version="1.0"
       description="Performs thorough research using multiple tools.">
    
    <state>
        <field name="research_query" type="string" required="true" />
        <field name="search_topics" type="list[string]" default="[]" />
        <field name="current_topic" type="string" default="" />
        <field name="findings" type="list[string]" reducer="append" />
    </state>
    
    <tools>
        <tool id="web_search" type="script_tool" src="web_search.py"
              description="Searches the web for information on a given query.">
            <parameters>
                <param name="query" type="string" description="Search query" />
            </parameters>
        </tool>
    </tools>
    
    <graph>
        <node id="plan_research" type="llm">
            <system_prompt>
                Break this research query into 2-4 specific search topics:
                "${research_query}"
                Return a list of focused search topics.
            </system_prompt>
            <output_schema>
                <field name="search_topics" type="list"
                       description="List of search topic strings" />
            </output_schema>
        </node>
        
        <node id="research_topic" type="llm">
            <system_prompt>
                Research the topic: ${current_topic}
                Use the search tool to find relevant information,
                then summarise your findings.
            </system_prompt>
            <tool_bind ref="web_search" />
            <output_schema>
                <field name="findings" type="list"
                       description="List of finding strings" />
            </output_schema>
        </node>
        
        <edge source="__START__" target="plan_research" />
        
        <edge source="plan_research" target="research_topic" type="map">
            <map_config items_field="search_topics" item_var="current_topic" />
        </edge>
        
        <edge source="research_topic" target="__END__" />
    </graph>
</agent>
```

**File: `agents/summariser.ajml`**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<agent name="summariser" version="1.0"
       description="Summarises research findings into a coherent report.">
    
    <state>
        <field name="source_material" type="list" required="true" />
        <field name="original_query" type="string" required="true" />
        <field name="summary" type="string" default="" />
    </state>
    
    <graph>
        <node id="synthesise" type="llm">
            <system_prompt>
                You are a research summariser. Given the following findings,
                create a clear, well-structured summary that directly answers
                the original query.
                
                Original query: ${original_query}
                
                Findings:
                ${source_material}
            </system_prompt>
            <output_schema>
                <field name="summary" type="string"
                       description="The synthesised summary" />
            </output_schema>
        </node>
        
        <edge source="__START__" target="synthesise" />
        <edge source="synthesise" target="__END__" />
    </graph>
</agent>
```

**File: `tools/web_search.py`**
```python
import os
import httpx

def run(query: str) -> dict:
    response = httpx.get(
        "https://api.search.example.com/search",
        params={"q": query},
        headers={"Authorization": f"Bearer {os.getenv('SEARCH_API_KEY')}"}
    )
    response.raise_for_status()
    return {"results": response.json().get("results", [])}
```

---

## 17. Future Roadmap

Features explicitly excluded from v2.0 but planned for future versions:

### 17.1 Human-in-the-Loop (Planned: v3.0)
LangGraph supports interrupt/resume patterns for human approval workflows. AJML will eventually support a `<node type="human_review">` that pauses the graph, exposes a review endpoint, and resumes on approval. This requires a persistence layer (checkpointing) which adds deployment complexity, hence the deferral.

### 17.2 Streaming (Planned: v2.1)
Server-Sent Events (SSE) streaming of intermediate state updates and LLM token generation. This would add a `/stream/{agent_name}` endpoint alongside the existing synchronous `/run/{agent_name}`.

### 17.3 Checkpointing & Persistence (Planned: v3.0)
LangGraph supports checkpointing to databases for long-running workflows. AJML will support a `<config><persistence backend="postgres" />` pattern to enable this.

### 17.4 Observability Hooks (Planned: v2.1)
Integration with LangSmith, OpenTelemetry, or custom logging. A `<config><observability provider="langsmith" />` pattern.

### 17.5 Visual Graph Editor (Planned: v3.0+)
A browser-based GUI for designing AJML graphs visually, generating `.ajml` files from drag-and-drop workflows.

### 17.6 Plugin System (Planned: v2.2)
Allow community-contributed tool types, LLM providers, and output formatters to be installed and referenced in AJML files. This would also enable adding new LLM providers without waiting for a spec update.

### 17.7 Advanced Type Constraints (Planned: v2.1)
Nested parameterised types (e.g., `list[list[string]]`), custom Pydantic validators on state fields, and runtime item-level type enforcement for parameterised types.

---

## Appendix A: Quick Reference — Tool Types vs Node Types

| Concept       | Declared in | Receives              | Returns     | State writeback        | Use case                                  |
|---------------|-------------|-----------------------|-------------|------------------------|-------------------------------------------|
| `script` node | `<graph>`   | Full state dict       | State updates | Yes (direct)          | Data transforms, routing sync, validation |
| `local_script` tool | `<tools>` | Kwargs per `SCHEMA` | dict        | Only via action node   | LLM-callable tool with self-contained schema |
| `script_tool` tool  | `<tools>` | Kwargs per AJML `<parameters>` | dict | Only via action node | LLM-callable tool with AJML-declared schema |
| `api_call` tool | `<tools>` | Kwargs per AJML `<parameters>` | dict (via `<returns>` mapping) | Only via action node | HTTP API integration with retries, timeout, backoff |
| `action` node | `<graph>`   | State → tool params   | State updates | Yes (via `<returns>`) | Deterministic tool execution (no LLM)    |
| `llm` + `tool_bind` | `<graph>` | Messages            | Messages    | No (message-only)      | LLM-driven tool use in conversation loop  |

## Appendix B: Interpolation Quick Reference

| Syntax              | Context                   | Example                          | Resolves to                              |
|---------------------|---------------------------|----------------------------------|------------------------------------------|
| `${field_name}`     | System prompts, URLs      | `${customer_id}`                 | Value of `state["customer_id"]` (or `""` if missing) |
| `${env:VAR_NAME}`   | Headers, URLs             | `${env:API_TOKEN}`               | Value of `os.getenv("API_TOKEN")`        |
| Bare `{` and `}`    | Anywhere                  | `{"key": "value"}`               | Literal text (unchanged)                 |

## Appendix C: Glossary

| Term          | Definition                                                                         |
|---------------|------------------------------------------------------------------------------------|
| Agent         | A self-contained AJML workflow defined in a single `.ajml` file.                   |
| Graph         | The directed graph of nodes and edges that defines an agent's execution flow.      |
| Node          | A unit of work: LLM call, tool execution, script, or subgraph delegation.          |
| Edge          | A connection between nodes defining execution order and routing.                   |
| State         | The shared data structure that flows through the graph, updated by each node.      |
| Reducer       | A function that determines how concurrent state updates are merged.                |
| Tool          | An action the agent can perform: API call or script execution.                     |
| Subgraph      | A child agent invoked as a node within a parent agent's graph.                     |
| Transpiler    | The AJML compiler that transforms `.ajml` files into deployable Python code.       |
| Super-step    | A single iteration of the graph where all ready nodes execute (potentially in parallel). |
| Tool-calling loop | The ReAct pattern where an LLM node invokes tools in a cycle until a final response. |
| Backoff        | Retry delay strategy: `fixed` (constant delay) or `exponential` (doubling delay).    |
| Parameterised type | A generic type like `list[string]` or `dict[int]` that constrains item types.      |

---

*AJML Specification v2.0 — February 2026*
