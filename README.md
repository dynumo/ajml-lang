# AJML — Agent Job Markup Language

A declarative, XML-based language for defining AI agent workflows. AJML transpiles `.ajml` files into production-ready Python servers powered by [LangGraph](https://github.com/langchain-ai/langgraph) and [FastAPI](https://fastapi.tiangolo.com/).

## Requirements

- Python 3.10+

## Installation

```bash
# Install from source
pip install .

# Or install in editable/development mode
pip install -e .
```

This installs the `ajml` CLI command.

## Quick Start

### 1. Create a new project

```bash
ajml init my_project
cd my_project
```

This scaffolds:

```
my_project/
├── agents/
│   ├── _project.ajml       # Project-level configuration
│   └── example_agent.ajml  # Example agent definition
├── tools/                   # Python tool implementations
├── build/                   # Generated output (gitignored)
├── .env.example
└── .gitignore
```

### 2. Configure your project

Edit `agents/_project.ajml` to set your LLM provider and server options:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project name="my_project" ajml_version="2.0">
    <config>
        <llm provider="openai" model="gpt-4o" />
        <server cors_origins="https://frontend.com" auth_env="AJML_API_KEY"
                docs_public="false" />
        <env>
            <var name="OPENAI_API_KEY" required="true" />
        </env>
    </config>
</project>
```

### 3. Define an agent

Create an `.ajml` file in the `agents/` directory:

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

### 4. Build and run

```bash
# Validate your project (no output generated)
ajml validate

# Build the project
ajml build

# Run the generated server
cd build
pip install -r requirements.txt
uvicorn main:app --reload
```

The server exposes:

- `POST /run/<agent_name>` — execute an agent
- `GET /health` — health check
- `/docs` — OpenAPI documentation

## CLI Reference

```
ajml init <project_name>           Create a new project scaffold
ajml build [dir] [options]         Transpile .ajml files to Python
ajml validate [dir]                Validate .ajml files without generating output
ajml --version                     Show version
```

**Build options:**

| Flag | Description |
|------|-------------|
| `--output <dir>` | Output directory (default: `build`) |
| `--verbose`, `-v` | Print per-agent compilation details |
| `--strict` | Treat warnings as errors |
| `--dry-run` | Validate only, emit no files |

You can also run the CLI as a Python module:

```bash
python -m ajml build .
```

## Language Overview

### State fields

Define typed state with optional defaults and reducers:

```xml
<state>
    <field name="messages" type="list[string]" default="[]" reducer="append" />
    <field name="count" type="int" default="0" reducer="add" />
</state>
```

Supported types: `string`, `int`, `float`, `bool`, `list`, `dict`, `enum`, and parameterised forms like `list[string]`, `dict[int]`.

Reducers: `overwrite` (default), `append`, `add`, `merge`, `concat`.

### Node types

| Type | Description |
|------|-------------|
| `llm` | Calls an LLM with a system prompt and optional structured output |
| `action` | Invokes a tool (API call, local script, etc.) |
| `script` | Runs a Python file directly |
| `subgraph` | Nests another agent as a sub-workflow |

### Edges and control flow

```xml
<!-- Unconditional -->
<edge source="nodeA" target="nodeB" />

<!-- Conditional -->
<edge source="router" type="conditional">
    <condition target="handleA">state['intent'] == 'refund'</condition>
    <condition target="handleB" default="true">True</condition>
</edge>

<!-- Parallel fan-out -->
<edge source="start" type="parallel" targets="taskA,taskB,taskC" join="merge_node" />
```

### Supported LLM providers

`openai`, `anthropic`, `google`, `mistral`, `groq`, `ollama`, `azure_openai`, `bedrock`

## Running Tests

```bash
pip install pytest
pytest
```

## Specification

See [SPEC.md](SPEC.md) for the full language reference.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
