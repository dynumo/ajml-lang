"""AJML CLI — build, validate, and init commands."""

import argparse
import os
import sys

from . import __version__
from .assembler import generate_env_example, generate_main_py, generate_requirements_txt
from .codegen import generate_agent_code
from .errors import AJMLCompilationError
from .preprocessor import count_sanitised_conditions, preprocess
from .validator import (
    AgentAST,
    ProjectAST,
    check_circular_dependencies,
    check_duplicate_agent_names,
    validate_agent,
    validate_project,
)


def main():
    """Entry point for the AJML CLI."""
    parser = argparse.ArgumentParser(
        prog="ajml",
        description="AJML — Agent Job Markup Language transpiler",
    )
    parser.add_argument(
        "--version", action="version", version=f"AJML Transpiler v{__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build the AJML project")
    build_parser.add_argument(
        "project_dir", nargs="?", default=".",
        help="Path to the project root (containing agents/)",
    )
    build_parser.add_argument(
        "--output", "-o", default="build",
        help="Output directory name (relative to project root)",
    )
    build_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging of each compilation phase",
    )
    build_parser.add_argument(
        "--strict", action="store_true",
        help="Treat warnings as errors",
    )
    build_parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate only — don't emit any files",
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate the AJML project")
    validate_parser.add_argument(
        "project_dir", nargs="?", default=".",
        help="Path to the project root (containing agents/)",
    )

    # Init command
    init_parser = subparsers.add_parser("init", help="Scaffold a new AJML project")
    init_parser.add_argument(
        "project_name",
        help="Name of the new project",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "build":
        sys.exit(cmd_build(args))
    elif args.command == "validate":
        sys.exit(cmd_validate(args))
    elif args.command == "init":
        sys.exit(cmd_init(args))


def cmd_build(args) -> int:
    """Execute the build command."""
    project_dir = os.path.abspath(args.project_dir)
    output_dir = os.path.join(project_dir, args.output)
    verbose = args.verbose
    strict = args.strict
    dry_run = args.dry_run

    print(f"AJML Transpiler v{__version__}")
    print("=" * 20)

    try:
        project, agents, all_warnings = _compile_project(project_dir, verbose)
    except AJMLCompilationError as e:
        print(f"\n{e}")
        return 1
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 2
    except Exception as e:
        print(f"\nInternal transpiler error: {e}")
        return 3

    # Print warnings
    for w in all_warnings:
        print(f"  ⚠ {w}")

    if strict and all_warnings:
        print(f"\nBuild failed ({len(all_warnings)} warning(s) treated as errors in strict mode).")
        return 1

    if dry_run:
        print(f"\nDry run complete (no files emitted). {len(all_warnings)} warning(s).")
        return 0

    # Phase 3 & 4: Generate code and assemble output
    print(f"\nAssembling server...")

    os.makedirs(output_dir, exist_ok=True)

    # Generate compiled agent files
    for name, agent in sorted(agents.items()):
        code = generate_agent_code(agent, project, agents)
        output_path = os.path.join(output_dir, f"compiled_{name}.py")
        with open(output_path, "w") as f:
            f.write(code)
        print(f"  ✓ {args.output}/compiled_{name}.py")

    # Generate main.py
    main_code = generate_main_py(project, agents)
    main_path = os.path.join(output_dir, "main.py")
    with open(main_path, "w") as f:
        f.write(main_code)
    print(f"  ✓ {args.output}/main.py ({len(agents)} endpoint{'s' if len(agents) != 1 else ''})")

    # Generate requirements.txt
    req_code = generate_requirements_txt(project, agents)
    req_path = os.path.join(output_dir, "requirements.txt")
    with open(req_path, "w") as f:
        f.write(req_code)
    print(f"  ✓ {args.output}/requirements.txt")

    # Generate .env.example
    env_code = generate_env_example(project)
    env_path = os.path.join(output_dir, ".env.example")
    with open(env_path, "w") as f:
        f.write(env_code)
    print(f"  ✓ {args.output}/.env.example")

    warning_str = f" ({len(all_warnings)} warning{'s' if len(all_warnings) != 1 else ''})" if all_warnings else ""
    print(f"\nBuild complete{warning_str}. Run with:")
    print(f"  cd {args.output} && uvicorn main:app --reload")

    return 0


def cmd_validate(args) -> int:
    """Execute the validate command (Phases 1 and 2 only)."""
    project_dir = os.path.abspath(args.project_dir)

    print(f"AJML Transpiler v{__version__}")
    print("=" * 20)

    try:
        project, agents, all_warnings = _compile_project(project_dir, verbose=True)
    except AJMLCompilationError as e:
        print(f"\n{e}")
        return 1
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 2
    except Exception as e:
        print(f"\nInternal transpiler error: {e}")
        return 3

    for w in all_warnings:
        print(f"  ⚠ {w}")

    print(f"\nValidation passed. {len(agents)} agent(s), {len(all_warnings)} warning(s).")
    return 0


def cmd_init(args) -> int:
    """Execute the init command — scaffold a new project."""
    project_name = args.project_name

    if os.path.exists(project_name):
        print(f"Error: Directory '{project_name}' already exists.")
        return 2

    # Create directory structure
    os.makedirs(os.path.join(project_name, "agents"))
    os.makedirs(os.path.join(project_name, "tools"))
    os.makedirs(os.path.join(project_name, "build"))

    # _project.ajml
    project_ajml = f'''<?xml version="1.0" encoding="UTF-8"?>
<project name="{project_name}" ajml_version="2.0">
    <config>
        <llm provider="openai" model="gpt-4o" />
        <env>
            <var name="OPENAI_API_KEY" required="true" />
        </env>
    </config>
</project>
'''
    with open(os.path.join(project_name, "agents", "_project.ajml"), "w") as f:
        f.write(project_ajml)

    # example_agent.ajml
    agent_ajml = f'''<?xml version="1.0" encoding="UTF-8"?>
<agent name="example_agent" version="1.0"
       description="An example AJML agent.">

    <state>
        <field name="user_input" type="string" required="true" />
        <field name="response" type="string" default="" />
    </state>

    <graph>
        <node id="respond" type="llm">
            <system_prompt>
                You are a helpful assistant. Respond to the user's input.
            </system_prompt>
            <output_schema>
                <field name="response" type="string" description="Your response" />
            </output_schema>
        </node>

        <edge source="__START__" target="respond" />
        <edge source="respond" target="__END__" />
    </graph>
</agent>
'''
    with open(os.path.join(project_name, "agents", "example_agent.ajml"), "w") as f:
        f.write(agent_ajml)

    # example tool
    tool_py = '''# Example tool script
def run(state: dict) -> dict:
    return {}
'''
    with open(os.path.join(project_name, "tools", "example_tool.py"), "w") as f:
        f.write(tool_py)

    # .env.example
    with open(os.path.join(project_name, ".env.example"), "w") as f:
        f.write("OPENAI_API_KEY=  # Required\n")

    # .gitignore
    gitignore = """build/
.env
__pycache__/
*.pyc
"""
    with open(os.path.join(project_name, ".gitignore"), "w") as f:
        f.write(gitignore)

    print(f"Created new AJML project: {project_name}/")
    print(f"  {project_name}/agents/_project.ajml")
    print(f"  {project_name}/agents/example_agent.ajml")
    print(f"  {project_name}/tools/example_tool.py")
    print(f"  {project_name}/build/           (empty, .gitignored)")
    print(f"  {project_name}/.env.example")
    print(f"  {project_name}/.gitignore")
    print(f"\nNext steps:")
    print(f"  cd {project_name}")
    print(f"  python -m ajml build .")

    return 0


def _compile_project(
    project_dir: str, verbose: bool = False
) -> tuple[ProjectAST, dict[str, AgentAST], list]:
    """Run Phases 1 and 2 on an AJML project.

    Returns:
        Tuple of (ProjectAST, dict of agent name -> AgentAST, list of warnings).
    """
    agents_dir = os.path.join(project_dir, "agents")
    if not os.path.isdir(agents_dir):
        raise FileNotFoundError(f"agents/ directory not found in {project_dir}")

    # Load _project.ajml
    project_path = os.path.join(agents_dir, "_project.ajml")
    if not os.path.exists(project_path):
        raise AJMLCompilationError(
            "E003",
            "Missing `_project.ajml` file. Every AJML project requires a project configuration file.",
            "agents/_project.ajml",
        )

    with open(project_path, "r", encoding="utf-8") as f:
        project_raw = f.read()

    project_root = preprocess(project_raw)
    project = validate_project(project_root, "agents/_project.ajml")

    if verbose:
        print(f"Project: {project.name} (ajml v{project.ajml_version})")

    # Scan for agent files
    ajml_files = sorted([
        f for f in os.listdir(agents_dir)
        if f.endswith(".ajml") and f != "_project.ajml"
    ])

    if verbose:
        print(f"Scanning agents/ ... found {len(ajml_files)} agent file{'s' if len(ajml_files) != 1 else ''} + _project.ajml.")
        print()

    # Phase 1 & 2 for each agent
    agents: dict[str, AgentAST] = {}
    all_warnings = []

    for i, ajml_file in enumerate(ajml_files, 1):
        filepath = os.path.join(agents_dir, ajml_file)
        rel_path = f"agents/{ajml_file}"

        if verbose:
            print(f"[{i}/{len(ajml_files)}] {ajml_file}")

        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()

        # Phase 1: Pre-process
        conditions_sanitised = count_sanitised_conditions(raw)
        root = preprocess(raw)
        if verbose:
            if conditions_sanitised > 0:
                print(f"  ✓ Pre-processed ({conditions_sanitised} condition{'s' if conditions_sanitised != 1 else ''} sanitised)")
            else:
                print(f"  ✓ Pre-processed")

        # Phase 2: Validate
        agent, warnings = validate_agent(root, rel_path, project, agents, project_dir)
        agents[agent.name] = agent
        all_warnings.extend(warnings)

        if verbose:
            node_count = len(agent.nodes)
            edge_count = len(agent.edges)
            tool_count = len(agent.tools)
            subgraph_count = sum(1 for n in agent.nodes if n["type"] == "subgraph")
            parts = [f"{node_count} node{'s' if node_count != 1 else ''}"]
            parts.append(f"{edge_count} edge{'s' if edge_count != 1 else ''}")
            parts.append(f"{tool_count} tool{'s' if tool_count != 1 else ''}")
            if subgraph_count:
                parts.append(f"{subgraph_count} subgraph{'s' if subgraph_count != 1 else ''}")
            print(f"  ✓ Validated ({', '.join(parts)})")

            for w in warnings:
                print(f"  ⚠ {w}")

    # Cross-agent checks
    check_duplicate_agent_names(agents)
    check_circular_dependencies(agents)

    # Second pass for subgraph validation now that all agents are loaded
    for name, agent in agents.items():
        for node in agent.nodes:
            if node["type"] == "subgraph":
                agent_ref = node["agent_ref"]
                if agent_ref not in agents:
                    raise AJMLCompilationError(
                        "E309",
                        f"Agent reference `{agent_ref}` does not match any .ajml file in the project.",
                        agent.filename,
                    )

    return project, agents, all_warnings
