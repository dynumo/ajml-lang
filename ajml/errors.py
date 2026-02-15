"""AJML Error Catalogue — compilation errors, warnings, and runtime errors."""


class AJMLCompilationError(Exception):
    """Raised when AJML compilation fails validation."""

    def __init__(self, code: str, message: str, filename: str = "", line: int = 0):
        self.code = code
        self.message = message
        self.filename = filename
        self.line = line
        detail = f"AJMLCompilationError [{code}]: {message}"
        if filename:
            detail += f"\n  → File: {filename}"
            if line:
                detail += f", Line: {line}"
        super().__init__(detail)


class AJMLWarning:
    """Represents a non-fatal compilation warning."""

    def __init__(self, code: str, message: str, filename: str = "", line: int = 0):
        self.code = code
        self.message = message
        self.filename = filename
        self.line = line

    def __str__(self):
        detail = f"AJMLWarning [{self.code}]: {self.message}"
        if self.filename:
            detail += f"\n  → File: {self.filename}"
            if self.line:
                detail += f", Line: {self.line}"
        return detail


# --- Error code constants ---

# Structure
E001 = "E001"  # Root element must be <agent> or <project>
E002 = "E002"  # Required block missing
E003 = "E003"  # Missing _project.ajml
E004 = "E004"  # Invalid ajml_version

# Uniqueness
E101 = "E101"  # Duplicate node ID
E102 = "E102"  # Duplicate tool ID
E103 = "E103"  # Duplicate state field name
E104 = "E104"  # Reserved word used

# Type validity
E105 = "E105"  # Invalid node type
E106 = "E106"  # Invalid tool type
E107 = "E107"  # Invalid state field type
E108 = "E108"  # Invalid reducer for type
E109 = "E109"  # Enum default not in values

# State
E201 = "E201"  # Cannot declare 'messages' field
E202 = "E202"  # Required field cannot have default

# Graph integrity
E301 = "E301"  # No entry point
E302 = "E302"  # Edge target not found
E303 = "E303"  # Edge source not found
E304 = "E304"  # Conditional group missing default
E305 = "E305"  # Default edge has condition
E306 = "E306"  # Multiple defaults from same source
E307 = "E307"  # Tool reference not found
E308 = "E308"  # Script file not found
E309 = "E309"  # Agent reference not found
E310 = "E310"  # Input map target not in child state
E311 = "E311"  # Output map source not in child state
E312 = "E312"  # Unreachable node
E313 = "E313"  # Node has no outgoing edges
E314 = "E314"  # Mixed edge types
E315 = "E315"  # Map items_field not a list
E316 = "E316"  # Action node params not in state

# Configuration
E401 = "E401"  # Unknown LLM provider
E402 = "E402"  # Circular subgraph dependency
E403 = "E403"  # Duplicate agent name

# Expression / Interpolation
E501 = "E501"  # Condition references undefined name
E502 = "E502"  # Invalid interpolation syntax

# Warnings
W301 = "W301"  # Parallel branches may write to same overwrite field

# Reserved words
RESERVED_WORDS = frozenset({
    "__START__", "__END__", "messages", "__root__", "__config__", "__state__",
})

# Valid node types
VALID_NODE_TYPES = frozenset({"llm", "action", "script", "subgraph"})

# Valid tool types
VALID_TOOL_TYPES = frozenset({"api_call", "local_script", "script_tool"})

# Valid simple types
VALID_SIMPLE_TYPES = frozenset({"string", "int", "float", "bool", "list", "dict", "enum"})

# Valid parameterised types
VALID_PARAM_TYPES = frozenset({
    "list[string]", "list[int]", "list[float]", "list[dict]",
    "dict[string]", "dict[int]", "dict[any]",
})

# All valid types
VALID_TYPES = VALID_SIMPLE_TYPES | VALID_PARAM_TYPES

# Valid reducers by type
VALID_REDUCERS = {
    "string": {"overwrite", "concat"},
    "int": {"overwrite", "add"},
    "float": {"overwrite", "add"},
    "bool": {"overwrite"},
    "list": {"overwrite", "append"},
    "dict": {"overwrite", "merge"},
    "enum": {"overwrite"},
}

# Parameterised types map to their base type for reducer validation
PARAM_TYPE_BASE = {
    "list[string]": "list",
    "list[int]": "list",
    "list[float]": "list",
    "list[dict]": "list",
    "dict[string]": "dict",
    "dict[int]": "dict",
    "dict[any]": "dict",
}

# Supported LLM providers
SUPPORTED_PROVIDERS = frozenset({
    "openai", "anthropic", "google", "mistral",
    "groq", "ollama", "azure_openai", "bedrock",
})

# Condition evaluation allowed names
CONDITION_ALLOWED_NAMES = frozenset({
    "state", "len", "any", "all", "abs", "min", "max",
    "str", "int", "float", "bool", "True", "False", "None",
})
