"""
Tool system following 12-factor agents principles.

Factor 4: Tools are just structured outputs.
Tools are simple function calls with JSON schemas, not framework magic.
"""

from dataclasses import dataclass
from typing import Dict, Any, Callable, List, Optional
import json


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None


@dataclass
class ToolCall:
    """
    A structured output representing a tool invocation.

    This is just JSON - no magic, full transparency.
    """
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "call_id": self.call_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCall':
        return cls(
            name=data["name"],
            arguments=data.get("arguments", {}),
            call_id=data.get("call_id")
        )


class Tool:
    """
    A tool is simply a callable with a schema.

    Factor 4: Treating tools as structured outputs means we explicitly
    define their schemas and handle execution transparently.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        function: Callable[[Dict[str, Any]], Any]
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function

    def get_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for this tool.

        This is what gets sent to the LLM to describe the tool.
        """
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

    def execute(self, arguments: Dict[str, Any]) -> Any:
        """Execute the tool with given arguments."""
        return self.function(arguments)


class ToolRegistry:
    """
    Registry of available tools.

    Inspired by Jazz's plugin architecture - tools can be registered
    and looked up dynamically.
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()

    def register(self, tool: Tool):
        """Register a tool."""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        return tool.execute(arguments)

    def get_schemas(self, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get schemas for specified tools (or all tools)."""
        if tool_names:
            tools = [self.tools[name] for name in tool_names if name in self.tools]
        else:
            tools = list(self.tools.values())

        return [tool.get_schema() for tool in tools]

    def _register_default_tools(self):
        """Register default tools available to all agents."""

        # Factor 7: Human-in-the-loop tool
        self.register(Tool(
            name="request_human_input",
            description="Request input or approval from a human operator",
            parameters=[
                ToolParameter(
                    name="question",
                    type="string",
                    description="The question or request for the human"
                ),
                ToolParameter(
                    name="context",
                    type="string",
                    description="Additional context about why human input is needed",
                    required=False
                ),
                ToolParameter(
                    name="urgency",
                    type="string",
                    description="Urgency level",
                    enum=["low", "medium", "high"],
                    required=False
                )
            ],
            function=lambda args: {
                "status": "requested",
                "question": args["question"],
                "context": args.get("context", ""),
                "urgency": args.get("urgency", "medium")
            }
        ))

        # Code execution tool
        self.register(Tool(
            name="execute_code",
            description="Execute code in a sandboxed environment",
            parameters=[
                ToolParameter(
                    name="language",
                    type="string",
                    description="Programming language",
                    enum=["python", "javascript", "bash"]
                ),
                ToolParameter(
                    name="code",
                    type="string",
                    description="Code to execute"
                ),
                ToolParameter(
                    name="timeout",
                    type="number",
                    description="Timeout in seconds",
                    required=False
                )
            ],
            function=self._execute_code
        ))

        # File operations
        self.register(Tool(
            name="read_file",
            description="Read contents of a file",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="File path to read"
                )
            ],
            function=self._read_file
        ))

        self.register(Tool(
            name="write_file",
            description="Write content to a file",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="File path to write"
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Content to write"
                ),
                ToolParameter(
                    name="mode",
                    type="string",
                    description="Write mode",
                    enum=["write", "append"],
                    required=False
                )
            ],
            function=self._write_file
        ))

        # API call tool
        self.register(Tool(
            name="api_call",
            description="Make an HTTP API request",
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="URL to call"
                ),
                ToolParameter(
                    name="method",
                    type="string",
                    description="HTTP method",
                    enum=["GET", "POST", "PUT", "DELETE", "PATCH"]
                ),
                ToolParameter(
                    name="headers",
                    type="object",
                    description="HTTP headers",
                    required=False
                ),
                ToolParameter(
                    name="body",
                    type="object",
                    description="Request body",
                    required=False
                )
            ],
            function=self._api_call
        ))

        # Task completion marker
        self.register(Tool(
            name="complete_task",
            description="Mark the current task as complete",
            parameters=[
                ToolParameter(
                    name="summary",
                    type="string",
                    description="Summary of what was accomplished"
                ),
                ToolParameter(
                    name="artifacts",
                    type="array",
                    description="List of created artifacts (files, URLs, etc.)",
                    required=False
                )
            ],
            function=lambda args: {
                "status": "COMPLETE",
                "summary": args["summary"],
                "artifacts": args.get("artifacts", [])
            }
        ))

    def _execute_code(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code safely (placeholder - implement actual execution)."""
        return {
            "status": "executed",
            "language": args["language"],
            "output": "[Code execution would happen here]",
            "error": None
        }

    def _read_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Read file (placeholder)."""
        try:
            with open(args["path"], "r") as f:
                content = f.read()
            return {
                "status": "success",
                "content": content,
                "path": args["path"]
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "path": args["path"]
            }

    def _write_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Write file (placeholder)."""
        try:
            mode = "a" if args.get("mode") == "append" else "w"
            with open(args["path"], mode) as f:
                f.write(args["content"])
            return {
                "status": "success",
                "path": args["path"],
                "bytes_written": len(args["content"])
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "path": args["path"]
            }

    def _api_call(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Make API call (placeholder - implement with requests)."""
        return {
            "status": "success",
            "url": args["url"],
            "method": args["method"],
            "response": "[API call would happen here]"
        }
