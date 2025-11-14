"""
Jazz Bridge - Integration with Jazz AI autonomous agents.

This module provides a bridge between AgentOps and Jazz (https://github.com/lvndry/jazz),
allowing our Python orchestration system to work with Jazz's TypeScript-based autonomous agents.

Factor 11: Trigger from anywhere - Jazz agents can be controlled via CLI, API, or programmatically.
Factor 10: Small, focused agents - Jazz agents are specialized tools with specific capabilities.
"""

import json
import subprocess
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class JazzAgent:
    """Represents a Jazz agent configuration."""

    id: str
    name: str
    purpose: str
    model: str
    provider: str
    tools: List[str]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "purpose": self.purpose,
            "model": self.model,
            "provider": self.provider,
            "tools": self.tools,
            "created_at": self.created_at
        }


class JazzBridge:
    """
    Bridge to Jazz AI autonomous agent system.

    Allows Python code to interact with Jazz agents via CLI commands.
    Jazz agents have real-world capabilities: email, git, file system, shell, web search, HTTP.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize Jazz bridge.

        Args:
            config_path: Path to jazz.config.json (optional, Jazz will use default locations)
        """
        self.config_path = config_path
        self._check_jazz_installed()

    def _check_jazz_installed(self) -> bool:
        """Check if Jazz CLI is installed."""
        try:
            result = subprocess.run(
                ["jazz", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Jazz CLI found: {result.stdout.strip()}")
                return True
            else:
                logger.warning("Jazz CLI not found. Install with: npm install -g jazz-ai")
                return False
        except FileNotFoundError:
            logger.warning("Jazz CLI not found. Install with: npm install -g jazz-ai")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Jazz CLI check timed out")
            return False

    def _run_jazz_command(self, args: List[str], input_data: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a Jazz CLI command and return the result.

        Args:
            args: Command arguments (e.g., ["agent", "list"])
            input_data: Optional stdin input for interactive commands

        Returns:
            Dictionary with stdout, stderr, and returncode
        """
        cmd = ["jazz"] + args

        if self.config_path:
            cmd.extend(["--config", str(self.config_path)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                input=input_data,
                timeout=30
            )

            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            logger.error(f"Jazz command timed out: {' '.join(cmd)}")
            return {
                "stdout": "",
                "stderr": "Command timed out",
                "returncode": -1,
                "success": False
            }
        except Exception as e:
            logger.error(f"Error running Jazz command: {e}")
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "success": False
            }

    def list_agents(self) -> List[JazzAgent]:
        """
        List all Jazz agents.

        Returns:
            List of JazzAgent objects
        """
        result = self._run_jazz_command(["agent", "list", "--json"])

        if not result["success"]:
            logger.error(f"Failed to list agents: {result['stderr']}")
            return []

        try:
            # Parse JSON output from Jazz
            agents_data = json.loads(result["stdout"])
            agents = []

            for agent_data in agents_data:
                agent = JazzAgent(
                    id=agent_data.get("id", ""),
                    name=agent_data.get("name", ""),
                    purpose=agent_data.get("purpose", ""),
                    model=agent_data.get("model", ""),
                    provider=agent_data.get("provider", ""),
                    tools=agent_data.get("tools", []),
                    created_at=agent_data.get("created_at", "")
                )
                agents.append(agent)

            return agents
        except json.JSONDecodeError:
            logger.error("Failed to parse Jazz agent list output")
            return []

    def get_agent(self, agent_id: str) -> Optional[JazzAgent]:
        """
        Get details of a specific Jazz agent.

        Args:
            agent_id: Agent ID or name

        Returns:
            JazzAgent object or None if not found
        """
        result = self._run_jazz_command(["agent", "get", agent_id, "--json"])

        if not result["success"]:
            logger.error(f"Failed to get agent {agent_id}: {result['stderr']}")
            return None

        try:
            agent_data = json.loads(result["stdout"])
            return JazzAgent(
                id=agent_data.get("id", ""),
                name=agent_data.get("name", ""),
                purpose=agent_data.get("purpose", ""),
                model=agent_data.get("model", ""),
                provider=agent_data.get("provider", ""),
                tools=agent_data.get("tools", []),
                created_at=agent_data.get("created_at", "")
            )
        except json.JSONDecodeError:
            logger.error("Failed to parse Jazz agent details")
            return None

    def create_agent(
        self,
        name: str,
        purpose: str,
        provider: str,
        model: str,
        tools: Optional[List[str]] = None
    ) -> Optional[JazzAgent]:
        """
        Create a new Jazz agent.

        Args:
            name: Agent name
            purpose: Agent's purpose/role
            provider: LLM provider (openai, anthropic, etc.)
            model: Model name
            tools: List of tool categories to enable

        Returns:
            Created JazzAgent or None if failed
        """
        # Note: Jazz CLI uses interactive wizard for creation
        # This is a simplified interface - actual creation may require manual setup
        logger.info(f"To create Jazz agent '{name}', run: jazz agent create")
        logger.info(f"Purpose: {purpose}")
        logger.info(f"Provider: {provider}, Model: {model}")
        if tools:
            logger.info(f"Enable tools: {', '.join(tools)}")

        # For now, we can't fully automate this due to Jazz's interactive wizard
        # Users should create agents via CLI, then list them here
        return None

    def chat(self, agent_id: str, message: str) -> str:
        """
        Send a message to a Jazz agent and get response.

        Args:
            agent_id: Agent ID or name
            message: Message to send

        Returns:
            Agent's response
        """
        result = self._run_jazz_command(
            ["agent", "chat", agent_id],
            input_data=message
        )

        if not result["success"]:
            logger.error(f"Failed to chat with agent {agent_id}: {result['stderr']}")
            return f"Error: {result['stderr']}"

        return result["stdout"]

    def delete_agent(self, agent_id: str) -> bool:
        """
        Delete a Jazz agent.

        Args:
            agent_id: Agent ID to delete

        Returns:
            True if successful, False otherwise
        """
        result = self._run_jazz_command(["agent", "delete", agent_id])

        if result["success"]:
            logger.info(f"Deleted Jazz agent: {agent_id}")
            return True
        else:
            logger.error(f"Failed to delete agent {agent_id}: {result['stderr']}")
            return False

    def get_available_tools(self) -> Dict[str, List[str]]:
        """
        Get available tool categories in Jazz.

        Returns:
            Dictionary mapping tool categories to their capabilities
        """
        return {
            "gmail": [
                "search", "get", "list_labels", "label", "unlabel",
                "archive", "unarchive", "trash", "send", "reply",
                "forward", "mark_read", "mark_unread", "star", "unstar", "batch_*"
            ],
            "git": [
                "status", "diff", "log", "commit", "push", "pull",
                "branch", "checkout", "add"
            ],
            "filesystem": [
                "read_file", "write_file", "list_directory", "search",
                "create_directory", "delete", "move", "copy", "stat",
                "permissions", "symlink"
            ],
            "shell": [
                "exec", "exec_streaming"
            ],
            "web": [
                "search"
            ],
            "http": [
                "request"
            ]
        }


# Example usage
if __name__ == "__main__":
    # Initialize bridge
    bridge = JazzBridge()

    # List available agents
    agents = bridge.list_agents()
    print(f"Found {len(agents)} Jazz agents:")
    for agent in agents:
        print(f"  - {agent.name} ({agent.id}): {agent.purpose}")

    # Show available tools
    tools = bridge.get_available_tools()
    print(f"\nAvailable tool categories: {', '.join(tools.keys())}")
