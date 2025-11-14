"""
Base Agent implementation following 12-factor agent principles.

Key principles applied:
- Factor 12: Agents are stateless reducers (pure functions)
- Factor 10: Small, focused agents
- Factor 6: Simple launch/pause/resume APIs
- Factor 5: Unified execution and business state
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import uuid
from datetime import datetime
import json


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_HUMAN = "awaiting_human"  # Factor 7: Contact humans


@dataclass
class AgentState:
    """
    Complete state snapshot for an agent (Factor 5: Unified state).

    This represents both execution state and business state in one place,
    making it easy to serialize, debug, and reason about.
    """
    agent_id: str
    agent_type: str
    status: AgentStatus
    current_step: int = 0
    max_steps: int = 10

    # Context window management (Factor 3)
    context: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)

    # Tool execution tracking (Factor 4)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)

    # Error handling (Factor 9)
    errors: List[Dict[str, str]] = field(default_factory=list)
    error_summary: Optional[str] = None

    # Human-in-the-loop (Factor 7)
    human_requests: List[Dict[str, Any]] = field(default_factory=list)
    human_responses: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    task_description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        state = asdict(self)
        state['status'] = self.status.value
        return state

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentState':
        """Deserialize state from dictionary."""
        data = data.copy()
        data['status'] = AgentStatus(data['status'])
        return cls(**data)

    def compact_errors(self, max_tokens: int = 500) -> str:
        """
        Factor 9: Compact errors into context window.

        Summarize errors efficiently to fit within token limits.
        """
        if not self.errors:
            return ""

        # Create compact error representation
        error_summary = []
        total_length = 0

        for i, error in enumerate(reversed(self.errors[-5:])):  # Last 5 errors
            err_msg = f"{i+1}. {error.get('type', 'Error')}: {error.get('message', '')[:100]}"
            total_length += len(err_msg)

            if total_length > max_tokens:
                error_summary.append(f"... and {len(self.errors) - i} more errors")
                break

            error_summary.append(err_msg)

        return "\n".join(error_summary)

    def add_to_context(self, key: str, value: Any, priority: int = 5):
        """
        Factor 3: Own your context window.

        Add information to context with priority for managing what gets included.
        """
        self.context[key] = {
            "value": value,
            "priority": priority,
            "added_at": datetime.now().isoformat()
        }

    def get_prioritized_context(self, max_items: int = 10) -> Dict[str, Any]:
        """Get context items sorted by priority."""
        sorted_items = sorted(
            self.context.items(),
            key=lambda x: x[1].get("priority", 0),
            reverse=True
        )
        return {k: v["value"] for k, v in sorted_items[:max_items]}


@dataclass
class AgentConfig:
    """
    Configuration for an agent instance.

    Factor 2: Own your prompts - prompts are explicitly defined here.
    """
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: str = "generic"
    name: str = "Agent"
    role: str = "Generic Assistant"

    # Factor 2: Owned prompts
    system_prompt: str = "You are a helpful assistant."
    task_prompt_template: str = "Task: {task}\n\nPlease complete this task."

    # Factor 10: Small, focused capabilities
    capabilities: List[str] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)

    # Control parameters
    max_steps: int = 10
    temperature: float = 0.7
    max_tokens: int = 2000

    # Factor 11: Trigger configuration
    triggers: List[Dict[str, Any]] = field(default_factory=list)


class Agent:
    """
    Factor 12: Stateless reducer pattern.

    The Agent is essentially a pure function: given a state and input,
    it produces a new state. All state is external and explicitly managed.
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    def step(
        self,
        state: AgentState,
        llm_client: Any,
        tool_registry: Any
    ) -> AgentState:
        """
        Factor 12: Core reducer function.

        Takes current state and returns new state. No side effects.
        This is the heart of the agent - a pure state transformation.

        Args:
            state: Current agent state
            llm_client: LLM interface for generating responses
            tool_registry: Available tools for the agent

        Returns:
            New agent state
        """
        # Create a new state (immutable pattern)
        new_state = AgentState(
            agent_id=state.agent_id,
            agent_type=state.agent_type,
            status=state.status,
            current_step=state.current_step + 1,
            max_steps=state.max_steps,
            context=state.context.copy(),
            conversation_history=state.conversation_history.copy(),
            tool_calls=state.tool_calls.copy(),
            tool_results=state.tool_results.copy(),
            errors=state.errors.copy(),
            error_summary=state.error_summary,
            human_requests=state.human_requests.copy(),
            human_responses=state.human_responses.copy(),
            created_at=state.created_at,
            updated_at=datetime.now().isoformat(),
            task_description=state.task_description
        )

        # Check if we've hit max steps
        if new_state.current_step >= new_state.max_steps:
            new_state.status = AgentStatus.COMPLETED
            return new_state

        # Check if waiting for human
        if state.status == AgentStatus.AWAITING_HUMAN:
            # Check if human has responded
            if len(state.human_responses) > len(state.human_requests):
                new_state.status = AgentStatus.RUNNING
            else:
                return new_state  # Still waiting

        try:
            # Factor 3: Build context-aware prompt
            prompt = self._build_prompt(new_state)

            # Factor 1: Get LLM response (natural language to tool calls)
            response = llm_client.generate(
                prompt=prompt,
                system_prompt=self.config.system_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            # Add to conversation history
            new_state.conversation_history.append({
                "role": "assistant",
                "content": response.get("content", ""),
                "tool_calls": response.get("tool_calls", [])
            })

            # Factor 4: Tools are just structured outputs
            tool_calls = response.get("tool_calls", [])

            if tool_calls:
                # Execute tools
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name")

                    # Factor 7: Check if requesting human
                    if tool_name == "request_human_input":
                        new_state.status = AgentStatus.AWAITING_HUMAN
                        new_state.human_requests.append(tool_call)
                        continue

                    # Execute regular tool
                    if tool_name in self.config.allowed_tools:
                        result = tool_registry.execute(
                            tool_name,
                            tool_call.get("arguments", {})
                        )

                        new_state.tool_calls.append(tool_call)
                        new_state.tool_results.append({
                            "tool": tool_name,
                            "result": result,
                            "timestamp": datetime.now().isoformat()
                        })
            else:
                # No more tool calls, task might be complete
                if "COMPLETE" in response.get("content", "").upper():
                    new_state.status = AgentStatus.COMPLETED

        except Exception as e:
            # Factor 9: Compact error handling
            error_entry = {
                "type": type(e).__name__,
                "message": str(e),
                "step": new_state.current_step,
                "timestamp": datetime.now().isoformat()
            }
            new_state.errors.append(error_entry)
            new_state.error_summary = new_state.compact_errors()

            # Decide if error is recoverable
            if len(new_state.errors) >= 3:
                new_state.status = AgentStatus.FAILED
            else:
                new_state.status = AgentStatus.RUNNING  # Try to recover

        return new_state

    def _build_prompt(self, state: AgentState) -> str:
        """
        Factor 3: Own your context window.

        Carefully construct the prompt with prioritized context.
        """
        parts = [
            f"Task: {state.task_description}",
            ""
        ]

        # Add prioritized context
        context = state.get_prioritized_context(max_items=5)
        if context:
            parts.append("Relevant Context:")
            for key, value in context.items():
                parts.append(f"- {key}: {value}")
            parts.append("")

        # Add recent conversation (last 3 exchanges)
        if state.conversation_history:
            parts.append("Recent Progress:")
            for msg in state.conversation_history[-3:]:
                content = msg.get("content", "")[:200]  # Limit length
                parts.append(f"- {content}")
            parts.append("")

        # Add error summary if present (Factor 9)
        if state.error_summary:
            parts.append("Previous Errors (learn from these):")
            parts.append(state.error_summary)
            parts.append("")

        # Add tool results
        if state.tool_results:
            parts.append("Tool Results:")
            for result in state.tool_results[-3:]:  # Last 3 results
                tool_name = result.get("tool", "unknown")
                tool_result = str(result.get("result", ""))[:100]
                parts.append(f"- {tool_name}: {tool_result}")
            parts.append("")

        parts.append("What is your next step? Use tools to make progress.")

        return "\n".join(parts)

    # Factor 6: Simple launch/pause/resume APIs

    @staticmethod
    def launch(config: AgentConfig, task_description: str) -> AgentState:
        """Launch a new agent with a task."""
        return AgentState(
            agent_id=config.agent_id,
            agent_type=config.agent_type,
            status=AgentStatus.RUNNING,
            max_steps=config.max_steps,
            task_description=task_description
        )

    @staticmethod
    def pause(state: AgentState) -> AgentState:
        """Pause agent execution."""
        paused_state = AgentState(**asdict(state))
        paused_state.status = AgentStatus.PAUSED
        paused_state.updated_at = datetime.now().isoformat()
        return paused_state

    @staticmethod
    def resume(state: AgentState) -> AgentState:
        """Resume agent execution."""
        resumed_state = AgentState(**asdict(state))
        resumed_state.status = AgentStatus.RUNNING
        resumed_state.updated_at = datetime.now().isoformat()
        return resumed_state
