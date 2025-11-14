"""
Agent Orchestrator following 12-factor principles.

Factor 8: Own your control flow.
The orchestrator explicitly manages how agents progress through states.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import logging

from .agent import Agent, AgentState, AgentConfig, AgentStatus
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Orchestrates multiple agents following 12-factor principles.

    This is inspired by Jazz's agent lifecycle management but with
    explicit state management and control flow.
    """

    def __init__(
        self,
        llm_client: Any,
        tool_registry: Optional[ToolRegistry] = None,
        state_store: Optional[Any] = None
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry or ToolRegistry()
        self.state_store = state_store  # For persistence (Redis, DB, etc.)

        # Factor 10: Small, focused agents
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.agents: Dict[str, Agent] = {}

        # Active states (in-memory, but can be persisted)
        self.states: Dict[str, AgentState] = {}

    def register_agent(self, config: AgentConfig):
        """
        Register an agent configuration.

        Similar to Jazz's agent creation, but with explicit config.
        """
        agent = Agent(config)
        self.agent_configs[config.agent_id] = config
        self.agents[config.agent_id] = agent
        logger.info(f"Registered agent: {config.name} ({config.agent_type})")

    def launch(
        self,
        agent_id: str,
        task_description: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> AgentState:
        """
        Factor 6: Simple launch API.

        Start an agent with a task.
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not registered")

        config = self.agent_configs[agent_id]
        agent = self.agents[agent_id]

        # Create initial state
        state = Agent.launch(config, task_description)

        # Add initial context if provided
        if initial_context:
            for key, value in initial_context.items():
                state.add_to_context(key, value)

        self.states[agent_id] = state

        # Persist if state store available
        if self.state_store:
            self.state_store.save(agent_id, state.to_dict())

        logger.info(f"Launched agent {config.name} with task: {task_description}")
        return state

    def pause(self, agent_id: str) -> AgentState:
        """Factor 6: Simple pause API."""
        if agent_id not in self.states:
            raise ValueError(f"No active state for agent {agent_id}")

        state = self.states[agent_id]
        paused_state = Agent.pause(state)
        self.states[agent_id] = paused_state

        if self.state_store:
            self.state_store.save(agent_id, paused_state.to_dict())

        logger.info(f"Paused agent {agent_id}")
        return paused_state

    def resume(self, agent_id: str) -> AgentState:
        """Factor 6: Simple resume API."""
        if agent_id not in self.states:
            raise ValueError(f"No active state for agent {agent_id}")

        state = self.states[agent_id]
        resumed_state = Agent.resume(state)
        self.states[agent_id] = resumed_state

        if self.state_store:
            self.state_store.save(agent_id, resumed_state.to_dict())

        logger.info(f"Resumed agent {agent_id}")
        return resumed_state

    async def step(self, agent_id: str) -> AgentState:
        """
        Execute one step of an agent.

        Factor 8: Explicit control flow.
        """
        if agent_id not in self.states:
            raise ValueError(f"No active state for agent {agent_id}")

        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not registered")

        current_state = self.states[agent_id]
        agent = self.agents[agent_id]

        # Factor 12: Agent.step is a pure reducer function
        new_state = agent.step(
            state=current_state,
            llm_client=self.llm_client,
            tool_registry=self.tool_registry
        )

        # Update stored state
        self.states[agent_id] = new_state

        # Persist
        if self.state_store:
            await self.state_store.save_async(agent_id, new_state.to_dict())

        return new_state

    async def run_until_complete(
        self,
        agent_id: str,
        max_iterations: Optional[int] = None
    ) -> AgentState:
        """
        Run an agent until it completes or fails.

        Factor 8: Explicit control flow with clear termination conditions.
        """
        iterations = 0
        max_iter = max_iterations or 100

        while iterations < max_iter:
            state = self.states[agent_id]

            # Check terminal states
            if state.status in [
                AgentStatus.COMPLETED,
                AgentStatus.FAILED,
                AgentStatus.AWAITING_HUMAN
            ]:
                logger.info(
                    f"Agent {agent_id} reached terminal state: {state.status.value}"
                )
                break

            # Check if paused
            if state.status == AgentStatus.PAUSED:
                logger.info(f"Agent {agent_id} is paused")
                break

            # Execute step
            new_state = await self.step(agent_id)

            iterations += 1

            # Small delay to prevent tight loops
            await asyncio.sleep(0.1)

        return self.states[agent_id]

    def provide_human_response(
        self,
        agent_id: str,
        response: Dict[str, Any]
    ) -> AgentState:
        """
        Factor 7: Provide human response to agent.

        Humans are integrated via the same mechanism as tools.
        """
        if agent_id not in self.states:
            raise ValueError(f"No active state for agent {agent_id}")

        state = self.states[agent_id]

        # Add human response
        state.human_responses.append({
            "response": response,
            "timestamp": asyncio.get_event_loop().time()
        })

        # Resume if was waiting
        if state.status == AgentStatus.AWAITING_HUMAN:
            state.status = AgentStatus.RUNNING

        self.states[agent_id] = state

        if self.state_store:
            self.state_store.save(agent_id, state.to_dict())

        return state

    def get_state(self, agent_id: str) -> Optional[AgentState]:
        """Get current state of an agent."""
        return self.states.get(agent_id)

    def get_status(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed status of an agent."""
        state = self.states.get(agent_id)
        if not state:
            return {"status": "not_found"}

        config = self.agent_configs.get(agent_id)

        return {
            "agent_id": agent_id,
            "agent_type": state.agent_type,
            "name": config.name if config else "Unknown",
            "status": state.status.value,
            "current_step": state.current_step,
            "max_steps": state.max_steps,
            "task": state.task_description,
            "awaiting_human": state.status == AgentStatus.AWAITING_HUMAN,
            "pending_requests": len(state.human_requests) - len(state.human_responses),
            "errors": len(state.errors),
            "tool_calls": len(state.tool_calls)
        }

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents and their states."""
        agents_info = []

        for agent_id, config in self.agent_configs.items():
            info = {
                "agent_id": agent_id,
                "name": config.name,
                "type": config.agent_type,
                "role": config.role,
                "capabilities": config.capabilities,
            }

            # Add state if active
            if agent_id in self.states:
                state = self.states[agent_id]
                info["status"] = state.status.value
                info["current_task"] = state.task_description
                info["current_step"] = state.current_step
            else:
                info["status"] = "idle"

            agents_info.append(info)

        return agents_info

    async def cleanup(self, agent_id: str):
        """Clean up agent state."""
        if agent_id in self.states:
            del self.states[agent_id]

        if self.state_store:
            await self.state_store.delete_async(agent_id)

        logger.info(f"Cleaned up agent {agent_id}")
