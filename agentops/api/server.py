"""
FastAPI server for AgentOps.

Factor 11: Trigger from anywhere, meet users where they are.
This API allows triggering agents from webhooks, CLI, web UI, etc.
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import logging

from agentops.core import AgentOrchestrator, ToolRegistry
from agentops.core.llm_client import LLMClient, MockLLMClient
from agentops.agents import (
    create_backend_dev_agent,
    create_devops_agent,
    create_frontend_dev_agent
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AgentOps API",
    description="Multi-agent development team following 12-factor principles",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator: Optional[AgentOrchestrator] = None


# Request/Response models
class TaskRequest(BaseModel):
    agent_type: str  # "backend_developer", "devops_engineer", "frontend_developer"
    description: str
    context: Optional[Dict[str, Any]] = None


class TaskResponse(BaseModel):
    task_id: str
    agent_id: str
    status: str
    message: str


class HumanResponseRequest(BaseModel):
    response: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator and register agents."""
    global orchestrator

    logger.info("Initializing AgentOps...")

    # Create LLM client (use mock for now)
    llm_client = MockLLMClient()

    # Create tool registry
    tool_registry = ToolRegistry()

    # Create orchestrator
    orchestrator = AgentOrchestrator(
        llm_client=llm_client,
        tool_registry=tool_registry
    )

    # Register agent configurations
    backend_config = create_backend_dev_agent()
    devops_config = create_devops_agent()
    frontend_config = create_frontend_dev_agent()

    orchestrator.register_agent(backend_config)
    orchestrator.register_agent(devops_config)
    orchestrator.register_agent(frontend_config)

    logger.info("AgentOps initialized with 3 agents")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "AgentOps",
        "version": "1.0.0",
        "description": "Multi-agent development team following 12-factor principles"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/tasks", response_model=TaskResponse)
async def create_task(task: TaskRequest):
    """
    Factor 11: Create a new task for an agent.

    This endpoint can be triggered from anywhere: CLI, webhooks, UI, etc.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Find agent by type
    agent_id = None
    for aid, config in orchestrator.agent_configs.items():
        if config.agent_type == task.agent_type:
            agent_id = aid
            break

    if not agent_id:
        raise HTTPException(
            status_code=404,
            detail=f"No agent found for type: {task.agent_type}"
        )

    # Factor 6: Launch the agent
    state = orchestrator.launch(
        agent_id=agent_id,
        task_description=task.description,
        initial_context=task.context
    )

    # Start running the agent in background
    asyncio.create_task(orchestrator.run_until_complete(agent_id))

    return TaskResponse(
        task_id=state.agent_id,
        agent_id=agent_id,
        status=state.status.value,
        message=f"Task assigned to {task.agent_type}"
    )


@app.get("/agents")
async def list_agents():
    """List all registered agents."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    agents = orchestrator.list_agents()
    return {"agents": agents}


@app.get("/agents/{agent_id}/status")
async def get_agent_status(agent_id: str):
    """Get status of a specific agent."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    status = orchestrator.get_status(agent_id)

    if status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Agent not found")

    return status


@app.get("/agents/{agent_id}/state")
async def get_agent_state(agent_id: str):
    """Get full state of an agent."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    state = orchestrator.get_state(agent_id)

    if not state:
        raise HTTPException(status_code=404, detail="Agent not found")

    return state.to_dict()


@app.post("/agents/{agent_id}/pause")
async def pause_agent(agent_id: str):
    """Factor 6: Pause agent execution."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        state = orchestrator.pause(agent_id)
        return {"status": "paused", "agent_id": agent_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/agents/{agent_id}/resume")
async def resume_agent(agent_id: str):
    """Factor 6: Resume agent execution."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        state = orchestrator.resume(agent_id)
        # Continue running
        asyncio.create_task(orchestrator.run_until_complete(agent_id))
        return {"status": "resumed", "agent_id": agent_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/agents/{agent_id}/human-response")
async def provide_human_response(agent_id: str, response: HumanResponseRequest):
    """
    Factor 7: Provide human response to agent.

    When an agent requests human input, respond here.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        state = orchestrator.provide_human_response(agent_id, response.response)

        # Resume agent if it was waiting
        asyncio.create_task(orchestrator.run_until_complete(agent_id))

        return {
            "status": "response_received",
            "agent_id": agent_id,
            "agent_status": state.status.value
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.

    Factor 11: Meet users where they are - provide real-time updates.
    """
    await websocket.accept()

    try:
        while True:
            # Send periodic status updates
            if orchestrator:
                agents = orchestrator.list_agents()
                await websocket.send_json({
                    "type": "status_update",
                    "agents": agents,
                    "timestamp": asyncio.get_event_loop().time()
                })

            await asyncio.sleep(2)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
