# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentOps is a multi-agent development team orchestration system designed to deploy autonomous AI agents that can perform software development tasks. The system is intended to run on RunPod.io using open-source large language models (120B parameter models or smaller alternatives).

## Current Repository State

This repository contains a working implementation of the AgentOps multi-agent orchestration system. The implementation follows the **12-Factor Agents** methodology (https://github.com/humanlayer/12-factor-agents) and incorporates patterns from **Jazz** (https://github.com/lvndry/jazz) for agent lifecycle management.

### Key Implementation Files

- `agentops/core/agent.py` - Core Agent class (stateless reducer pattern)
- `agentops/core/orchestrator.py` - Agent orchestrator with lifecycle management
- `agentops/core/tools.py` - Tool system with registry (plugin architecture)
- `agentops/core/llm_client.py` - LLM provider interface (OpenAI, Anthropic, RunPod, Ollama)
- `agentops/agents/` - Specialized agent configurations (backend, devops, frontend)
- `agentops/api/server.py` - FastAPI server with REST and WebSocket endpoints

## Architecture

The system implements all 12 factors from the 12-Factor Agents methodology:

### 12-Factor Principles Implementation

1. **Factor 1 - Natural Language to Tool Calls**: LLM client converts natural language to structured tool calls
2. **Factor 2 - Own Your Prompts**: Agent configurations have explicit system prompts (not hidden in framework)
3. **Factor 3 - Own Your Context Window**: AgentState manages conversation history with token-aware compaction
4. **Factor 4 - Tools Are Just Structured Outputs**: Tools return JSON schemas, no framework magic
5. **Factor 5 - Unify Execution State and Business State**: Single AgentState dataclass contains everything
6. **Factor 6 - Launch/Pause/Resume**: Simple APIs (`orchestrator.launch()`, `pause()`, `resume()`)
7. **Factor 7 - Contact Humans with Tool Calls**: `request_human_input` tool for human-in-the-loop
8. **Factor 8 - Own Your Control Flow**: Explicit orchestrator control flow, no hidden event loops
9. **Factor 9 - Compact Errors**: Error summarization to fit in context window
10. **Factor 10 - Small, Focused Agents**: Three specialized agents (backend, devops, frontend)
11. **Factor 11 - Trigger from Anywhere**: FastAPI server with REST, WebSocket, and CLI support
12. **Factor 12 - Stateless Reducer**: Agent.step() is a pure function: state → state

### Core Components

1. **Agent (`agentops/core/agent.py`)**
   - Stateless reducer pattern (Factor 12)
   - AgentState dataclass for unified state (Factor 5)
   - Error compaction for context management (Factor 9)
   - Six status states: IDLE, RUNNING, PAUSED, COMPLETED, FAILED, AWAITING_HUMAN

2. **AgentOrchestrator (`agentops/core/orchestrator.py`)**
   - Jazz-inspired lifecycle: register, launch, pause, resume, list
   - Explicit control flow (Factor 8)
   - Manages multiple concurrent agents
   - Provides human response integration (Factor 7)

3. **Tool System (`agentops/core/tools.py`)**
   - Jazz-inspired plugin architecture (ToolRegistry)
   - Tools as pure JSON schemas (Factor 4)
   - Default tools: `execute_code`, `read_file`, `write_file`, `api_call`, `request_human_input`, `complete_task`

4. **LLM Client (`agentops/core/llm_client.py`)**
   - Natural language to tool calls (Factor 1)
   - Multi-provider support: OpenAI, Anthropic, RunPod, Ollama
   - MockLLMClient for testing without API costs

5. **Specialized Agents (`agentops/agents/`)**
   - Backend Developer: Python, FastAPI, Django, databases, APIs
   - DevOps Engineer: Docker, Kubernetes, Terraform, CI/CD, cloud platforms
   - Frontend Developer: React, TypeScript, Vue, CSS, accessibility, performance
   - Each agent owns its prompts (Factor 2) and has focused capabilities (Factor 10)

6. **API Server (`agentops/api/server.py`)**
   - Trigger from anywhere (Factor 11)
   - REST endpoints: `/tasks`, `/agents`, `/agents/{id}/status`, `/agents/{id}/pause`, `/agents/{id}/resume`
   - Human response endpoint: `/agents/{id}/human-response`
   - WebSocket endpoint: `/ws` for real-time updates

### Frontend Dashboard

**Status**: Not yet implemented. Planned features:
- Location: `agentops/static/index.html`
- Real-time monitoring of agent activity
- Task assignment interface
- Activity stream showing agent communications
- Built with TailwindCSS and Alpine.js

### Deployment

**Current Status**: Core implementation complete, deployment configurations pending.

Planned deployment modes:
- Docker-based deployment via `docker-compose.yml`
- Three deployment modes:
  1. Full RunPod deployment (~$40-50/month)
  2. Hybrid (RunPod LLM + local orchestrator, ~$20-30/month)
  3. Local with Ollama (free)

## Development Commands

### Running the Application

```bash
# Local development with mock LLM (no API key required)
python -m agentops.api.server

# Or directly
python agentops/api/server.py

# Server will start on http://localhost:8000
```

### API Interactions

```bash
# List all registered agents
curl http://localhost:8000/agents

# Create a task for backend developer
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "backend_developer",
    "description": "Build user authentication API with JWT tokens",
    "context": {}
  }'

# Get agent status
curl http://localhost:8000/agents/{agent_id}/status

# Get full agent state
curl http://localhost:8000/agents/{agent_id}/state

# Pause an agent
curl -X POST http://localhost:8000/agents/{agent_id}/pause

# Resume an agent
curl -X POST http://localhost:8000/agents/{agent_id}/resume

# Provide human response to waiting agent
curl -X POST http://localhost:8000/agents/{agent_id}/human-response \
  -H "Content-Type: application/json" \
  -d '{"response": {"approved": true, "feedback": "Looks good"}}'

# Health check
curl http://localhost:8000/health
```

### Using with Real LLM Providers

To use with real LLM providers instead of the mock, modify the startup code in `agentops/api/server.py`:

```python
# For OpenAI
llm_client = LLMClient(provider="openai", api_key="your-key", model="gpt-4")

# For Anthropic
llm_client = LLMClient(provider="anthropic", api_key="your-key", model="claude-3-opus-20240229")

# For RunPod
llm_client = LLMClient(provider="runpod", endpoint="https://your-endpoint.runpod.io", api_key="your-key")

# For Ollama (local)
llm_client = LLMClient(provider="ollama", endpoint="http://localhost:11434", model="llama2")
```

### Tool System (Factor 4: Tools as Structured Outputs)

The tool system uses pure JSON schemas without framework magic:

```python
# Define a tool
tool = Tool(
    name="write_file",
    description="Write content to a file",
    parameters=[
        ToolParameter(name="path", type="string", description="File path", required=True),
        ToolParameter(name="content", type="string", description="File content", required=True)
    ],
    function=lambda path, content: Path(path).write_text(content)
)

# Register with registry
tool_registry.register(tool)

# Tool returns OpenAI-compatible JSON schema
schema = tool.get_schema()
# {
#   "name": "write_file",
#   "description": "Write content to a file",
#   "parameters": {
#     "type": "object",
#     "properties": {...},
#     "required": [...]
#   }
# }
```

**Default Tools**:
- `execute_code`: Run Python code in isolated environment
- `read_file`: Read file contents
- `write_file`: Write file contents
- `api_call`: Make HTTP requests
- `request_human_input`: Request human decision (Factor 7)
- `complete_task`: Mark task as complete

### Environment Variables

**Current Implementation**: Uses mock LLM by default, no environment variables required.

**For Production** (when using real LLM providers):
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `RUNPOD_API_KEY`: RunPod API key
- `RUNPOD_ENDPOINT`: RunPod inference endpoint URL
- `OLLAMA_ENDPOINT`: Ollama endpoint (default: http://localhost:11434)
- `LLM_PROVIDER`: Provider to use (openai, anthropic, runpod, ollama)
- `LLM_MODEL`: Model name to use

**Future** (planned):
- `REDIS_URL`: Redis connection for state persistence
- `MAX_MONTHLY_COST`: Cost limit threshold
- `ENABLE_MONITORING`: Enable/disable monitoring features

## Key Architectural Patterns

### Agent System (Factor 12: Stateless Reducer)

Each agent is a pure function that transforms state:

```python
class Agent:
    def step(self, state: AgentState, llm_client: LLMClient, tool_registry: ToolRegistry) -> AgentState:
        """Pure function: state → state"""
        # 1. Generate LLM response with tool calls
        # 2. Execute tool calls
        # 3. Return new state (immutable pattern)
```

Agent configurations define:
- **agent_type**: Unique identifier (e.g., "backend_developer")
- **system_prompt**: Owned, explicit prompt (Factor 2)
- **capabilities**: List of skills (e.g., python, docker, react)
- **allowed_tools**: Whitelist of tools agent can use
- **max_steps**: Maximum iterations before requiring human input
- **temperature**: LLM sampling temperature

### AgentState (Factor 5: Unified State)

Single dataclass containing everything:

```python
@dataclass
class AgentState:
    agent_id: str
    agent_type: str
    status: AgentStatus  # IDLE, RUNNING, PAUSED, COMPLETED, FAILED, AWAITING_HUMAN
    current_step: int
    max_steps: int
    context: Dict[str, Any]  # Business state
    conversation_history: List[Dict]  # Execution state
    tool_calls: List[Dict]
    tool_results: List[Dict]
    errors: List[Dict]
    human_response: Optional[Dict]
    task_description: str
    result: Optional[str]
```

### Task Flow

1. **Launch** (Factor 6): `orchestrator.launch(agent_id, task_description)` creates initial state
2. **Execute**: `orchestrator.run_until_complete(agent_id)` runs agent steps
3. **Step Execution**:
   - Call `agent.step(state, llm_client, tool_registry)` (pure function)
   - LLM converts natural language to tool calls (Factor 1)
   - Execute tools (Factor 4: just structured outputs)
   - Update state immutably
4. **Human-in-the-Loop** (Factor 7): Agent can call `request_human_input` tool
   - State becomes `AWAITING_HUMAN`
   - Execution pauses
   - Human provides response via API
   - Agent resumes with response in context
5. **Completion**: Agent calls `complete_task` tool or reaches max steps
6. **Error Handling** (Factor 9): Errors compacted to fit context window

### Communication Pattern (Factor 11: Trigger from Anywhere)

- **REST API**: Create tasks, check status, pause/resume agents
- **WebSocket**: Real-time updates (broadcasts agent state changes)
- **CLI**: Can be triggered from command line (future)
- **Webhooks**: Can be triggered from external events (future)

### Jazz-Inspired Patterns

1. **Lifecycle Management**: register → launch → run → pause/resume → complete
2. **Plugin Architecture**: ToolRegistry allows dynamic tool registration
3. **Agent Registry**: Orchestrator maintains registry of agent configurations
4. **Isolation**: Each agent has its own state, no shared mutable state

## Technology Stack

- **Backend**: Python 3.10+, FastAPI, asyncio, dataclasses
- **Real-time Communication**: WebSockets
- **LLM Integration**: OpenAI, Anthropic, RunPod, Ollama (multi-provider)
- **State Management**: In-memory (Redis planned for persistence)
- **Testing**: MockLLMClient for API-free testing
- **Deployment**: Docker, Docker Compose (planned)
- **AI Models**: Open-source LLMs (configurable size)

## Testing and Development

### Testing Without API Costs

The system includes a `MockLLMClient` that simulates LLM behavior without making actual API calls:

```python
llm_client = MockLLMClient()
orchestrator = AgentOrchestrator(llm_client=llm_client, tool_registry=tool_registry)
```

The mock client recognizes keywords in prompts and generates appropriate tool calls for testing.

### Development Workflow

1. **Start the server**: `python agentops/api/server.py`
2. **Create a task**: POST to `/tasks` with agent_type and description
3. **Monitor via WebSocket**: Connect to `/ws` for real-time updates
4. **Check status**: GET `/agents/{agent_id}/status`
5. **Provide human input**: POST to `/agents/{agent_id}/human-response` when agent is `AWAITING_HUMAN`

### Adding New Agents

1. Create agent configuration in `agentops/agents/your_agent.py`:

```python
from agentops.core import AgentConfig

def create_your_agent() -> AgentConfig:
    return AgentConfig(
        agent_type="your_agent_type",
        name="Your Agent Name",
        role="Your Agent Role",
        system_prompt="Your explicit system prompt...",
        task_prompt_template="Your task template with {task}...",
        capabilities=["skill1", "skill2"],
        allowed_tools=["execute_code", "read_file", "write_file"],
        max_steps=20,
        temperature=0.7
    )
```

2. Register in `agentops/api/server.py` startup:

```python
from agentops.agents.your_agent import create_your_agent

orchestrator.register_agent(create_your_agent())
```

### Adding New Tools

1. Create tool in your code or in `agentops/core/tools.py`:

```python
from agentops.core import Tool, ToolParameter

def my_custom_function(param1: str, param2: int) -> str:
    # Your tool implementation
    return f"Result: {param1} {param2}"

my_tool = Tool(
    name="my_custom_tool",
    description="Description of what this tool does",
    parameters=[
        ToolParameter(name="param1", type="string", description="First param", required=True),
        ToolParameter(name="param2", type="integer", description="Second param", required=True)
    ],
    function=my_custom_function
)
```

2. Register with the tool registry:

```python
tool_registry.register(my_tool)
```

3. Add to agent's `allowed_tools` list

## Cost Considerations

The system is designed to be cost-effective:
- **Mock LLM for Development**: No API costs during development and testing
- **Multi-Provider Support**: Choose the most cost-effective provider for your needs
- **Configurable Model Sizes**: Trade performance for cost (120B models vs smaller alternatives)
- **RunPod Serverless**: Endpoints scale to zero when idle (planned)
- **Local Ollama Option**: Completely free local inference
- **Cost Tracking**: Built-in monitoring (planned)
- **Three Deployment Tiers**: Different budgets supported

## Design Principles and Best Practices

This codebase strictly follows the **12-Factor Agents** methodology for production-ready LLM agents:

### Key Design Decisions

1. **No Hidden Magic** (Factors 2, 4, 8)
   - All prompts are explicit and owned
   - Tools are just JSON schemas, not framework abstractions
   - Control flow is explicit, no hidden event loops
   - Easy to understand, debug, and modify

2. **Immutable State** (Factor 12)
   - Agent.step() is a pure function
   - State transformations create new state objects
   - No shared mutable state between agents
   - Enables time-travel debugging and easy testing

3. **Human-in-the-Loop** (Factor 7)
   - Agents can request human input at any time
   - Explicit tool call for human interaction
   - Execution pauses until human responds
   - Enables safe deployment with human oversight

4. **Explicit Everything** (Factors 2, 3, 8)
   - Owned prompts (not hidden in framework)
   - Owned context window (explicit history management)
   - Owned control flow (explicit orchestrator logic)
   - Enables full control and customization

5. **Small, Focused Agents** (Factor 10)
   - Each agent has specific, limited capabilities
   - Backend, DevOps, Frontend specialists
   - Easier to reason about and debug
   - Can compose multiple agents for complex tasks

6. **Composable Tools** (Factor 4)
   - Tools are just functions with JSON schemas
   - Easy to add new tools
   - ToolRegistry enables plugin architecture
   - Tools can be shared across agents

### Jazz-Inspired Design

- **Lifecycle Management**: Clean APIs for register, launch, pause, resume, complete
- **Plugin Architecture**: ToolRegistry for dynamic tool registration
- **Agent Registry**: Centralized management of agent configurations
- **Isolation**: Each agent operates independently with its own state

### Code Organization

```
agentops/
├── core/               # Core orchestration logic
│   ├── agent.py       # Agent (stateless reducer)
│   ├── orchestrator.py # Orchestrator (lifecycle management)
│   ├── tools.py       # Tool system (plugin architecture)
│   └── llm_client.py  # LLM interface (multi-provider)
├── agents/            # Specialized agent configurations
│   ├── backend_dev.py
│   ├── devops.py
│   └── frontend_dev.py
└── api/               # API server (trigger from anywhere)
    └── server.py
```

### When Working with This Codebase

- **Adding Features**: Follow the 12-factor principles
- **Modifying Agents**: Update agent configs in `agentops/agents/`
- **Adding Tools**: Register new tools in the ToolRegistry
- **Testing**: Use MockLLMClient to avoid API costs
- **Debugging**: All state is explicit and inspectable
- **State Management**: Never mutate state, always create new state objects
- **Control Flow**: Keep it explicit in the orchestrator

### References

- **12-Factor Agents**: https://github.com/humanlayer/12-factor-agents
- **Jazz**: https://github.com/lvndry/jazz
- **Deployment Guide**: See `AGENT_OPS_DEPLOY.md` for RunPod deployment
