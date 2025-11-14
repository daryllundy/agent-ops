# AgentOps

A production-ready multi-agent orchestration system for autonomous AI development teams, built following the [12-Factor Agents](https://github.com/humanlayer/12-factor-agents) methodology and inspired by [Jazz](https://github.com/lvndry/jazz).

## Features

- **12-Factor Agents Methodology**: Production-ready LLM agents following industry best practices
- **Stateless Reducer Pattern**: Agents as pure functions enabling easy testing and debugging
- **Multi-Provider LLM Support**: OpenAI, Anthropic, RunPod, and Ollama
- **Human-in-the-Loop**: Built-in support for human oversight and decision-making
- **Specialized Agents**: Backend, DevOps, and Frontend developer agents with focused capabilities
- **Explicit Control Flow**: No hidden magic, all state and control flow is transparent
- **Plugin Architecture**: Extensible tool system with dynamic registration
- **Real-time Updates**: WebSocket support for monitoring agent activity
- **Cost-Effective**: Mock LLM for development, multi-provider support for production

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/daryllundy/agent-ops.git
cd agent-ops

# Install dependencies (coming soon)
pip install -r requirements.txt
```

### Running the Server

```bash
# Start with mock LLM (no API key required)
python agentops/api/server.py

# Server runs on http://localhost:8000
```

### Create Your First Task

```bash
# Create a backend development task
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "backend_developer",
    "description": "Build a REST API for user authentication with JWT tokens",
    "context": {}
  }'

# List all agents
curl http://localhost:8000/agents

# Check agent status
curl http://localhost:8000/agents/{agent_id}/status
```

## Architecture

### 12-Factor Agents Implementation

This system implements all 12 factors for production-ready LLM agents:

1. **Natural Language to Tool Calls** - LLM client converts natural language to structured tool calls
2. **Own Your Prompts** - Explicit system prompts in agent configurations
3. **Own Your Context Window** - Token-aware conversation history management
4. **Tools Are Just Structured Outputs** - Pure JSON schemas, no framework magic
5. **Unify Execution State and Business State** - Single AgentState dataclass
6. **Launch/Pause/Resume** - Simple lifecycle APIs
7. **Contact Humans with Tool Calls** - request_human_input tool for human oversight
8. **Own Your Control Flow** - Explicit orchestrator logic, no hidden event loops
9. **Compact Errors** - Error summarization for context window management
10. **Small, Focused Agents** - Specialized agents for specific domains
11. **Trigger from Anywhere** - REST API, WebSocket, CLI support
12. **Stateless Reducer** - Agent.step() as a pure function

### Core Components

```
agentops/
├── core/
│   ├── agent.py          # Stateless reducer agent implementation
│   ├── orchestrator.py   # Agent lifecycle management
│   ├── tools.py          # Tool system with plugin architecture
│   └── llm_client.py     # Multi-provider LLM interface
├── agents/
│   ├── backend_dev.py    # Backend developer agent
│   ├── devops.py         # DevOps engineer agent
│   └── frontend_dev.py   # Frontend developer agent
└── api/
    └── server.py         # FastAPI server with REST + WebSocket
```

## API Reference

### Endpoints

- `POST /tasks` - Create a new task for an agent
- `GET /agents` - List all registered agents
- `GET /agents/{agent_id}/status` - Get agent status
- `GET /agents/{agent_id}/state` - Get full agent state
- `POST /agents/{agent_id}/pause` - Pause agent execution
- `POST /agents/{agent_id}/resume` - Resume agent execution
- `POST /agents/{agent_id}/human-response` - Provide human response to waiting agent
- `GET /health` - Health check
- `WS /ws` - WebSocket for real-time updates

### Available Agents

- **backend_developer** - Python, FastAPI, Django, databases, API design
- **devops_engineer** - Docker, Kubernetes, Terraform, CI/CD, cloud platforms
- **frontend_developer** - React, TypeScript, Vue, CSS, accessibility

### Default Tools

- `execute_code` - Run Python code in isolated environment
- `read_file` - Read file contents
- `write_file` - Write file contents
- `api_call` - Make HTTP requests
- `request_human_input` - Request human decision/feedback
- `complete_task` - Mark task as complete

## Using Real LLM Providers

The system uses a mock LLM by default for development. To use real LLM providers, modify `agentops/api/server.py`:

```python
from agentops.core.llm_client import LLMClient

# For OpenAI
llm_client = LLMClient(
    provider="openai",
    api_key="your-openai-key",
    model="gpt-4"
)

# For Anthropic
llm_client = LLMClient(
    provider="anthropic",
    api_key="your-anthropic-key",
    model="claude-3-opus-20240229"
)

# For RunPod
llm_client = LLMClient(
    provider="runpod",
    endpoint="https://your-endpoint.runpod.io",
    api_key="your-runpod-key"
)

# For Ollama (local)
llm_client = LLMClient(
    provider="ollama",
    endpoint="http://localhost:11434",
    model="llama2"
)
```

## Extending the System

### Adding a New Agent

1. Create agent configuration:

```python
# agentops/agents/my_agent.py
from agentops.core import AgentConfig

def create_my_agent() -> AgentConfig:
    return AgentConfig(
        agent_type="my_agent",
        name="My Agent",
        role="My Specialized Role",
        system_prompt="Your explicit system prompt...",
        task_prompt_template="Task: {task}\n\nApproach...",
        capabilities=["skill1", "skill2"],
        allowed_tools=["execute_code", "read_file", "write_file"],
        max_steps=20,
        temperature=0.7
    )
```

2. Register in `agentops/api/server.py`:

```python
from agentops.agents.my_agent import create_my_agent

orchestrator.register_agent(create_my_agent())
```

### Adding a New Tool

```python
from agentops.core import Tool, ToolParameter

def my_tool_function(param1: str) -> str:
    return f"Processed: {param1}"

my_tool = Tool(
    name="my_tool",
    description="What this tool does",
    parameters=[
        ToolParameter(
            name="param1",
            type="string",
            description="Parameter description",
            required=True
        )
    ],
    function=my_tool_function
)

tool_registry.register(my_tool)
```

## Design Principles

### No Hidden Magic

- All prompts are explicit and owned (not hidden in frameworks)
- Tools are pure JSON schemas
- Control flow is explicit in the orchestrator
- State is transparent and inspectable

### Immutable State

- `Agent.step()` is a pure function: state → state
- No shared mutable state between agents
- Enables time-travel debugging and easy testing

### Human-in-the-Loop

- Agents can request human input at any time
- Execution pauses until human responds
- Enables safe deployment with human oversight

## Development

### Testing Without API Costs

The `MockLLMClient` simulates LLM behavior without making API calls:

```python
from agentops.core.llm_client import MockLLMClient

llm_client = MockLLMClient()
```

### Development Workflow

1. Start the server: `python agentops/api/server.py`
2. Create tasks via REST API
3. Monitor via WebSocket at `/ws`
4. Check agent status and state
5. Provide human responses when agents are `AWAITING_HUMAN`

## Deployment

See [AGENT_OPS_DEPLOY.md](AGENT_OPS_DEPLOY.md) for detailed deployment instructions on RunPod and other platforms.

Planned deployment modes:
- Full RunPod deployment (~$40-50/month)
- Hybrid (RunPod LLM + local orchestrator, ~$20-30/month)
- Local with Ollama (free)

## Documentation

- [CLAUDE.md](CLAUDE.md) - Comprehensive guide for AI assistants working with this codebase
- [AGENT_OPS_DEPLOY.md](AGENT_OPS_DEPLOY.md) - Deployment guide for RunPod and other platforms

## References

- [12-Factor Agents](https://github.com/humanlayer/12-factor-agents) - Methodology for production-ready LLM agents
- [Jazz](https://github.com/lvndry/jazz) - Autonomous agent management inspiration

## License

MIT License

## Contributing

Contributions welcome! This project follows the 12-Factor Agents methodology. Please ensure:
- Agents remain stateless reducers
- Prompts are explicit and owned
- Control flow is explicit
- State is immutable
- Tools are pure JSON schemas

## Roadmap

- [ ] Complete implementation of real LLM provider integrations
- [ ] Redis persistence layer for state management
- [ ] Web dashboard for real-time monitoring
- [ ] Docker and docker-compose configurations
- [ ] CLI interface for task management
- [ ] Cost tracking and alerting
- [ ] Multi-agent collaboration workflows
- [ ] Plugin marketplace for community tools
- [ ] Agent performance analytics
