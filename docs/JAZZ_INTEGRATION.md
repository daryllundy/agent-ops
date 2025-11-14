# Jazz Integration Guide

AgentOps integrates with [Jazz](https://github.com/lvndry/jazz), a TypeScript-based CLI tool for creating autonomous AI agents with real-world capabilities.

## What is Jazz?

Jazz transforms AI assistants from passive conversation partners into active agents that can:

- **Email Management**: 16 Gmail tools (search, send, label, archive, batch operations)
- **Git Operations**: 9 tools for repository management (commit, push, branch, status)
- **File System**: 15 tools for file/directory operations
- **Shell Execution**: Run commands with security validation
- **Web Search**: Information retrieval via Linkup service
- **HTTP Client**: REST API integration

## Why Use Jazz with AgentOps?

AgentOps provides:
- **12-Factor Agent methodology** for production-ready systems
- **Python-based orchestration** with stateless reducers
- **Multi-provider LLM support** (OpenAI, Anthropic, RunPod, Ollama)
- **Human-in-the-loop** integration

Jazz provides:
- **Real-world tool execution** (email, git, shell, filesystem)
- **Security-first design** with explicit approval system
- **TypeScript implementation** with type safety
- **Mature tool integrations** for common workflows

Together, they enable sophisticated multi-agent systems with both planning/orchestration (AgentOps) and execution capabilities (Jazz).

## Installation

### 1. Install Jazz CLI

```bash
# Using npm
npm install -g jazz-ai

# Or using the project's package.json
npm install

# Verify installation
npx jazz --version
```

### 2. Install Node.js Dependencies

```bash
# In the agentops project directory
npm install
```

### 3. Configure Jazz

Create a `jazz.config.json` file (or copy from example):

```bash
cp jazz.config.example.json jazz.config.json
```

Edit `jazz.config.json` with your API keys:

```json
{
  "llm": {
    "openai": {
      "api_key": "sk-your-openai-api-key"
    },
    "anthropic": {
      "api_key": "sk-ant-your-anthropic-api-key"
    }
  },
  "google": {
    "clientId": "your-google-client-id.apps.googleusercontent.com",
    "clientSecret": "your-google-client-secret"
  },
  "linkup": {
    "api_key": "your-linkup-api-key"
  }
}
```

**Note**: `jazz.config.json` is gitignored for security.

## Creating Jazz Agents

### Via CLI (Recommended)

```bash
# Create a new agent interactively
npx jazz agent create

# Or using npm script
npm run agent:create
```

The wizard will prompt you for:
- **Name**: Agent identifier (e.g., "email-manager")
- **Purpose**: What the agent does (e.g., "Manage and triage emails")
- **Provider**: LLM provider (openai, anthropic, etc.)
- **Model**: Specific model (gpt-4, claude-3-opus-20240229)
- **Tools**: Which capabilities to enable (gmail, git, filesystem, etc.)

### Example Agent Configurations

#### Email Triage Agent
```
Name: email-triage
Purpose: Automatically label and organize incoming emails
Provider: openai
Model: gpt-4
Tools: gmail, web
```

#### Git Assistant Agent
```
Name: git-helper
Purpose: Help with git operations and commit messages
Provider: anthropic
Model: claude-3-opus-20240229
Tools: git, filesystem, shell
```

#### Research Agent
```
Name: researcher
Purpose: Research topics and create documentation
Provider: openai
Model: gpt-4
Tools: web, filesystem, http
```

## Using Jazz Agents

### Via CLI

```bash
# List all agents
npx jazz agent list
npm run agent:list

# Chat with an agent
npx jazz agent chat email-triage
npm run agent:chat email-triage

# Get agent details
npx jazz agent get email-triage

# Delete an agent
npx jazz agent delete email-triage
```

### Via Python Bridge

The `JazzBridge` class provides programmatic access to Jazz agents from Python:

```python
from agentops.integrations import JazzBridge

# Initialize bridge
bridge = JazzBridge()

# List all Jazz agents
agents = bridge.list_agents()
for agent in agents:
    print(f"{agent.name}: {agent.purpose}")

# Get specific agent
agent = bridge.get_agent("email-triage")
if agent:
    print(f"Found agent: {agent.name}")
    print(f"Model: {agent.provider}/{agent.model}")
    print(f"Tools: {', '.join(agent.tools)}")

# Send a message to an agent
response = bridge.chat("email-triage", "Label all GitHub emails as 'dev'")
print(response)

# Get available tool categories
tools = bridge.get_available_tools()
print(f"Available tools: {list(tools.keys())}")
```

### Integrating with AgentOps Orchestrator

You can use Jazz agents alongside AgentOps Python agents:

```python
from agentops.core import AgentOrchestrator
from agentops.integrations import JazzBridge

# Initialize both systems
orchestrator = AgentOrchestrator(llm_client=llm_client, tool_registry=tool_registry)
jazz_bridge = JazzBridge()

# Use Python agents for planning and orchestration
state = orchestrator.launch("backend_developer", "Design API schema")
await orchestrator.run_until_complete("backend_developer")

# Use Jazz agents for execution tasks
jazz_response = jazz_bridge.chat("git-helper", "Commit the API schema changes")

# Combine workflows
# 1. Python agent designs the solution
# 2. Jazz agent executes file/git operations
# 3. Python agent reviews and completes
```

## Authentication

### Gmail Integration

```bash
# Login to Gmail
npx jazz auth gmail login

# Check status
npx jazz auth gmail status

# Logout
npx jazz auth gmail logout
```

Follow the OAuth flow to grant Jazz access to your Gmail account.

## Security Considerations

### Approval System

Jazz implements a "user approval system" where agents must request confirmation before:
- Sending emails
- Modifying files
- Executing shell commands
- Pushing to git repositories
- Other write operations

This ensures human oversight while maintaining agent autonomy for read operations.

### Command Validation

Jazz validates shell commands to prevent:
- Destructive operations without approval
- Unsafe command execution
- Unintended system modifications

### Audit Logging

All agent actions are logged for audit and debugging purposes.

## Tool Categories Reference

### Gmail (16 tools)
- **Read**: `search`, `get`, `list_labels`
- **Label**: `label`, `unlabel`
- **Organize**: `archive`, `unarchive`, `trash`
- **Compose**: `send`, `reply`, `forward`
- **Modify**: `mark_read`, `mark_unread`, `star`, `unstar`
- **Batch**: `batch_*` operations

### Git (9 tools)
- **Info**: `status`, `diff`, `log`
- **Changes**: `add`, `commit`, `push`, `pull`
- **Branches**: `branch`, `checkout`

### Filesystem (15 tools)
- **Read**: `read_file`, `list_directory`, `search`, `stat`
- **Write**: `write_file`, `create_directory`
- **Modify**: `delete`, `move`, `copy`, `permissions`, `symlink`

### Shell (2 tools)
- `exec`: Execute command
- `exec_streaming`: Execute with streaming output

### Web (1 tool)
- `search`: Web search via Linkup

### HTTP (1 tool)
- `request`: Make HTTP requests

## Example Workflows

### Email Workflow Automation

```bash
# Create email agent
npx jazz agent create
# Name: email-manager
# Tools: gmail, web

# Use it
npx jazz agent chat email-manager
> "Search for emails from GitHub, label them as 'dev', and archive"
```

### Git Assistant

```bash
# Create git agent
npx jazz agent create
# Name: git-helper
# Tools: git, filesystem

# Use it
npx jazz agent chat git-helper
> "Check git status, suggest a commit message, and commit the changes"
```

### Research & Documentation

```bash
# Create research agent
npx jazz agent create
# Name: researcher
# Tools: web, filesystem, http

# Use it
npx jazz agent chat researcher
> "Research TypeScript 5.5 features and create a summary document"
```

## Best Practices

1. **Start Small**: Begin with read-only operations to understand agent behavior
2. **Review Approvals**: Always review approval requests before confirming
3. **Test in Sandbox**: Test agents with non-critical data first
4. **Monitor Logs**: Review agent actions regularly
5. **Limit Permissions**: Only enable tools agents actually need
6. **Combine Systems**: Use AgentOps for orchestration, Jazz for execution

## Troubleshooting

### Jazz CLI Not Found

```bash
# Install globally
npm install -g jazz-ai

# Or use local install
npm install
npx jazz --version
```

### Configuration Issues

```bash
# Check config location
echo $JAZZ_CONFIG_PATH

# Or create in project directory
cp jazz.config.example.json jazz.config.json
```

### Authentication Errors

```bash
# Re-authenticate with Gmail
npx jazz auth gmail logout
npx jazz auth gmail login
```

### Python Bridge Errors

Ensure Jazz CLI is in your PATH:

```python
import subprocess
result = subprocess.run(["which", "jazz"], capture_output=True, text=True)
print(result.stdout)  # Should show path to jazz binary
```

## References

- **Jazz GitHub**: https://github.com/lvndry/jazz
- **Jazz Documentation**: See Jazz repository docs/
- **12-Factor Agents**: https://github.com/humanlayer/12-factor-agents
- **AgentOps Documentation**: See CLAUDE.md and README.md

## Contributing

Jazz agents can be extended with custom tools and integrations. See Jazz's contributing guidelines for:
- Adding new tool categories
- Implementing custom LLM providers
- Enhancing security features
- Improving the CLI experience
