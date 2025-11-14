"""
Backend Developer Agent - Factor 10: Small, focused agent.

Specialized in backend development tasks: APIs, databases, business logic.
"""

from agentops.core import AgentConfig


def create_backend_dev_agent() -> AgentConfig:
    """
    Create a backend developer agent configuration.

    Factor 2: Own your prompts - explicitly defined here.
    Factor 10: Small, focused capabilities.
    """

    # Factor 2: Owned system prompt
    system_prompt = """You are an expert Backend Developer specializing in:
- API design and implementation (REST, GraphQL, gRPC)
- Database design and optimization (PostgreSQL, MongoDB, Redis)
- Business logic and data modeling
- Authentication and authorization
- Testing (unit, integration, end-to-end)
- Performance optimization

You write clean, maintainable, production-ready code.
You always consider:
- Security (SQL injection, auth vulnerabilities)
- Performance (N+1 queries, caching)
- Testability (dependency injection, mocking)
- Documentation (clear docstrings, API docs)

When you complete a subtask, use the complete_task tool.
When you need human input or approval, use request_human_input.
"""

    # Factor 2: Task prompt template
    task_prompt_template = """Backend Development Task: {task}

Approach this systematically:
1. Analyze requirements
2. Design the solution (data models, API endpoints, business logic)
3. Implement the code
4. Write tests
5. Document the implementation

Use available tools to execute code, read/write files, and make API calls.
"""

    return AgentConfig(
        agent_type="backend_developer",
        name="Backend Dev",
        role="Backend Developer",
        system_prompt=system_prompt,
        task_prompt_template=task_prompt_template,
        capabilities=[
            "python",
            "fastapi",
            "django",
            "flask",
            "postgresql",
            "mongodb",
            "redis",
            "api_design",
            "database_design",
            "testing",
            "authentication",
            "authorization",
        ],
        allowed_tools=[
            "execute_code",
            "read_file",
            "write_file",
            "api_call",
            "request_human_input",
            "complete_task",
        ],
        max_steps=20,
        temperature=0.7,
    )
