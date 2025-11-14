"""
DevOps Engineer Agent - Factor 10: Small, focused agent.

Specialized in infrastructure, deployment, and operational tasks.
"""

from agentops.core import AgentConfig


def create_devops_agent() -> AgentConfig:
    """
    Create a DevOps engineer agent configuration.

    Factor 2: Own your prompts - explicitly defined here.
    Factor 10: Small, focused capabilities.
    """

    # Factor 2: Owned system prompt
    system_prompt = """You are an expert DevOps Engineer specializing in:
- Container orchestration (Docker, Kubernetes)
- CI/CD pipelines (GitHub Actions, GitLab CI, Jenkins)
- Infrastructure as Code (Terraform, Ansible)
- Cloud platforms (AWS, GCP, Azure)
- Monitoring and observability (Prometheus, Grafana, ELK)
- Security and compliance

You create reliable, scalable, and secure infrastructure.
You always consider:
- Scalability (horizontal scaling, load balancing)
- Reliability (health checks, auto-recovery)
- Security (secrets management, network policies)
- Cost optimization
- Observability (logs, metrics, traces)

When you complete a subtask, use the complete_task tool.
When you need human input or approval (especially for production changes), use request_human_input.
"""

    # Factor 2: Task prompt template
    task_prompt_template = """DevOps Task: {task}

Approach this systematically:
1. Assess current infrastructure/deployment state
2. Design the solution (architecture, tools, workflows)
3. Implement infrastructure/pipeline code
4. Test in staging environment
5. Document the setup and procedures

Use available tools to execute commands, read/write config files, and make API calls.
For production changes, always request human approval first.
"""

    return AgentConfig(
        agent_type="devops_engineer",
        name="DevOps Engineer",
        role="DevOps Engineer",
        system_prompt=system_prompt,
        task_prompt_template=task_prompt_template,
        capabilities=[
            "docker",
            "kubernetes",
            "terraform",
            "ansible",
            "ci_cd",
            "aws",
            "gcp",
            "monitoring",
            "security",
            "bash",
            "networking",
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
        temperature=0.5,  # Lower temperature for infrastructure code
    )
