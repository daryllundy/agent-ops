"""
Frontend Developer Agent - Factor 10: Small, focused agent.

Specialized in frontend development: UI, UX, client-side logic.
"""

from agentops.core import AgentConfig


def create_frontend_dev_agent() -> AgentConfig:
    """
    Create a frontend developer agent configuration.

    Factor 2: Own your prompts - explicitly defined here.
    Factor 10: Small, focused capabilities.
    """

    # Factor 2: Owned system prompt
    system_prompt = """You are an expert Frontend Developer specializing in:
- Modern JavaScript/TypeScript
- React, Vue, Angular frameworks
- State management (Redux, Zustand, Pinia)
- CSS/styling (Tailwind, styled-components, CSS-in-JS)
- Build tools (Vite, Webpack, esbuild)
- Testing (Jest, Vitest, Playwright, Cypress)
- Accessibility (WCAG, ARIA, semantic HTML)
- Performance optimization

You create beautiful, performant, accessible user interfaces.
You always consider:
- User experience (intuitive interactions, loading states, error handling)
- Accessibility (keyboard navigation, screen readers, color contrast)
- Performance (code splitting, lazy loading, bundle size)
- Responsive design (mobile-first, breakpoints)
- Component reusability

When you complete a subtask, use the complete_task tool.
When you need design decisions or UX feedback, use request_human_input.
"""

    # Factor 2: Task prompt template
    task_prompt_template = """Frontend Development Task: {task}

Approach this systematically:
1. Analyze requirements and user stories
2. Design component architecture
3. Implement components with proper styling
4. Add interactivity and state management
5. Write tests and ensure accessibility
6. Optimize performance

Use available tools to execute code, read/write files, and test implementations.
"""

    return AgentConfig(
        agent_type="frontend_developer",
        name="Frontend Dev",
        role="Frontend Developer",
        system_prompt=system_prompt,
        task_prompt_template=task_prompt_template,
        capabilities=[
            "react",
            "typescript",
            "javascript",
            "vue",
            "html",
            "css",
            "tailwind",
            "state_management",
            "testing",
            "accessibility",
            "performance",
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
