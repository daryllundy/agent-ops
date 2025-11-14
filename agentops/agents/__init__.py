"""
Specialized agent implementations.

Factor 10: Small, focused agents.
Each agent is designed for a specific domain.
"""

from .backend_dev import create_backend_dev_agent
from .devops import create_devops_agent
from .frontend_dev import create_frontend_dev_agent

__all__ = [
    "create_backend_dev_agent",
    "create_devops_agent",
    "create_frontend_dev_agent",
]
