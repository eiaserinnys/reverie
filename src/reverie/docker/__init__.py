"""Docker management module for Reverie Setup Wizard.

This module provides functionality to:
- Check Docker installation
- Generate docker-compose.yml for PostgreSQL+pgvector
- Manage container lifecycle (start, stop, status)
"""

from reverie.docker.compose_template import generate_compose_file
from reverie.docker.docker_manager import DockerManager

__all__ = ["generate_compose_file", "DockerManager"]
