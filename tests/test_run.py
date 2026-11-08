"""Kedro project bootstrapping does not depend on external data."""

from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


def test_project_context_loads():
    bootstrap_project(Path.cwd())
    with KedroSession.create(project_path=Path.cwd()) as session:
        assert session.load_context().catalog is not None
