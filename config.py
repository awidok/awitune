"""
Project configuration loader for awitune.

Each project is a directory with a config.yaml describing the task,
metrics, resources, and paths to project-specific files.
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ProjectConfig:
    name: str
    project_dir: Path

    # Docker
    docker_image: str = "awitune-agent"

    # Optimization — keys in eval_results.json (must match what evaluate.py produces)
    test_metric_key: str = "test_score"
    val_metric_key: str = "val_score"
    metric_direction: str = "maximize"  # "maximize" or "minimize"

    # Resources
    gpus: list[int] = field(default_factory=lambda: [0])
    timeout_minutes: int = 480
    max_turns: int = 500

    # Proxy
    proxy_port: int = 8081

    # Resolved paths (set in __post_init__)
    data_dir: Path = field(default=None, repr=False)
    solutions_dir: Path = field(default=None, repr=False)
    experiments_dir: Path = field(default=None, repr=False)
    evaluate_script: Path = field(default=None, repr=False)
    agent_prompt: Path = field(default=None, repr=False)
    analyst_prompt: Optional[Path] = field(default=None, repr=False)
    reference_dir: Optional[Path] = field(default=None, repr=False)

    def __post_init__(self):
        self.project_dir = Path(self.project_dir).resolve()
        self.data_dir = self.data_dir or self.project_dir / "data"
        self.solutions_dir = self.solutions_dir or self.project_dir / "solutions"
        self.experiments_dir = self.experiments_dir or self.project_dir / "experiments"
        self.evaluate_script = self.evaluate_script or self.project_dir / "evaluate.py"
        self.agent_prompt = self.agent_prompt or self.project_dir / "AGENT.md"

        analyst = self.project_dir / "ANALYST.md"
        if self.analyst_prompt is None and analyst.exists():
            self.analyst_prompt = analyst

        ref = self.project_dir / "reference"
        if self.reference_dir is None and ref.exists():
            self.reference_dir = ref

    def is_better(self, new_score: float, old_score: float) -> bool:
        if self.metric_direction == "maximize":
            return new_score > old_score
        return new_score < old_score

    def best_score_sort_key(self):
        """SQL ORDER BY direction for the primary metric."""
        return "DESC" if self.metric_direction == "maximize" else "ASC"


def load_config(project_dir: str | Path) -> ProjectConfig:
    """Load a ProjectConfig from a project directory's config.yaml."""
    project_dir = Path(project_dir).resolve()
    config_path = project_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml in {project_dir}")

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    raw["project_dir"] = project_dir
    return ProjectConfig(**raw)
