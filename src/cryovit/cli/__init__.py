"""Command-line interface for CryoViT."""

## Necessary imports to register CLI commands in pyproject.toml. DO NOT REMOVE!
from .cli import cli  # noqa: F401
from .dino_cli import features  # noqa: F401
from .eval_cli import evaluate  # noqa: F401
from .infer_cli import infer  # noqa: F401
from .train_cli import train  # noqa: F401
