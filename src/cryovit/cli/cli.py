"""Defines the command-line interface (CLI) for CryoViT.

Copied from https://github.com/teamtomo/fidder/blob/main/src/fidder/_cli.py."""

import typer
from click import Context
from typer.core import TyperGroup


class OrderCommands(TyperGroup):
    """Return list of commands in the order appear."""

    def list_commands(self, ctx: Context):
        """Return list of commands in the order appear."""
        return list(self.commands)  # get commands using self.commands


cli = typer.Typer(
    cls=OrderCommands,
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)


@cli.callback()
def callback():
    """
    CryoViT's command line interface (CLI) for training, evaluation, and inference.

    You can choose between the different options listed below.
    To see the help for a specific command, run:

    cryovit <command> --help
    """
