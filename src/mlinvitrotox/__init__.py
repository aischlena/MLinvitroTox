import click

from mlinvitrotox.apply.runmodels import run_models
from mlinvitrotox.train.evaluatemodels import run_evaluation
from mlinvitrotox.train.providemodels import export_model


@click.group()
def main():
    """MLinvitroTox command-line interface."""
    pass


main.add_command(run_models)
main.add_command(run_evaluation)
main.add_command(export_model)