from pathlib import Path

import click

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from mlinvitrotox.utils.model import Model


@click.command("export")
@click.option(
    "--model",
    "-m",
    required=True,
    default="models_v3",
    help="name of the model",
)
@click.option(
    "--output", 
    "-o", 
    required=True, 
    help="output filename (.itox) or directory"
)
@click.option(
    "--export_stats",
    "-s",
    required=True,
    type=click.BOOL,
    default=False,
    help="include stats for app",
)
def export_model(model, output, export_stats):
    """
    Export model

    """
    model_instance = Model()
    model_instance.load(model)
    model_instance.export(output, export_stats)


if __name__ == "__main__":
    export_model()

