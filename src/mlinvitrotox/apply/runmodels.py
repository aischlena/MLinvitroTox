from pathlib import Path

import click

import sys

# Project root:
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Make sure the local src/ directory is searched before site-packages
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

import mlinvitrotox.utils.predict as predict
from mlinvitrotox.utils.model import Model

MODELS_DIR = PROJECT_ROOT / "data" / "results" / "models"
### add option to choose the aeids based on mechanistic targets

@click.command("run")
@click.option(
    "--model", 
    "-m", 
    required=True, 
    default="models_v3.itox",
    help="name of the model or path to model file"
)
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    help="Input fingerprint CSV",
)
@click.option(
    "--id",
    "-d",
    help="column name of id",
    default=None,
    required=False,
)
@click.option(
    "--aeid",
    multiple=True,
    help="One or more AEIDs to predict. Can be specified multiple times.",
)
@click.option(
    "--mechanistic-target",
    "-MT",
    multiple=True,
    help="Filter assays by mechanistic target.",
)
@click.option(
    "--signal-direction",
    default=None,
    help="Filter assays by signal direction.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="path to directory where the processed output should be stored. It will be created if it does not exist.",
)
@click.option(
    "--similarity",
    "-s",
    type=click.Choice(["cosine", "jaccard"], case_sensitive=False),
    default="cosine",
    show_default=True,
    help="Similarity method used for predictions",
)

def run_models(model, input, id, aeid, mechanistic_target, signal_direction, output, similarity):
    """
    Run models on molecular fingerprints.
    """
    model_path = Path(model)

    # First check whether the model exists at the supplied path.
    if not model_path.exists():
        # Then check the package's local models directory.
        local_model_path = MODELS_DIR / model_path

        if local_model_path.exists():
            model_path = local_model_path
        elif not model_path.is_absolute():
            # Keep only the model filename for the Zenodo download.
            model_path = Path(model_path.name)

    model_instance = Model()
    model_instance.load(model_path)

    # Load input fingerprints

    df_predfps = (predict.load_and_prepare_pred_fps(input, id=id))

    # Run predictions

    df_predictions = model_instance.predict(
        df_predfps,
        similarity_method=similarity,
        aeids=list(aeid) if aeid else None,
        mechanistic_target=list(mechanistic_target) if mechanistic_target else None,
        signal_direction=signal_direction,
    )

    # Save predictions

    output_predictions_file = Path(output)

    output_predictions_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving predictions to: {output_predictions_file.resolve()}")

    df_predictions.to_csv(output_predictions_file, index=False)

    print(f"Predictions stored to: {output_predictions_file.resolve()}")


if __name__ == "__main__":
    run_models()
