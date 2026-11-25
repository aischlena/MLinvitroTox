from tqdm import tqdm

from pathlib import Path
import click

import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from mlinvitrotox.utils.model import Model


@click.command("eval")
@click.option("--model", "-m", default="models_v3", help="name of the model")
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="path to directory where the metric file should be stored. It will be created if it does not exist.",
)
def run_evaluation(model, output):
    """
    Evaluate models

    """
    model_instance = Model()
    model_instance.load(model)

    metrics_dfs_list = []

    # for each aeid
    print("Loading metric files")
    aeids = model_instance.df_assay_info["aeid"].astype("str").to_list()
    for aeid in tqdm(aeids, desc="Processing AEIDs"):
        aeid_path = (
            Path(model_instance.ml_algorithm)
            / str(aeid)
        )
        preproc_path = aeid_path / "FS_RandomForest"

        # load metrics
        metrics_path = preproc_path / "mb_val_structure" / "metrics.csv"
        metrics_path = model_instance.get_path(metrics_path)
        met_df = pd.read_csv(metrics_path)
        met_df.insert(loc=0, column="aeid", value=aeid)
        metrics_dfs_list.append(met_df)

    # concatenate metrics and predictions
    metrics_df = pd.concat(metrics_dfs_list)
    metrics_df = metrics_df.sort_values("aeid")

    # Save metrics
    output_metrics_file = Path(output)
    
    output_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving metrics to: {output_metrics_file.resolve()}")
    
    metrics_df.round(5).to_csv(output_metrics_file, index=False)
    print("Metrics stored.")


if __name__ == "__main__":
    run_evaluation()
