import joblib
import zipfile
import shutil
import requests
import toml
from io import StringIO

from pathlib import Path

import pandas as pd

import click
from tqdm import tqdm

import mlinvitrotox.utils.training as ml
import mlinvitrotox.utils.predict as predict

SELECTED_AEIDS = "selected_aeids.csv"
SELECTED_CHEMICALS = "selected_chemicals.csv"


class Model:
    model_repo = "https://zenodo.org/records/23262638/files/"

    def __init__(
        self, ml_algorithm: str = "class"
    ) -> None:
        self.ml_algorithm = ml_algorithm

    def load(self, path: str | Path):
        model_path = Path(path)

        if not model_path.exists():
            try:
                model_name = model_path.with_suffix("")
                model_path = self.download(model_name)

            except RuntimeError as e:
                raise FileNotFoundError(
                    f"Could not find model {path}. "
                    "To use the provided models, specify '-m mlinvitrotox_model'."
                ) from e

        # Store the resolved model path
        self.model_path = Path(model_path)

        #print(f"Resolved model path: {self.model_path.resolve()}")
        #print(f"Model exists: {self.model_path.exists()}")

        if self.model_path.suffix == ".itox":
            zip_ref = zipfile.ZipFile(self.model_path, "r")

            def get_path(sub_path):
                try:
                    return zip_ref.open(Path(sub_path).as_posix())
                except KeyError as e:
                    raise FileNotFoundError(e)

        elif self.model_path.is_dir():

            def get_path(sub_path):
                return self.model_path / sub_path

        else:
            raise RuntimeError(f"Failed to open file {self.model_path}")

        self.get_path = get_path

        # load true training fingerprints
        training_fps_path = get_path("ToxCast_True_FPS.csv")
        self.df_truefps = ml.get_fingerprint_df(training_fps_path)

        # load features to keep
        features_to_keep_path = get_path("features_to_keep.csv")
        self.features_to_keep = pd.read_csv(features_to_keep_path)["fingerprints"].tolist()

        # load logged assay information overview
        assay_info_path = get_path(SELECTED_AEIDS)
        df_assay_info = pd.read_csv(assay_info_path)
        df_assay_info["aeid"] = df_assay_info["aeid"].astype("str")
        self.df_assay_info = df_assay_info

    def download(self, model_name: str):
        url = self.model_repo + str(model_name) + ".itox"

        # Find project root (directory containing pyproject.toml)
        project_root = Path.cwd()

        for parent in [project_root, *project_root.parents]:
            if (parent / "pyproject.toml").exists():
                project_root = parent
                break

        # Default model directory inside the repository
        default_value = project_root / "results"
        default_value.mkdir(parents=True, exist_ok=True)

        # Optional configuration override
        appdir = Path(click.get_app_dir("mlinvitrotox"))
        appdir.mkdir(parents=True, exist_ok=True)

        toml_file = appdir / "config.toml"

        if toml_file.exists():
            with open(toml_file, "r") as file:
                config = toml.load(file)

            model_cache = Path(
                config.get("model_cache", str(default_value))
            )
        else:
            config = {"model_cache": str(default_value)}

            with open(toml_file, "w") as file:
                toml.dump(config, file)

            model_cache = default_value

        model_cache.mkdir(parents=True, exist_ok=True)

        local_filename = model_cache / f"{model_name}.itox"

        print(f"Model directory: {model_cache.resolve()}")

        # Avoid downloading an existing model again
        if local_filename.exists():
            print(f"Using existing model: {local_filename}")
            return local_filename

        response = requests.get(url, stream=True, timeout=60)

        if not response.ok:
            raise RuntimeError(
                f"Could not find model {model_name} in the online repository."
            )

        print(f"Downloading model {model_name}...")

        with open(local_filename, "wb") as file:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                if chunk:
                    file.write(chunk)

        print(f"Downloaded model to: {local_filename.resolve()}")

        return local_filename

    def predict(self, df_predfps, similarity_method: str | None = None, aeids: list[str] | None = None, mechanistic_target: str | None = None, signal_direction: str | None = None,):
        """
        Run all models on the prepared prediction fingerprints.
        """

        # Make sure chem_id is retained
        columns_to_keep = [
            str(col)
            for col in self.features_to_keep
        ]

        if "chem_id" not in columns_to_keep:
            columns_to_keep.insert(0, "chem_id")

        df_pred = df_predfps.loc[:, columns_to_keep].copy()

        chem_ids = df_pred.pop("chem_id")

        # AEIDs to predict
        df_assays = self.df_assay_info.copy()

        # Filter by AEID
        if aeids is not None:
            aeids = [str(aeid) for aeid in aeids]
            df_assays = df_assays[
                df_assays["aeid"].astype(str).isin(aeids)
            ]

        # Filter by mechanistic target
        if mechanistic_target is not None:

            # Match any of the provided search terms
            mask = pd.Series(False, index=df_assays.index)

            for target in mechanistic_target:
                mask |= df_assays["MechanisticTarget"].str.contains(
                    target,
                    case=False,
                    na=False,
                    regex=False,
                )

            matched_targets = (
                df_assays.loc[mask, "MechanisticTarget"]
                .dropna()
                .unique()
            )

            print(f"Mechanistic targets matching {mechanistic_target}:")

            for target in matched_targets:
                print(f"  - {target}")

            df_assays = df_assays[mask]

        # Filter by signal direction
        if signal_direction is not None:
            df_assays = df_assays[
                df_assays["signal_direction"] == signal_direction
            ]

        # AEIDs to predict
        aeids = df_assays["aeid"].astype(str)

        if df_assays.empty:
            raise ValueError("No assays match the selected filters.")

        all_predictions = []

        # Cache training fingerprints
        train_fp_cache = {}

        for aeid in aeids:
            print(f"Running model for AEID {aeid} ...")

            pred_df = predict.run_model(
                aeid=aeid,
                df_pred=df_pred,
                chem_ids=chem_ids,
                true_fps=self.df_truefps,
                get_path=self.get_path,
                train_fp_cache=train_fp_cache,
                similarity_method=similarity_method,
            )

            all_predictions.append(pred_df)

        # Combine all AEIDs
        all_predictions_df = pd.concat(
            all_predictions,
            ignore_index=True
        )

        # Merge assay information
        all_predictions_df["aeid"] = (
            all_predictions_df["aeid"].astype(str)
        )

        df_assay = self.df_assay_info.copy()
        df_assay["aeid"] = df_assay["aeid"].astype(str)

        predictions_df = pd.merge(
            all_predictions_df,
            df_assay,
            on="aeid",
            how="left",
        )

        # Calculate endpoint scores
        predictions_df = predict.calculate_endpoint_score(
            predictions_df
        )

        return predictions_df

    def export(self, filepath: str | Path, export_stats=False):
        """Exports current model to an .itox file at the given path

        Args:
            filepath (str | Path): the path to the .itox file to be created
            export_stats (boolean): if the statistics files should be exported as well

        """
        filepath = Path(filepath)
        if filepath.suffix != ".itox" and not filepath.is_dir():
            raise RuntimeError(
                f"Wrong file type of target. Export targets can only be .itox file or an existing directory.\nProvided filepath is {filepath}"
            )

        if self.model_path.suffix == ".itox":
            # since we already have an (immutable) itox file we can just copy it
            shutil.copy(
                self.model_path,
                filepath,
            )
        elif self.model_path.is_dir():
            # We need to collect all relevant files for export
            if filepath.is_dir():
                # make the output have the same filename as the original
                filepath = filepath / self.model_path.with_suffix(".itox").name

            include_list = []

            # add base
            include_list += [self.model_path / "modeltraining_metrics.csv"]
            include_list += [self.model_path / "ToxCast_True_FPS.csv"]
            include_list += [self.model_path / "features_to_keep.csv"]
            # selected_aeids.csv will be created later

            # add logs
            include_list += [self.model_path / "log" / "modelfitting.log"]

            # for each aeid
            failed_aeids = []
            aeid_folders = self.model_path / self.ml_algorithm
            for aeid_folder in aeid_folders.glob("*/"):
                if (
                    aeid_folder / "FS_RandomForest"
                    not in aeid_folder.iterdir()
                ):
                    continue
                if aeid_folder / "failed.txt" in aeid_folder.iterdir():
                    failed_aeids += [int(aeid_folder.stem)]
                    continue

                include_list += [aeid_folder / "selected_chemicals.csv"]
                feature_selection_path = (
                    aeid_folder / "FS_RandomForest"
                )
                include_list += [feature_selection_path / "preprocessing_model.joblib"]
                estimator_path = (
                    aeid_folder
                    / "FS_RandomForest"
                    / "XGBClassifier"
                )
                include_list += [estimator_path / "best_estimator_full_data.joblib"]
                include_list += [estimator_path / "selected_features.csv"]

                # add metric folders
                if export_stats:
                    wanted_files = [
                        "cm_default.svg",
                        "metrics.csv",
                        "report_default.csv",
                        "roc_curve.svg",
                        "test_predictions_default.csv",
                        "threshold.csv",
                    ]

                    for folder in [
                        "mb_val_srs_neg",
                        "mb_val_srs_pos",
                        "mb_val_structure",
                    ]:
                        folder_path = feature_selection_path / folder

                        include_list += [
                            folder_path / filename
                            for filename in wanted_files
                            if (folder_path / filename).is_file()
                        ]

            # update selected aeids files to not contain failed aeids
            selected_aeids_path = self.get_path("selected_aeids.csv")
            df = pd.read_csv(selected_aeids_path)
            df = df[~df["aeid"].isin(failed_aeids)]

            # Writing the files to the zipfile
            with zipfile.ZipFile(
                filepath, "w", compression=zipfile.ZIP_DEFLATED
            ) as zipf:
                for path in tqdm(include_list):
                    zipf.write(path, arcname=path.relative_to(self.model_path))

                # write the selected_aeids.csv
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                zipf.writestr("selected_aeids.csv", csv_buffer.getvalue())
        else:
            raise RuntimeError(f"Unkown model file {self.model_path}")
