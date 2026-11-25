import pandas as pd
import joblib
import numpy as np

from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

### define 3 functions: load and prepare fps, calculate endpoint score, run model for predictions ###

def load_and_prepare_pred_fps(input_file, id=None):
    predfps = pd.read_csv(input_file)

    # Detect & rename chem_id column
    possible_id_columns = ["dtxsid", "guid", "features", "sirius_id", "folder"]
    if id is not None:
        possible_id_columns.append(id)

    found_id_column = None
    col_map = {c.lower(): c for c in predfps.columns}

    for col in possible_id_columns:
        if col in col_map:
            found_id_column = col_map[col]  # Store the found ID column name
            predfps.rename(columns={col_map[col]: "chem_id"}, inplace=True)
            break  # Assuming only one such column exists, exit the loop after renaming

    # If no predefined ID column is found, prompt the user for selection
    if found_id_column is None:
        print(
            "No predefined ID column found. Please select the ID column from the list below:"
        )
        for index, column in enumerate(predfps.columns, start=1):
            print(f"{index}. {column}")
        selected_index = int(input("Enter the number of the ID column: ")) - 1
        if 0 <= selected_index < len(predfps.columns):
            selected_column = predfps.columns[selected_index]
            predfps.rename(columns={selected_column: "chem_id"}, inplace=True)
            found_id_column = selected_column
        else:
            print("Invalid selection. Exiting without renaming columns.")
            return predfps  # Early return in case of invalid selection

    # Keep fingerprint columns + chem_id
    fp_cols = [
        col for col in predfps.columns
        if col == "chem_id" or any(ch.isdigit() for ch in col)
    ]
    predfps = predfps[fp_cols]

    # Binarize fingerprints
    bit_cols = [c for c in predfps.columns if c != "chem_id"]
    predfps[bit_cols] = (predfps[bit_cols] >= 0.5).astype(int)

    return predfps


def calculate_endpoint_score(predictions_df):
    # remove not annotated mechanistic targets
    filtered_predictions_df = predictions_df[predictions_df["MechanisticTarget"] != "not_annotated"].copy()

    # group and calculate endpoint score
    cols_gb = ["chem_id", "signal_direction", "MechanisticTarget"]
    filtered_predictions_df["endpoint_score"] = (
        filtered_predictions_df.groupby(cols_gb)["prediction"]
        .transform("mean")
    )
    predictions_df = predictions_df.merge(
        filtered_predictions_df[cols_gb + ["endpoint_score"]],
        on=cols_gb,
        how="left"
    )
    predictions_df["endpoint_score"] = predictions_df["endpoint_score"].fillna(np.nan)

    # calculate number of endpoints per mechanistic target
    predictions_df["endpoints_count"] = (
        predictions_df.groupby(cols_gb[1:])["aeid"]
        .transform("nunique")
    )
    # weighted endpoint score based on model balanced accuracy
    filtered_predictions_df["weighted_prob"] = (filtered_predictions_df["probability"]* filtered_predictions_df["model_BA"])

    weighted_endpoint_score = (
        filtered_predictions_df
        .groupby(cols_gb)
        .agg(
            weighted_prob_sum=("weighted_prob", "sum"),
            ba_sum=("model_BA", "sum")
        )
        .reset_index()
    )

    weighted_endpoint_score["weighted_endpoint_score"] = (weighted_endpoint_score["weighted_prob_sum"]/ weighted_endpoint_score["ba_sum"])   

    weighted_endpoint_score = weighted_endpoint_score[cols_gb + ["weighted_endpoint_score"]]

    predictions_df = predictions_df.merge(
        weighted_endpoint_score,
        on=cols_gb,
        how="left"
    )

    predictions_df = predictions_df.drop_duplicates()

    return predictions_df


def run_model(
    aeid: str,
    df_pred: pd.DataFrame,
    chem_ids: pd.Series,
    true_fps: pd.DataFrame,
    get_path,
    train_fp_cache: dict,
    similarity_method: str | None = None,
):
    # Paths inside the model archive or directory
    preprocessing_path = (f"class/{aeid}/FS_RandomForest/preprocessing_model.joblib")
    classifier_path = (f"class/{aeid}/FS_RandomForest/XGBClassifier/best_estimator_full_data.joblib")
    selected_chemicals_path = f"class/{aeid}/selected_chemicals.csv"
    metrics_path = (f"class/{aeid}/FS_RandomForest/mb_val_structure/metrics.csv")

    # Load model components
    preprocessor = joblib.load(get_path(preprocessing_path))
    classifier = joblib.load(get_path(classifier_path))

    # Apply preprocessing
    X_pre = preprocessor.transform(df_pred)

    # Get feature names
    if hasattr(preprocessor, "get_feature_names_out"):
        final_features = preprocessor.get_feature_names_out()
    else:
        final_features = [f"f{i}" for i in range(X_pre.shape[1])]

    # Load training fingerprints once per AEID
    if aeid not in train_fp_cache:
        with get_path(selected_chemicals_path) as f:
            selected_df = pd.read_csv(f)

        selected_ids = selected_df["DTXSID"]
        filtered_true = true_fps[
            true_fps["DTXSID"].isin(selected_ids)
        ].loc[:, final_features]

        train_fp_cache[aeid] = filtered_true.to_numpy(dtype=bool)
    else:
        # Reconstruct the cached training fingerprint array
        X_train_bool = train_fp_cache[aeid]
        filtered_true = pd.DataFrame(
            X_train_bool,
            columns=final_features,
        )

    # Similarity
    similarity = None

    if similarity_method == "cosine":
        sim_matrix = cosine_similarity(
            X_pre, filtered_true.to_numpy(dtype=bool)
        )
        similarity = sim_matrix.max(axis=1)

    elif similarity_method == "jaccard":
        X_new_bool = X_pre.astype(bool, copy=False)
        X_train_bool = train_fp_cache[aeid]

        max_sim = np.zeros(X_new_bool.shape[0])
        chunk_size = 1000

        for i in range(0, X_train_bool.shape[0], chunk_size):
            chunk = X_train_bool[i:i + chunk_size]
            jaccard_sim = 1 - pairwise_distances(
                X_new_bool,
                chunk,
                metric="jaccard",
            )
            max_sim = np.maximum(
                max_sim, jaccard_sim.max(axis=1)
            )

        similarity = max_sim

    elif similarity_method is not None:
        raise ValueError(
            "similarity_method must be None, 'cosine', or 'jaccard'"
        )

    # Predictions
    predictions = classifier.predict(X_pre)
    probabilities = classifier.predict_proba(X_pre)[:, 1]

    # Model balanced accuracy
    with get_path(metrics_path) as f:
        metrics_file = pd.read_csv(f)

    metrics = metrics_file.iloc[0]

    return pd.DataFrame({
        "chem_id": chem_ids,
        "aeid": aeid,
        "prediction": predictions,
        "probability": probabilities,
        "similarity": similarity,
        "model_BA": metrics.get("balanced_accuracy"),
    })


