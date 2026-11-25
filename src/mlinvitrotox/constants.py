from pathlib import Path

# Project root:
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directories
CONFIG_DIR_PATH = PROJECT_ROOT / "src" / "mlinvitrotox" / "config"
DATA_DIR_PATH = PROJECT_ROOT / "data"
INPUT_DIR_PATH = DATA_DIR_PATH / "input"
OUTPUT_DIR_PATH = DATA_DIR_PATH / "processed"
RESULTS_DIR_PATH = DATA_DIR_PATH / "results"


# IDs
CHEM_ID = "DTXSID"
SMILES_ID = "smiles"
SIRIUS_ID = "sirius_id"

# Step 2 input
INVITRO_INPUT_DIR_PATH = INPUT_DIR_PATH / "invitro"
INVITRODB_CHEMICAL_LIST = "Chemical_List_ToxCast_invitroDB_v4_1-2024-04-29.csv"
SIRIUS_FPS_DEFINITIONS = "csi_fingerid.tsv"

# Step 2 output
FPS_OUTPUT_DIR_PATH = OUTPUT_DIR_PATH / "chemical_fingerprints"
INTERMEDIATE_FPS_OUTPUT_DIR_PATH = FPS_OUTPUT_DIR_PATH / "intermediate"
TRUE_FPS_TRAINING = "ToxCast_True_FPS.csv"
TRUE_HDF_FPS_TRAINING = "HDF_1024_fingerprints.csv"


# Step 4 output
VALIDATION_FPS_OUTPUT_DIR_PATH = FPS_OUTPUT_DIR_PATH / "validation"
VALIDATION_SELECTED_ABSINDEX = "fingerprint_absindex.csv"

# Step 3 output
PRED_FPS_TEST_POS = "Pred_FPs_pos.csv"
PRED_FPS_TEST_NEG = "Pred_FPs_neg.csv"

# Modeling

# Modeling input
CONFIG_FILE = "config.yaml"
CONFIG_MODELS_FILE = "config_classification.yaml"
PYTCPL_INPUT_FOLDER = "pytcpl"

ICE_REFERENCES = "ice_reference_file_mechtargets_aeid.csv"
PYTCPL_OVERVIEW_FILE = "assay_overview_selected.csv"

# Modeling output
MODELS_RESULTS_DIR_PATH = RESULTS_DIR_PATH / "models"
SELECTED_AEIDS = "selected_aeids.csv"
SELECTED_CHEMICALS = "selected_chemicals.csv"


# Data abundance
AEIDS_ABUNDANCE = OUTPUT_DIR_PATH / "aeids_data_abundance.csv"


# Retrained models

RETRAINED_MODELS_DIR_PATH = MODELS_RESULTS_DIR_PATH / "final_models_retrained"
RETRAINED_MODELS_CLASS_DIR_PATH = MODELS_RESULTS_DIR_PATH / "final_models_retrained" / "class"
TOTAL_CHEMICALS = "selected_chemicals.csv"
TEST_CHEMICALS = "test_true_chemicals.csv"
AEIDS_WORKING = "selected_aeids.csv"