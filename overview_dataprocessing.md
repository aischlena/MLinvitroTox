# MLinvitroTox data processing

In data processing, all input input data is prepared for modeling. 

The data is open source and provided by us. The only exception is the application in the `data/input/sirius/sources_fractions`.

Script: `src/mlinvitrotox/train/processinputdata.py`

The directory structure and the correspondings paths are organized and specified in `src/mlinvitrotox/constants.py`.

TODO load big data sets from external sources


## Step 0: Extract SIRIUS directories

The SIRIUS directories need to be extracted before they can be processed with MLinvitroTox. Run `src/mlinvitrotox/apply/extractsiriusdata path/to/siriusfolder`. This takes a few minutes, depending on the folder size.



## Step 1: Filter and extract MassBank data

### Code:
- `src/mlinvitrotox/utils/extract_hrms_massbank_data.py`

### Input:

#### MassBank files (open-access)

1. `data/input/hrms_massbank/published/` 

Source:  `https://zenodo.org/records/10213786` or `https://github.com/MassBank/MassBank-data`

The folder contains 85'758 files downladed from the MassBank repository (462 MB) (02.10.2024).

delete_gc: all .txt folders in Published are being deleted that dont have AC$INSTRUMENT_TYPE =        ESI,     APCI, APPI, MALDI and deleted files are logged in delted_files.csv

split_pos_neg: put folders in Positive or Negative folder based on their ionization

group_by_inchikey: group the different spectra by their inchikeys into one folder, saved in combined_neg/pos

2. true fps and predicted fps for massbank

True fps were calculated over the file sirius_cal_fps_true 




### Output:

#### MassBank summary grouped by SMILES

11. `data/processed/hrms_massbank/massbank_published-data_smiles.csv`

A file with the SMILES, ACCESSION_LIST and GUID, grouped by SMILES resulting in several ACCESSIONs per SMILES

#### MassBank summary complete

12. `data/processed/hrms_massbank/massbank_published-data_complete.csv`

The complete filtered massbank output with 1 entry per ACCESSION

#### File to match GUID and DTXSID

10. `data/processed/hrms_massbank/massbank_GUID-DTXSID.csv`



## Step 2: Get true fingerprints

These step is done twice, for training and validation.

### Code:
- `src/utils/get_true_fps.py`

### Input (for training):

#### CSI FingerID definitions

3. `data/input/sirius/csi_fingerid.tsv`

NEWSIRIUS this will likely change with the new SIRIUS version

#### List of chemicals

4. `data/input/invitro/Chemical_List_Toxcast_invitroDB_v4_1-2024-04-29.csv`

This file is downloaded from `https://comptox.epa.gov/dashboard/chemical-lists`.

### Input (for validation):

#### CSI FingerID definitions

3. `data/input/sirius/csi_fingerid.tsv`

#### SMILES from summarized MassBank data

11. `data/processed/hrms_massbank/massbank_published-data_smiles.csv`

This is an output from step 1.


### Output (for training):

#### True fingerprints by DTXSID

13. `data/processed/chemical_fingerprints/ToxCast_true_fps.csv`

#### File with DTXSID, SMILES, standardized_smiles

xx. `data/processed/chemical_fingerprints/intermediate/Chemical_List_ToxCast_invitroDB_v3_5-2023-07-07_clean.csv`


### Output (for validation):

#### True fingerprints by GUID

14. `data/processed/chemical_fingerprints/validation_true-fps.parquet`


#### File with DTXSID, SMILES, standardized_smiles

xx`data\processed\chemical_fingerprints\intermediate\Chemical_List_ToxCast_invitroDB_v4_1-2025-08-17_stdsmiles.csv`



## Step 3: Get predicted fingerprints from SIRIUS

These step is done twice, for validation and testing.

### Code:
`src/utils/get_predicted_fps.py`

#### Input (for validation):

3. `data/input/sirius/csi_fingerid.tsv`


12. `data/processed/hrms_massbank/massbank_published-data_complete.csv`

Output from step 1

13. `data/processed/chemical_fingerprints/training_true-fps.parquet`

Output from Step 2

13. `data/processed/chemical_fingerprints/training_true-fps.parquet`

Output from Step 2

### Output (for validation):

15. `data/processed/chemical_fingerprintsPred_FPs_neg.csv`
15. `data/processed/chemical_fingerprintsPred_FPs_pos.csv`




## Step 4: MassBank validation of SIRIUS

### Code:
`src/utils/validate_massbank.py`

### Input:

14. `data/processed/chemical_fingerprints/validation_true-fps.parquet`

Output from Step 2

15. `data/processed/chemical_fingerprints/validation_sirius_pred-fps.parquet`

Output from Step 3

10. `data/hrms_massbank/massbank_GUID-DTXSID.csv`

Output from Step 1

### Output:

#### Validation metrics per fingerprint bit

xx. `data/processed/chemical_fingerprints/validation/massbank_validation_metrics_absindex.csv`

#### Some mappings from chemicals (as DTXSIDs) to AbsoluteIndexes

20. `data/processed/chemical_fingerprints/hrms_massbank/massbank_validation_predfps_dtxsid.csv`

#### AbsoluteIndexes passed validation

Only keep fingerprint bits with good enough recall and precision

21. `data/processed/chemical_fingerprints/validation/massbank_validation_selected_absindex.csv`
