# MLinvitroTox

MLinvitroTox performs high-throughput hazard-based prioritization of high-resolution mass spectrometry data.


## A. Project description

MLinvitroTox is an open-source Python package developed for hazard-driven prioritization of toxicologically relevant signals detected in complex environmental samples through nontarget high-resolution mass spectrometry (NTS HRMS/MS). MLinvitroTox is a machine learning (ML) framework comprising 300 independent XGBoost classifiers trained on molecular fingerprints from chemical structures and target specific endpoints from the ToxCast/Tox21 [invitroDBv4.1 database](https://www.epa.gov/comptox-tools/exploring-toxcast-data). MLinvitroTox predicts a bioactivity for each specific endpoint for every unidentified HRMS feature (a distinct m/z ion) based on the molecular fingerprints derived from MS2 fragmentation spectra. The binary bioactivity fingerprints are used as the basis for prioritizing the HRMS features towards further elucidation and analytical confirmation. This approach adds toxicological relevance to environmental analysis by focusing the time-consuming molecular identification efforts on features most likely to cause adverse effects. 
MLinvitroToc was first developed by Arturi et al. [MLinvitroTox 2 github]([https://github.com/aischlena/MLinvitroTox/tree/main](https://github.com/aischlena/MLinvitroTox/tree/legacy_mlinvitrotox_2/src/mlinvitrotox)) using Sirius 5.8.2 for the positive ionization mode. The model was retrained using fingerprints computed with SIRIUS 6.1.0 and expanded to include both the positive and negative electroionization modes. Following information relates to the latest model release.

The package contains 
- scripts that were used to build the models
- modeling results
- scripts for users to run the models on their data


## B. Getting started using git

```
git clone https://github.com/aischlena/MLinvitroTox.git
cd MLinvitroTox

python -m venv .venv
source .venv/Scripts/activate

python -m pip install --upgrade pip
python -m pip install .

mlinvitrotox --help
```


## C. Example / Usage

Have a look at the [tutorial](https://renkulab.io/projects/expectmine/mlinvitrotox-tutorial). 


## D. Development

If you are interested in the project and the package, please reach out to <lena.aisch@eawag.ch>.


## References
- Arturi et al. (2024) "MLinvitroTox reloaded for high-throughput hazard-based prioritization of HRMS data." (In preparation).
- Arturi, Katarzyna, and Juliane Hollender. "Machine learning-based hazard-driven prioritization of features in nontarget screening of environmental high-resolution mass spectrometry data." Environmental Science & Technology 57, no. 46 (2023): 18067-18079.
- Dührkop, Kai, Markus Fleischauer, Marcus Ludwig, Alexander A. Aksenov, Alexey V. Melnik, Marvin Meusel, Pieter C. Dorrestein, Juho Rousu, and Sebastian Böcker. "SIRIUS 4: a rapid tool for turning tandem mass spectra into metabolite structure information." Nature methods 16, no. 4 (2019): 299-302.
- Abedini, Jaleh, Bethany Cook, Shannon Bell, Xiaoqing Chang, Neepa Choksi, Amber B. Daniel, David Hines et al. "Application of new approach methodologies: ICE tools to support chemical evaluations." Computational Toxicology 20 (2021): 100184.
- Richard, Ann M., Richard S. Judson, Keith A. Houck, Christopher M. Grulke, Patra Volarath, Inthirany Thillainadarajah, Chihae Yang et al. "ToxCast chemical landscape: paving the road to 21st century toxicology." Chemical research in toxicology 29, no. 8 (2016): 1225-1251.
