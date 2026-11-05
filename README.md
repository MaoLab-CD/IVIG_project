# IVIG resistance in Kawasaki disease — machine-learning reproduction package

Code and de-identified data for the manuscript:

> **Predictive factors and interactions in intravenous immunoglobulin resistance in Kawasaki
> disease: a machine-learning study**
>
> Qiming Tang, Shuran Shao, Haoyang Mi, Yimin Hua, Kaiyu Zhou, Xiaoliang Liu, Chuan Wang,
> Xiaowei Mao
>
> *BMC Medical Informatics and Decision Making* — submitted, revision in progress.

This repository contains the two analysis notebooks, the de-identified prepared datasets they
read, and the scripts that produced those datasets.

---

## Contents

| Path | What it is |
| :--- | :--- |
| [`kd_ivig_analysis.ipynb`](kd_ivig_analysis.ipynb) | Main notebook. Reads only `data/prepared/`. Recomputes the model benchmark (Figure 2) and the SPA threshold-sensitivity heatmap, then displays the stored result tables and figures for the downstream analyses (feature selection, Table 1, SHAP interactions, partial dependence, extreme-score table). |
| [`data_preparation.ipynb`](data_preparation.ipynb) | The pipeline that produced `data/prepared/`: cleaning, imputation, the patient-level split, Tabula synthesis and SPA filtering. |
| [`data/prepared/`](data/prepared/) | De-identified prepared datasets: five splits × two feature panels, plus `manifest.json` and `README.md`. |
| [`results/notebook/`](results/notebook/) | Result tables read by the main notebook: the benchmark table, the SPA grid, and the summary tables behind Figures 3–5 and Tables 1–2. |
| [`figures/notebook/`](figures/notebook/) | Figure panels displayed by the main notebook. |
| [`tools/retained_data.py`](tools/retained_data.py) | Reference implementation of the cleaning, unit-repair and imputation stage. |
| [`tools/tabula_synthesis.py`](tools/tabula_synthesis.py) | Reference implementation of the Tabula synthesis stage. |

The benchmark and the SPA sensitivity grid are recomputed from `data/prepared/` when the notebook
runs. The downstream panels are displayed from the stored tables in `results/notebook/`; those
tables are the outputs of the analysis reported in the manuscript.

The scripts in `tools/` implement the first two preparation stages. They read the raw 2025
snapshots and the trained Tabula checkpoints, which are not part of this repository.

---

## Data

All files under `data/prepared/` are de-identified. There are no names, admission numbers, dates
of birth or free-text fields; `row_id` is a sequential row number assigned at export.

The cohort comprises 1,943 children with complete Kawasaki disease treated at the West China
Second University Hospital, Sichuan University (2013–2023), of whom 250 had initial IVIG
resistance. The split is patient-level and fixed: 1,356 training records and 587 validation
records. The validation partition is never resampled and is not used for model selection.

| Split | File stem | n | IVIG-resistant |
| :--- | :--- | ---: | ---: |
| Original training set | `original_train_panel_{a,b}.csv` | 1,356 | 172 |
| SMOTE-balanced training set | `smote_train_panel_{a,b}.csv` | 2,368 | 1,184 |
| Tabula synthetic training set | `synthetic_train_panel_{a,b}.csv` | 9,568 | 4,322 |
| SPA-filtered synthetic training set | `spa_train_panel_{a,b}.csv` | 5,993 | 3,307 |
| Validation set | `validation_panel_{a,b}.csv` | 587 | 78 |

**Panels.** Panel A holds the 47 base clinical and laboratory features; Panel B adds ten
documented composites derived from them (57 features in total). Both panels are supplied for every
split so the benchmark can be reproduced with and without the composites. The formulas for the ten
composites are listed in [`data/prepared/README.md`](data/prepared/README.md) and reproduce exactly
from the released Panel A columns.

**Labels.** `Label = 1` is IVIG-resistant; `Label = 0` is IVIG-responsive.

**Provenance.** Original, validation and SMOTE files carry integer day variables and
training-median imputation. The synthetic and SPA files keep fractional day variables and were
filtered only for physiologically impossible values, so plausible values outside the training
range are retained. `data/prepared/manifest.json` records the feature lists, counts and the exact
composition of each split.

The source workbook and the trained Tabula checkpoints are not included in this repository. Re-use
of the data beyond reproduction of this study requires approval from the Ethics Committee of
Sichuan University (No. 201312160121).

---

## Environment

Python 3.12. The reported run used:

```bash
conda create -y -n kd-ivig python=3.12 pip
conda activate kd-ivig
python -m pip install \
  "numpy==2.2.5" "pandas==2.2.3" "scipy==1.15.3" "scikit-learn==1.6.1" \
  "matplotlib==3.10.3" "seaborn==0.13.2" "openpyxl==3.1.5" "tqdm==4.67.1" \
  "xgboost==2.1.0" "imbalanced-learn==0.13.0" "shap==0.47.2" "tabpfn==2.1.3" \
  "torch==2.7.0" jupyterlab ipykernel
```

TabPFN downloads its classifier checkpoint on first use. A GPU is optional: the notebook selects
CUDA when `torch.cuda.is_available()` and falls back to CPU otherwise, where TabPFN inference is
substantially slower. Behind a restricted network, set `HF_ENDPOINT=https://hf-mirror.com` before
starting Jupyter.

---

## Reproducing the analysis

```bash
jupyter lab kd_ivig_analysis.ipynb
```

Run all cells top to bottom on a fresh kernel. Every stochastic step is seeded (`SEED = 42`).

The expensive cell is the benchmark: seven models (logistic regression, elastic net, SVM, random
forest, XGBoost, histogram gradient boosting, TabPFN) × four training sets × two panels. On a
single RTX A6000 the whole notebook takes about thirteen minutes — roughly eight for the benchmark
and five for the SPA sensitivity grid. `RUN_BENCHMARK` in the Part 1 cell controls the benchmark:
leave it `True` to refit, or set it to `False` to redraw Figure 2 from the committed
`results/notebook/figure2_benchmark.csv`.

`data_preparation.ipynb` covers the stages that produced `data/prepared/`. Its inputs — the raw
2025 snapshots and the Tabula checkpoints — are not part of this repository, so the prepared
datasets are shipped ready to use.

---

## Method summary

1. **Cohort and split.** Complete-KD cohort; missingness screen at 30%; stratified 70:30
   patient-level split; KNN imputation fitted on the training partition only.
2. **Augmentation.** SMOTENC on the training partition; Tabula (DistilGPT-2 derived,
   middle-padding) trained on the SMOTE-balanced set to generate 10,000 candidate records.
3. **SPA filtering.** Generated records are perturbed feature-wise and retained when the label is
   consistent with the baseline model and the prediction is perturbation-sensitive, using the
   confidence and perturbation-change-sensitivity (PCS) criteria documented in
   `data_preparation.ipynb`.
4. **Benchmarking.** The seven models above are trained on the original, SMOTE, synthetic and SPA
   training sets and evaluated on the held-out validation partition. ROC AUC is threshold-free;
   specificity and sensitivity use the Youden threshold on the validation partition.
5. **Interpretation.** SHAP interaction values for the histogram gradient boosting model,
   compared between the original and SPA training sets, with two-dimensional partial dependence
   for the strongest pairs.

The synthesis step builds on:

> Zhao Z, Birke R, Chen L. *Tabula: Harnessing Language Models for Tabular Data Synthesis.*
> arXiv:2310.12746, 2023.

---

## Citation

If you use this code or data, please cite the manuscript above and the Tabula paper. Full citation
details will be updated on acceptance.

## License

Code is released under the MIT License. The de-identified data are provided for reproduction of
this study only; see the ethics note above.
