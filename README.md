# IVIG resistance in Kawasaki disease — machine-learning analysis

Code for the manuscript:

> **Predictive factors and interactions in intravenous immunoglobulin resistance in Kawasaki
> disease: a machine-learning study**
>
> Qiming Tang, Shuran Shao, Haoyang Mi, Yimin Hua, Kaiyu Zhou, Xiaoliang Liu, Chuan Wang,
> Xiaowei Mao
>
> *BMC Medical Informatics and Decision Making* — submitted, revision in progress.

This repository contains the two analysis notebooks and the two scripts behind the data-preparation
pipeline.

---

## Contents

| Path | What it is |
| :--- | :--- |
| [`kd_ivig_analysis.ipynb`](kd_ivig_analysis.ipynb) | Main notebook, with its executed outputs: model benchmark (Figure 2), SPA threshold sensitivity, and the downstream feature-selection and interpretation analyses. |
| [`data_preparation.ipynb`](data_preparation.ipynb) | The pipeline that produced the prepared datasets: cleaning, imputation, the patient-level split, Tabula synthesis and SPA filtering. |
| [`tools/retained_data.py`](tools/retained_data.py) | Cleaning, unit repair and imputation stage. |
| [`tools/tabula_synthesis.py`](tools/tabula_synthesis.py) | Tabula synthesis stage. |
| [`LICENSE`](LICENSE) | MIT licence for the code. |

The notebooks read from two directories that are not part of this repository:

* `data/prepared/` — the de-identified prepared datasets (see [Data](#data));
* `results/notebook/` and `figures/notebook/` — the result tables and figure panels that the
  downstream sections display.

Both notebooks include their stored outputs, so the analyses can be read without rerunning them.

---

## Data

The cohort comprises 1,943 children with complete Kawasaki disease treated at the West China
Second University Hospital, Sichuan University (2013–2023), of whom 250 had initial IVIG
resistance. The split is patient-level and fixed: 1,356 training records and 587 validation
records. The validation partition is never resampled and is not used for model selection.

| Split | n | IVIG-resistant |
| :--- | ---: | ---: |
| Original training set | 1,356 | 172 |
| SMOTE-balanced training set | 2,368 | 1,184 |
| Tabula synthetic training set | 9,568 | 4,322 |
| SPA-filtered synthetic training set | 5,993 | 3,307 |
| Validation set | 587 | 78 |

**Panels.** Panel A holds the 47 base clinical and laboratory features; Panel B adds ten
documented composites derived from them (57 features in total). Both panels exist for every split,
so the benchmark can be reproduced with and without the composites.

**Labels.** `Label = 1` is IVIG-resistant; `Label = 0` is IVIG-responsive.

**Provenance.** Original, validation and SMOTE tables carry integer day variables and
training-median imputation. The synthetic and SPA tables keep fractional day variables and were
filtered only for physiologically impossible values, so plausible values outside the training
range are retained.

The tables are de-identified: no names, admission numbers, dates of birth or free-text fields;
`row_id` is a sequential row number. They are distributed separately from this repository under
the terms of the manuscript's data availability statement. Re-use beyond reproduction of this study
requires approval from the Ethics Committee of Sichuan University (No. 201312160121).

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

## Running the notebooks

Both notebooks are provided with their executed outputs. To rerun them, place the prepared datasets
in `data/prepared/` and the result tables in `results/notebook/`.

`kd_ivig_analysis.ipynb` runs from a fresh kernel and every stochastic step is seeded
(`SEED = 42`). The expensive cell is the benchmark: seven models (logistic regression, elastic net,
SVM, random forest, XGBoost, histogram gradient boosting, TabPFN) × four training sets × two
panels, about eight minutes on one RTX A6000, plus five minutes for the SPA sensitivity grid.
`RUN_BENCHMARK` in the Part 1 cell controls the benchmark; leave it `True` to refit.

`data_preparation.ipynb` covers the stages that produced the prepared datasets. Its inputs — the
raw 2025 snapshots and the Tabula checkpoints — are not part of this repository.

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

If you use this code, please cite the manuscript above and the Tabula paper. Full citation details
will be updated on acceptance.

## License

Code is released under the MIT License; see [LICENSE](LICENSE). The de-identified data described
above are distributed separately and are provided for reproduction of this study only.
