# Prepared datasets (de-identified)

The main notebook reads only these files. `row_id` is a sequential row number, not a patient
identifier; the files contain no names, admission numbers, dates of birth or free-text fields.

## Files

- `original_train_panel_a.csv` / `original_train_panel_b.csv`: n=1356, resistant=172, responsive=1184
- `smote_train_panel_a.csv` / `smote_train_panel_b.csv`: n=2368, resistant=1184, responsive=1184
- `synthetic_train_panel_a.csv` / `synthetic_train_panel_b.csv`: n=9568, resistant=4322, responsive=5246
- `spa_train_panel_a.csv` / `spa_train_panel_b.csv`: n=5993, resistant=3307, responsive=2686
- `validation_panel_a.csv` / `validation_panel_b.csv`: n=587, resistant=78, responsive=509

## Labels
- `Label = 1`: IVIG-resistant (KD).
- `Label = 0`: IVIG-responsive.

## Panels
- Panel A: 47 base features.
- Panel B: 57 features (47 base + 10 documented composites).

## Panel B composites
Every composite is derived from the Panel A columns of the same row; all ten reproduce exactly
from the released files.

| Composite | Formula | Notes |
| :--- | :--- | :--- |
| `ALB-dNLR` | `expit(ALB * (WBC - ANC) / ANC / 10)` | logistic-transformed |
| `ALBI` | `expit(0.66 * log10(TB) - 0.085 * ALB)` | logistic-transformed |
| `CALLY` | `ALB * Lym` | |
| `RPR` | `PLT / RBC` | |
| `PVR` | `PLT / ALB` | |
| `BA` | `GLB / ALB` | |
| `CAR` | `CRP / ALB` | |
| `CLI` | `ANC / Lym` | |
| `L1` | `Lym / WBC` | |
| `N1` | `ANC / WBC` | |

`expit(x) = 1 / (1 + exp(-x))`; `TB` is in µmol/L and `ALB` in g/L.

## Notes
- Validation (587 rows / 78 resistant) is fixed; do not use it for model selection.
- The SMOTE training set is the prespecified 1,184:1,184 design. One hundred generated
  IVIG-resistant records were discarded (seed 20261104) to restore that balance; the empirical
  records were left untouched. The synthetic and SPA sets were generated before this correction
  and were not regenerated.
- Syn/SPA keep fractional day variables; Ori/SMOTE/Val use integer days.
- All columns are numeric; no free-text or identifiers.

## Feature lists
### Panel A
```
FeverDay, LabTestDay, IVIGDay, Sex, Age, Conj, Lips, Limbs, Rash, LymphSym, DigeSys, NervSys, RespSys, MuscSys, UrineR, WBC, Lym, ANC, RBC, HGB, PLT, HCT, RDW-CV, RDW-SD, PDW, MPV, P-LCR, PCT, CRP, ESR, ALT, AST, TB, ALB, UN, Cr, K, Na, DBIL, IDIL, GLB, r-GT, LDH, P, Cl, Ca, Mg
```
### Panel B
```
FeverDay, LabTestDay, IVIGDay, Sex, Age, Conj, Lips, Limbs, Rash, LymphSym, DigeSys, NervSys, RespSys, MuscSys, UrineR, WBC, Lym, ANC, RBC, HGB, PLT, HCT, RDW-CV, RDW-SD, PDW, MPV, P-LCR, PCT, CRP, ESR, ALT, AST, TB, ALB, UN, Cr, K, Na, DBIL, IDIL, GLB, r-GT, LDH, P, Cl, Ca, Mg, ALB-dNLR, ALBI, BA, CALLY, CAR, CLI, L1, N1, PVR, RPR
```
