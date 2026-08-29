# Literature Validation: EGNN Ensemble for Tp Prediction in PU–Cloisite 30B Nanocomposites

Actual-vs-predicted validation of an SE(3)-equivariant graph neural
network (EGNN) ensemble against literature-reported peak thermal
decomposition temperatures (**Tp**) for polyurethane–Cloisite 30B (PU/C30B)
nanocomposite samples. Part of the pipeline for:

> *Physics-Informed Synthetic Pretraining and Few-Shot Literature Calibration
> of an SE(3)-Equivariant Graph Neural Network Ensemble for Thermal
> Decomposition Prediction in Polyurethane–Cloisite 30B Nanocomposites*

## What this does

`validate_literature.py` takes the raw EGNN and Ridge-regression predictions
produced by the full pipeline's `validate_literature()` run and:

1. Builds a clean actual-vs-predicted table (raw predictions + leave-one-out
   cross-validated (LOOCV) affine-calibrated predictions).
2. Computes error metrics (MAE, RMSE, MAPE, hit-rate within ±10 °C).
3. Generates two plots: a predicted-vs-actual scatter (with y = x reference
   line and raw-prediction error bars) and a per-sample calibrated-error bar
   chart.
4. Exports everything to `results.csv`, `results.md`, and
   `tp_actual_vs_predicted.png` for easy viewing on GitHub.

### Why LOOCV calibration?

The raw EGNN output is a **zero-shot** prediction and carries a systematic
offset (it over-predicts Tp relative to literature values on this sample
set). Reporting raw error alone would overstate the model's accuracy problem.
Leave-one-out affine calibration fits a simple linear correction
(`y = a·x + b`) on all-but-one samples and evaluates on the held-out sample,
repeated for every sample — so the calibration itself is never trained on
the point it's judged against. This gives an honest, non-leaky estimate of
how well the model generalizes to literature data after removing the
systematic offset.

## Results

| Metric | Value |
|---|---|
| LOOCV MAE | 5.92 °C |
| LOOCV RMSE | 7.92 °C |
| Relative-error accuracy (100 − MAPE) | 98.5% |
| Within ±10 °C | 81.8% (9/11 samples) |
| Largest miss | PU5CD (actual 401.0 °C, predicted 383.7 °C, error −17.3 °C) |

Full per-sample table: [`results.md`](results.md) / [`results.csv`](results.csv)

![Actual vs Predicted Tp](tp_actual_vs_predicted.png)

*Left: predicted vs. actual Tp, showing the raw zero-shot offset (red,
uncalibrated) alongside honest LOOCV-calibrated predictions (blue), against
the y = x line of perfect prediction. Right: per-sample calibrated error,
with samples exceeding ±10 °C tolerance flagged in red.*

## Usage

### Google Colab
Paste the full contents of `validate_literature.py` into a single cell and
run it. Only `numpy`, `pandas`, `scikit-learn`, and `matplotlib` are
required, all of which are preinstalled on Colab.

### Local
```bash
pip install numpy pandas scikit-learn matplotlib tabulate
python validate_literature.py
```

### Using your own data
Replace the `lit_tp`, `egnn_raw`, `egnn_std`, and `ridge_raw` arrays near the
top of the script with your own values (keeping the same sample order as
`names`) to plug in results from a live pipeline run instead of the
hardcoded literature-validation numbers.

## Repository contents

| File | Description |
|---|---|
| `validate_literature.py` | Main script: builds table, computes metrics, generates plots and exports |
| `results.csv` | Per-sample actual/predicted values and errors (machine-readable) |
| `results.md` | Same table, Markdown-formatted, plus summary statistics |
| `tp_actual_vs_predicted.png` | Scatter + error-bar chart figure |

## Sample naming

Sample codes follow `PU<wt%><clay/dispersion code>`, e.g. `PU3CD` = PU with
3 wt% clay via CD processing, `PU5BUH` = PU with 5 wt% clay via BUH
processing. Adjust this key to match your actual naming convention if it
differs.

## Citation

If you use this validation approach, please cite the associated paper (see
above) and note the specific software versions used, e.g.:

- RDKit: Open-Source Cheminformatics Software, version X.X.X. https://www.rdkit.org
- Fey, M.; Lenssen, J. E. Fast Graph Representation Learning with PyTorch
  Geometric. *arXiv* **2019**, arXiv:1903.02428 (PyTorch Geometric vX.X.X used).
