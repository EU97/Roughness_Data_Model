# Project Report: Roughness_Data_Model

This report summarizes how to run the tooling, quality gates status, and requirements coverage, mirroring the current repo state.

## How to run

All commands assume Windows PowerShell and the repository venv.

```powershell
# 1) Activate venv (once per session)
& C:/Users/edgar/Documents/GitHub/Roughness_Data_Model/.venv/Scripts/Activate.ps1

# 2) Single specimen (with optional ISO 16610 filtering)
python ./src/Single.py ./data/GrupoI/EspeI --apply-filter --cutoff-mm 0.8 --filter-source primary

# 3) Batch across all specimens under ./data
python ./src/Batch.py ./data --apply-filter --cutoff-mm 0.8 --filter-source primary
# Outputs: ./data/batch_summary.json and ./data/batch_summary.csv

# 4) Comparative analysis (reads batch summary)
python ./src/Compare.py ./data --output-dir reports --metrics Ra Rq Rz_ISO RSm Rpk Rk Rvk Mr1 Mr2 --rank-metric Ra
# Outputs in: ./data/reports (CSVs, PNGs, report.md)
```

Notes:
- Inputs come from Surfcom-like .tx1/.tx2/.tx3 files in each specimen folder.
- CSVs are written with UTF-8 BOM for Excel compatibility.

## Quality gates

- Build/syntax: PASS
  - Verified with `py_compile` on `src/Single.py`, `src/Batch.py`, `src/Compare.py`.
- Runtime smoke tests: PASS
  - Single, Batch, and Compare executed successfully on the provided dataset. Outputs were generated as expected.
- Lint/style: Minor non-blocking warnings
  - Long lines, naming conventions for module names, and chained plotting statements. Functional, not breaking.
- Performance/robustness:
  - Matplotlib figures are closed after saving to avoid excessive memory use in batch runs.
  - File encodings handled (latin-1 inputs, UTF-8 BOM outputs).

## Requirements coverage

- ISO 4287:1997 parameters (Ra, Rq, Rp, Rv, Rz/Rt, Rz ISO by 5 segments, Rsk, Rku): Done.
- Axis and encoding fixes (Spanish chars, correct X from headers): Done.
- Slope correction profiles exported without altering prior outputs: Done.
- Single canonical processor (`Single.py`) with reusable `procesar_carpeta()`: Done.
- ISO 13565-2 (Rk, Rpk, Rvk, Mr1, Mr2) via MRC core line, with plots: Done.
- Optional ISO 16610 Gaussian filtering, metrics and plots appended: Done.
- Batch processing (`Batch.py`) with JSON and CSV summaries: Done.
- Comparative analysis (`Compare.py`) per-group stats, rankings, and cross-group visuals and report: Done.
- Documentation: README includes methodology and usage; this REPORT complements with how-to-run and gate coverage.

## Extras and next steps

- You can choose the ranking metric via `--rank-metric` in `Compare.py`.
- Optional future work: combined PDF report, stricter linting, and CLI presets for common cutoffs.
