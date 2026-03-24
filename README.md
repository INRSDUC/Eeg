# EEG Sparse Score-Based Attack (Research Scaffold)

This repository provides a staged implementation scaffold for developing and evaluating a sparse score-based black-box perturbation pipeline on EEG classification models built with Braindecode.

## Scope and ethics

Use this code only for authorized security research in controlled environments.
Do not use it against real systems or data without explicit permission.

## Project structure

- `src/run_baseline.py`: Stage 1 baseline EEGNet training.
- `src/attack/support.py`: Stage 2 channel-window support representation.
- `src/attack/basis.py`: Stage 3 smooth raised-cosine basis.
- `src/attack/greedy_attack.py`: Stage 4 greedy score-based support search + Stage 5 SPSA refinement.
- `src/evaluate_attack.py`: Stage 6 attack evaluation over budget settings.
- `src/defense/lightweight.py`: Lightweight defense primitives.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Stage 1 baseline

```bash
python -m src.run_baseline
```

Outputs:
- `outputs/eegconformer_baseline.pt`
- `outputs/baseline_metrics.json`
- `outputs/baseline_scores.npz`
- `outputs/train_val_loss_curve.png`
- `outputs/train_val_accuracy_curve.png`
- `outputs/learning_rate_curve.png`

## Run BNCI2014_001 human-recognition baseline

```bash
python -m src.run_human_recognition_baseline
```

Outputs:
- `outputs/bnci2014_001_human_recognition/subject_recognition_baseline.pt`
- `outputs/bnci2014_001_human_recognition/subject_recognition_metrics.json`
- `outputs/bnci2014_001_human_recognition/subject_recognition_scores.npz`
Notes:
- Uses subject-ID recognition labels.
- Uses a full cross-session split: trains on `0train` and validates on `1test`.

## Run multi-seed baseline stability sweep

```bash
python -m src.run_multiseed_baseline
```

Outputs:
- `outputs/eegconformer_baseline_seed*.pt`
- `outputs/baseline_metrics_seed*.json`
- `outputs/baseline_scores_seed*.npz`
- `outputs/baseline_multiseed_summary.json`

## Run a single attack demo

```bash
python -m src.run_attack_demo
```

Output:
- `outputs/attack_demo_result.json`

## Run evaluation grid

```bash
python -m src.evaluate_attack
```

Output:
- `outputs/attack_eval_report.json`

## Notes

- Default data source is `BNCI2014_001` via MOABB/Braindecode.
- Default baseline preprocessing now combines a 4-38 Hz bandpass, exponential moving standardization, optional Euclidean Alignment, and training-only augmentation.
- Hyperparameters are defined in `src/config.py`.
- Attack evaluation now reports pre-defense ASR, post-denoising ASR, post suspicious-window filtering ASR, and query-budget exhaustion rate.
- This is a development scaffold with clean module boundaries so you can iterate each stage independently.
