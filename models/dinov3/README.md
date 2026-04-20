# DINOv3 VisDrone Compare Toolkit

This folder contains a self-contained minimal experiment toolkit for training and evaluating a DINOv3-based detector on VisDrone and exporting the comparison metrics:

- P/%
- R/%
- mAP50/%
- GFLOPS
- Params/M

## Files

- `config.yaml`: main experiment config
- `common.py`: shared utilities and model builder
- `dataset.py`: VisDrone dataset loader (YOLO labels)
- `losses.py`: Hungarian matching + detection losses
- `train.py`: training entrypoint
- `evaluate.py`: evaluation entrypoint
- `profile_model.py`: GFLOPS + Params export
- `export_report.py`: merge metrics/profile into one report
- `check_dataset.py`: dataset integrity checker
- `run_pipeline.py`: one-command pipeline runner

## Quick Start

1. Edit `config.yaml` paths if needed.
2. Optional data check:

   `python experiments/visdrone_dinov3_compare/check_dataset.py --config experiments/visdrone_dinov3_compare/config.yaml`

3. Train:

   `python experiments/visdrone_dinov3_compare/train.py --config experiments/visdrone_dinov3_compare/config.yaml`

4. Evaluate:

   `python experiments/visdrone_dinov3_compare/evaluate.py --config experiments/visdrone_dinov3_compare/config.yaml`

5. Profile:

   `python experiments/visdrone_dinov3_compare/profile_model.py --config experiments/visdrone_dinov3_compare/config.yaml`

6. Export final report:

   `python experiments/visdrone_dinov3_compare/export_report.py --config experiments/visdrone_dinov3_compare/config.yaml`

## Notes

- This is a minimal baseline implementation intended for controlled comparison experiments.
- For multi-GPU, start with single GPU baseline first, then extend with DDP.
- GFLOPS requires optional package `thop`. If unavailable, the script still exports Params/M.
