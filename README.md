# ACCFormer: Predicting Analog Circuit Performance Metrics via Topology-Aware Transformers - Enhanced Framework

This is the official, refactored, and actively maintained repository for our paper: *ACCFormer: Predicting Analog Circuit Performance Metrics via Topology-Aware Transformers*.

## ⚠️ Code Structure & Reproducibility Notice
To provide a more rigorous foundation for future research and fairer feature alignment, **we have upgraded the baseline models in this repository by equipping them with the same decoder architecture used in our proposed model.** 

**Due to foundry-related and industrial confidentiality restrictions, the raw circuit simulation datasets used in the paper cannot be publicly released.** 

If you are looking for the exact, unrefactored code used to generate the tables in the original manuscript, please refer to our Paper Archive Repository:
👉 **https://github.com/SetsunaFSeiei21/Origin_ACCFormer**

## Usage

### Data Structure

Place circuit data under `data/<circuit_name>/`:

```text
data/
└── <circuit_name>/
    ├── device_messages.json
    ├── source/
    │   ├── pretrain_design_features.csv
    │   └── pretrain_targets.csv
    └── target/
        ├── target_design_features.csv
        └── target_targets.csv
```

### Environment

```bash
cd ACCFormer
conda activate newbase
export PYTHONPATH=$(pwd):${PYTHONPATH}
```

### Source Training

Smoke test:

```bash
python -m src.experiment.source_experiment \
  model=mlp \
  dataset.circuit_name=two_stage_folded_opamp \
  dataset.mission_type=source \
  dataset.num_workers=0 \
  exp.epochs=2 \
  exp.log_interval=1 \
  scheduler.warmup_epochs=1
```

Full source training:

```bash
python -m src.experiment.source_experiment \
  model=mlp \
  dataset.circuit_name=two_stage_folded_opamp \
  dataset.mission_type=source
```

To use another model:

```bash
python -m src.experiment.source_experiment \
  model=res_mlp \
  dataset.circuit_name=two_stage_folded_opamp \
  dataset.mission_type=source
```

### Target Fine-tuning

Smoke test:

```bash
python -m src.experiment.target_experiment \
  model=mlp \
  target_init=mlp \
  dataset.circuit_name=two_stage_folded_opamp \
  dataset.mission_type=target \
  dataset.num_workers=0 \
  source_checkpoint_path=model_weight/mlp/two_stage_folded_opamp/source/mlp_two_stage_folded_opamp_source.pt \
  exp.epochs=2 \
  exp.log_interval=1 \
  scheduler.warmup_epochs=1
```

Full target fine-tuning:

```bash
python -m src.experiment.target_experiment \
  model=mlp \
  target_init=mlp \
  dataset.circuit_name=two_stage_folded_opamp \
  dataset.mission_type=target \
  source_checkpoint_path=model_weight/mlp/two_stage_folded_opamp/source/mlp_two_stage_folded_opamp_source.pt
```

For date-tagged source checkpoints, specify the full checkpoint path:

```bash
python -m src.experiment.target_experiment \
  model=mlp \
  target_init=mlp \
  dataset.circuit_name=two_stage_folded_opamp \
  dataset.mission_type=target \
  source_checkpoint_path=model_weight/2026-05_09/mlp/two_stage_folded_opamp/source/mlp_two_stage_folded_opamp_source.pt \
  result_root=experiment_result/2026-05_09 \
  weight_root=model_weight/2026-05_09 \
  log_root=logs/2026-05_09
```

### Slurm Submission

Make scripts executable:

```bash
chmod +x scripts/run/scut_slurm_run.sh
chmod +x scripts/run/submit_source_grid.sh
chmod +x scripts/run/submit_target_grid.sh
```

Submit source experiments:

```bash
bash scripts/run/submit_source_grid.sh
```

Submit target experiments:

```bash
bash scripts/run/submit_target_grid.sh
```

If the source checkpoint was generated on a previous date:

```bash
SOURCE_DATE_TAG=2026-05_09 bash scripts/run/submit_target_grid.sh
```

### Outputs

Default outputs are saved to:

```text
experiment_result/<model_name>/<circuit_name>/<source_or_target>/
model_weight/<model_name>/<circuit_name>/<source_or_target>/
logs/<model_name>/<circuit_name>/<source_or_target>/
```

Slurm scripts save outputs with date tags:

```text
experiment_result/<DATE_TAG>/<model_name>/<circuit_name>/<source_or_target>/
model_weight/<DATE_TAG>/<model_name>/<circuit_name>/<source_or_target>/
logs/<DATE_TAG>/<model_name>/<circuit_name>/<source_or_target>/
hpc_logs/<DATE_TAG>/
```