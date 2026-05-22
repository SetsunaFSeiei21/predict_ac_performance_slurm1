## 使用方法

### 数据结构

请将电路数据放在 `data/<circuit_name>/` 下：

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

### 环境准备

```bash
cd ACCFormer
conda activate newbase
export PYTHONPATH=$(pwd):${PYTHONPATH}
```

### Source 训练

快速测试：

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

正式训练：

```bash
python -m src.experiment.source_experiment \
  model=mlp \
  dataset.circuit_name=two_stage_folded_opamp \
  dataset.mission_type=source
```

切换模型：

```bash
python -m src.experiment.source_experiment \
  model=res_mlp \
  dataset.circuit_name=two_stage_folded_opamp \
  dataset.mission_type=source
```

### Target 微调

快速测试：

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

正式微调：

```bash
python -m src.experiment.target_experiment \
  model=mlp \
  target_init=mlp \
  dataset.circuit_name=two_stage_folded_opamp \
  dataset.mission_type=target \
  source_checkpoint_path=model_weight/mlp/two_stage_folded_opamp/source/mlp_two_stage_folded_opamp_source.pt
```

如果使用带日期目录的 source checkpoint，请指定完整路径：

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

### Slurm 批量提交

运行前请确认脚本具有执行权限：

```bash
chmod +x scripts/run/scut_slurm_run.sh
chmod +x scripts/run/submit_source_grid.sh
chmod +x scripts/run/submit_target_grid.sh
```

提交 source 实验：

```bash
bash scripts/run/submit_source_grid.sh
```

提交 target 实验：

```bash
bash scripts/run/submit_target_grid.sh
```

如果 source checkpoint 是之前某一天生成的：

```bash
SOURCE_DATE_TAG=2026-05_09 bash scripts/run/submit_target_grid.sh
```

### 输出路径

默认输出路径：

```text
experiment_result/<model_name>/<circuit_name>/<source_or_target>/
model_weight/<model_name>/<circuit_name>/<source_or_target>/
logs/<model_name>/<circuit_name>/<source_or_target>/
```

Slurm 脚本会使用带日期的输出路径：

```text
experiment_result/<DATE_TAG>/<model_name>/<circuit_name>/<source_or_target>/
model_weight/<DATE_TAG>/<model_name>/<circuit_name>/<source_or_target>/
logs/<DATE_TAG>/<model_name>/<circuit_name>/<source_or_target>/
hpc_logs/<DATE_TAG>/
```