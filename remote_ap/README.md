# 远程 AP VLA：训练与正在使用的推理服务

从 huoshan `/cs/kyy/behavior_ap_experiment` 导出。当前推理权重为 `runs/pi05_goal_horizon_v2_b8/checkpoint_024000`，不是最新训练步数自动替换。`active_checkpoint_metadata` 是该权重的原始配置、归一化和数据契约。

## 与原仓库已有代码的关系

检查发现训练代码尚未交付；当前推理服务源码、tokenizer和LeRobot实现已经逐字节存在于`../pi05_harness_20261017`。因此`vendor`、`serving_runtime`和`tokenizer`采用相对符号链接复用，不重复上传。`scripts/pi05_executor.py`保留训练工作目录里的原始版本；实际服务使用的是`serving_runtime/pi05_executor.py`（tokenizer相对路径兼容版本）。

## 内容

- `scripts/train_pi05.py`：原始pi0.5微调入口、带时间mask的loss、优化器/RNG恢复。
- `scripts/training_data.py`：按原轨迹读取状态、动作和3路RGB，25维状态选择、分位归一化、32帧有效动作mask。
- `scripts/training_version.py`：数据契约及续训校验。
- `scripts/prepare_training.py`、`build_semantic_index.py`、`labels.py`、`build_horizon_filtered.py`、`audit_training_horizon.py`：语义索引、切分、归一化及时间对齐过滤。
- `training_metadata/`：horizon_filtered_v2的数据规模/采样/验证及真实b8运行参数快照，不包含parquet或原始视频。
- `requirements-observed.txt`：实际远程环境包版本快照。`lerobot`由仓库提供；标有原机器路径的依赖需按本机实际位置配置，不要直接照搬editable路径。
- `provenance.json`：来源与原文件SHA256。为了部署，train_pi05.py/training_data.py增加了环境变量路径覆盖，其余计算和训练行为保持原样；原hash代表导出时源文件。

## 推理部署

ModelScope公开推理包：`YouyiKou/behavior-ap-pi05-checkpoint-024000`。上传是否已完成以该库文件及服务端校验结果为准，不以此README存在为准。下载到自己的目录后：

```bash
export AP_PYTHON=/path/to/venv/bin/python
export AP_CHECKPOINT=/path/to/checkpoint_024000
bash remote_ap/serve.sh
```

接口绑定127.0.0.1，使用SSH本地转发。`GET /health`确认实际checkpoint；`POST /predict`仅接收61D原始proprio、按头/左腕/右腕顺序的3张Base64 RGB及`instruction`。返回有限的32×23原生R1Pro控制量，harness默认只执行8步前缀。归一化、状态维度和语言格式不得自行替换。服务只推理，远程不安装Isaac。

## 训练部署

设置`AP_PYTHON`、`AP_PROJECT_ROOT`、`AP_RAW_DATA_ROOT`、`AP_TRAINING_INDEX`、`AP_BASE_MODEL_PATH`，可选`AP_TOKENIZER_PATH`（默认复用仓库tokenizer）。示例：

```bash
bash remote_ap/train.sh --steps 30000 --batch-size 8 --accumulation 4 \
  --workers 4 --dtype float32 --memory-fraction 0.82 --run-name NEW_UNIQUE_NAME
```

精确历史参数见`training_metadata/run-config.json`；上面仅为启动示例，不能替代实际配置。训练不会自动下载原始数据和基座。使用新的run-name，续训需显式`--resume`并匹配数据契约。推理包不含约25GB的`training_state.pt`，所以它不支持精确恢复原Adam/RNG状态；如需精确续训应另行获取该文件。

原始预处理脚本保留了远端绝对路径，重建训练集前应检查顶部ROOT/OUT等路径；脚本拒绝覆盖既有训练版本。数据集分片和源视频不在Git/ModelScope推理包中。

## 模型与评测边界

31类组合标签，来自99任务，421211训练窗口；radio是留出任务。开门/抽屉/盖子标签不等于已验证掌握该容器操作，也不等于完整原任务成功。完整任务后续交给冠军模型，这里未上传冠军权重。

基座和第三方依赖沿用上游许可。该导出不重新授予上游数据、资产或模型的使用许可。令牌、密钥、官方仿真资产和原始示范均未包含。
