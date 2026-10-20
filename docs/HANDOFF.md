# 部署与接手步骤

## 1. 仿真环境（本地）

当前实测基础：OmniGibson 3.9.3、Isaac Sim 5.1.0，官方源码commit见`environment/simulator-source.json`。依赖导出见`environment/local-conda.yml`和`local-pip-freeze.txt`。这些是环境快照，含原机器editable安装路径，**不能原样作为跨机器一键安装清单**。

获取官方源码并固定版本：

```bash
git clone git@github.com:StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K
git checkout a8247a8cc1633fe1ca0cc66aa07243d46c64f155
```

按该提交官方安装说明安装OmniGibson/Isaac及BDDL，参考导出依赖对齐版本。当前官方仓库无本地修改，`local-simulator.patch`为空。不要在远程推理主机安装Isaac。

通过官方授权流程获取BEHAVIOR资产、机器人资产、2026 challenge任务实例。资产目录要求见simulator-source.json；解密密钥由接收方自己获取，不在仓库中。实例、目标和初始化变更均来自本仓库JSON及`ap_harness/ood99.py`、`fixtures.py`，不依赖运行输出目录。

`deps/`是原机补充依赖而非源码，本仓库不复制其平台二进制。可在目标Python下执行 `python -m pip install --target deps -r environment/harness-extra-requirements.txt` 安装对应平台版本。需要jsonschema、tokenizers等依赖时，参照版本导出安装到合适环境/专用目录，不要随意覆盖Isaac锁定的核心依赖。

## 2. 机器路径配置

```bash
python scripts/configure_handoff_paths.py \
  --python /YOUR/ENV/bin/python --assets /YOUR/BEHAVIOR-ASSETS \
  --remote-root /YOUR/REMOTE/behavior_ap_experiment --local-user YOUR_USER --ssh-host YOUR_HOST
# 先看修改清单，确认后同命令增加 --write，再 git diff 检查
```

还需手动审核：`runtime/behavior_harness/planner.py`及`ap_harness/run.py`中的Codex CLI路径、用户PATH；`scripts/run_sim.sh`中的DISPLAY/XAUTHORITY；`remote/scripts/start*.sh`中的远端环境/模型/cache路径与GPU编号。原脚本里引用的`sudo_run.sh`属于原机器个人工具，不随仓库分发；接收方应让资产在其用户下可读、直接运行，或使用自己的受控权限流程。不要复制原用户的登录文件。

Planner需要接收方自己的Codex CLI及登录。`harness_handoff.md`和bundle内`HANDOFF.md`说明AP观察/动作契约。

## 3. VLA服务（远端）

- AP：checkpoint_024000，自训练权重**未公开且未包含**，需要另外从提供方取得；原位置`runs/pi05_goal_horizon_v2_b8/checkpoint_024000`。其推理代码和tokenizer在`pi05_harness_20261017/`。不要把该目录的010000历史元数据冒充024000权重。
- 冠军：`https://github.com/IliaLarchenko/behavior-1k-solution`，固定源码commit见champion-source-status.txt；执行`git submodule update --init --recursive`，核对openpi提交。主仓库补丁`champion-local-changes.patch`，openpi子目录补丁`champion-openpi.patch`（为空则不应用），将`remote/eval_utils_2025.py`放入src/b1k/shared/。这些是本次运行所需兼容改动，保留发布方wrapper动作修正。
- 权重：HF `IliaLarchenko/behavior_submission`，固定revision `89545bc1b7aa7f2e687bc0032d091f132d715d4e`，任务到4个checkpoint映射在`configs/champion_search_suite/task_checkpoint_mapping.json`。
- 远端目录中部署`remote/scripts/`，根据实际路径调整启动脚本。18770为AP，18774为多任务冠军；SSH隧道绑定localhost。
- 下载脚本校验大小及LFS SHA256。验证标记存在且真实推理成功后才能进入闭环。
- 冠军服务每组重启，task_id严格对应任务，不复用前一组历史。50固定任务嵌入不支持99任务全集。

## 4. 验证顺序

```bash
PYTHONPATH=deps:. /YOUR/ENV/bin/python -m unittest discover -s tests -q
bash scripts/run_sim.sh --output outputs/new_preflight \
  --scenario-spec configs/champion_search_suite/assembling_gift_baskets.json \
  --initialize-only --initial-visibility-audit --max-steps 10000
```

检查preflight中的几何关系、初始不可见及fingerprint，再运行`run_champion_search_suite.py`；该批脚本会创建输出目录并持有锁。**不要在原机器已有批测运行时从此快照启动第二份。**原始活动目录与本快照完全独立。

## 5. 接收方必须知道

训练原数据和提取数据不包含在Git；AP训练集为horizon_filtered_v2。此仓库面向推理与仿真评测，不包含完整训练流水线。训练数据预览网页和所有历史视频留在原机，不在本源码包中。

每次初始化会重新生成场景状态，不提供跨版本可通用的烘焙USD世界。需要相同官方资产/实例版本才能复现。当前照片/结果摘要仅说明已发生过的实验，不替代接收方环境的验证。
