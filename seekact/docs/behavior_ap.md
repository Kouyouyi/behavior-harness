> SeekAct 整合版：部署与新命令以 [主README](../../README.md) 为准。下文保留历史开发协议与实验路径说明；它们不是新机器的默认目录。

# BEHAVIOR R1Pro 仿真后端

本后端把 SeekAct 的 Codex Agent 与本地 BEHAVIOR/OmniGibson R1Pro 仿真连接起来。Agent 观察三路 RGB-D、读取机器人反馈，再通过有界的底盘、手臂、夹爪和躯干工具执行任务。默认模式只使用原有机器人工具；启用下文的 `vla_tools` 后，可以调用主动感知 VLA 和冠军 VLA。此后端不控制真实 G1/RB-Y1 硬件。

本文说明接口、实验约束与复现入口，不代表三个任务已经成功。以每次运行保存的 `trial_record.json` 和官方任务判定为准；单元测试、接口校准、工具 ACK 或视频可播放均不能证明任务完成。

## 进程与部署

```text
scripts/run_behavior_experiment.py（普通用户 supervisor）
  ├─ harness.codex_cli.behavior → Codex Agent
  │    └─ services.controller.frontend（MCP、标准库）
  │         └─ services.behavior_ap.backend（私有 Unix socket 客户端）
  └─ services.behavior_ap.worker（唯一 Isaac 所有者、主线程执行）
       └─ behavior_ap_sim.environment.BehaviorEnvironment → OmniGibson / R1Pro
```

- Agent、MCP frontend 和 supervisor 使用 `SeekAct` Python 环境。BEHAVIOR 客户端/前端走标准库传输，不需要在前端安装 Isaac、机器人 SDK 或加载仿真资产。
- worker 复用已有 `behavior-2026` 环境，以及独立的 `behavior_ap` 仿真仓库。NumPy、OmniGibson、Isaac 及其依赖只在此侧使用。当前部署通过已有 `sudo_run.sh` 启动 worker，以读取本机 `/root` 下的仿真资产；Agent 仍以普通用户运行。
- 仿真固定使用 `render_profile=fast`、headless 和视频记录。`fast` 是渲染配置，不是任务成功条件，也不是速度提升的实测结论。
- Isaac 和 planner/Agent 在本地运行。远程 `huoshan` 可提供已下载的官方专家数据；不要在那里安装或运行 Isaac。
- GPU 租约与等待由仿真仓库负责。等待其他实验释放 GPU，不中断其他实验。`runner.lock` 防止同一个输出目录的重复调度。
- Agent 复用调用用户已有的 Codex 登录；不复制认证文件。需要可用的 `codex`，或设置 `CODEX_BIN` 指向可执行文件。

2026-10-24 按用户要求，后续新 BEHAVIOR Agent 会话默认使用 `gpt-6-astra`，显式传入 `model_reasoning_effort="low"`、`service_tier="fast"` 和 `features.fast_mode=true`。显式 `agent_model` 仍优先。模型 Fast 服务层与仿真 `render_profile=fast` 是两个独立设置；启动记录保存请求值，不把它当作服务端实际处理层已被验证。正在运行的长 `codex exec` 回合保持原参数，新设置从下一次新会话生效。旧记录未显式保存的推理强度、服务层保持未知，不追溯填成 low/Fast。

当前路径在 [运行脚本](../scripts/run_behavior_experiment.py) 中是机器相关常量，迁移到其他机器时必须先逐项修改，而不是照搬 `/root` 或用户名：

| 常量/依赖 | 当前机器上的位置 |
|---|---|
| 本仓库 | `/media/cody/软件/SeekAct` |
| `SIM_ROOT` | `/media/cody/软件/behavior_ap` |
| `SIM_PYTHON` | `/home/cody/miniconda3/envs/behavior-2026/bin/python` |
| `AGENT_PYTHON` | `/home/cody/miniconda3/envs/SeekAct/bin/python` |
| `SUDO_HELPER` | `/home/cody/.codex/skills/environment/scripts/sudo_run.sh` |
| `GPU_LOCK` | `/media/cody/软件/behavior_harness/outputs/ood99_gpu.lock` |
| `ASSET_ROOT` | `/root/download/datasets/BEHAVIOR-1K` |
| 视频验收程序 | `/usr/bin/ffmpeg`、`/usr/bin/ffprobe` |

`worker_command()` 设置两个仓库的 `PYTHONPATH`、资产目录、共享 GPU 锁和 headless 环境变量。`behavior_ap` 自己的安装配置也必须与本机 OmniGibson 源码、资产和解释器一致。仿真资产、官方专家数据、Conda 环境、Codex 登录及运行输出不随本仓库分发；需在目标机器单独准备。当前 runner 不是自动安装器。

## 本次固定的三个原场景

本机 protocol：

```text
/media/cody/软件/behavior_ap/outputs/seekact_agent_20261023/protocol.json
```

三项均为官方 **ID 原场景**：`variant=original`、`split=public_test`、`instance_id=301`、seed `20261020`。`301` 是场景实例号，不是 task index。没有为本次实验添加收音机 OOD 或其他人工布局变体。

| 任务 | 官方 task name | task index | 官方完整任务提示 |
|---|---|---:|---|
| 整理鞋子 | `putting_shoes_on_rack` | 22 | Pick up the two gym shoes and the two sandals from the corridor floor and place them onto the hallstand (shoe rack) in the corridor, making sure they are on the rack and not on the floor. Arrange them so the two gym shoes are next to each other and the two sandals are next to each other. |
| 整理玩具 | `putting_away_toys` | 54 | Pick up all the toy figures from the floor and put them inside the toy boxes. |
| 开收音机 | `turning_on_radio` | 0 | Turn on the radio receiver that's on the table in the living room. |

每个 attempt 重新启动 worker、初始化同一个任务实例；最多执行 10,000 个物理控制步。当前 Agent session 的默认墙钟预算为 14,400 秒；GPU/初始化等待另行记录，不以仿真秒替代真实耗时。protocol 固定了完整提示、初始实例、种子和专家选择元数据，runner 保存其 SHA-256 用于比较。

## 零样本与单条专家经验

1. 每个任务先独立执行 `zero_shot`。Agent 仅得到任务描述、机器人接口说明与当前观察，不提供专家录像、历史技能库或任务经验。
2. 只有当前任务的最新零样本 attempt 已封存为**有效任务失败**，才允许读取该任务预先固定的一条专家轨迹并提取经验。网络、进程、ACK、视频或仿真基础设施错误不能作为这个资格条件；先修复并用新 attempt 完成有效零样本试验。
3. 专家选择提前按对应任务最小 `episode_index` 固定，不根据结果挑选示范：鞋子为 4400、玩具为 10800、收音机为 0。这些是元数据选择，不表示已观看或分析内容。
4. 将这一条轨迹的经验写入一个 UTF-8 `.md` 或 `.txt` 文件，包含来源 episode、观察到的动作效果、适用条件与不确定性。只在失败后的分析阶段读内容；不重放示范动作、不检索其他示范。
5. 用 `one_expert_retry` 从全新初始场景执行。runner 再检查零样本判定及 protocol SHA，快照经验文本并记录 SHA、字节数和来源零样本记录。旧 attempt 保留。

[`export_one_behavior_expert.py`](../scripts/export_one_behavior_expert.py) 已实现失败门槛后的单轨迹导出。它先复核选定任务最新封存的有效零样本失败及 protocol SHA，再连接 `huoshan`，按固定 episode 筛选 parquet、复制唯一 annotation，并从三路原视频的第 0 帧各精确截取预定帧数。输出转为 H.264 High / `yuv420p` / 30fps，传输核验 SHA，最后逐路完整解码和检查实际帧数。不会复制包含其他示范的整段视频，也不会自动改选专家。

导出使用 `<task_name>/expert_export.lock`，不占用仿真批次的全局 `runner.lock`。已有效失败的鞋子任务可以导出专家，同时玩具或收音机继续各自的零样本；正在运行的其他任务不会获得该经验。新导出目录通过原子创建保留唯一性，现有目录拒绝覆盖或续传。导出前、传输前及最终验收时都重新检查选定任务的失败证据；同任务新零样本或 protocol 改变会拒绝完成。不要把同一个任务的新零样本与专家导出并行调度；不同任务可以并行。

当前代码实现的是这条 **BEHAVIOR 专用的受控经验注入路径**。没有把上游主分支或 G1 路径描述的通用经验学习机制原样迁移到此后端，也没有自动经验提取、跨任务记忆检索或共享技能库合并。`--experience` 只消费一份非空、最多 128 KiB 的摘要；代码可以校验文件数量、哈希和零样本资格，不能仅凭一个文本文件证明作者实际只看过一条轨迹。示范来源与分析边界仍需由实验操作者保留证据。

专用 Agent 会话禁用外部 shell、web、其他 MCP、记忆/插件及历史技能库，只允许本后端工具。其界面提供 RGB-D 与机器人状态，不向 Agent 返回特权对象位置或场景状态真值。下层单独调用 `harness.codex_cli.behavior --experience` 不具备 supervisor 的失败资格校验；正式实验使用下面的 runner 入口。

## 复现命令

以下命令为本机路径示例。迁移部署请先修改上述常量，复制并检查 protocol。输出目录由 `--protocol` 的父目录决定，没有单独的 `--output` 参数。

运行全部三个零样本任务，按 protocol 中的顺序执行：

```bash
cd '/media/cody/软件/SeekAct'
/home/cody/miniconda3/envs/SeekAct/bin/python scripts/run_behavior_experiment.py \
  --protocol '/media/cody/软件/behavior_ap/outputs/seekact_agent_20261023/protocol.json' \
  --phase zero_shot --attempt v1
```

只运行一个任务时，`--task` 接受该 protocol 的正式名称或 task index：

```bash
/home/cody/miniconda3/envs/SeekAct/bin/python scripts/run_behavior_experiment.py \
  --protocol '/media/cody/软件/behavior_ap/outputs/seekact_agent_20261023/protocol.json' \
  --phase zero_shot --task turning_on_radio --attempt v1
```

上面两条是全量与单任务两种选择，不要对已经存在的同名 attempt 再运行。遇到基础设施故障，先修复原因，再改用新的标签，例如 `--attempt infra_repair_v2`；不恢复旧物理进度、不覆盖旧证据。

该任务已封存为有效零样本失败后，导出预先固定的一条专家：

```bash
/home/cody/miniconda3/envs/SeekAct/bin/python scripts/export_one_behavior_expert.py \
  --output '/media/cody/软件/behavior_ap/outputs/seekact_agent_20261023' \
  --task turning_on_radio
```

远端解释器默认是 `/cs/kyy/behavior_ap_experiment/.venv/bin/python`，需要 PyArrow；可用 `--remote-python` 修改。`--ffmpeg` 指定远端编码程序，`--local-ffmpeg`、`--local-ffprobe` 指定本地验证程序。依赖需要预先准备，脚本不会安装。只有 `export_complete.json` 标明 `export_verified` 且 `source_count=1` 才能将导出视为完整；失败时保留本地及远端部分文件，不自动重试或换专家。

完成这唯一专家的分析，保存来源明确的摘要后，执行全新重试：

```bash
/home/cody/miniconda3/envs/SeekAct/bin/python scripts/run_behavior_experiment.py \
  --protocol '/media/cody/软件/behavior_ap/outputs/seekact_agent_20261023/protocol.json' \
  --phase one_expert_retry --task turning_on_radio --attempt v1 \
  --experience '/absolute/path/to/radio_single_expert_experience.md'
```

经验重试必须明确指定一个任务和一个摘要；`zero_shot` 拒绝 `--experience`。已有 attempt 目录和已有执行证据会被拒绝复用。runner 遇到无效评测会停止后续任务并保留证据；它不会自动重放部分动作或反复选择成功结果。

如果同一 protocol 的 runner 仍在执行，可加 `--wait-for-runner` 排队；默认不等待，锁被占用时立即报错。等待只改变锁获取方式，获得锁后仍重新检查全新 attempt 目录与有效零样本失败证据，之后才读取经验并初始化新场景。例如后台排队鞋子任务的单专家重试（日志路径须使用新文件名）：

```bash
nohup /home/cody/miniconda3/envs/SeekAct/bin/python scripts/run_behavior_experiment.py \
  --protocol '/media/cody/软件/behavior_ap/outputs/seekact_agent_20261023/protocol.json' \
  --phase one_expert_retry --task putting_shoes_on_rack --attempt v1 \
  --experience '/absolute/path/to/shoes_single_expert_experience.md' --wait-for-runner \
  > '/absolute/path/to/shoes_retry_queue_v1.log' 2>&1 < /dev/null &
```

## 物理执行与成功判定

Agent 可调用 `behavior_observe`、`behavior_state`、`behavior_pixel_to_position`、`behavior_move_base`、`behavior_move_arm`、`behavior_gripper`、`behavior_torso` 和 `behavior_finish`。手臂工具通过当前 R1Pro 的 Jacobian IK 和物理关节控制接近显式位姿；动作接口有范围与步数上限。位置和夹爪命令不直接修改物体位姿或任务状态。

worker 是唯一执行物理步的进程。Agent 推理及等待工具请求期间仿真不前进；没有后台空转物理。每步先持久化提交记录，再执行一步并核验步数，最后记录 ACK。提交的动作不会因网络重连自动重放；同一请求 ID、相同 payload 返回缓存结果，改变 payload 被拒绝。ACK 不确定时封锁后续动作。像素投影绑定当前 RGB-D `frame_id`，物理动作提交即使失败也会使旧帧失效。

最终任务成功取自官方 `BehaviorTask.success` / BDDL。以下指标应分开报告：

| 证据 | 能说明什么 |
|---|---|
| Agent 声明、`behavior_finish` | Agent 的判断；声明 success 只有官方 success 为真才会被接受 |
| IK 残差、工具完成、逐步 ACK | 控制执行及记账是否完整；不能证明抓取、放置或开机 |
| `official_success` | 官方任务谓词结果；仍需同时检查评测完整性 |
| 视频完整解码 | 录像是否完整可用；不能替代官方任务成功 |
| `evaluation_valid` / `status` | 将有效 `task_success` / `task_failure` 与基础设施错误分开 |

有效结果要求 Agent 正常结束、worker 正常结束、官方 success 是明确布尔值、没有仿真错误或未确定动作、提交数/ACK 数/步数一致，并且视频验收通过。0 步运行明确记录 `no_frames`，不伪造视频。凡存在未完成进程、损坏录像或不完整 ACK，都不能作为有效能力失败或成功统计。

## 真实墙钟计时

[`WallClockLedger`](../services/behavior_ap/timing.py) 使用 `time.perf_counter_ns()` 记录单调真实墙钟，并附 UTC 时间戳。输出显式标注 `wall_clock_not_simulation_time=true`。物理步数和按控制频率换算的仿真时间是独立指标。

- supervisor 记录 `worker_startup`、`agent_session`、`worker_finalize`，其总时间是该次调度范围的实际经过时间。
- worker 记录 GPU 等待、初始化、工具分发、观察、动作执行和视频收尾。嵌套类别同时提供 inclusive 与 exclusive 耗时；未被 span 覆盖的间隙列为 `idle_unclassified_seconds`。
- Agent 日志记录事件抵达时间与整个 Agent 进程的墙钟。它包含模型等待、网络重连、工具执行、协议及输出开销，**无法据此独立测量纯 LLM 推理耗时**。即使 protocol 中列了 `agent_inference`，也不能把整个 `agent_session` 改名为纯推理时间。
- supervisor、Agent、worker 的计时有重叠，不能相加作为端到端耗时；同一 ledger 的 inclusive 类别也不能直接求和。后台视频编码诊断只作为独立字段，不累加到前台总墙钟。worker 计时范围以其 ledger 的起止为准；当前结束点在 Isaac 最后 shutdown 前，完整进程收尾由 supervisor 包含。
- 专家导出器在 `export_timing/` 单独记录本地真实总墙钟，以及 `expert_export`、`expert_transfer`、`expert_verification` 三阶段。远端编码等耗时是重叠诊断，不能再加到本地总时间。专家内容分析仍需另行记录真实起止；未记录的分析耗时应标为未测量，不能从其他阶段扣减或估算填充。

## 输出与验收材料

本机输出根目录为 `/media/cody/软件/behavior_ap/outputs/seekact_agent_20261023/`。每次运行使用 `<task_name>/<phase>_<attempt>/`，例如 `turning_on_radio/zero_shot_v1/`。主要文件：

| 路径 | 内容 |
|---|---|
| 根目录 `protocol.json`、`summary.json` | 固定协议、全部 attempt 的汇总；计数单位是 attempt，不挑选成功样本 |
| `trial_started.json`、`trial_record.json` | 来源、参数、开始/结束状态及最终评测分类 |
| `worker_config.json`、`agent_config.json`、`launch_commands.json` | 本次具体部署与运行参数 |
| `execution.jsonl`、`actions.npy` | 逐步提交/ACK 与实际提交动作数组 |
| `observations/` | 原始相机图像、RGB-D 几何及 Agent 观察记录 |
| `episode_closed.json`、`agent_finish.json` | 官方终态、动作完整性、视频验收与 Agent 判断 |
| `sim/videos/robot_cameras.mp4`、`sim/videos/third_person.mp4` | 机器人相机与第三人称全过程视频，以最终验收记录中的实际路径为准 |
| `worker_timing/`、`supervisor_timing/` | `timing_events.jsonl` 与 `wall_timing.json` |
| `agent/agent_events.jsonl`、`agent/agent_messages.md`、`agent/agent_timing.json` | Agent 原始事件、可见动作说明与进程墙钟 |
| `single_expert_experience.md` | 仅经验重试保存的唯一摘要快照 |
| `worker_error.json`、`uncertain_action.json`、`finalization_error.json`、`video_verification_error.json` | 出错时保留的基础设施、动作或视频诊断 |

单条专家保存在独立的 `<task_name>/expert_episode_<episode_index>/`，不混入物理执行 attempt。该目录包含 `episode.parquet`、`annotation.json`、`videos/head.mp4` / `left_wrist.mp4` / `right_wrist.mp4`、远端来源清单、`export_complete.json` 和 `export_timing/`；失败则写 `export_error.json`。远端独立临时目录在 `/cs/kyy/behavior_ap_experiment/reports/seekact_exports/`，以任务、episode 和唯一标识命名并保留来源/失败记录。

对于有物理步的运行，验收要求机器人相机和第三人称两路完整视频，检查 H.264 / `yuv420p`、实际帧数与执行步数一致，并用 FFmpeg 从头到尾完整解码。录像/状态文件包含实验记录，默认留在本地输出目录；本 runner 不自动上传它们。

实现入口：[worker](../services/behavior_ap/worker.py)、[工具与参数](../services/behavior_ap/schema.py)、[R1Pro 控制](../services/behavior_ap/robot.py)、[RGB-D 投影](../services/behavior_ap/geometry.py)、[视频验收](../services/behavior_ap/video.py)、[专用 Agent launcher](../harness/codex_cli/behavior.py)。相应 `local_tests/unit/test_behavior_*.py` 是 CPU/mock 或小型合成视频契约检查，不会自动构成三任务真实仿真成功证据。


## Learned AP / champion VLA tools

The optional operator `backend_options.vla_tools` enables `behavior_active_perception_vla`, `behavior_champion_vla` and `behavior_vla_continue`. Existing robot tools remain available. AP uses original checkpoint_024000, ordered head/left/right RGB and raw61D proprioception, predicts32x23 and executes only prefix8. Its language is rendered from demonstrated qualitative motion tags; it does not have arbitrary grasp/place/switch language skills.

Champion uses the verified public checkpoint_2 and the unchanged published wrapper (including resampling and action corrections). It executes20x23 per chunk, takes a fixed skill ID, and does not consume text. This checkpoint covers radio0 and shoes22, plus `picking_up_toys`7 and `collecting_childrens_toys`21. Original `putting_away_toys`54 is unsupported; using another toy skill requires an explicit recorded skill mapping and must be reported as cross-task identity transfer, including whether its selection was human-confirmed or an announced operational default. No silent task/model substitution.

A start tool creates a bounded job (at most120 steps). Repeatedly call `behavior_vla_continue(job_id, wait_seconds=20)` until complete, preferably in a code loop. Only the Isaac owner thread admits one saved chunk prefix per continue call. HTTP inference runs in a background thread with immutable retry bytes, while physics is frozen; ordinary motion is blocked while a job is pending. A completed job returns fresh images. Every action still has a durable submitted/ACK record and controller/job/chunk provenance. On handback, numeric controls preserve the learned gripper commands. Champion history persists across consecutive champion calls but resets on episode, skill or intervening controller changes. A failed/uncertain action is never replayed.

Remote inference only: deploy `services/behavior_ap/champion_server.py` with the existing huoshan champion venv/vendor PYTHONPATH and checkpoint assets, on a dedicated loopback port18778, then forward it through SSH. Keep the original AP server18770. Never install or run Isaac on huoshan. The study output `server_deployment.json` records the content-addressed deployed script and PID. Neither weights nor normalization statistics are changed.

The prepared study lives at `/media/cody/软件/behavior_ap/outputs/seekact_vla_tools_20261024`. `models_ready.json` is real inference plus duplicate/reset verification, not task success. `worker_calibration_v1/acceptance.json` validates30 real integration steps (8AP+20champion+2numeric), separately from task trials. Run the three-task supervisor with:

```bash
/home/cody/miniconda3/envs/SeekAct/bin/python scripts/run_behavior_vla_suite.py \
  --protocol /media/cody/软件/behavior_ap/outputs/seekact_vla_tools_20261024/protocol.json
```

It starts fresh scenes, preserves previous attempts, waits for an explicit `toy_decision.json` before the unsupported original toy task, and writes `index.html`/`vla_usage.json`. A valid ability failure is not cherry-picked for another attempt. Infrastructure repairs require a new attempt name and `--retry-infrastructure-fresh-episode`. The toy decision must record its provenance honestly. An explicitly stated operational default may preserve the original task and allow related fixed skills under the user's tool-integration authorization; it is not a human confirmation. A human reply overrides the pending default before execution. Model waits remain separate from capability failures. All finalized videos are H264 High/avc1/yuv420p and fully decoded. The existing GPU lease waits for other compute owners and never terminates them.

### Confirmed replacement toy task (2026-10-24)

The user subsequently confirmed three trials total with both VLA tools and requested a champion-native toy task. The current cohort is shoes22, radio0, and `collecting_childrens_toys`21. `protocol_native_toy21_v2.json` freezes that change; `protocol_current.json` selects the active report. Shoes/radio keep their unchanged, verified records. The original toy54 cross-skill trial remains in `index_cross_task_history.html` and its original directory, excluded from the current three-task aggregate. This later confirmation does not retroactively authorize the historical cross-skill choice.

Task21 exposes only champion skill21 (checkpoint2), with transfer disabled, and starts a fresh official public_test/301 scene. The English task instruction is derived from the installed official BDDL problem0; its source and SHA256 are in the protocol. No extra expert trajectory/summary is supplied. The original AP024000 remains available. Run only through the existing GPU lease, without interrupting other owners:

```bash
/home/cody/miniconda3/envs/SeekAct/bin/python scripts/run_behavior_vla_suite.py \
  --protocol /media/cody/软件/behavior_ap/outputs/seekact_vla_tools_20261024/protocol_native_toy21_v2.json \
  --attempt native_toy21_v1
```

Do not start a second supervisor while this attempt is active. A valid task failure remains a failure; infrastructure retries require a new attempt and fresh initial scene, preserving old evidence.

### VLA-only scheduling (latest user protocol)

The current study is `/media/cody/软件/behavior_ap/outputs/seekact_vla_only_20261024`, not the older mixed-control cohort. All three native-supported tasks (shoes22, radio0, toys21) start fresh. Enable `vla_tools.enabled=true` and `vla_tools.vla_only=true` in the frozen operator protocol. The GPT backend remains gpt-6-astra/low/Fast requested.

The original numerical implementations remain available to other modes. In this mode the frontend hides all four native motion tools and both the backend and Worker reject them. Worker.step additionally refuses physical actions whose provenance is not `ap_vla` or `champion_vla`. Read-only state, images and pixel projection remain available. Every physical step must match a saved VLA output prefix and ACK.

The initial phase is search: only AP can execute. After a completed AP job, GPT observes a fresh frame and calls `behavior_handoff_to_champion(frame_id,target_description,reason)` to record its visual target-found assessment. This call adds zero physical steps and does not prove task success. Champion execution requires this authorization. Another AP job returns control to search and requires a new handoff. Final task success always uses official BDDL; a search failure need not invent a champion execution.

The 28-step `worker_calibration_v1` is an integration fixture (8 AP + 20 champion), not a benchmark or evidence that the shoe target was found. It verifies exact prefixes, a semantic handoff fixture, and rejection of a native movement request without physical advance. 87 CPU tests cover the control gates, ownership, stale handoffs and task acceptance. The supervisor `scripts/run_vla_only_study.py` then runs three fresh trials, audits action attribution and independently decodes the six final videos. Do not start another copy while it runs. Old mixed-control results and the user-stopped20-step trial remain historical and are excluded.


### Repeated visual closed-loop AP / champion scheduling

Set `vla_tools.closed_loop=true` together with `enabled=true,vla_only=true`.
This retains the original modes while enforcing per-segment decisions in this mode.
Both start tools require `frame_id`, `step_budget` (positive physical steps, no per-call upper bound),
`visual_progress`, `reason`, and `next_goal`; omit `chunks`. The frame must be
from the current physical step. AP still predicts32 actions and executes at most8
per fresh inference; champion returns20 and executes at most20. The last prefix
is truncated to the remaining segment budget. Requests larger than the episode
remaining budget are clipped at that episode limit;120 is not a per-call cap. Saved raw outputs are unchanged,
unused tails are discarded, and a partial champion prefix resets its temporal
history before the next inference. It never replays an old tail.

Every completed segment returns current camera images to GPT. GPT chooses the
next policy and budget after inspecting them. Three consecutive champion
no-progress/lost-target decisions block a fourth unchanged decision; the agent
must perform a safe AP view/approach recovery and renew the visual handoff.
This is a planner assessment, not an automatic detector or proof of grasp.
Held-object safety must be assessed from wrist images before AP recovery.
Champion still receives only its fixed task identity, not the planner reason
or next-goal text. AP receives only its trained qualitative prompt.

Temporary stagnation cannot seal an early `failure`: the default10000-step
budget is retained unless official success, environment termination, or a
concrete safety/infrastructure `blocked` outcome occurs. No native movement
fallback is admitted. Scheduler decisions are saved as `vla/decision_*.json`
and inside `vla/summary.json`; raw predictions, prefix lengths, every physical
ACK, and full-process videos remain auditable.


### Interaction-only champion ownership and wall-time-only trials

`vla_tools.interaction_only=true` requires a single `interaction_goal` for each
champion job. After it ends, GPT must inspect fresh images and call
`behavior_assess_interaction`. `in_progress` permits continuing exactly the same
interaction; `completed` or `need_ap` clears champion authorization and requires
an executed AP job plus fresh handoff before the next interaction. `unsafe`
blocks further physical calls. These assessments are GPT's visual claims, not
automatic grasp/placement detectors or official BDDL success. Champion weights
and raw outputs remain unchanged and the fixed task model still does not accept
interaction_goal as a natural-language control input.

Use `max_steps:null` and `wall_limit_seconds:3600` to remove both the host and
native OmniGibson step-count Timeout. The official PredicateGoal remains intact.
The wall deadline starts when the task trial is created and includes simulator
startup, agent/inference waits and physical execution; video encoding/final audit
can finish afterward without further physical actions. Deadline checks occur
before each physical prefix action and while polling model responses. Expected
wall expiry is recorded as `task_timeout`, distinct from infrastructure errors.
Previous finite-step protocols retain their original behavior.

For this run only, `BEHAVIOR_ALLOW_TODESK_DISPLAY=1` permits the exact ToDesk
Session display process to coexist when at least20000MiB is free. The default
exclusive gate is unchanged; any other GPU compute process and the shared lease
still block execution. No desktop or unrelated experiment is terminated.

### Controller provenance in saved videos

Saved robot-camera and third-person MP4s now have a56-pixel header by default.
A cyan magnifier/`AP` identifies AP actions; a gold star/`CHAMPION VLA` identifies
champion actions. The header includes the physical step and job/chunk identity.
Native controls are purple; missing ownership is explicitly shown as unknown,
never inferred from intended phase. `Worker.step` copies the current action owner
to the recording backend before executing that action. Frame0 is the camera
result after physical action1. The ribbon is added only to recording composites,
not RGB/RGB-D policy observations. Physics/action values and recording fps are
unchanged. Both baseline and asynchronous recording paths use the same ribbon.

To annotate an existing sealed study without overwriting its original evidence:

```bash
/home/cody/miniconda3/envs/behavior-2026/bin/python scripts/annotate_behavior_controller_videos.py --output /path/to/study
```

Use the existing controlled sudo helper if `sim/` is privileged-owned. The script
requires contiguous submitted-action/ACK sequences and exactly one frame per
executed action. It writes `sim/videos_annotated/`, preserves original `sim/videos/`
and hashes, fully decodes both outputs, and creates `video_annotation.json` with
frame-aligned controller ranges and source/output hashes. The report selects the
annotated version only after this verification receipt exists and keeps a link
to the original. `controller_video_annotations.json` is the study-level receipt.

### Qwen local interaction completion monitor

`scripts/run_qwen_interaction_study.py --output <new-study-root>` waits for the
configured OpenAI-compatible Qwen service, discovers its served model ID, verifies
an actual image request, runs baseline and monitored-worker calibrations, then
starts three fresh trials. Secrets remain in a mode-600 file outside the repo;
protocols contain only `vla_tools.qwen_monitor.api_key_file`, never the key.

AP exclusively owns search, viewpoint changes, approach and travel. The champion
receives its original fixed task identity; GPT's `interaction_goal` is for the
monitor, not a new champion language input. It must identify one pick/lift,
place/release or activation with an observable completion condition.

After each champion chunk (at most20 physical steps), physics freezes and Qwen
receives one current RGB composite (head and both wrists) plus the local goal.
No previous frames, previous answers, proprioception, BDDL predicates or oracle
are supplied. Only exact `继续` and `已完成` are accepted. Invalid outputs and
transient network errors retry the same request without advancing physics or
adding a vote. A `继续` clears the positive streak; duplicate physical-frame
votes are rejected. Two consecutive positive votes on increasing physical frames
interrupt the champion before the next inference/action and return fresh images
to GPT. AP must execute before another champion handoff. This is a visual
assessment, not verified grasp stability or official BDDL task success.

All inputs, frames, raw replies and parsed verdicts are saved in each job's
`qwen_*` directories. `vla/qwen_checks.json` and the final VLA summary record
streaks and switch evidence. Video badges continue to identify the controller
that actually emitted each action. Network/Qwen waiting is absent from the
physical-process video and remains in wall-clock records.

Explicit `unlimited_duration_authorized: true`, `max_steps: null` and
`wall_limit_seconds: null` remove both episode step and wall-clock limits from
simulator and BEHAVIOR Codex launcher. Transport requests still have retryable
connection timeouts; these do not end the experiment. Official success, genuine
environment termination, explicit cancellation or concrete safety blockage can
still end a trial. Other robot launchers retain their finite session requirement.
