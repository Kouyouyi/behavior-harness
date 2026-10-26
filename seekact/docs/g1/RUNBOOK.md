# 启动与恢复

当前推荐入口见 [DEPLOYMENT.md](DEPLOYMENT.md) 和 [VLA.md](VLA.md)。以下按实验保留历史命令，不代表当前默认配置。

当前实现过程中。可执行命令随实测更新；不要使用继承的 `scripts/rby1` 启动 G1。

当前真机已按 O 正常退出并由支撑承重，上层 owner 已关闭。E025 离髋、E026 前伸回位通过本轮粗调试验收；实际目标误差 2.70 cm、回位误差 1.94 cm，后续仍需重复性和精度验证。以下动作命令需要本轮现场状态明确，不能在无人确认时直接启动。

隔离仿真的五组对照及 E021/E022 正常碰撞离髋伸手、启动命令和视频见 [SIMULATION.md](SIMULATION.md)。仿真 `--case reach` 不能绕过真机 executor 使用。仿真入口固定 DDS 77/lo，不能改成真机 domain/interface；碰撞关闭参数仅用于仿真诊断。

## 目录

- 源码：`/home/user/code/rax/worktrees/g1-sonic`
- 运行证据：源码目录下 `runtime/g1/`，Git 忽略
- Python 初期复用 `/home/user/GR00T-WholeBodyControl/.venv_data_collection/bin/python`（已有 numpy、OpenCV、PyAV、ZMQ）；不修改该环境。
- 真机视频服务 `g1-deploy-video.service`：13579 为相机控制，5555 为左目 RGB 预览。双目接收占用相机 PICO 请求，不能同时连接 PICO 相机客户端。

## 当前运动接入边界

CoMotion 的 `run_pico_real.sh --run` 会真正启动控制器并释放原运动服务，不是只读探测，不能在未知站姿时执行。其 `--check` 只做本地检查。RAX 首次动作前须实测直接参考通道的接管与退出，不能把 token/PICO 通道的历史验证自动延伸到该通道。

若原生控制器已在维持机器人，不要通过关闭窗口或杀进程加载新代码；使用既定现场停止流程。没有实际反馈时不自动重放动作。

## 环境及离线回归

```bash
cd /home/user/code/rax/worktrees/g1-sonic
export G1_PY=/home/user/GR00T-WholeBodyControl/.venv_data_collection/bin/python
PYTHONPATH=runtime/g1/dev-python:. "$G1_PY" -m pytest \
  local_tests/unit/test_g1_reference.py local_tests/unit/test_g1_backend.py \
  local_tests/unit/test_g1_commission.py local_tests/unit/test_workspace.py \
  local_tests/unit/test_frontend.py -q \
  --junitxml=runtime/g1/software-reference-verification.xml
```

pytest 依赖已隔离在 `runtime/g1/dev-python`。测试需要允许本机回环/Unix socket；不连接 DDS 或真实控制端口。软件通过不能替代真机验收。

新增伸手流程测试：

```bash
PYTHONPATH=runtime/g1/dev-python:. "$G1_PY" -m pytest \
  local_tests/unit/test_g1_reach.py -q
```

E023 的 41＋3 项分别归档为 `E023-reach-legacy-regression.xml`、`E023-reach-flow-tests.xml`，不要覆盖它们来表示另一源码版本的验证。

## 只读基线与相机

```bash
bash /home/user/code/CoMotion/experiments/shared_control/scripts/run_pico_real.sh --check
"$G1_PY" -m services.g1_sonic.camera_service \
  --receiver-ip 192.168.123.76 --output runtime/g1/rgbd-NEXT --seconds 20
```

`--check` 不启动控制；相机服务只控制视频采集，不发送电机指令。相机采集前核实 PICO 客户端未占用 13579。每轮用新目录，停止后 `health.json` 的 `running=false`，旧图像不再被视作实时观察。

## 准备新一轮（此步不启动控制器）

```bash
"$G1_PY" scripts/g1_prepare_native.py \
  --run runtime/g1/EXXX-reference --reference-hold
```

生成 `controller.sh`、`owner.sh`、`manifest.json`、`runtime_source/`。检查二进制 SHA256、DDS domain 0、网卡 eno1、原生看门狗 350 ms；owner 从归档源码启动。原生模型与两个复用 Python 源文件仍是外部固定依赖；按基线清单核对版本，不能把准备目录当作完整容器镜像。

## 现场控制阶段（会动作）

只在本轮复验方案及现场状态明确后执行。两个命令分别在独立终端运行，保留原生终端的键盘操作：

```bash
bash runtime/g1/EXXX-reference/owner.sh
```

```bash
bash runtime/g1/EXXX-reference/controller.sh
```

原生启动会进行默认姿态初始化，不是只读进程。owner 等待 `Init Done` 后请求 planner idle，连续稳定约 3 秒再接管零偏移参考，验证约 3 秒。查看 `executor_status.json` 和现场实际状态；必须达到 `reference_ready`，不能把仅启动成功当作就绪。

owner 只有本机 ZMQ 17586 发布权；原生 `g1_field` 是唯一 LowCmd 发布者。租约默认 600 秒，过期退回锁定 idle。修改工作树不会改变这一轮归档 owner；需要更新时先按现场流程结束本轮，再准备新目录。

## RAX 一次往返（会动作，当前暂不重试）

```bash
"$G1_PY" scripts/g1_run_controller.py \
  --executor runtime/g1/EXXX-reference --run runtime/g1/EXXX-rax-roundtrip \
  --side right --delta-rad 0.03
```

启动相机和 MCP，运行 `examples/g1/controller.py`，结束时关闭这两个服务。脚本不负责停止原生或 owner。保留 `result.json`/`failure.json`、原生 trace、owner events/results、参考发送/回读日志和 agent 程序快照。没有结果或反馈不确定时不重放。

完成须同时通过原生回读、真实响应、全身回位与速度稳定；`submitted`、发送计时结束或看到图像都不是成功。`reference_review_required` 不再接收动作，必须查看误差报告；它仍在发布保持参考，并未进入阻尼。

## 分段离髋与前伸（会动作，clear_reach_v1）

E024–E026 已执行的流程：现场先确认双脚承重、吊装仅防跌落、手臂不受吊带限制。仅触地但主要由吊装承重不等同于该条件；不从关节反馈自行推断承重。

```bash
"$G1_PY" scripts/g1_prepare_native.py \
  --run runtime/g1/NEW-clear-reach --reference-hold --reach-profile
```

上面只准备脚本。按前文现场控制阶段启动 owner 与有原生 O 的桌面 controller 窗口。站立和参考接管达到 `reference_ready` 后：

```bash
"$G1_PY" scripts/g1_run_controller.py \
  --executor runtime/g1/NEW-clear-reach --run runtime/g1/NEW-rax-clear \
  --action clear_arms
```

离髋成功保持后，现场观察双侧 BrainCo/手腕与大腿、髋部、吊带无接触或限制，且双脚仍承重。头部相机未看到该处时不能用图像或模型编造确认。记录在本轮 `operator_clearance.json`，字段：`baseline_id`、`phase="clear_arms"`、`monotonic`、当时实测 `q`、`feet_bearing_weight`、`both_arms_clear`、现场原话与来源；未知的厘米间隙应记为 null。该记录须对应当前明确现场答复，120 秒后或姿态明显变化后无效。

```bash
"$G1_PY" scripts/g1_run_controller.py \
  --executor runtime/g1/NEW-clear-reach --run runtime/g1/NEW-rax-reach \
  --action reach_return
```

前伸与返回完成后按下面流程结束本轮。不因结果接近门槛而自动重复或增加幅度。每个 MCP 程序都会自行结束相机和 frontend，但不停止原生或 owner。数值和范围见 [CONTRACT.md](CONTRACT.md)。

停止后可离线审计（不连接机器人）：

```bash
"$G1_PY" scripts/g1_audit_reach.py --run runtime/g1/NEW-clear-reach
```

## 结束与异常

1. 按现场既定支撑流程准备承重，再在原生窗口按 **O**。O 会进入阻尼并退出，不能假定仍维持站立。
2. 确认现场承重和原生正常退出，再退出 owner。不要以杀死原生进程、关闭原生窗口代替停机。
3. `fault_idle` 只撤回上层参考、锁定后续动作并回到原生 planner idle；不意味着机器人已被机械支撑或电机关闭。
4. 人工挪动、外部接触及其时刻尽量记入实验。失联、过速和手动干预后的证据与正常跟踪区间分开解释，不自动解除锁定。

历史失败和新版语义见 EXPERIMENTS E008–E011。当前没有外参标定、通用末端规划、BrainCo 控制或携物行走验收。

## E060 起的保持与步态规则

- 保持姿态时单次关节速度超过 1.5 rad/s 只记录，不再立即故障锁定；新动作必须等实测稳定至少一秒且当前关节速度低于 0.3 rad/s。接管/手臂动作速度限制、下肢偏离、倾斜、反馈和原生保护仍有效。人工挪脚不等于允许忽略实际过热。
- scripts/g1_read_motor_health.py --output <新文件> 使用 teleop Python，仅订阅 LowState，零发布者，读取绕组/驱动温度及 motorstate。温度回落不能替代带载观察。
- 迈步参考完成、实测停稳与实际走近分别验收。E062 的参考完成但物理走近失败，不能依据 completed=true 就继续假定抓取距离已经缩短。每轮仅一次单步，无自动重复。

## 踝温观测服务（不会抬脚）

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user/GR00T-WholeBodyControl/.venv_teleop/bin/python -m services.g1_sonic.ankle_monitor --run runtime/g1/NEW-ankle-observer --seconds 20
```

该服务只订阅 LowState，任意踝绕组/驱动温度严格 >91°C 时播报对应侧告警，每侧每个高温周期一次；同侧 pitch/roll 都 <85°C 持续五秒重布防。日志会明确记录动作未启用。`--replay 文件` 使用独立标记的回放源，零 DDS 连接，播报以 Replay test 开头。

新 prepare 输出 thermal-observer.sh（600 秒观测），不自动启动它或任何控制器。该入口不执行抬脚，不能当作已经完成的真机高温自动处理 skill。

## Pico 踝部示范离线分析与仿真

以下在 `/home/user/code/rax/worktrees/g1-sonic` 执行，只读取源数据、创建新输出。`NEW-*` 必须为不存在的目录；不覆盖 E076–E078。

```bash
/home/user/GR00T-WholeBodyControl/.venv_data_collection/bin/python scripts/g1_review_ankle_demonstrations.py export \
  --dataset /home/user/GR00T-WholeBodyControl/outputs/ankle_heat \
  --output runtime/g1/NEW-ankle-demo --episodes 13 15

MUJOCO_GL=egl MPLCONFIGDIR=/tmp/rax-ankle-matplotlib \
  /home/user/GR00T-WholeBodyControl/.venv_teleop/bin/python scripts/g1_review_ankle_demonstrations.py review \
  --source runtime/g1/NEW-ankle-demo/episode_000015.npz --render

/home/user/code/CoMotion/.venv-sim/bin/python scripts/g1_sim_reference.py \
  --case recorded_token --recorded-episode runtime/g1/NEW-ankle-demo/episode_000015.npz \
  --heel-side right --run runtime/g1/NEW-sim-ankle-right \
  --sim-build runtime/g1/build-stream-timing-sim-v2

/home/user/GR00T-WholeBodyControl/.venv_data_collection/bin/python scripts/g1_audit_pico_ankle_sim.py \
  --run runtime/g1/NEW-sim-ankle-right --side right

MUJOCO_GL=egl /home/user/code/CoMotion/.venv-sim/bin/python scripts/g1_render_heel.py \
  --run runtime/g1/NEW-sim-ankle-right --demo-side right
```

episode 13 对应左侧，将文件编号、side 与新输出名同时替换。物理仿真入口强制检查 DDS77/lo 的 sim 构建身份，不能用于 field 二进制；`recorded_token` 尚未注册到真机 owner。仅运动学重建的视频明确标记推定地面与无物理推进，与后续 SONIC 动力学视频分开。

## 原始 Pico 抬跟执行器（E085 起）

会发送真实身体动作；与旧 reference owner、Pico 遥操及其他身体控制器互斥。沿用原生 O、watchdog、模型与增益。现场恢复双脚稳定承重、吊装仅防跌落且手脚无障碍后启动。不要复用已有日志的实验目录。以下 E085 命令用于说明已冻结入口；新一轮应生成新目录。

```bash
cd /home/user/code/rax/worktrees/g1-sonic
# 新目录准备，无运动；替换 E_NEW 为新实验编号
/home/user/GR00T-WholeBodyControl/.venv_teleop/bin/python scripts/g1_prepare_native.py --help
# 终端 1：独占高层 owner，等待原生 Init Done 后进入站立
bash runtime/g1/E085-real-ankle-demo/owner.sh
# 终端 2：原生控制器，保留可操作 O 的窗口
bash runtime/g1/E085-real-ankle-demo/controller.sh
# 终端 3：ankle_status.json 为 ready 后，单次右侧实际播报/抬跟/返回
bash runtime/g1/E085-real-ankle-demo/test-right.sh
```

`test-left.sh` 为左侧独立试验。结果写入 ankle_results；completed 仅表示片段提交与返回稳定，实际后跟动作由现场观察确认。动作结束保持示范首帧。无需追求毫米级回位。

确认该侧现场动作后，可用同目录 `arm-right.sh` 布防右侧；`arm-both.sh` 要求当前会话左右均完成手动流程。初始无自动侧；`disarm.sh` 撤销后续温度触发，不作为正在进行动作的紧急停止。严格 >91°C 且该侧布防后先播报，成功结束才动作，每侧每高温周期最多一次；两踝温度均 <85°C 连续 5 s 才解除锁存。音频失败、状态过期或电机故障不继续动作。不要主动加热来测试事件。

本 owner 已含温度订阅和路由，无需同时运行旧 thermal-observer.sh。记录包括 ankle_events、ankle_temperatures、ankle_execution、controller_trace 和实际 WAV。手动测试播报说明是测试，不冒充真实高温告警。

结束时按现场流程让支撑承重，再在原生窗口按 O（进入阻尼并退出）。核实原生退出后才停止 owner。不得用 kill 原生进程代替现场 O 流程。

## 双侧自动启动模式（E087 起，用户已要求默认响应两侧）

E085 的“默认不布防/必须先做本会话手动测试”仅适用于未配置自动侧的入口。以下新配置明确授权双侧温度动作，不要求先手动测试两侧。

```bash
cd /home/user/code/rax/worktrees/g1-sonic
# 准备新实验目录；已存在的目录不能覆盖
/home/user/GR00T-WholeBodyControl/.venv_teleop/bin/python scripts/g1_prepare_native.py \
  --run runtime/g1/E087-real-ankle-bilateral-auto \
  --native-build runtime/g1/build-stream-timing-field-v2 \
  --ankle-profile --thermal-auto-sides left right
# 终端 1（现场恢复承重后）
bash runtime/g1/E087-real-ankle-bilateral-auto/owner.sh
# 终端 2（原生 O 停止窗口）
bash runtime/g1/E087-real-ankle-bilateral-auto/controller.sh
```

E087 已生成时只运行两个启动脚本，不重复准备。ready 后查看 ankle_status.json 的 thermal_auto_sides 应为 [left,right]。无需额外 arm-both；发生真实 >91°C 时，owner 自己播放对应侧语音，再执行抬跟/返回。原生自身 >=90°C 的语音仍存在，它不是本 owner 的动作完成信号。不会主动加热测试，也不把两套阈值混为一谈。

两侧同时热时串行处理，每次开始前重新检查温度。`disarm.sh` 禁用后续自动响应，`arm-both.sh` 可在当前已配置会话恢复双侧；同一高温周期不会因此重复动作。动作中的紧急停止仍用原生 O。

E087 已结束归档，原生 22:07:06 正常 O 退出，owner 已清理。上例 E087 启动文件仅用于追溯；下一轮请将准备命令中的目录换成未使用的新编号，再运行新目录脚本，保留 `--thermal-auto-sides left right`，避免覆盖本轮证据。

## E089 手动启动与每秒全身温度窗口

E089 已生成，身体控制器尚未启动。此轮左右自动响应均已配置，继承 E087 的 >91°C 对应侧播报/抬跟逻辑。不要同时运行 Pico 或其他身体 owner。按以下顺序开三个终端；启动第三个前恢复双脚稳定承重、吊装仅防跌落，并留人操作原生 O。

终端 1：温度面板（只读，可提前打开）。

```bash
bash /home/user/code/rax/worktrees/g1-sonic/runtime/g1/E089-manual-ankle-bilateral-auto/temperature-watch.sh
```

终端 2：自动动作 owner，等待原生初始化。

```bash
bash /home/user/code/rax/worktrees/g1-sonic/runtime/g1/E089-manual-ankle-bilateral-auto/owner.sh
```

终端 3：原生 SONIC，初始化后自动进入站立；在此窗口按 O 停止。

```bash
bash /home/user/code/rax/worktrees/g1-sonic/runtime/g1/E089-manual-ankle-bilateral-auto/controller.sh
```

面板每秒刷新 29 个身体关节，不包含独立 BrainCo 手指电机。W=绕组温度，D=驱动温度，M=两者最大值，Err=电机错误码。左右腿/腰和两臂并排，温度 >=85°C 黄色、>91°C 红色；这些颜色仅用于显示，非踝关节不会触发抬脚。踝 >91°C 标记星号，电机错误标记感叹号。DDS 反馈过期会明确标记 STALE 和旧值，不能当成实时温度。

Owner 为 ready、Auto 为 left,right 时双侧自动待命；顶部还显示执行阶段与 remaining 剩余秒数。身体控制初始化前显示 waiting 是正常的，温度仍独立订阅。到达严格 >91°C 后，先成功播报再执行对应侧动作；双侧热则依次处理，同一高温周期不重复抬脚。原生 >=90°C 播报与动作阈值不同。

原有会话时限仍为启动策略后 600 s；到期退回原生站立并关闭自动响应，不会自动停机。结束时先让支撑承重，在终端 3 按 O，原生退出后再 Ctrl+C 结束终端 2/1。不要先关闭 owner。关闭温度窗口只结束显示，不影响控制器。

温度窗口每秒采样记录到 temperature_dashboard.jsonl；控制事件、温度、动作记录也在本轮目录。E089 仅供这一轮使用，之后用新目录准备同样配置，不覆盖历史日志。

验证 E088：真实 DDS 只读 4 秒，4 次显示（启动首帧等待数据、随后 3 帧有效），全部 29 关节、约 1.000 s 间隔、0 运动指令。显示的实时/过期/无数据和关节编号已检查；原 25 项自动动作回归全部通过。

## E096 原生报警重复响应与放大动作

E096 已准备、尚未启动。此配置替代 E089 的严格 >91°C / 每高温周期一次：逐电机 >=90°C 进入报警、<85°C 清除，与原生一致；第一次成功返回后至少等 30 s，报警仍有效且身体稳定才再次播报/抬对应侧。原生每五秒重复的语音不会累积成多个动作。每次实际动作前仍有独立成功播报；它不是原生 TTS 完成事件订阅。

左右采用 CoMotion leg-delta gain=1.5 的仿真候选。仿真后跟抬升约 5.5/6.1 cm，实机幅度未验证。旧 E089 原生已 O 退出、残留 owner 已清理；不要同时启动其他身体控制器。

终端 1（只读温度，更新后的规则和 next repeat 倒计时）：

```bash
bash /home/user/code/rax/worktrees/g1-sonic/runtime/g1/E096-manual-ankle-delta-repeat/temperature-watch.sh
```

终端 2（自动执行器）：

```bash
bash /home/user/code/rax/worktrees/g1-sonic/runtime/g1/E096-manual-ankle-delta-repeat/owner.sh
```

终端 3（现场恢复双脚承重、吊装仅防跌落后启动；原生 O 停止窗口）：

```bash
bash /home/user/code/rax/worktrees/g1-sonic/runtime/g1/E096-manual-ankle-delta-repeat/controller.sh
```

面板应显示 Auto:left,right、Rule:native90、repeat gap 30s。该侧仍在85–89°C且之前达到过90°C时，原生报警仍有效，下一次动作可能继续；低于85°C时清除。软件故障会关闭自动侧，不用重复启动旧目录掩盖故障。会话仍限600s，结束仍先支撑承重、终端3按O，退出后再Ctrl+C关闭owner和温度面板。

新一轮复现准备（换未使用目录）：

```bash
cd /home/user/code/rax/worktrees/g1-sonic
/home/user/GR00T-WholeBodyControl/.venv_teleop/bin/python scripts/g1_prepare_native.py \
  --run runtime/g1/NEW-ankle-delta-repeat \
  --native-build runtime/g1/build-stream-timing-field-v2 \
  --ankle-profile --thermal-auto-sides left right \
  --ankle-demo-source runtime/g1/E095-selected-ankle-delta \
  --thermal-rule native90 --thermal-repeat-seconds 30
```

若仅对照原示范幅度，省略 --ankle-demo-source；若对照旧规则则使用 --thermal-rule strict91 --thermal-repeat-seconds 0。不要修改 E089/E096 的冻结源或覆盖已运行日志。
