# Agent 调用家庭 VLA

VLA 是 RAX toolbox 中的可选工具。Agent 观察和选择任务，RTC 生成动作，
既有 owner 统一发送 SONIC token 和 BrainCo 目标。踝温动作在同一 owner 内优先处理。
不运行第二套 RTC 身体发布器或键盘发布器，不改变现有 SONIC 模型。

## 模型与任务

固定模型：`/media/user/My_PSSD/202_models/home_rtc_quality_bs64_8ep_final`。
启动器在 RAX `runtime/g1/vla-model` 生成配置副本和权重链接，不改 SSD 或外部仓库。
服务端提供模型路径与配置/统计/权重索引指纹，客户端在每次建立 RTC 队列时核对。
原 `pick_cola_rtc.py quality/old` 不是这个家庭模型。

| task | 模型包 tasks.json 中的训练指令 |
|---|---|
| chips | pick up the chips on the table |
| cola | pick up the cola on the table |
| mickey | pick up the mickey on the table |
| milk | pick up the milk |
| pie | pick up the pie |
| wahaha | pick up the wahaha |

数据目录 `/media/user/My PSSD/20261016` 包含这六类物品。
这个工具不把“从桌子拿起”宣称为已经学会“放到沙发/茶几”。

## 统一 Bash 启动入口（当前版本）

在本机桌面终端执行：

```bash
cd /home/user/code/rax/worktrees/g1-vla-retest
bash scripts/start_g1_experiment.sh
```

从个人仓库克隆后，在克隆目录执行同一个脚本。脚本默认使用本机已有模型、
Python 环境、Codex 登录和原生 SONIC 构建，不安装依赖、不复制认证。
原生构建路径可通过 `--native-build /绝对路径/build-stream-timing-field-v2` 指定。
默认相机接收网卡地址是 `192.168.123.76`，可用 `--receiver-ip` 修改。

启动顺序：

1. 检查依赖和重复控制进程，新建 `runtime/g1/agent-时间戳/` 并冻结本轮代码。
2. 复用正确的家庭模型服务，或后台启动它；核对模型身份及 RTC delay=8、interval=30。
3. 复用已有 RAX 相机和 eno1 电机遥测，或在后台启动；等待新鲜图像及健康遥测。
4. 后台启动本轮 MCP 和支持消息提交的 UI。
5. 打开 owner、原生控制两个实验终端，打开 Firefox，等待 `standing_ready`。
6. 后台启动 UI agent，显示“启动就绪”。此后 UI 提交任务即唤醒 agent。

模型、相机、MCP、遥测、UI 和 agent 都在后台，不额外弹出终端。
脚本不会自动进行 VLA prepare、抓取或手部开合；收到 UI 任务后由 agent 观察并
按需准备。`prepare` 会打开双手，持物时不得调用。

模型启动时间可能较长，脚本每10秒报告等待状态。日志、进程清单和源码版本保存在
本轮目录；成功启动后公共环境文件是 `runtime/g1/current.env`。
`startup-ready.json` 表示启动检查通过，不代表抓取成功。
UI 地址含本机登录 token，保存在本轮 `ui-background.log`，启动成功时也会打印。

### 只检查或只生成配置

```bash
# 只读检查，不启动、不停止进程
bash scripts/start_g1_experiment.sh --check

# 仅生成新目录与配置，不连接机器人、不启动后台服务
bash scripts/start_g1_experiment.sh --prepare-only

# 指定其他 UI 端口
bash scripts/start_g1_experiment.sh --ui-port 8783
```

`--run` 可以指定本工作树 `runtime/g1/` 下的新目录，不能复用已存在目录。
`--prepare-only` 生成的是检查用快照；正式启动时另建新目录。

### 再次启动与停止

原生控制仍在运行时，脚本拒绝重复启动，也不会自动 O。上一轮由操作者手动 O、
确认原生退出后，再运行入口即可。脚本只在核对当前进程身份和对应轮次的
`operator_keyboard_O` 正常退出记录后清理旧 owner、MCP、UI 和空终端。
仍在执行任务的旧 UI agent、未知端口占用或没有手动 O 记录的 owner 会阻止启动，
不会被强行关闭。模型和相机继续复用。

若本轮启动失败，脚本不会关闭已经启动的 owner 或原生控制。查看本轮日志，
按现场流程手动结束控制后再重试。仅在尚未尝试启动控制窗口时，脚本会释放
本次创建的 MCP/UI 后台进程，便于再次启动；模型和传感器服务保留。

会话租约默认不限时，移动速度参考固定0.3（平移单位m/s），每段 VLA 最多15秒。
原生终端的150秒、130°C、遥测过期和 owner 故障仅告警，不发送 O。
原生／硬件保护保持不变。停止由操作者手动 O 或明确授权当次停止。
不要同时启动独立 RTC runner、keyboard_pub 或另一个交互式控制 agent。

### UI 任务示例

> 抓取桌上的米老鼠。先观察，需要时移动调整位置，再调用 VLA；每段最多15秒，
> 执行后观察是否抓起。抓起后保持。

后续可输入“松开右手”“闭合左手”或“暂停 VLA”。当前入口沿用已有 Codex 登录；
场景图像和工具状态会交给所配置的 agent 模型，GR00T 权重推理在本地 GPU。
身体端口为17586/17587，模型5550，VLA相机5555，UI默认8782。

历史多终端手动入口与 E111 自动 O 启动器仅供复现实验记录，不作为当前启动方式。
完整后台命令保存在本轮 `background-services.json`，身体启动命令在 `owner.sh`
和 `controller.sh`。当前 Bash 入口的编排有离线测试及本机只读／配置生成验证；
自动编排整条链尚未据此声称通过新增实机试验。

## 工具时序与暂停

`g1_vla_status` → `g1_vla(action="prepare")` → `g1_vla_result` → 观察 →
`g1_vla(action="run", task="mickey", steps=750, evidence="当前图像中的交接依据")` →
结果 → 观察。`examples/g1/vla_controller.py` 展示标准 agent Context 调用。

prepare 是六秒身体过渡并张开双手；手中有物体时不能当作无副作用复位。
run 为 1–1500 步，50 Hz、40 步 horizon，默认 delay=8、request interval=30；
其他受训练支持的延迟和对应间隔见上文。
按用户真机调试要求，VLA 共用执行器不使用固定关节速度门槛（上身2 rad/s、
腿部3 rad/s），涵盖 prepare/run 和同一执行器内的踝温响应，避免热响应再次触发
相同中断。独立非VLA抬跟模式保留原检查；有效反馈、关节位置范围、身体倾角、
原生 watchdog 和 O 流程保留。

身体反馈兼容整数 `state_monotonic_ns` 和当前原生的一元素 `trace_monotonic_s`
源时间戳，统一检查不超过0.2秒；不会用接收时刻刷新旧样本。
请求过期、错会话、执行忙、模型不匹配或剩余会话不足都会拒绝；
推理超时/控制漏步暂停并丢弃队列，不反复播放最后一块动作。

`pause` 保持最后身体目标并停止发送新手部目标，不撤回已送达动作、不松手，
也不替代原生 O。踝温优先时暂停 VLA、丢弃队列，反馈稳定后先播报再抬对应侧。
之后不会自动恢复 VLA；需重观察，处理手中物体后才能重新 prepare。
这条组合路径尚未通过真机抓取中断试验。

## 本次验证与边界

里程碑标签：`g1-agent-vla-milestone-20261019`。Git 内保存
[E115结果](evidence/E115/experiment-result.json)、[抓取确认图](evidence/E115/held-confirmation.png)、
[工具调用记录](evidence/E115/tool-calls.json)、[模型/外部依赖指纹](evidence/E115/provenance.json)
及[本地时延摘要](evidence/E115/latency-summary.json)。完整运行日志和模型权重仍在本机。

- E115：真实Agent/MCP → VLA prepare → mickey 750步（15秒）完成，无RTC超时；
  两帧确认右手抓住米老鼠并离开桌面，用户确认接稳后O停机。单次成功证据见
  `runtime/g1/E115-mickey-vla-130/experiment-result.json`，不代表重复成功率。
- E099：通过实际 RAX MCP frontend 调用 `g1_vla`，指定家庭模型真实 GPU 推理，
  合成 RGB/统计均值状态，250 个 RTC 周期通过；输出 64D token + 6D/6D 手目标。
  无身体/手部发布器、零机器人命令。报告在 `runtime/g1/E099-vla-gpu-smoke/`。
- E100：真实相机 640×360 RGB、左右手各 6D 反馈读取通过，零命令。
  测试时原生未运行，未获得实时身体反馈；没有声称完成真机 VLA 闭环。
- 独立温度桥接五秒读到 97 组数据；现有踝温实机验收保留。
- 共 86 项 G1 测试和 6 项工作区测试通过（分批运行），包含后续会话的工具及 skill 继承。
- 工具/生命周期测试覆盖初始化、有限步结束、暂停抢先、热响应中断后禁止旧队列
  恢复、过期/错会话、忙、会话不足和推理错误。

复现 GPU 接口验证（需先启动模型服务；不会连接硬件）：

```bash
/home/user/GR00T-WholeBodyControl/.venv_inference/bin/python \
  scripts/g1_vla_smoke.py --output runtime/g1/vla-smoke-NEXT --steps 250
```

真实抓取重复成功率、相机/身体/手部联合周期统计和抓取时踝温动作仍需现场验证。
