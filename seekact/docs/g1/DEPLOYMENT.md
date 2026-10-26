# G1 真机部署

当前集成工作树为 `worktrees/g1-vla-retest`，分支为 `fix/g1-ui-hand-release-no-auto-stop`。实时数据、模型、录像和冻结实验目录放在
`runtime/`；代码、部署说明和小型抬脚 token 资产纳入 Git。

可选家庭 VLA 的统一部署与 agent 工具见 [VLA.md](VLA.md)。`--vla` 自动启用以下踝温默认值。

当前 UI 实验统一使用 `bash scripts/start_g1_experiment.sh`：
两个实验终端和 Firefox，其余服务后台运行。`--check` 只读检查，
`--prepare-only` 只生成新快照。下面的独立踝温命令用于该调试 profile；
完整 UI + VLA + 移动 + 手部工具的启动步骤见 [统一入口](VLA.md#统一-bash-启动入口当前版本)。

## E109 实测问题（2026-10-19）

E109 密集双侧响应期间左踝绕组达到130°C并报告电机故障。当前动作结束后保持
示范起始token，而不是恢复动作前站姿；成功回位不代表卸载或散热有效。
10秒配置是当前实现，不应解读为已验证的持续散热策略；返回/等待姿态及双侧
调度问题尚未修复，不能仅凭E096的单侧结果认定密集双侧重复可靠。
详见 [实验复核](EXPERIMENTS.md#e109-温升复核2026-10-19)。
温度原始两列按宇树SDK为外壳[0]/绕组[1]，历史字段名和界面误称绕组/驱动。

## 默认踝温响应

用户已在真机确认本功能好用，后续部署保留：左右各自触发、先播报再动作、
CoMotion 腿部 delta gain=1.5、成功回位后最短间隔 10 秒。
采用原生报警迟滞：单电机绕组/驱动任一 >=90°C 进入报警，<85°C 清除；
一侧任一踝关节仍处于报警时可再次响应。不是每次五秒原生语音都排队抬脚。
本机播报完成是动作前提；没有监听机器人扬声器音频。

E096 日志记录两次左侧温度动作完成并停稳，用户报告实际效果良好。
这不是量化散热实验；不把仿真的 5–6 cm 脚跟高度当作真机测量。
选定动作已存入 `services/g1_sonic/data/ankle_relief`，不依赖 E095 临时目录。
清单保存 token 文件 SHA256、源示范 SHA256 和幅度。示范 13 为左，15 为右。

## 准备与启动

以下在本工作树根目录执行。准备只生成冻结配置，不启动真机：

```bash
python3 scripts/g1_prepare_native.py \
  --run runtime/g1/ankle-NEXT \
  --native-build runtime/g1/build-stream-timing-field-v2 \
  --ankle-profile --thermal-auto-sides left right \
  --thermal-rule native90 --thermal-repeat-seconds 10
```

每轮使用新目录。恢复双脚承重、吊装防跌落且不限制动作，现场有人可按 O 后，
分别开三个终端（第三个只读温度，每秒刷新）：

```bash
bash runtime/g1/ankle-NEXT/owner.sh
bash runtime/g1/ankle-NEXT/controller.sh
bash runtime/g1/ankle-NEXT/temperature-watch.sh
```

原生窗口执行现有初始化流程。owner 在原生 Init Done 后启动站立并监测。
保持阶段允许人工调整脚位；动作前等稳定，不关闭状态、倾角、原生看门狗。
会话默认不限时（owner `--lease-seconds 0`），不再使用原来的 600 秒期限。结束时先按现场流程支撑，
再在原生窗口 O 停止，确认原生退出后关闭 owner。

### 会话租约管理

`g1_session_lease(action="status")` 查询租约；`action="set", seconds=0` 取消期限；
若用户明确要求限时，可用 `seconds>=30` 从当前时刻重新计时，没有 600 秒上限。
提交后用 `action="result", job_id="..."` 确认 owner 已应用。
不限时状态表示为 `lease_unlimited=true, lease_remaining_seconds=null`，不是剩余 0 秒。

只有单纯租约到期造成的 `fault_idle` 可以恢复：owner 先检查新鲜健康反馈及稳定状态，
然后回到 `ready`，不会自动续跑已结束的 VLA、移动或手部动作，也不重启原生控制。
硬件、姿态、通信故障不能通过租约接口清除。VLA 每段仍为用户要求的 15 秒；
原生通信看门狗和硬件保护保持不变。旧冻结 owner 没有此接口，必须在操作者
手动 O、确认原生退出后换用新冻结运行，不能热换发布器或仅编辑状态文件。

`g1_prepare_native.py` 的历史参考/步态 profile 是独立调试入口，不能与此 owner
同时运行。后续任务入口应复用该 owner 的踝温响应，禁止启动第二个身体发布器。


## 试验窗口管理

新生成的 `owner.sh` 在启动身体 owner 前调用冻结的终端清理器。它只处理本仓库
`runtime/g1/*/window-*.sh` 打开的 GNOME 窗口：关闭已退出控制/agent 的空窗口和
旧的只读温度、决策监视器；保留本轮窗口、仍在运行的模型/相机及其他未知进程。
检测到任一原生控制器或身体 owner 仍在运行时跳过清理，不代替原生 O 停机。
原始日志和试验目录不删除；普通用户终端不处理。

可在工作树根目录手动预览或执行清理：

```bash
python3 scripts/g1_manage_terminals.py --runs-root runtime/g1
python3 scripts/g1_manage_terminals.py --runs-root runtime/g1 --apply
```

保留指定试验全部窗口可加 `--keep-run runtime/g1/试验目录`。模型和相机服务继续复用，
新试验最多显示 owner、Native O 两个终端和 Firefox UI；模型、相机、MCP、温度及决策服务在后台运行。若相机复用旧目录，
`backend-options.json` 的 `camera_directory` 直接指向该目录。历史冻结 owner 不会自动更新。

高温持续时，10 秒从上一次动作**成功回位**开始计算；间隔到期且报警仍有效，
再次播报同侧并动作，可连续多次。两侧串行，等待身体稳定可能延后动作；
报警解除则停止，播报失败、回位失败或 owner 锁定不属于正常重复条件。

## UI messages and single-hand release (2026-10-19)

The console's default `readonly` mode does not accept chat. To attach the live
operator inbox, add `--message-directory "$RUN"` (or `message_directory=RUN`
to `Console`) using the same current run as the MCP backend's
`control_directory`. The UI sends an idempotent request ID, persists the text,
and shows queued, read and acknowledged states. It never maps text directly to
motor commands. A running agent reads `g1_operator_messages` and receives
pending instructions with ordinary G1 tool results. It acknowledges them with
a public reply before submitting new movement. Pause remains available.
The inbox alone stores messages. The background UI agent described below now
wakes Codex automatically; it does not depend on an active chat in the coding app.
Pending instructions expire after 120 seconds or an owner-session change.

The integrated owner exposes `g1_hand(side="left"|"right"|"both",
action="open"|"close", evidence="...")` and `g1_hand_result(job_id)`.
Selected hands blend to configured poses over two seconds. Any unselected hand
retains its commanded target; no body reference is replaced. The existing paired
BrainCo bridge remains the only hand publisher. Fresh feedback, settled idle
owner and known hand targets are required. Initial VLA prepare establishes those
targets; after locomotion, hand tools work without another body reset/prepare.
Pause active VLA or stop/settle movement first. Observe after completion: a
completed command does not prove grasp/release. Right-hand opening has two live
UI trials; left/both and closing are offline-tested only.

`g1_move` adds bounded forward/backward/left/right, diagonal, rotation and stepping
primitives; see [LOCOMOTION.md](LOCOMOTION.md). Its `stop` means planner idle,
never native shutdown. New integrated runs must enable both capabilities:

```bash
python3 scripts/g1_prepare_native.py --run runtime/g1/agent-NEXT \
  --native-build /home/user/code/rax/worktrees/g1-sonic/runtime/g1/build-stream-timing-field-v2 \
  --agent-tools
```

This only freezes scripts/configuration; it starts no hardware. Old frozen owners
cannot gain these tools by restarting MCP alone. Never overwrite trial snapshots.

The operator prohibits automatic native O shutdown. The terminal relay now
forwards O only from its operator keyboard. The 150-second threshold, 130 C,
stale telemetry, owner faults and legacy STOP_O files produce warnings, never
an injected O. Startup health checks and native protections remain. An agent
may pause upper-level VLA commands, but must obtain explicit permission before
closing native motion control by O or any equivalent means. Historical runtime
launchers/logs retain their original behavior for provenance and must not be
reused as current launchers.

Real UI/hand-release evidence: `docs/g1/evidence/ui-hand-release-20261020/`.
The run exceeded 150 seconds with `automatic_stop=false`, then exited with
`operator_keyboard_O`, code 0. The requested backward steps were not executed:
that historical frozen VLA owner exposed no backward locomotion tool. The current
source now integrates it; physical validation remains outstanding.

## UI 提交后自动唤醒 Agent（2026-10-20）

在当前 UI、共享 MCP frontend、身体 owner 分别就位后，后台运行：

```bash
/home/user/GR00T-WholeBodyControl/.venv_data_collection/bin/python \
  -m harness.codex_cli.ui_agent \
  --config "$RUN/operator-input.json" \
  --socket /tmp/rax-g1-milestone-retest.sock
```

新 `g1_prepare_native.py --agent-tools` 还生成 `ui-agent.sh`；设置
`RAX_MCP_SOCKET` 为该 frontend 的 socket 后调用即可。这个脚本不生成 operator
配置，也不启动身体/原生/第二个 MCP。后台运行它，不增加桌面终端；同一 session
不要同时运行交互式 Codex。现有 UI `--message-directory` 与配置中的
`control_directory` 必须指向同一运行目录。

UI 显示 **Agent 待命 · 发送任务自动开始** 时，直接输入“抓取桌上的米老鼠”。
后台 Codex 会读取/回复消息、观察、选择 VLA/移动/手部工具、等待结果并回报。
“抓取”授权包含必要准备及后续抓取，不能仅做 prepare 就退出。本轮返回后，
下一条消息会自动开始新一轮；无需到 Codex 聊天里另说“开始”。
执行中补充指令由 agent 在工具返回、等待和15秒分段边界读取；不是硬实时抢占。
仍由唯一 owner 串行动作。UI 待命不等于机器人就绪：手动 O 后不能自动重启。

实现采用 `codex exec --json --ephemeral`，复用已有登录及模型配置，输出公开回复和
实际 MCP 事件；不展示模型内部推理。依据 [OpenAI 非交互模式文档](https://learn.chatgpt.com/docs/non-interactive-mode)。
每轮独立上下文含最近收件箱历史，重新观察当前状态；不恢复并重放旧动作。
同一运行目录及 session 有排他锁，消息 request_id 去重、持久化派发记录、owner
会话校验；服务中断后不自动重放已派发指令。启动失败/推理失败会公开报告，
需要根据现场重新提交。默认单轮处理上限240秒，退出只可暂停上层动作，绝不 O。
UI 代理还在 MCP 转发层限制 VLA 每段最多750步，禁止其它工具和跨 owner 会话调用。

验证使用真实 Codex + 独立假机器人 MCP + 历史图像回放：自动执行 prepare/run，
随后第二次 HTTP 提交再次唤醒。本测试没有实机动作，不代表实机成功率。

复现离线模型验收（仅假接口，不连接机器人）：

```bash
/home/user/code/Isaac-GR00T-rtc/.venv/bin/python \
  local_tests/unit/smoke_g1_ui_agent.py --run runtime/g1/ui-agent-smoke-NEW
```
