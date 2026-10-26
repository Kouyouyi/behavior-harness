# G1 工程文件导读：从用户任务到执行、记录与经验

本文面向第一次接触 Agent 工程的读者，按 2026-10-20 本机代码说明。
当前主工程绝对路径为：

`/home/user/code/rax/worktrees/g1-vla-retest`

下文的相对路径都接在这个根目录后。例如 `harness/codex_cli/ui_agent.py`
完整路径是 `/home/user/code/rax/worktrees/g1-vla-retest/harness/codex_cli/ui_agent.py`。

## 1. 先分清代码、说明和实验记录

Agent 是根据指令和当前观察选择工具的模型；工具背后的 Python 程序负责执行。
用户输入“抓米老鼠”，当前 UI 入口会唤醒 Agent，由它选择已注册工具，
不是直接按这句话找到并运行一个同名 Python 文件。

| 文件类型 | 通常是什么 | 是否会自动起作用 |
|---|---|---|
| `.md` | 人看的文档，或者给 Agent 的操作指令 | 必须被启动器注入或被 Agent 读取；放进目录不等于被加载 |
| `.py` | 可执行程序、工具实现或辅助脚本 | 必须被调用或导入；存在不等于正在运行 |
| `.sh` | 启动命令的封装 | 执行脚本时才启动相应流程 |
| `.json` | 配置、请求、结果或路线配方 | 必须有对应程序读取 |
| `.jsonl` | 每行一条事件的日志 | 用于追踪过程；不会因为存下来了就自动成为经验 |
| `.sqlite3` | 数据库文件 | 当前消息队列使用它；不代表已建好长期经验库 |
| 图片与深度图 | 观察证据 | 要结合时间、动作和帧来源判断，不能代替当前现场观察 |

工程目录总表见 [REPOSITORY_LAYOUT.md](../../REPOSITORY_LAYOUT.md)，
多个仓库之间的关系见 [WORKSPACE.md](../../WORKSPACE.md)。

## 2. 顶层目录各管什么

```text
g1-vla-retest/
  README.md                 项目总入口
  G1.md                     G1 当前能力、部署入口与验证范围
  REPOSITORY_LAYOUT.md      简短目录地图与旧路径对应关系
  WORKSPACE.md              本机其他仓库和工作树的关系
  scripts/                  启动、配置、检查及离线处理
  harness/codex_cli/        Agent 如何启动、接收消息和调用工具
  services/                 控制服务、机器人能力与用户界面
  agent_skills/             预置操作规范
  agent_blocks/             可复用的 Agent 指令组件
  auto_research_agent/      通用工作区模板和工具客户端
  data/tasks/mickey_grasp/  米老鼠任务资产
  examples/g1/              Python 使用示例
  docs/g1/                  工程说明、设计、实验复盘与精选证据
  runtime/g1/               本机运行产物，Git 忽略
  local_tests/unit/         离线软件测试
```

`services/rby1/` 和 `docs/rby1/` 是继承的 RB-Y1 支持。
不能因为当前用 G1 就把它们当成无用文件；通用框架仍可能引用跨机器人组件。

## 3. 一条“抓米老鼠”指令怎么流转

```text
scripts/start_g1_experiment.sh
  → scripts/g1_start_experiment.py 准备并启动实验服务
  → services/console/ 接收用户在网页提交的指令
  → services/g1_sonic/messages.py 保存消息
  → harness/codex_cli/ui_agent.py 唤醒 Agent
  → harness/codex_cli/ui_bridge.py 提供工具
  → task_steps.py / replay_route.py / 直接机器人 MCP 调用
  → services/controller/frontend.py 分发底层请求
  → services/g1_sonic/backend.py 分发 G1 能力
  → 运动执行器、VLA 运行时、手部与相机服务
```

MCP 在这里可以理解为“Agent 调用工具的接口协议”。
模型决定何时观察、移动、抓取或检查；Python 程序执行工具约定的工作。
路线中的多个固定步骤可以由程序连续执行，不必每一步都让模型重新编写程序。

### 启动与配置

| 文件 | 作用 |
|---|---|
| `scripts/start_g1_experiment.sh` | 统一 Bash 入口，设置运行环境并转入 Python 启动器 |
| `scripts/g1_start_experiment.py` | 组织配置、已有服务检查与复用、实验终端和 UI 启动 |
| `scripts/g1_prepare_native.py` | 生成原生运行配置和相关脚本、准备运行源码副本；准备配置本身不等于执行抓取 |

历史运行可能使用 `runtime_source/` 中的源码副本。排查旧实验时，应先确定它
实际运行的版本，不能只看今天工作树里的代码。

### 用户界面与消息

| 文件或目录 | 作用 |
|---|---|
| `services/console/server.py` | 提供网页和 HTTP 接口 |
| `services/console/static/` | 网页的 HTML、JavaScript 和样式 |
| `services/console/runtime.py` | 界面运行状态；实机模式下通过队列传递操作消息 |
| `services/console/conversation.py` | 把消息和事件整理成界面会话 |
| `services/console/adapters.py` | 读取运行产物，供界面展示 |
| `services/console/telemetry.py`、`thermal.py` | 状态遥测与温度展示 |
| `services/g1_sonic/messages.py` | 消息持久化、领取和状态管理，底层使用 `operator-messages.sqlite3` |

网页展示“完成”需要看具体指的是消息处理完成、动作完成还是抓取确认。
它们不是同一种结果。

## 4. Agent 层：指令、判断与步骤

| 文件 | 作用与边界 |
|---|---|
| `harness/codex_cli/launcher.py` | 准备 Agent 工作区、工具配置和 `AGENTS.md`；把 G1 robot-interface 规范写入会话指令 |
| `harness/codex_cli/ui_agent.py` | 处理 UI 消息、构造提示、启动模型轮次、记录状态与输出 |
| `harness/codex_cli/ui_bridge.py` | 注册可用工具，检查当前会话，记录工具调用并转发 |
| `harness/codex_cli/task_steps.py` | `g1_task_step` 的实现，组织观察、移动、准备、抓取、检查、暂停和持物确认 |
| `harness/codex_cli/replay_route.py` | `g1_replay_route` 的实现，读取路线配方并执行，返回当前抓取检查点 |
| `harness/codex_cli/relative_perception.py` | 从当前 RGB-D、朝向等观测提取相对测量，供路线判断使用 |

`ui_agent.py` 当前组合固定操作说明、`FAST_TASK_INSTRUCTIONS`、待处理消息和
最近 12 条消息。最近消息提供短期上下文，不等于跨实验长期记忆。
当前没有扫描 `docs/g1/experiments/` 或整个 `runtime/g1/` 的经验检索步骤。

`task_steps.py` 保存动作与帧的关联，并检查操作所需条件。
例如持物确认要求相应观察证据；检查帧编号和流程条件并不等于程序拥有独立的
视觉抓取成功检测器，仍需模型对当前图像作判断。

`replay_route.py` 使用八段已整理的路线动作及当前朝向测量，
允许有限的转向纠正。平移仍是整理过的动作原语，没有实现完整的位置闭环。
历史图片用于来源审计，不是用旧图片证明当前可抓取。
路线走完和准备完成后，还要继续进行当前抓取判断与结果检查。

## 5. 机器人能力层：真正执行请求

| 文件 | 作用 |
|---|---|
| `services/controller/frontend.py` | MCP 前端：工具分发、会话连接、错误与调用记录 |
| `services/g1_sonic/backend.py` | G1 能力入口，将工具请求交给对应模块 |
| `services/g1_sonic/vla.py` | VLA 请求检查、任务提交和运行状态接口 |
| `services/g1_sonic/vla_runtime.py` | VLA 模型运行时与动作执行衔接；不是模型权重文件 |
| `services/g1_sonic/locomotion.py` | 有时限的移动任务及状态管理 |
| `services/g1_sonic/ankle_executor.py`、`executor.py` | 不同配置下的原生执行路径；具体启用哪个由部署配置决定 |
| `services/g1_sonic/hand.py` | 手部控制能力 |
| `services/g1_sonic/state.py` | 读取机器人运行状态 |
| `services/g1_sonic/rgbd.py`、`camera_service.py` | 图像、深度和相机服务 |
| `services/g1_sonic/lease.py` | 会话租约相关管理 |

这些文件让动作可执行、结果可追踪；它们没有承担自动总结历史经验的职责。
模型权重和原生构建位置由部署配置引用，通常在仓库外或本机运行目录，
并没有包含在这些 Python 文件里。

## 6. 米老鼠任务资产与离线工具

任务资产绝对路径：

`/home/user/code/rax/worktrees/g1-vla-retest/data/tasks/mickey_grasp/`

```text
mickey_grasp/
  README.md
  routes/mickey_right_side_v1/
    recipe.json
    <来源帧ID>/
      rgb.png
      depth.png
      meta.json
```

`recipe.json` 描述路线步骤、来源动作、测量、资产校验等信息。
来源动作执行过，不代表重新组合的路线已完成无人干预实机验收；
应保留配方里的验证状态。

离线脚本绝对目录：

`/home/user/code/rax/worktrees/g1-vla-retest/scripts/tasks/mickey_grasp/`

| 文件 | 作用 |
|---|---|
| `analyze_session.py` | 分析已有运行的消息、动作和耗时，帮助人工复盘 |
| `compile_right_route.py` | 从特定历史来源整理右侧路线及其证据，不是通用经验总结器 |
| `README.md` | 使用方式和路径说明 |

`examples/g1/mickey_workflow.py` 是工具使用示例，不是 UI 收到“抓米老鼠”
后自动执行的入口。其他控制示例也要通过明确的调用路径才会执行。

旧路径 `data/replay/mickey_right_side_v1/`、
`scripts/g1_analyze_task_session.py` 和 `scripts/g1_compile_right_route.py`
保留相对符号链接，指向新位置，只有一份实际内容。

## 7. Markdown 操作规范与 Markdown 复盘有何不同

`agent_skills/` 保存预置操作规范，按任务、控制、感知等组织。
当前 G1 入口明确加载：

`/home/user/code/rax/worktrees/g1-vla-retest/agent_skills/control/g1_sonic/robot-interface/SKILL.md`

启动器将它的内容写入会话 `AGENTS.md`，所以它会影响 Agent 行为。
不能由此推断其他目录中的每个 `SKILL.md` 都被这个入口加载。

`agent_blocks/` 和 `auto_research_agent/` 提供通用指令组件、工作区模板与客户端。
其中 `auto_research_agent/api/runtime.py` 的 `Context` 是程序连接工具服务的客户端。
当前 G1 UI 的指令还由 `launcher.py` 和 `ui_agent.py` 专门构造。

`docs/g1/experiments/` 则保存供工程人员查阅的复盘。
目前没有自动把这些 Markdown 内容送入下一轮 G1 UI Agent 的读取机制。

## 8. 一次实验留下什么

原始运行目录：

`/home/user/code/rax/worktrees/g1-vla-retest/runtime/g1/<运行目录>/`

常见产物如下；具体文件随启动入口和代码版本而不同。

| 相对运行目录的位置 | 内容与用途 |
|---|---|
| `operator-messages.sqlite3` | 用户消息和处理状态 |
| `agent-events.jsonl` | Agent 消息、工具调用等事件 |
| `session/mcp.jsonl` | 底层工具调用审计 |
| `session/agent-workspace/AGENTS.md` | 当时注入的操作规范 |
| `session/agent-workspace/runtime/frames/` | 实际观察的 RGB、深度及元信息 |
| `ui-agent-turns/<轮次>/prompt.txt` | 该轮实际发送给 Agent 的提示 |
| `ui-agent-turns/<轮次>/command.json` | Agent 启动参数 |
| `ui-agent-turns/<轮次>/events.jsonl`、`stderr.log`、`result.json` | 模型输出、错误和轮次结束状态 |
| `ui-agent-turns/<轮次>/compiled-task.json` | 使用分步工具时生成的步骤状态 |
| `vla_requests/`、`vla_results/` | VLA 请求与结果 |
| `move_requests/`、`move_results/` | 移动请求与结果 |
| `runtime_source/` | 运行源码副本，帮助还原当时版本 |

运行目录、用户任务、UI 轮次、抓取尝试和底层动作 job 是不同层级。
一轮模型对话可能调用多个动作；一次抓取任务也可能经过多个 UI 轮次。
`result.json` 中进程正常退出不能独立证明抓取成功。

`ui_bridge.py` 的公开事件会截短工具文本。排查细节应结合底层 MCP 日志、
动作结果与原图，不要只看界面摘要。

完整运行记录保留本机。`docs/g1/evidence/` 保存随文档归档的精选证据，
`docs/g1/experiments/` 保存人的分析，二者不能代替完整 rollout。
`G1.md` 描述当前工程入口；历史计划和运行手册须结合日期及版本阅读。

## 9. 历史经验现在如何复用

**当前有人工整理后的复用，没有接通自动经验闭环。**

```text
实验产生日志和图片
  → 人或明确启动的离线脚本分析
  → 写实验复盘
  → 开发者将有效做法写入指令、工具步骤或路线配方
  → 后续 Agent 使用更新后的指令和工具
```

例如，整理路线后让 `replay_route.py` 读取配方，就是经验的工程化复用。
但它不是系统自动浏览全部旧实验、找到相似失败、总结新经验并更新路线。
新增复盘 Markdown，也不会自动改变下一轮行为。

拟实现的流程是：

```text
确认一次尝试结束
  → 关联动作、观察、人工介入和结果
  → 复盘得到有来源和适用条件的候选经验
  → 入库，保留待验证状态
  → 下一轮按阶段、症状和部署条件检索
  → 记录使用的经验版本及实际效果
  → 根据复测和反例修订，必要时再固化成技能
```

这个流程的文件位置与验收条件见 [EXPERIENCE_SYSTEM.md](EXPERIENCE_SYSTEM.md)。
其中 `services/experience/`、`experience_tools.py`、`runtime/experience/`
及任务 `task.json`、`GUIDE.md`、`skills.json` 都是计划项，不是已经接通的功能。

与本机官方 RAX、ASPIRE、HumanoidHarnessDeploy 的具体区别也见该设计文档。
它们已有的能力并不完全相同，不能把“三方对齐”理解为复制同一套现成系统。

## 10. 测试检查什么

`local_tests/unit/test_g1_task_steps.py`、`test_g1_replay_route.py`、
`test_g1_mickey_workflow.py` 和 `test_g1_ui_agent.py`
检查步骤状态、路线流程、示例和消息调度等软件行为。
这类离线测试使用模拟后端或假模型进程，不能证明现场抓取成功率。

## 11. 想查什么，就从哪里开始

| 问题 | 起点 |
|---|---|
| 整个工程有什么 | `README.md`、`REPOSITORY_LAYOUT.md` |
| 如何启动实验 | `scripts/start_g1_experiment.sh`、`G1.md` |
| 用户消息为什么没被处理 | `messages.py`、`ui_agent.py` 和运行消息记录 |
| Agent 实际看到了什么指令 | 会话 `AGENTS.md` 和轮次 `prompt.txt` |
| 为什么选择这个步骤 | `ui_agent.py` 的提示、轮次事件与当前图像 |
| 工具怎么调用到底层 | `ui_bridge.py`、`frontend.py`、`backend.py` |
| VLA 请求如何提交和执行 | `vla.py`、`vla_runtime.py` |
| 上次为什么失败 | 对应运行的工具日志、动作结果、图片和人工反馈 |
| 已写下来的复盘在哪里 | `docs/g1/experiments/` |
| 下次是否会读到复盘 | 当前不会自动读取；查看本节经验复用说明 |
| 今后经验系统放哪里 | `docs/g1/EXPERIENCE_SYSTEM.md` |

推荐阅读顺序：项目 README → 本文 → 米老鼠任务 README →
`ui_agent.py` → `task_steps.py` / `replay_route.py` → 对应机器人能力模块 →
一份真实实验复盘与其证据。
