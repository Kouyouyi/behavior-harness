# G1 主工程文件布局

本机当前工程根目录：

`/home/user/code/rax/worktrees/g1-vla-retest`

从[米老鼠任务入口](data/tasks/mickey_grasp/README.md)查看当前任务。
逐层讲解见[工程文件导读](docs/g1/FILE_GUIDE.md)。
运行方式见 [G1.md](G1.md)，启动入口仍是 `scripts/start_g1_experiment.sh`。

## 现有目录职责

| 相对工程根目录的位置 | 保存什么 | 何时修改 |
|---|---|---|
| `harness/codex_cli/` | Agent 启动、UI 工具桥接、步骤和路线编排 | 修改 Agent 如何调用能力 |
| `services/g1_sonic/` | G1 状态、运动、手部、VLA 和相机服务 | 修改机器人能力实现 |
| `services/controller/`、`services/console/` | MCP 前端、UI 和服务连接 | 修改通信或用户入口 |
| `agent_skills/` | 按任务、控制、感知分类的预置操作规范 | 修改给 Agent 的工具使用说明 |
| `auto_research_agent/` | RAX 通用工作区模板及客户端 | 修改通用模板；当前 G1 UI 的指令还由 launcher 与 ui_agent 构造 |
| `data/tasks/mickey_grasp/` | 米老鼠任务导航、路线配方及来源帧 | 增加或修订任务资产 |
| `scripts/tasks/mickey_grasp/` | 米老鼠离线分析和路线生成 | 修改复盘辅助工具 |
| `scripts/` 其余入口 | 启动、部署、仿真和设备检查 | 按各脚本用途维护 |
| `examples/g1/` | 控制器调用示例 | 编写示例；不把它当作 UI 自动选择的入口 |
| `docs/g1/` | 接口、部署、现状和设计说明 | 修改工程文档 |
| `docs/g1/experiments/` | 按实验归档的复盘 | 新增历史事实和解释 |
| `docs/g1/evidence/` | 文档引用的精选图片和统计 | 保存可随代码携带的证据 |
| `runtime/g1/` | 本机完整实验、运行快照和日志 | 运行时生成，Git 忽略 |
| `local_tests/unit/` | 离线回归测试 | 验证软件行为 |
| `services/rby1/`、`docs/rby1/` | 继承的 RB-Y1 后端和说明 | 保留跨机器人支持及依赖 |

`agent_skills/` 是预置规范；自动从实验中提炼的长期经验尚未接入当前 G1 UI。
后续目录与接入点见 [EXPERIENCE_SYSTEM.md](docs/g1/EXPERIENCE_SYSTEM.md)。

## 当前调用链

```text
UI 用户消息
  → harness/codex_cli/ui_agent.py
  → harness/codex_cli/ui_bridge.py
  → task_steps.py / replay_route.py / 已注册机器人 MCP 工具
  → services/g1_sonic/
  → 当前 owner 与底层运行时
```

`replay_route.py` 读取 `data/tasks/mickey_grasp/routes/mickey_right_side_v1/recipe.json`。
路线完成后仍需当前抓取判断，不能把路线返回视为抓取成功。

## 2026-10-20 目录归类

| 旧位置 | 新位置 | 兼容方式 |
|---|---|---|
| `data/replay/mickey_right_side_v1/` | `data/tasks/mickey_grasp/routes/mickey_right_side_v1/` | 相对符号链接，共用同一份资产 |
| `scripts/g1_analyze_task_session.py` | `scripts/tasks/mickey_grasp/analyze_session.py` | 相对符号链接，命令和导入保持可用 |
| `scripts/g1_compile_right_route.py` | `scripts/tasks/mickey_grasp/compile_right_route.py` | 相对符号链接，命令和导入保持可用 |
| `docs/g1/AFTERNOON_TASK_COMPILATION.md` | `docs/g1/experiments/mickey-grasp-20261020.md` | 旧文档保留导航页 |

现有实验日志、证据内容、示例模块路径、启动入口、原生构建和模型位置保留。
历史证据中的源码路径、哈希和运行目录描述的是实验当时的状态，不随目录整理改写。
本次归类没有创建经验数据库，也没有启用自动复盘或自动检索。

## 本机工作树关系

`/home/user/code/rax` 是官方索引分支。
`/home/user/code/rax/worktrees/g1-sonic` 是另一条保留中的开发工作树，
当前启动器的默认原生构建仍来自其 `runtime/g1/build-stream-timing-field-v2`。
工作区总览见 [WORKSPACE.md](WORKSPACE.md)；两个工作树分别查看 Git 状态。
