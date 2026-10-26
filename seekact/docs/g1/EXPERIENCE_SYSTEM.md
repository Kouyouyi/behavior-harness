# 米老鼠任务经验系统接入设计

状态：目录已按任务归类；下面的经验模块、工具和数据结构尚未实现。
当前已有日志记录、离线时间分析和特定来源的路线生成，尚无完整经验学习闭环。

工程根目录是 `/home/user/code/rax/worktrees/g1-vla-retest`。

## 当前复用方式与三方机制的差距

当前链路是：日志与图像记录 → 人工或显式运行离线脚本分析 → 写复盘 →
开发者把有效做法写入提示、步骤代码或路线配方 → 后续任务使用这些改动。
`docs/g1/experiments/` 不会被 G1 UI 自动扫描；最近 12 条消息也不是长期记忆。
机器人能力模块 `backend.py`、`vla.py`、`vla_runtime.py` 负责执行，不负责总结。

下面依据 2026-10-20 本机代码与规范，不能代表三个上游仓库未来所有版本。

| 项目 | 已有经验机制 | 与当前 G1 工程尚未接通的部分 |
|---|---|---|
| 官方 RAX 本机副本 | experience-memory 规范要求即时记录 `memory/TIPS.md`，启动或恢复时读 `INDEX.md` 和技巧；按条件读取技能和证据，遇反例修订；可再提升为带证据的技能及长期库条目 | 当前 G1 启动器只明确注入 G1 操作规范，没有因此自动获得这套记忆读写流程 |
| ASPIRE 本机副本 | iterative-debugging 流程提供共享技能、提案验证和使用报告；反馈区分 useful、misleading、wrong、not-applicable | 当前 G1 没有接入该技能提案、验证、反馈和修订流程 |
| HumanoidHarnessDeploy | 离线实验复盘可读取旧记忆、保存反思和候选策略；部分 G1 SONIC 实时入口在启用条件满足时加载记忆并提供反思工具 | 当前 G1 没有接入其任务和部署范围隔离、反思持久化及后续加载；固定策略入口本身也不等于会用记忆 |
| 当前 SeekAct G1 UI | 日志、人工复盘、预置操作说明、已整理步骤与路线 | 候选经验入库、阶段检索、使用反馈和反例修订尚未串联 |

因此，“尚未对齐”准确地指：尚未把这些机制落实到当前 G1 的执行入口并完成验收。
不是三个参考项目都有一套完全相同的“自动复盘 → 按阶段检索 → 效果回写”流水线。
官方 RAX 也不只是固化技能，它明确保留尚未提升为技能的局部技巧；
其 Markdown 规范要求 Agent 执行相应操作，不等同于独立后台服务必然自动运行。

HumanoidHarness 的 `RobotTaskMemory.load()` 默认取同任务、同范围的最近 6 条记录，
不是通用的语义或阶段检索器。复盘得到候选策略、通过格式检查，
也不表示该策略已经通过实机验证。

本机核对入口：

- `/home/user/code/rax/skills/experience-memory/SKILL.md`
- `/home/user/code/rax/skills/continual-learning/SKILL.md`
- `/home/user/code/ASPIRE/.claude/skills/iterative-debugging/scripts/campaign.py`
- `/home/user/code/ASPIRE/tests/test_iterative_debugging_pipeline.py`
- `/home/user/code/HumanoidHarnessDeploy/robots/g1/review_experiment.py`
- `/home/user/code/HumanoidHarnessDeploy/robots/g1/sonic_run.py`
- `/home/user/code/HumanoidHarnessDeploy/robots/g1/sonic_toolkit.py`
- `/home/user/code/HumanoidHarnessDeploy/rpent/harness/robot_memory.py`

## 待实现的代码位置

| 位置（相对工程根目录） | 职责 |
|---|---|
| `services/experience/schema.py` | 区分事实、技巧、失败模式、假设、恢复方法及验证范围 |
| `services/experience/ingest.py` | 将 run、用户任务、UI batch、动作 job 和证据关联；一轮 UI 不等于一次抓取 |
| `services/experience/store.py` | 经验版本、来源和证据存储；幂等导入及可重建索引 |
| `services/experience/review.py` | 从已结束的尝试与图片生成有来源的候选经验，标明人工介入及未知项 |
| `services/experience/retrieve.py` | 按任务、当前阶段、失败症状和部署兼容性选取经验，返回适用条件及未知项 |
| `services/experience/curate.py` | 合并重复结论，记录反例、修订范围和复测状态 |
| `harness/codex_cli/experience_tools.py` | 经验查询、证据读取和使用反馈的 MCP 工具 |
| `scripts/g1_import_experience.py` | 导入已有实验 |
| `scripts/g1_review_experience.py` | 单次离线复盘 |
| `scripts/g1_experience_worker.py` | 后台处理已结束任务的复盘队列 |

待新增任务定义位于 `data/tasks/mickey_grasp/task.json`、`GUIDE.md` 和 `skills.json`。
创建这些文件时须同时实现消费者与格式检查；文件存在不代表 Agent 已加载。

## 现有接入点

- `harness/codex_cli/ui_agent.py`：在任务开始时注入短摘要，结束后排队复盘。
- `harness/codex_cli/ui_bridge.py`：注册查询、取证和反馈工具，沿用当前会话检查。
- `harness/codex_cli/task_steps.py`：关联动作、帧和执行结果，保留既有动作检查。
- `harness/codex_cli/replay_route.py`：作为已注册执行能力；新经验不能直接覆盖路线。

历史经验只提供判断依据。当前任务指令、工具契约、实际观察和模型部署兼容性仍需检查。
将“记录合规”“执行完成”“视觉确认”“无人干预成功”和“跨场景复现”分别记录。

## 计划中的数据目录

默认本机路径：`/home/user/code/rax/worktrees/g1-vla-retest/runtime/experience/`。
实现时允许配置稳定的数据根目录，避免工作树变化造成经验丢失；当前尚无该配置接口。

```text
runtime/experience/
  index.sqlite                         可从记录重建的查询索引
  tasks/mickey_grasp/
    episodes/<任务执行ID>/             manifest、timeline、attempts、evidence
    lessons/<经验ID>/v001.json          有来源、适用条件和状态的经验版本
    validations/                       独立复测及反例
```

原始日志继续在 `runtime/g1/<运行>/`。重要证据须有稳定副本或可解析的归档引用，
不能依赖未来可能被归档删除的临时文件。记录不确定的任务关联，避免把不同任务或
一次人工辅助成功混成多次独立成功。

计划将提供给 Agent 的经验快照和反馈保存在
`runtime/g1/<运行>/ui-agent-turns/<轮次ID>/memory-context.json` 和
`memory-feedback.jsonl`，以便比较使用经验前后的实际效果。

## 第一阶段验收

1. 同一实验重复导入，不增加重复证据。
2. 动作结束、观察到持物、人工报告和未知结果分别记录。
3. 失败重试能检索到相关旧案例，并显示仍需当前确认的条件。
4. 每次使用经验可以追溯版本、动作和结果；反例能影响后续检索。
5. 复盘进程只处理记录，不启动动作；待验证候选经独立检查后才成为可执行技能。
