# 米老鼠任务离线脚本

本机目录：

`/home/user/code/rax/worktrees/g1-vla-retest/scripts/tasks/mickey_grasp/`

| 文件 | 输入 | 输出 |
|---|---|---|
| `analyze_session.py` | 已保存的 UI、MCP 和动作结果 | 任务时间、调用和执行结果统计 |
| `compile_right_route.py` | 已人工复核的来源运行 | 路线配方与 RGB-D 来源帧 |

这两个脚本只读已有实验并写指定的输出目录，不连接机器人、不发布动作。
路线生成器当前针对已复核的十段来源动作取前八段，不能用于任意运行的自动经验提炼。

从工程根目录运行，例如将新结果写到本地分析目录：

```bash
python3 scripts/tasks/mickey_grasp/analyze_session.py \
  runtime/g1/agent-20261020-141425 \
  --output runtime/analysis/mickey_grasp/afternoon-timing.json

python3 scripts/tasks/mickey_grasp/compile_right_route.py \
  runtime/g1/agent-20261020-162610 \
  --output runtime/analysis/mickey_grasp/right-route-candidate
```

路线输出目录必须尚不存在。复核并测试后的资产再放入
`data/tasks/mickey_grasp/routes/`，避免覆盖现有路线或把新生成内容自动当成已验证技能。

旧入口 `scripts/g1_analyze_task_session.py` 和
`scripts/g1_compile_right_route.py` 通过符号链接保留；新文档使用本目录路径。
