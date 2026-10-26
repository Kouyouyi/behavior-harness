# 本机机器人 Agent 工作区

当前项目在本机的绝对路径为：

`/home/user/code/rax/worktrees/g1-vla-retest`

项目入口：[文件导读](docs/g1/FILE_GUIDE.md)、
[目录地图](REPOSITORY_LAYOUT.md)、[米老鼠任务](data/tasks/mickey_grasp/README.md)。
以下绝对路径描述这台开发机，不是克隆本仓库后自动创建的目录。

| 绝对路径 | 本机职责 |
|---|---|
| `/home/user/code/rax` | RAX 官方索引分支、通用技能规范与本机导航 |
| `/home/user/code/rax/worktrees/g1-vla-retest` | 本项目当前 G1 UI、米老鼠抓取、步骤和路线编排 |
| `/home/user/code/rax/worktrees/g1-sonic` | 另一条 G1 工作树；当前启动器仍依赖其中的原生构建 |
| `/home/user/code/HumanoidHarnessDeploy` | G1 混合策略与复盘机制参考；部分配置引用其运行环境 |
| `/home/user/code/ASPIRE` | 技能发现、验证与维护机制参考 |
| `/home/user/code/SoL-Pi` | Agent 工具与上下文效率机制参考 |
| `/home/user/code/Isaac-GR00T-rtc` | VLA 运行所用的外部工程及环境 |

两个 G1 目录是 Git worktree，各自有分支、未提交修改和运行产物。
工作树不是普通备份，不应通过复制目录来合并开发。

```bash
git -C /home/user/code/rax worktree list
git -C /home/user/code/rax/worktrees/g1-vla-retest status --short
git -C /home/user/code/rax/worktrees/g1-sonic status --short
```

当前个人远端为 `jiachengliu3/SeekAct`，官方来源为
`robot-agent-exploration/rax`。它们承担不同角色。
移动整个工作树之前须检查原生构建、模型、运行环境及历史证据中的绝对路径。
