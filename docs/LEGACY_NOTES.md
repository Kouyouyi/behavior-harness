## 99 任务主动感知扰动测试（2026-10-19，进行中）

`configs/ood99_manifest.json` 与训练 parquet 的 99 个任务集合已逐项核对一致。每任务配置位于 `configs/ood99/`：明确目标类别、全部同类目标实例、原始支撑/容纳关系、扰动种子和实例 301。终点仍为找到并接近，不是完成完整原任务。18 项配置存在容纳关系祖先，只有运行时确认祖先可开合才执行关容器操作，其余为原支撑面重新摆放后的搜索。

本批固定 checkpoint_024000、每轮预测 32 步执行 8 步、最多 800 个实际物理步、100 个 VLA 批次。导航最多沿用 2 批，纯转向 4 批，开容器 12 批；每批重新获取观测并调用 VLA。每任务一个实例、一个种子，不能视为该任务的稳健成功率。

初始场景先经过独立进程的几何与三路相机不可见检查；闭环重建时比对所有目标和机器人位姿，容差 2 cm 和四元数差 0.02，配置必须逐字段一致。多目标验收允许同类任意一个，但距离和可见性必须来自同一个物体。目标物体只在初始化重摆放，执行阶段不传送物体或机器人。

本机旧语义分割映射在 `picking_up_trash` 场景可复现原生崩溃；分阶段挂接也失败。此批改用安装版 Replicator 的 `instance_id_segmentation_fast`，以完整 USD 物体子树匹配网格实例 ID，仍统计实际渲染像素而非几何可见性替代。初始/终点 `.npy` 掩码和 ID→prim 映射保存在私有审计目录，绝不传给 planner。已在首个真实场景验证映射包含三个易拉罐的网格路径且像素均为零。成功仍要求 planner 声明 reached、同一个目标至少 16 像素且水平距离不超过 1 m。

执行状态与结果：`outputs/ood99_20261019_v5/summary.json`、`index.html`。每项分别记录预检失败、基础设施失败、预算内未成功和成功。此前 `ood99_20261019`、`_v3`、`_v4` 是修复过程证据，不作为独立任务结果。尚未跑完全部 99 项，不应把 pending 算失败。原任务成功独立保留。

局限：这些是人工规则生成并待真实验证的任务相关场景扰动，不声称已证明与全部训练布局统计不重叠；采用官方放置采样和稳定后关系检查，尚无独立的三角网格穿透深度验收。大批运行需要保持本地仿真、模型服务及 SSH 隧道在线。`STOP_AFTER_CURRENT` 文件可让批量脚本在当前任务结束后停止，已提交动作不自动重放。

## 本次真实验证结论（2026-10-18）

最终检查：23 项测试通过，2840 个真实物理步，20 个视频均完整解码且符合 H.264 High / avc1 / yuv420p。所有已提交动作与预测前缀逐项一致，无未确认提交。

**尚未全通过，不能声称已完成自主搜索开柜任务。**

- 陌生位置有效搜索轮次 `search_unfamiliar_20261018_03` 从初始视野外扫描发现收音机，执行 352 个真实步，终点目标有 2422 个头部相机像素，水平距离 1.000414 米。因超过严格 1.0 米阈值约 0.414 毫米，保留失败结论，没有舍入为成功。
- 全房间闭柜轮次出现上层持续扫描、绕路以及规划通信超时；未得到完整开柜成功证据。
- 柜前局部诊断分别测试 checkpoint_010000 与 checkpoint_024000，并测试持续开柜指令；它们使用更近的初始位置，不能替代全房间搜索结果。
- `fixed_drawer_024000_20261018` 是明确标记的固定指令诊断，不是自主视觉规划。在更近的起点，持续 400 个真实步（13.33 秒）执行“拉开抽屉”；抽屉关节未发生有效变化，终点距离 0.655473 米，但三路相机目标像素均为 0。排除上层换目标和等待后，仍未完成开柜。
- 所有结果由 `scripts/audit_rollout.py` 复核原始预测前缀、动作确认数和视频完整解码；汇总由 `scripts/summarize_suite.py` 生成 `outputs/real_simulation_report_20261018.json`。这些是参数迭代与故障隔离试验，不应计算为独立 benchmark 成功率。

剩余问题需要继续排查：柜前接近姿态、目标高度和抓持阶段的策略表现，以及训练示范与当前柜体/相机分布的匹配。关节跟随记录、训练分位检查与预处理检查没有显示明显的整体动作错序或归一化错误，但不能因此断言所有物理接口与视觉分布都已完全等价。固定指令诊断的机械臂跟随误差中位数约 0.0079 rad、95 分位约 0.0421 rad、最大约 0.1435 rad。

额外诊断参数：`--opening-commit-chunks 12` 表示一次开柜语言目标最多保持 12 个新观测 VLA 批次（3.2 秒）；默认仍为 1。`--diagnostic-open-drawer` 生成固定诊断指令，完全不调用视觉规划器，并在终点进行分割审计。`--initial-offset-x`、`--initial-offset-y` 仅修改初始位姿；执行中不传送物体或机器人。不得将这些诊断配置的结果冒充原始起点的自主任务成功。

024000 的配置、归一化统计与数据契约已与 010000 逐字节比对一致，READY 标记完整，真实 HTTP 推理输出有限的 32×23 动作。权重哈希见 `outputs/checkpoint_024000_weight_sha256.txt`。使用 `scripts/remote_server.sh 024000` 与 `scripts/tunnel.sh 18770` 启动其服务和隧道，再传 `--checkpoint-step 24000`；客户端拒绝未核验权重或回合内切换权重。

## 2026-10-18 真实仿真复测与修复

实际试验分别保存在 `outputs/search_unfamiliar_20261018_*` 与 `outputs/search_closed_20261018_*`，每个已结束轮次的 `acceptance.json` 包括动作逐项比对、提交/确认计数、原始遥测、终点审计和视频完整解码结果。通信失败、人工中止诊断轮次和未达验收标准的轮次保留原结果，不能当作成功。

新增 `--initial-yaw-offset-deg 150` 用于陌生位置搜索：目标先移到电视柜，评测器最后一次初始化后改变底盘初始朝向；经过 30 个初始化稳定步检查朝向误差，然后才开始记录任务步。策略执行中不传送机器人或物体。前三张机载 RGB 与 `initial_heading_validation.json` 保留起始证据。

默认 `--planner-max-chunks 4` 是新的分层规划配置：纯转向指令最多保持 4 个 VLA 批次，其他导航最多 2 个，开容器每批规划。每个 VLA 批次仍从当前 61 维状态与三张 RGB 重新推理 32×23 动作，只执行前 8 步，剩余丢弃。上层沿用指令时标记 `reused_instruction` 和原证据步号，不能视为新的视觉判断。使用 `--planner-max-chunks 1` 可复现旧的逐批上层规划配置。

上层使用已有 Codex 登录，通过 HTTPS 请求；默认继承用户模型，也可用 `AP_HARNESS_PLANNER_MODEL=gpt-6.1-sol` 显式选择。每个新规划结果记录模型和代码哈希。网络失败最多重试一次，每次限时 85 秒，仿真暂停且不提交动作；不完整响应不执行。重复失败会封存视频并结束为基础设施错误。只在完整 `turn.completed` 和答案已写出后结束 CLI 收尾等待，并验证无工具调用和训练指令契约。

真实末端审计前先封存视频，再添加实例分割；之后禁止物理步进。约一米是严格工程阈值，无隐式舍入或容差。规划提示采用约 0.8 米的视觉停止余量，但独立验收仍为 1.0 米与至少 16 个目标像素。

运行示例（已有远端服务和 SSH 隧道时）：

```bash
cd /media/cody/软件/behavior_harness
/home/cody/.codex/skills/environment/scripts/sudo_run.sh env AP_HARNESS_PLANNER_MODEL=gpt-6.1-sol bash scripts/run_sim.sh --output outputs/new_search_run --ood radio_on_tv_cabinet_facing_room --initial-yaw-offset-deg 150 --max-steps 1600 --max-rounds 200 --planner-max-chunks 4
/home/cody/.codex/skills/environment/scripts/sudo_run.sh env AP_HARNESS_PLANNER_MODEL=gpt-6.1-sol bash scripts/run_sim.sh --output outputs/new_closed_run --ood radio_inside_closed_container --container-name bottom_cabinet_rhdbzv_0 --max-steps 1600 --max-rounds 200 --planner-max-chunks 4
/home/cody/.codex/skills/environment/scripts/sudo_run.sh /home/cody/miniconda3/envs/behavior-2026/bin/python scripts/audit_rollout.py outputs/new_search_run outputs/new_closed_run
```

20 项测试已通过，包含原始动作不变、当前观测刷新、开柜不复用、失败规划可重试而工具事件不可重试。历史测试数以下按当时版本保留。

## 2026-10-18 终点确认：找到并接近目标

用户确认无需抓取、放置或开启收音机。当前终点验收要求：规划器基于当前 RGB 报告 reached，停止所有物理动作后，独立实例分割确认任一机载相机中精确目标对象至少 16 像素，且底盘与目标中心水平距离不超过 1.0 米。1.0 米为当前工程默认值，并非训练数据或 benchmark 的官方阈值；不证明机械臂可达或无障碍接触。

结果字段 `find_and_approach_success` 与 `official_task_success` 分开。前者才是此次用户目标，后者保留原场景任务遥测。`perception_reached` 仅表示规划器提出到达判断。距离或可见性不满足时记录 `arrival_verification_failed`；不继续执行动作。终点分割启用后禁止再次调用物理步进。原生崩溃或审计异常属于基础设施失败，不视为找到目标。

16 项单元测试通过，包括近但不可见、可见但太远、独立验收成功但原始任务未完成。新的终点审计尚未在真实到达轨迹中验证；此前 32 步普通场景和 8 步闭柜场景仅证明调用链路，不能作为搜索到达成功率。终点动态启用分割仍可能触发已知 Isaac 错误，保留原始失败记录。

# 本地主动感知 Harness + huoshan π0.5

本地目录：`/media/cody/软件/behavior_harness`。仿真复用 `behavior-2026` Conda 环境和本机已有 BEHAVIOR 资产。模型只运行在 huoshan GPU 1；不安装远端 Isaac、不复制权重到本地、不改动训练。

2026-10-17 验证：14 项针对性测试通过；本地 fixture 预处理、远端严格加载、真实视觉规划和 SSH 模型推理通过。原场景完成 4 轮/32 步，闭柜场景完成 1 轮/8 步，每步提交与模型前缀逐项一致，两轮双路录像均完整解码通过。两轮都是短预算诊断，原生 success=false，尚未证明搜索、开柜或完整原任务成功。汇总见 `outputs/validation_report.json`。失败的摆放和原生渲染崩溃已独立保留。

## 输入与调度

当前头部/左腕/右腕 RGB → 本地 Codex 视觉规划 → 冻结训练 renderer → SSH 转发 → huoshan π0.5 → 原始 32×23 动作 → 本地同步执行前 8 步 → 新观测。

完整 61D 状态由官方 `robot_r1::proprio` 提供。使用交接包原版 `r1pro.yaml`，包含原有控制器、关节顺序、base 缩放和 grasping_mode。模型端完成一次归一化/反归一化；本地不裁剪、不再次缩放、不插入全零动作。物理频率为 30 Hz，模型和规划等待期间不调用物理 step。每轮余下 24 步永久丢弃。仿真执行失败、推理超时、模型身份改变或陈旧观测均中止，不自动重放。

视觉 planner 使用当前登录的桌面用户及其默认模型设置。工具、MCP、插件、规则和记忆已在 CLI 注册层禁用；模型只得到三张当前 RGB、用户目标、允许的动作组合和最近八轮历史。原生成功、目标位置、分割及场景配置单独记录，绝不进入规划请求。旧 harness 代码已按来源哈希冻结到 `runtime/`，不再依赖旧实验代码的后续修改。

## 能力边界

支持训练覆盖的 31 种动作组合，包括搜索方向、接近、抬高/降低观察位置，以及单独开门、抽屉和盖子。开容器指令的目标是容器，打开后重新观察，再定位原目标。隐藏位置不传给 planner。

**当前训练契约没有任意搬障、抓取、放置、按开关的动作标签。** `reached` 表示 planner 认为已到达目标的可观察/交互位置；harness 会停止当前感知阶段，并保留环境原生成功值，不伪报原任务完成。搜索、开柜能力是否成功取决于模型闭环表现，接口和单元测试通过不构成任务成功。要可靠执行完整操作任务，需要后续操作策略或相应训练覆盖。

`radio_inside_closed_container` 是额外的场景生成入口。要求显式选择容器，采样失败、关门后不满足 Inside 或目标包围盒越界时拒绝运行。抽屉从打开到关闭的位置变换仅发生于初始化，并保留目标的抽屉局部坐标；策略运行时不移动场景物体。

本机 Isaac 的 `omni.syntheticdata` 在启用分割并推进物理时出现原生段错误，启动时创建分割相机也曾崩溃。因此实际动作轮使用 RGB；实时 `target_visible` 记录为 null。可使用 `--initialize-only --initial-visibility-audit` 单独验证三路实例分割遮挡；在该审计中出现任何目标像素会拒绝闭柜场景。它不能与物理执行同时启用。已有一次静态审计测得三路目标像素均为 0，但不能替代后续轨迹的逐帧可见性。当前也没有独立穿透深度审计，不能将此入口称为已完整验证的 OOD 基准。

## 启动

先检查远端 GPU 1 空闲显存，避免和新的远端任务冲突。以下服务和隧道分别保持在独立终端，结束后 Ctrl-C。

```bash
cd '/media/cody/软件/behavior_harness'
bash scripts/remote_server.sh
```

```bash
cd '/media/cody/软件/behavior_harness'
bash scripts/tunnel.sh
```

健康检查：`curl http://127.0.0.1:18769/health`。若服务和隧道已就绪，不要重复启动。默认严格固定 `checkpoint_010000`，更换检查点必须重新配套 metadata、归一化、tokenizer 和验收，不应只改服务路径。

运行本地闭环（输出目录必须未存在）：

```bash
cd '/media/cody/软件/behavior_harness'
/home/cody/.codex/skills/environment/scripts/sudo_run.sh bash scripts/run_sim.sh \
  --output outputs/search_new_01 --max-steps 800 --max-rounds 100 \
  --ood radio_on_tv_cabinet_facing_room \
  --goal '寻找收音机，必要时探索可见通路，观察并接近收音机。'
```

已有仿真资产位于 `/root`，因此仿真复用受控 sudo；视觉规划子进程自动切回 `cody`，不复制任何凭证。运行的是原生 R1Pro，无虚拟云台。首次场景加载通常需要几分钟。800 步是本地诊断预算，不是官方成功率评测。

`--initialize-only` 仅加载并记录初始观测，不执行 VLA。成功初始化后的 `sim/container_inventory_private.json` 列出可打开容器，只供测试场景配置使用。闭柜入口示例（将实际名称替换为该清单中的容器）：

其他官方任务可指定 `--task-name`、`--instance-id`、`--target-key` 和 `--goal`；`target-key` 只用于独立评分，不传给 planner。内置重摆放与闭柜场景仅针对收音机，其他任务必须另做保持任务语义的场景设计，不能直接套用闭柜扰动。

```bash
/home/cody/.codex/skills/environment/scripts/sudo_run.sh bash scripts/run_sim.sh \
  --output outputs/occluded_new_01 --max-steps 800 --max-rounds 100 \
  --ood radio_inside_closed_container --container-name ACTUAL_CONTAINER_NAME \
  --goal '寻找并接近收音机；目标不可见时探索可见通路，必要时检查柜子内部。'
```

先用 `--initialize-only --initial-visibility-audit` 验证新容器摆放；再用全新的输出目录运行 RGB 动作轮。采样器只在初始化阶段使用，不在策略执行过程中移动物体。

## 验证与审计

```bash
cd '/media/cody/软件/behavior_harness'
PYTHONPATH=. /home/cody/miniconda3/envs/behavior-2026/bin/python scripts/verify_local.py
/home/cody/miniconda3/envs/behavior-2026/bin/python -m unittest discover -s tests -v
```

- `outputs/local_preprocessing.json`：本地完整 fixture 预处理对照。
- `outputs/remote_acceptance.json`：远端冻结包严格加载和真实 GPU 推理验收记录。
- `outputs/remote_http_validation.json`：真实视觉指令经 SSH 返回模型动作，包含延迟；未执行物理动作。
- `outputs/planner_fixture/`：真实视觉规划输入、输出和 CLI 事件。
- 每次运行 `source_manifest.json`、`run_config.json`：源文件哈希和参数。
- `control/round_*/`：当前 RGB、61D 状态、VLA 完整 prompt/token 长度、32 步预测、每步提交及返回确认、执行后观测。
- `control/summary.json`：感知阶段停止原因；`sim/result.json`：原生 `BehaviorTask.success` 和仿真终态。
- `scorer_private.json`：独立目标位置和容器打开状态，不作为模型输入；像素可见性仅在静态审计中测量，物理执行期间为 null。
- `sim/videos/`：H.264 High / avc1 / yuv420p 机器人和第三视角录像。每个实际物理步对应一帧。

源码接口使用 [Codex 非交互模式](https://learn.chatgpt.com/docs/non-interactive-mode) 的图像附件和 JSON schema 输出，并额外进行本地语义校验。项目交接原文见 `harness_handoff.md`；冻结适配包完整性结果见 `sha256_check.log`。
