# SONIC 参考往返仿真对照（2026-10-18）

## 最新：用户 Pico 抬跟示范 E076–E078

用户指定 `ankle_heat` episode 13/15 后，已分别定位左/右小幅抬跟。目标沿用用户实际动作，不要求整脚离地；使用原始全身 token，而非重新生成固定上身的 IK 抬脚参考。

| 产物 | 内容与结论 |
|---|---|
| [episode 13 实测状态重建](../../runtime/g1/E076-ankle-pico-demonstrations/episode_000013_review/measured_state_reconstruction.mp4) | 真机关节＋IMU 做 FK；左后跟相对前掌增量约 10.9 mm；无物理推进，根高度/地面为显示假设 |
| [episode 15 实测状态重建](../../runtime/g1/E076-ankle-pico-demonstrations/episode_000015_review/measured_state_reconstruction.mp4) | 同上；右后跟相对前掌增量约 8.1 mm |
| [E077 左侧 token 物理仿真](../../runtime/g1/E077-sim-pico-ankle-13/review-final/announced_heel.mp4) | 正常碰撞/重力，原始 token；复现屈膝，但该初态左后跟仅升约 0.43 mm |
| [E078 右侧 token 物理仿真](../../runtime/g1/E078-sim-pico-ankle-15/review-final/announced_heel.mp4) | 右后跟抬升约 10.7 mm，前掌接触，补充返回后落稳；先实际播报，再动作 |

两段录制末尾均未完全恢复初始站姿，因此仿真追加了 3 s token 返回插值。MP4 包含实际仿真的进入、播放、返回和退出，视频由录下的物理状态渲染。录制本身没有温度字段，MuJoCo 没有热模型；此处验证动作和传输，不宣称降温。E077 漏发一个源帧，E078 全部源帧发出；接收端回读到的 token 与源记录精确匹配。手部 stock Dex3 在 idle/进入退出段存在手腿接触，示范/返回段无此接触，不能把整段当作 BrainCo 真机间隙验收。详见实验记录与各 run 的 `demonstration_comparison.json`。

目的：检查 E008 的参考播放时序与身体回位，区分参考、实际关节和腕部位置。用户同意先仿真，并补充现场可能挡住手臂；现场接触状态现记为不确定，保留之前“无接触”的原始陈述，不据此判定唯一原因。

## 实现与边界

- 入口 `scripts/g1_sim_reference.py`；复用生产 `services/g1_sonic/reference.py` 的 55 帧窗口和 7 秒、+0.03 rad 右肘包络。
- 使用 CoMotion 已有原生 `g1_sim`，校验二进制 SHA256 `bc56b32b3bb68ed2113088f36e20b085a48ba2e2a948733dcfaed02cd68e2600`。该二进制硬编码 DDS 77、接口 `lo`；真机二进制未启动。ZMQ 使用 17576/17577。
- encoder、decoder 和 planner 使用原有 ONNX；TensorRT 缓存复制到每轮目录。仿真与真机 C++ 主要区别为 DDS/网卡/CRC、trace 字段、source blend 配置；仿真 blend 显式设为同样的 0.6 秒。
- 使用标准 `scene_43dof.xml`、既有 ground-start 初态，等待测量稳定后零偏移接管；没有基座辅助支撑。它不是 E008 真机的精确初态或历史快照。
- 标准模型带 Dex3，未校准为 BrainCo 的质量、惯量和碰撞几何；没有加入家庭家具。此次由仿真实验程序直接提交同一参考生成器的输出，不是重测完整 RAX MCP/agent 链路。
- 每轮新建模拟器和控制器进程，记录源代码、配置、原始物理状态、原生 encoder/decoder 日志。结束后只清理本轮仿真进程。

## 已完成的标准模型实验

最终误差采用动作后 15–17 秒的中位数相对动作前状态。腕部指 `right_wrist_yaw_link` 原点、相对机身坐标，不是 BrainCo 指尖或已标定抓取点。

| 实验 | 接管后保持 | 右肘最大变化 | 右肘最终偏差 | 全身最大最终关节偏差 | 腕部最终位置偏差 |
|---|---:|---:|---:|---:|---:|
| E012 零偏移对照 | 6 s | 0.00738 rad | 0.00728 rad | 0.00806 rad | 0.40 mm |
| E013 +0.03 rad 往返 | 6 s | 0.00728 rad | 0.00230 rad | 0.00876 rad | 0.40 mm |
| E014 +0.03 rad 往返 | 47 s | 0.00842 rad | 0.00051 rad | 0.00825 rad | 1.57 mm |
| E016 关闭手指碰撞后往返 | 6 s | 0.00551 rad | 0.00508 rad | 0.00776 rad | 0.25 mm |
| E017 关闭上肢碰撞后往返 | 6 s | 0.01032 rad | 0.00032 rad | 0.00575 rad | 0.11 mm |

以上没有复现真机约 0.107 rad 的回位偏差；参考增量也不等于实际关节增量。零偏移自身有缓慢漂移，不能将往返中的全部位移都归因于该参考动作。末尾 0.5 秒全身最大速度分别约 0.0324、0.0391、0.0397 rad/s。

**接触复核发现：标准模型并非手臂完全无接触。** E014 在约 10 Hz 的 165 个记录状态重建中，右手食指/右髋部和左手拇指/左髋部始终有接触几何，最大穿透约 0.270 mm / 0.128 mm。这是记录姿态的几何重建，不是历史接触力测量；不能忽略这些接触来宣称已完成无接触验证。

E016 在相同质量、惯量、增益下关闭 14 个手指几何的碰撞参与，实际物理重跑完成。166 个记录状态重建中仍有 94 个包含右腕/右髋接触，最大穿透约 0.030 mm，因此追加关闭上肢碰撞的 E017。该诊断改变接触模型，不能部署，也不能声称复现了实际挡臂物体。

E017 进一步关闭手、腕、肘、肩几何的碰撞参与，保留质量、惯量和控制增益；166 个记录状态重建未发现上肢接触。该组同样没有复现真机回位偏差。它支持“当前无上肢接触仿真条件下可恢复”，不证明现场有没有接触，也不证明接触是唯一原因。各轮是从同一规定 qpos/qvel 新启进程，原生 planner 及异步时序没有做到逐状态完全一致，不能把轮间差异当作精确因果效应。

E013/E014/E016 的原生 encoder token 均有变化（最大分量变化 0.0625）；不是完全未进入策略的空参考。但参考与实际响应并非一比一，微小位移和基线漂移需分开解释。

## 播放时序

E013、E014 当前参考曲线的固定时移拟合值分别约 1.94 s、3.48 s，拟合 RMSE 约 0.00119、0.00290 rad。后者并非精确恒定延迟；不要将拟合值当作每一帧的延迟。

完整未来 q/dq 窗口在约 8.02 s、8.60 s 首次回到基线，两者都晚于发送端 7 秒结束。仿真支持保留原生回读验收；尚未证明滑窗积压/追帧的唯一根因。

## 真机误差的任务空间解释

用 E008 已记录的实测 29 个关节做模型 FK，第 7/10/12 秒的腕部相对机身位移约 18.9/18.9/19.2 mm，朝向变化约 0.357 rad（20.4°）。这是**模型估计**，并非相机测得的末端误差，也不含未标定的 BrainCo 抓取点偏置。

0.107 rad 本身不足以判定任务能否成功；应结合位置、朝向、物品尺寸和抓取容差。旧的 0.05 rad 粗调试门槛保持不变，不以事后放宽阈值将 E008 改记为通过。

## 复现与证据

工作目录 `/home/user/code/rax/worktrees/g1-sonic`，Python `/home/user/code/CoMotion/.venv-sim/bin/python`。所有 `--run` 必须用新目录。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user/code/CoMotion/.venv-sim/bin/python \
  scripts/g1_sim_reference.py --run runtime/g1/NEW-zero --case zero
PYTHONDONTWRITEBYTECODE=1 /home/user/code/CoMotion/.venv-sim/bin/python \
  scripts/g1_sim_reference.py --run runtime/g1/NEW-pulse --case pulse
PYTHONDONTWRITEBYTECODE=1 /home/user/code/CoMotion/.venv-sim/bin/python \
  scripts/g1_sim_reference.py --run runtime/g1/NEW-long --case pulse --hold-seconds 47
PYTHONDONTWRITEBYTECODE=1 /home/user/code/CoMotion/.venv-sim/bin/python \
  scripts/g1_sim_reference.py --run runtime/g1/NEW-no-hand-contact --case pulse --disable-hand-collisions
PYTHONDONTWRITEBYTECODE=1 /home/user/code/CoMotion/.venv-sim/bin/python \
  scripts/g1_sim_reference.py --run runtime/g1/NEW-no-arm-contact --case pulse --disable-arm-collisions
```

- E012 第一次启动因事件日志函数参数重名而失败，仿真子进程已清理，未启动身体策略。保留 `runtime/g1/E012-sim-zero/failure.json`；修正后的有效结果是 `E012-sim-zero-v2`。
- 每轮 `result.json`、`manifest.json`、`samples.jsonl`、`scene/trajectory.jsonl`、`controller_trace.jsonl` 和 `events.jsonl` 为原始证据。
- E014 录像 `runtime/g1/E014-sim-elbow-long-hold/review/roundtrip.mp4` 和 `contact_sheet.jpg` 由记录的 qpos/qvel 渲染；没有额外推进物理。`contact_audit/report.json` 为接触几何审计。
- `scripts/g1_compare_sim.py` 汇总数值及真机 FK；`scripts/g1_render_sim.py` 渲染；`scripts/g1_audit_sim_contacts.py` 重建接触几何。原始三组汇总是 `runtime/g1/E015-sim-comparison/report.json`。
- 最终五组汇总 `runtime/g1/E018-sim-final-comparison/report.json`。E017 的 `review/roundtrip.mp4` 明确标注关闭上肢碰撞，不能脱离该条件展示成标准模型实测。

五组对照后的离髋伸手验证见下节；BrainCo 模型、真实接触和播放积压仍需继续核对。尚未验证家庭抓取，也未重新启动真机。

## 正常碰撞下离髋伸手（E019 起）

用户同意从手臂离开髋部的姿态验证，并希望真机使用更明显的动作。E019/E020 的准备失败与修正保留在 [实验记录](EXPERIMENTS.md)，不覆盖原数据。以下新增目标不改变旧 E008 的验收结论，也未修改真机 executor。

流程：双腕相对原参考向前 8 cm、向外 8 cm、向上 6 cm，四秒平滑进入离髋姿态；等待原生完整未来 q/dq 到达目标、速度稳定与无接触，再从**实测右腕位置**规划向前 10 cm、向上 4 cm；随后回到离髋参考。全过程保留正常碰撞，不回到最初贴髋姿态。IK 只改变手臂参考，其余 15 个身体关节及根姿态参考保持，实际下肢仍由 SONIC 平衡策略控制。

预先写入每轮 `reach_criteria.json` 的任务判据：前向位移至少 7 cm、目标位置误差至多 3 cm、回位误差至多 2 cm、伸手及回位朝向变化至多 0.20 rad；每阶段速度低于 0.15 rad/s 持续约一秒、原生完整未来 q/dq 已到目标。它们是本轮仿真诊断标准，不是已经标定的抓取容差。

E021 `runtime/g1/E021-sim-clear-reach-up` 完成 27.14 秒动作流程：

| 指标 | 实测 |
|---|---:|
| 右腕前向 / 横向 / 向上位移 | 97.22 / 1.25 / 51.24 mm |
| 目标位置误差 | 11.65 mm |
| 回到离髋姿态的位置误差 | 4.15 mm |
| 伸手朝向变化 / 回位朝向误差 | 4.80° / 1.23° |
| 全身关节最大回位偏差 | 0.07834 rad |
| 整轮最大机身倾斜 | 4.71° |

该轮任务空间判据通过；关节最大偏差仍大于旧单肘试验的 0.05 rad 门槛，说明需要区分身体姿态冗余与腕部任务效果。它不证明所有 0.107 rad 误差均可忽略，更不证明 BrainCo 抓取点已准确定位。

**接触和距离审计：** 离髋前有原标准模型的手/髋接触；离髋稳定之后到结束，645 个独立记录状态重建均无上肢接触。世界坐标包围盒给出手/腕对髋/躯干/骨盆的保守距离下界：左侧至少 29.99 mm，右侧至少 23.74 mm。该审计覆盖约 50 Hz 记录，不能声称穷尽所有 200 Hz 物理子步，且不是历史接触力测量。

原始 `result.json` 的最小 `mj_geomDistance` 是 0；25 个采样点出现距离返回 0、但最近点分离的异常，其中复核的右拇指/右髋几何对没有接触。保留原值，不能把它解释为已确认接触，也不能直接忽略。上述独立包围盒下界收于 `contact_audit/report.json`；后续在线验收改用该保守下界，并要求离髋后整段采样无上肢接触、间距至少 1 cm、整轮倾斜不超过 0.25 rad。

复现：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user/code/CoMotion/.venv-sim/bin/python \
  scripts/g1_sim_reference.py --run runtime/g1/NEW-clear-reach --case reach
PYTHONDONTWRITEBYTECODE=1 /home/user/code/CoMotion/.venv-sim/bin/python \
  scripts/g1_audit_sim_contacts.py --run runtime/g1/NEW-clear-reach
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /home/user/code/CoMotion/.venv-sim/bin/python \
  scripts/g1_render_sim.py --run runtime/g1/NEW-clear-reach
```

E021 录像与四阶段关键帧：`review-v2/roundtrip.mp4`、`review-v2/contact_sheet.jpg`。第一版 `review/` 最后一张关键帧未选到，保留但不作为推荐入口。所有录像都由记录状态渲染，不追加物理推进。

本实验是原生 SONIC 接收身体参考的物理仿真，不包含 agent/MCP 调用、RGB-D 闭环、BrainCo 手部模型、家庭家具、物品抓放或真机复验。当前证据足以支持准备更可见的离髋伸手试验，不能升级为家庭搬运已完成。

### E022：修正距离判据后的完整复跑

`runtime/g1/E022-sim-clear-reach-repeat` 使用同一目标与正常碰撞模型，在线改用保守包围盒下界，并将整段离髋后接触/间距及最大机身倾斜纳入最终验收；15.80 秒完成。没有放宽 IK、位置、朝向或速度判据。

| 指标 | E021 | E022 |
|---|---:|---:|
| 右腕前伸 | 97.22 mm | 92.19 mm |
| 右腕抬高 | 51.24 mm | 49.68 mm |
| 目标位置误差 | 11.65 mm | 12.85 mm |
| 回位位置误差 | 4.15 mm | 4.46 mm |
| 回位朝向误差 | 1.23° | 0.99° |
| 全身关节最大回位偏差 | 0.07834 rad | 0.04090 rad |
| 整轮最大倾斜 | 4.71° | 1.91° |
| 离髋后上肢接触（记录采样） | 0 | 0 |
| 手/身体保守间距下界 | 23.74 mm（独立审计） | 27.20 mm（在线） |
| 任务空间验收 | 通过 | 通过 |

E022 录像见 `review/roundtrip.mp4`。两轮执行耗时和姿态响应不同，包含异步原生播放、不同阶段等待与策略历史，不能把差异全部归因于距离查询修正，也不能将两次完成视为长期重复性统计。

## 高温播报后单侧抬跟（E063–E065）

需求：先有侧别明确的音频播报，再进行对应侧脚跟轻抬、脚尖可保持接触，并提供第三视角 MP4。此处只验证动作与播报顺序，不启用真机自动响应。

`g1_sim_reference.py --case heel` 在站稳和离髋后，生成单次左右侧参考：静止 2 s、向支撑侧转移 3 s、保持 1 s、抬跟 3 s、保持 3 s、落跟 3 s、回中 3 s、停稳 4 s，追加记录 8 s。E064/E065 名义足部俯仰角 0.40 rad、名义侧移 65 mm；由两腿 IK 生成关节参考，实际物理响应单独计量。没有直接修改实际 qpos、禁用碰撞或施加支撑力。

播报为 `Simulation. Warning! High temperature at left/right ankle. Lifting the left/right heel.`，以 espeak 合成 WAV，paplay 实际播放；播放成功事件先于 heel_started，失败则不开始该动作。MuJoCo 没有电机热模型，告警为明确标记的模拟事件。真实温度/原生音频与动作的连接尚未实现。

E063 轻幅参考（0.20 rad / 35 mm）未使脚跟离地，保留失败。E064/E065 均完成实际抬跟、脚尖接触及落回：

| 轮次 | 实测平均抬跟 | 保持期前掌接触 | 对应脚承重 | 踝 pitch 力矩 RMS | 踝 roll 力矩 RMS |
|---|---:|---:|---:|---:|---:|
| E064 left | 19.92 mm | 100% | 182.6 → 126.4 N | 9.83 → 13.95 Nm | 0.46 → 1.56 Nm |
| E065 right | 28.97 mm | 100% | 172.1 → 94.1 N | 12.50 → 10.31 Nm | 0.91 → 2.01 Nm |

两轮整个动作记录中对应侧前掌均持续有接触力；最大机身倾斜分别 0.0402/0.0461 rad；末两秒漂移 0.084/0.689 mm，上肢接触样本零，外加支撑力零。末端参考回到初始值，上身参考在抬跟期间不变。两侧实际最大关节速度约 0.421/0.377 rad/s。

**动作通过不等于散热通过。** 左侧 pitch/roll 负载力矩均上升，右侧 roll 也上升；不能把减少总足底承重等同于降低过热踝电机的负载。尤其现有真机高温涉及左踝 roll，当前结果不支持将这组抬跟动作直接作为其自动散热策略。

原始状态由带只读接触/力矩记录的 `scripts/g1_sim_heel_scene.py` 采集，沿用 CoMotion 的物理/控制路径；新源哈希的说明见每轮 simulation_provenance.json。标准 G1/Dex3 仍不是 BrainCo 负载校准模型。

视频由 `scripts/g1_render_heel.py` 回放实际 qpos/qvel，没有推进新物理。左侧为全身第三视角，右侧为同一时刻足部碰撞几何近景，隐藏外观鞋底网格以看清接触；橙点标后跟、绿点标前掌。这一改变仅影响渲染，不改变物理碰撞。音频为同一次实际播报 WAV，按照 events.jsonl 的单调时钟对齐；无动作加速。

- 左侧：`runtime/g1/E064-sim-left-announced-heel/review-final/announced_heel.mp4`
- 右侧：`runtime/g1/E065-sim-right-announced-heel/review-final/announced_heel.mp4`

复现（每次换新目录，sim Python）：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user/code/CoMotion/.venv-sim/bin/python scripts/g1_sim_reference.py --run runtime/g1/NEW-left-heel --case heel --heel-side left --heel-angle .4 --heel-shift .065 --sim-build runtime/g1/build-stream-timing-sim-v2
MUJOCO_GL=egl /home/user/code/CoMotion/.venv-sim/bin/python scripts/g1_render_heel.py --run runtime/g1/NEW-left-heel
```

右侧将 heel-side 改为 right。运行需要宿主机 GPU、DDS77/lo 和可正常播放的音频服务；音频不可用时不会进入该动作。

左右合辑：`runtime/g1/E066-announced-heel-review/left_then_right_final.mp4`。最终相机置于动作侧，使对应脚更靠近观察者；关闭地面反光仅改善可见性，不改变物理。首次外观近景和第二版接触几何渲染分别保存在 review、review-contact；交付版为 review-final。

## 整脚离地卸载尝试（E071–E072）

用户指出希望解除地面接触、让踝电机放松，因此改用足部保持朝向、抬高整脚的 IK 参考。E071 名义抬高 45 mm、向支撑侧移 75 mm，仍经相同 SONIC 和正常物理：实际前掌没有离地，保持期该脚平均 128.88 N，完全卸载比例 0；踝 pitch/roll RMS 分别由 9.84/0.458 增至 14.70/0.646 Nm。动作结果为失败，不能把参考足高当作真实足高。

E072 尝试名义抬高 80 mm、侧移 120 mm，现有参考 IK 在动作发送前拒绝（左腿约 4.34 mm 位置残差、0.0557 rad 朝向残差）；无卸载动作，不放宽原有残差和关节边界。两轮均非真机动作。

E071 录像：`runtime/g1/E071-sim-left-whole-foot-unload/review-final/announced_heel.mp4`（沿用文件命名，但画面明确 WHOLE-FOOT UNLOAD: FAIL，语音为 lifting the left foot）。当前无法据此启用真机 >91°C 自动抬脚。

E073 离线筛选后，E074 使用可解的名义足高 65 mm / 侧移 80 mm 复测：保持期仍约 127.60 N 前掌承重，完全卸载比例 0，roll RMS 0.456→1.644 Nm，故仍失败。所有动作均先播放侧别明确的模拟告警。现有结果不支持将这些参考直接部署成真机过热自处理动作，也不排除人工调整脚位能改善真实负载；二者不可视作已验证等价。

## CoMotion 腿部 delta 放大（E092–E094）

左右选择 1.5 倍对应腿相对旋转增量，实际后跟抬升约 54.83/61.06 mm，返回稳定；左 2 倍约 65.21 mm 未选用。不是直接乘 token 或电机目标。完整数据、局限和真实运行区别见 EXPERIMENTS。

- [左侧 1.5 倍：第三视角与脚部近景 MP4](../../runtime/g1/E092-sim-left-delta-1p5/review-final/announced_heel.mp4)
- [右侧 1.5 倍：第三视角与脚部近景 MP4](../../runtime/g1/E094-sim-right-delta-1p5/review-final/announced_heel.mp4)

视频包含仿真播报，正常重力/碰撞、无支撑力。右侧出现短暂整脚卸载；未建立热模型。新幅度尚未经真机验证，不能把这两个高度承诺为实机高度。
