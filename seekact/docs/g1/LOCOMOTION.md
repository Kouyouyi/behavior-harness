# G1 基础移动接口

`g1_move` 将八方向平移、左右原地旋转、原地踏步集中在一个 MCP 工具中。
方向以**该段开始时机器人朝向**为准；平移期间朝向参考固定，斜向单位向量归一化。
参考实现：`HumanoidHarnessDeploy/robots/g1/sonic_actions.py`、`sonic_contract.py`，
以及 `HumanoidHarnessDeploy929/robots/g1/obstacle/turning.py` 的旋转朝向渐变。

## 调用契约

| action | 输入 | 返回 |
|---|---|---|
| status | 只有 action | owner 状态、移动能力、限值、当前任务、上身保持目标 |
| run | direction、frame_id、evidence；可选 speed、duration_s、mode、yaw_rate_deg_s | job_id、submitted |
| result | job_id | pending_or_unknown 或任务结果 |
| stop | 只有 action | 停止请求 job_id；随后查询 result 确认停稳 |

run 的 direction：`forward`、`backward`、`left`、`right`、`forward_left`、
`forward_right`、`backward_left`、`backward_right`、`rotate_left`、`rotate_right`、`step_in_place`。

| 参数 | 默认与范围 | 语义 |
|---|---|---|
| mode | 平移默认 walk；旋转/踏步默认且仅允许 slowWalk | SONIC 模式 2 / 1 |
| speed | 按用户要求固定为 0.3；可省略，显式传值只接受 0.3 | planner 的步态/速度参考，平移单位 m/s；不等于实测速度；agent 不再调整此参数 |
| duration_s | 默认 2；平移/踏步 0.1–3，旋转 0.1–2 | 从移动阶段开始计时，交接和停稳另外计时 |
| yaw_rate_deg_s | 旋转默认 15，范围 1–30 | 朝向目标变化率，deg/s；乘以 duration_s 不超过30度 |
| frame_id | 必填 | 本会话 g1_observe 保存的新图像，最长30秒、每帧仅可提交一次；提交时相机仍须新鲜 |
| evidence | 必填，1–2000字符 | 实际观察到的路径、手与物体状态；后退时不能把前视画面当作后方无障碍证据 |

前进示例（把 frame_id 替换为实际观测 ID）：

```python
g1_move(action="run", direction="forward", speed=0.3, duration_s=2,
        frame_id="<g1_observe返回的frame_id>", evidence="当前路径与手部状态的实际观察")
g1_move(action="result", job_id="<返回的job_id>")
```

原地左转：

```python
g1_move(action="run", direction="rotate_left", speed=0.3,
        yaw_rate_deg_s=15, duration_s=2,
        frame_id="<新frame_id>", evidence="当前转身空间与持物状态的实际观察")
g1_move(action="stop")
```

旋转指定的是朝向参考渐变，实际角度由 IMU 反馈积分报告。
接口没有里程计，`actual_distance_m=null`；不得用速度×时间冒充实际移动距离。
`completed=true` 只表示定时执行、源确认和反馈停稳，不代表到达目标或仍抓牢物体。
结果中的 `upper_tracking_max_error_rad` 报告该段上身实际关节相对保持目标的最大误差。

## 上半身与手部保持

第一次移动或从 VLA token 切换时，从新鲜实测状态锁存腰部3关节、双臂14关节。
通过 SONIC 原生 `upper_body_position/upper_body_velocity` 字段持续发送17关节位置和零速度，
覆盖 planner 的上身参考，保留下肢运动规划。先发 idle，再等待原生 planner 和上身参考回读确认，
随后执行移动。到时或停止后取消未完成的转向目标，维持当前朝向与同一上身目标。
连续移动不重新锁存漂移后的实际关节值。

这是全身策略跟踪下的参考保持，不是将上半身焊死或逐电机改写输出。
实际跟踪偏差由仿真及观测评估。工具不发送手部指令，原 BrainCo owner 的目标继续保持；
仿真使用 stock G1/Dex3，无 BrainCo 或真实持物负载，因此不能据此宣称物体保持验证通过。

## 执行与 VLA 交接

唯一身体出口仍是 `ankle_executor`。VLA 运行中拒绝移动；需要先 pause 并观察。
移动任务未停稳时拒绝 VLA prepare/run。移动接受时废弃旧 VLA 队列并使 preparation 失效，
不自动恢复。重新 prepare 会打开手，因此持物时不能直接用它恢复 VLA。
停稳后仍可直接调用 `g1_hand` 开合左/右/双手，无需重新 prepare；它保持现有身体参考。
踝温处理先打断移动、确认停稳再执行原有动作。遥测、故障及租期检查只影响上层动作；不得因此自动发送 O 或终止原生控制，停机决定权归用户。
planner 上身保持不改变原有全身踝温示范的语义。

`run` 和 `stop` 都是异步请求，不存在第二套身体或手部发布器。
参数、观测年龄、会话 ID、队列冲突、原生模式/参考回读、停止后的反馈均有检查。
停稳超时进入 owner 故障，不自动重试。退出 Agent 会尝试停止正在执行的移动；原生 O 仍独立。

## 部署与复现

完整 agent 运行使用 `--agent-tools`，等价于 `--vla --locomotion`：

```bash
python scripts/g1_prepare_native.py --run runtime/g1/<新的运行目录> \
  --native-build /home/user/code/rax/worktrees/g1-sonic/runtime/g1/build-stream-timing-field-v2 --agent-tools
```

此命令仅生成脚本，不启动硬件。旧运行目录及旧 owner 不会自动获得此能力。
Codex 白名单随工作区工具表包含 `g1_move`，故需使用新会话配置/重启工具连接。

隔离仿真固定 DDS domain 77、lo、端口17576/17577，使用实有 SONIC 模型、
自由根、正常重力和接触，测试阶段外部支撑为零：

```bash
/home/user/code/CoMotion/.venv-sim/bin/python scripts/g1_sim_reference.py \
  --case locomotion --sim-build runtime/g1/build-stream-timing-sim-v2 \
  --run runtime/g1/<新的仿真目录> --move-mode walk --move-speed 0.3
MUJOCO_GL=egl /home/user/code/CoMotion/.venv-sim/bin/python \
  scripts/g1_render_locomotion.py --run runtime/g1/<仿真目录>
```

保留 `result.json`、`trials.json`、`samples.jsonl`、原生回读、MuJoCo完整轨迹，
以及 `third-person/locomotion-upper-hold.mp4`、逐方向截图和视频解码校验信息。
渲染只回放实际记录的物理状态，不生成或修改运动轨迹。本次导入的既有仿真结论与来源见 [证据](evidence/locomotion/provenance.json)，不是本轮重新执行的实机验证。

已保存的第三视角视频：

- [E117：离髋姿态，八方向与旋转](/home/user/code/rax/worktrees/g1-sonic/runtime/g1/E117-sim-locomotion-upper-hold/third-person/locomotion-upper-hold.mp4)。
- [E119：抬臂姿态，八方向与旋转](/home/user/code/rax/worktrees/g1-sonic/runtime/g1/E119-sim-carry-pose-walk03/third-person/locomotion-upper-hold.mp4)。

两组均观察到正确移动方向。抬臂组上身参考固定，但实际关节误差最大0.285 rad、
手腕位置漂移最大2.64 cm，未通过本轮0.15 rad跟踪误差诊断。
详见 [E119 诊断](evidence/locomotion/E119-validation.json)。因此目前只确认上身参考保持，
实际持物姿态精度仍需改进，并另做带负载验证。
