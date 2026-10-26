# Humanoid Agent · SeekAct Robotics · G1 16:9 展示大屏

面向电视与宽屏展示的单屏界面，基准画布 1920×1080，可等比适配 1080p、4K 等尺寸。
**全部 29 个身体电机温度同时显示，不分页、不滚动、不需要点击展开。**
非 16:9 窗口使用留边，保持内容完整和比例不变。

下一轮接入 Agent 请先读 [交接与接口说明](HUMANOID_AGENT_HANDOFF.md)，其中区分已实现接口与后续调度设计，并提供 [可运行的事件样例](examples/agent-events.example.jsonl)。

## 启动与展示

在 `worktrees/g1-sonic` 目录运行（Python 3.10+，服务只需标准库）：

```bash
bash scripts/run_g1_console.sh
```

打开终端输出的 `http://127.0.0.1:8780/?token=...` 完整地址。使用浏览器 F11 或
右下角的全屏按钮进入大屏。`Ctrl+C` 结束服务；端口冲突时使用 `--port 8782`。

界面固定展示：

- 顶部用户原始指令、任务状态、Agent 当前动作和用时；不预设任务拆解或百分比。
- 左上第一视角；右侧贯穿主体高度的 Codex 风格会话：用户气泡、处理状态、公开摘要、工具卡片和逐字回复。
- 左下紧凑温度区：左右臂各 7 个、左右腿各 6 个、腰部 3 个电机，共 29 个。
- 右下固定文本输入框、语音输入按钮和发送按钮；左上品牌为 Humanoid Agent，下方标注 SeekAct Robotics。
- 每个电机的外壳 / 绕组温度、黄色预警、红色告警或灰色失效状态。
- 数据来源、连接状态和时间。

保留单屏布局，无侧栏、分页、筛选器和详情展开。任务控制与记录导出仍保留 HTTP 接口。
输入框提交到 `/api/v1/messages`。匹配搬运样例的输入会启动完整交互演示，其他输入得到演示会话确认；实机只读模式可以编辑草稿，
发送按钮停用并明确提示发送服务尚未接入。

默认演示模式自动开始「把桌上的米老鼠放到沙发上」，42 秒完成六个阶段，停留 8 秒后重新
演示。不需要观众操作。通过 API 暂停或取消后不会自动恢复；使用 `--no-autoplay` 可关闭
自动播放。发送消息后会停止自动创建下一轮演示，当前演示仍可结束，对话不会被后续演示反馈覆盖。实机只读模式永不自动创建任务。

演示使用历史照片、模拟温度和确定性任务流程，**不调用真实模型或发送机器人动作**。
温度中包含一个黄色、一个红色的踝部示例，以展示告警效果。它们不是实测温度。
照片来自：
`runtime/g1/E050-walk-grasp-survey/ed08c869fc074ea3a9cb0894a2d83381/rgb.png`。
`services/console/static/observation.png` 是原样副本；目标框是演示标注，非实时检测。

## 文本与语音输入

- Enter 发送，Shift + Enter 换行；中文输入法的候选确认回车不会发送。
- 消息最多 1000 字；发送期间锁定编辑和重复提交。失败或断线保留草稿，断线禁用发送。
- 普通演示回复明确标记尚未连接真实 Agent。会话保留最近四轮历史，不将演示确认当作机器人执行成功。
- 输入「把桌上的米老师玩偶放到沙发上」会启动模拟搬运；原文完整保留，计划注释明确将「米老师玩偶」映射到米老鼠玩偶样例。米老鼠 / mickey 到沙发 / sofa 同样支持。新样例输入会结束当前模拟任务并开始新一轮，不会控制真机。
- 计划仅在会话中出现，顶部没有预设步骤。每步依次呈现公开判断摘要、模拟工具调用、返回数据和结果回复；42 秒后完成模拟验收。步骤计划和工具反馈为确定性脚本，不是模型实际规划。
- 点击麦克风启动中文识别，再次点击结束；临时识别结果会替换更新，最终文字留在草稿中，**不自动发送**。
- 识别中有醒目标识，输入暂时只读。30 秒后停止，离开页面时中止录音；权限拒绝、无语音、设备不可用、网络失败及浏览器不支持时都有文本提示。

语音使用浏览器的 `SpeechRecognition` / `webkitSpeechRecognition`，遵循
[Web Speech API](https://webaudio.github.io/web-speech-api/) 的 start / stop / abort 与结果回调。
需要浏览器提供识别能力并授权麦克风，服务可能在线处理音频；启动提示和按钮说明对此作了标注。
服务器不接收或存储音频，仅在点击发送时接收转写文本。真实麦克风与识别服务效果仍需在展示电脑验收。

## 温度显示规则

规则由后端 `services/console/thermal.py` 统一计算，前端直接展示 `level`，不另设阈值。
基于已有 `temperature_watch.py` 的颜色阈值与 `ankle_thermal.py` 的 native90 踝部迟滞逻辑：

| 状态 | native90（默认） | strict91（兼容旧模式） |
| --- | --- | --- |
| 正常 | 两列温度最大值 <85°C，且无保留告警 | 最大值 <85°C |
| 黄色预警 | 最大值 ≥85°C 且 <90°C，且无保留告警 | 最大值 ≥85°C 且 ≤91°C |
| 红色过热 | 最大值 ≥90°C | 最大值 >91°C |
| 红色电机故障 | 有效样本 `motorstate != 0`，不依赖温度 | 相同 |
| 红色告警保持 | 踝部曾达到 90°C 后，必须降到 <85°C 才解除 | 不启用 native90 迟滞 |
| 灰色失效 | 样本缺失、时间过期或浏览器断线 | 相同 |

温度使用 **max(外壳, 绕组)**，不是只看绕组。
故障、过热、告警保持和预警均有文字标注，避免只靠颜色辨认。
过期值可以保留查看，但标灰、不计为当前有效温度；过期样本不能清除已保留的踝部告警。
踝部迟滞在本服务观察到阈值越界后建立，服务重启不会推测此前未知的告警状态。

若存在新鲜 `ankle_status.json`，优先采用 owner 的 `thermal_rule`；否则使用启动参数：

```bash
bash scripts/run_g1_console.sh --thermal-rule native90
# 兼容旧严格阈值：
bash scripts/run_g1_console.sh --thermal-rule strict91
```

图例随实际规则更新。此模块仅负责展示，不触发任何抬脚或保护动作，不修改原控制器。

## 实机观测（只读）

```bash
bash scripts/run_g1_console.sh --mode readonly \
  --camera-directory /absolute/path/to/camera-output \
  --telemetry-file /absolute/path/to/console-telemetry.json \
  --owner-file /absolute/path/to/run/ankle_status.json \
  --agent-events-file /absolute/path/to/agent-events.jsonl
```

所有路径由本地操作员指定。控制台只读取文件，不启动相机、采集器、Agent 或 owner。
缺失数据保持 unavailable，任务/对话提交及动作接口返回 409。
真实模型、持续任务驱动和运动执行尚未接入。自动演示成功不代表实机任务成功。

### 相机

兼容 `services.g1_sonic.camera_service` 的 `latest.json` 和 `<frame_id>/rgb.png`。
前端每次完成请求后等待 250 ms 再拉取状态；图像随上游新 frame_id 更新，当前是观测预览，不是 30 fps 视频。
依据 `received_monotonic`、`host_boot_id` 校验同机同次开机且 ≤1 秒；不会以文件修改时间
刷新旧图。32 位十六进制 frame_id 和解析后的文件路径均被校验。
后续可用 WebRTC/MJPEG 替换画面通道，保留快照证据接口。

### 29 电机只读采集器

在已安装 Unitree SDK 的环境中单独运行：

```bash
PYTHONPATH="$PWD" /path/to/unitree-env/bin/python -m services.console.telemetry \
  --interface eno1 --output /absolute/path/to/console-telemetry.json
```

只订阅 `rt/lowstate`。DDS 回调时保存源采样时间和 boot ID，每 0.2 秒原子写入 JSON；
断流不会刷新源时间戳。没有构造运动命令发布器。

```json
{
  "schema_version": 1,
  "source": "live_lowstate",
  "sampled_at": 1792400000.0,
  "monotonic": 12345.0,
  "host_boot_id": "same-host-boot-id",
  "motors": [{
    "hardware_index": 0,
    "name": "left_hip_pitch_joint",
    "temperature_case_winding_c": [35, 48],
    "motorstate": 0,
    "tau_est": 1.5
  }]
}
```

实际采集包含 29 个身体电机；不伪造尚未接入的手部电机温度。
旧字段 `temperature_winding_driver_c` 兼容读取，但明确按 SDK 顺序显示为外壳 / 绕组。
缺少可信时钟或 boot ID 的旧日志只作为过期值展示；源样本超过 2.5 秒标灰。

### Agent 事件

读取 App Server `item/started`、`item/completed` 中的 `mcpToolCall` 和 `agentMessage`，
或归一化 JSONL：

```json
{"id":"obs-1","time":1792400000.0,"kind":"summary","title":"抓取检查","text":"玩偶仍在桌面，抓取未通过，需要重新定位。"}
```

允许 kind：`summary | tool | observation | verification | error | task | message | plan`。

自定义任务通过 `task` 事件传入完整任务快照，不依赖搬运步骤。每条任务相关事件都携带
同一个 `task_id`；切换任务后不会沿用上一任务的回复或工具记录。示例：

```jsonl
{"id":"t1-update","task_id":"t1","kind":"task","status":"running","details":{"instruction":"检查窗户是否关闭，然后向我报告","current_step":"查看窗户状态","elapsed_s":12}}
{"id":"tool1","task_id":"t1","kind":"tool","tool":"inspect_window","status":"running","details":{"arguments":{"location":"window"}}}
{"id":"reply1","task_id":"t1","kind":"message","role":"assistant","time":1792400000.0,"status":"streaming","text":"我正在检查"}
{"id":"reply1","task_id":"t1","kind":"message","role":"assistant","time":1792400001.0,"status":"completed","text":"我正在检查窗户，稍后报告结果。"}
```

外部 Agent 可追加任意步骤计划（最多展示 12 步），与样例的六步拆解无关：

```jsonl
{"id":"plan1","task_id":"t1","kind":"plan","text":"检查窗户并报告","details":{"steps":[{"id":"look","label":"查看窗户","status":"completed"},{"id":"report","label":"报告结果","status":"running"}]}}
```

`plan.details.steps[].status` 为 `pending | running | completed | failed`。
`state.conversation.plan` 和 `plan_note` 驱动会话内计划；模拟工具结果在 `conversation.tool.result`
显式包含 `simulated: true`，另记录 `tool_result` 事件用于复核。

`task.status` 支持 `running | paused | completed | cancelled | failed | waiting`；未知值显示
状态未知。没有用时或当前动作时显示等待，不猜测进度。`message.text` 是累计全文，复用
消息 ID 更新；不是 token delta。生产端须写入当前 UNIX 秒时间，streaming 标记超过 5 秒
没有新消息即停止。文件是公开展示事件源，请仅写入适合展示的摘要和工具参数。
只读模式标识为事件记录，不把日志中的历史动作伪装成实时模型思考。
原始 reasoning blocks 不读取、不展示；公开判断摘要与工具记录直接呈现在大屏。
每次只读取文件尾部至多 256 KiB、120 行，输出最多 60 条事件；进程内保留最近任务元数据，避免长回复挤掉任务标题。生产端应周期性追加完整 task 快照，保证展示服务重启后仍能从尾部恢复任务。此适配器不管理 App Server 生命周期。

## HTTP API v1

接口沿用前版，不依赖屏幕操作。首次访问设置 HttpOnly、SameSite=Strict cookie 后移除 URL
中的 token。所有资源和 API 需 cookie 或 `X-Operator-Token` 认证，只绑定 loopback。
POST 需 `application/json` 且请求体 ≤16 KiB；错误形如 `{"error":"..."}`。

| Method | Path | 行为 |
| --- | --- | --- |
| GET | `/api/v1/state` | 聚合展示快照 |
| GET | `/api/v1/events?after=123` | 本控制台持久化事件游标 |
| GET | `/api/v1/history` | 最近结束的演示任务 |
| GET | `/api/v1/camera/frame?frame_id=...` | 已发布相机帧 PNG |
| POST | `/api/v1/tasks` | 演示任务提交，带 request_id 去重 |
| POST | `/api/v1/tasks/{id}/actions` | 演示 pause / resume / cancel |
| POST | `/api/v1/messages` | 搬运样例触发模拟任务；其他消息进入演示会话，不提交机器人动作 |
| GET | `/api/v1/export` | 下载当前快照 JSON |
| GET | `/api/v1/schema` | OpenAPI 3.1 定义 |

`state.conversation` 包含 `source`、`phase`、通用 `task`、`user`、`summary`、`tool`、
处理耗时 `elapsed_s` 和 `output: {id, text, streaming}`。手动会话的 `phase` 为 `replied`，
`history` 最多包含当前轮之前的 8 条演示消息。前端每 250 ms 拉取快照，
按同一 ID 增量展示回复，不反复重播。演示依次呈现思考状态、模拟工具调用、逐段输出；
暂停和断线冻结动态状态，支持系统减少动态效果偏好。对话区自动跟随最新输出，
长指令与长回复仅在会话区域内部滚动，不挤占底部温度区域。
`state.task` / `stages` 保留旧演示协议；新展示优先使用 `conversation.task`。
真实自定义任务由外部 Agent 执行服务负责，这里先提供观测展示接口；`POST /tasks`
仍仅运行内置搬运样例，不能据此声称任意任务已接通执行。

`state.presentation` 包含 `aspect_ratio`、基准宽高与 `autoplay`。
`state.telemetry.joints` 始终包含 29 个电机，并增加：

```json
{"id":4,"case_c":65,"winding_c":87,"max_c":87,"fresh":true,
 "level":"critical","reason":"alarm_latched","alarm_latched":true}
```

`level` 为 `normal | warning | critical | unknown`，`reason` 为
`temperature | motor_fault | alarm_latched | stale | null`。
telemetry 还包含 `counts`、`thermal_rule`、`rule_source` 和 `thresholds`。
客户端断线时必须把旧数据转为 unknown，而非持续显示旧的健康状态。

提交任务：`{"instruction":"把桌上的米老鼠放到沙发上","request_id":"client-uuid"}`。
同一进程内相同 request_id 和指令返回同一任务；不同指令复用 ID 或并发新任务返回 409。
去重缓存最多 200 条，不承诺跨进程去重。动作：`{"action":"pause"}`。
暂停冻结演示时长，取消终止当前自动播放；关闭服务记录 interrupted，重启不重放旧任务。
默认自动播放重启后可创建全新演示任务。
演示完成仅设置 `simulated_verified: true`，`verified` 始终 false。

`state.events` 在只读模式来自 Agent 日志；`events?after=` 始终来自控制台自身日志，
两者游标不可混用。`capabilities` 保留用于后续操作端接线；`physical_motion` 始终 false。
记录保存在忽略目录 `.local-runtime/g1-console/demo/` 或 `readonly/`。
原 `scripts/run_dashboard.sh` 保持独立。

远程观看可通过 SSH 转发：

```bash
ssh -L 8780:127.0.0.1:8780 user@workstation
```

## 配套交互演示视频

成片：`.local-runtime/console-preview/Humanoid-Agent-SeekAct-Robotics-demo.mp4`。
本次成片 56.4 秒、约 3.8 MB，1920×1080、30 fps、16:9、H.264 MP4，无音轨，带品牌片头和片尾；真实浏览器操作录屏，过程为：
输入指定原文 → 发送 → 展示计划 → 逐步处理状态与公开判断摘要 → 六种工具模拟调用与返回 → 模拟验收。
画面采用历史实拍样本，温度采用全部正常的合成数据；不代表机器人动作录像。
视频同时保留 WebM 原片、`.trace.json` 事件快照和 `.json` 章节时间及媒体检查结果。

可重复录制（另需 Playwright、Chromium、ffmpeg 与 ffprobe）：

```bash
python3 scripts/record_g1_console_demo.py
# 使用已有 Chromium 时：
python3 scripts/record_g1_console_demo.py --browser /absolute/path/to/chrome
```

录制器只启动临时 loopback 展示服务，真实调用 `/api/v1/messages`，验证每一步工具、
反馈和最终模拟验收标识；录制结束自动关闭本机服务。不会连接模型、机器人或麦克风。

## 验证

```bash
env -u PYTHONPATH /path/to/test-env/bin/python -m pytest -q \
  local_tests/unit/test_g1_console.py local_tests/unit/test_dashboard.py
```

本轮 53 项 Python 检查通过；录制时验证全部六种工具的调用与返回、6/6 计划完成、模拟验收和全部电机温度，无页面错误。MP4 已完整解码并抽查片头和验收帧。此前输入与温度回归包含 1920×1080、3840×2160、
1366×768、非 16:9 留边、全部 29 电机可见且无裁切、自动播放、黄红边界、故障标红、
踝部迟滞、owner 规则切换、数据过期及断线恢复、中文输入法、发送接口、历史对话、失败保留草稿，以及语音开始/停止、权限拒绝和不支持降级，无页面 JavaScript 错误。
本轮预览在 `.local-runtime/console-preview/composer-1080p.png`，输入后效果为 `composer-message.png`，语音状态图为 `composer-voice.png`；语音图使用模拟识别回调。
测试使用合成数据，没有连接模型、麦克风服务或运行真机动作。语音回调测试不能替代实际麦克风识别验收。
