# π0.5 权重适配与 Harness 交接

日期：2026 年 10 月 17 日。目标是在另一台机器复现当前模型的输入处理、加载与动作输出，先通过离线接口验收，再接模拟器。当前阶段不要求实现 live planner 或运行闭环实验。

参考检查点为 checkpoint_010000，已完整保存并完成 200 条固定样本离线评测。训练仍在继续，计划 30,000 步；这不是最终模型。更换检查点时，必须同时替换其 config.json、norm_stats.json、dataset_contract.json，并确认 READY.json 中 complete=true。

## 文件交付

训练主机 SSH 别名为 huoshan。适配包：

    /cs/kyy/behavior_ap_experiment/handoff/pi05_harness_20261017.tar.gz

参考权重目录：

    /cs/kyy/behavior_ap_experiment/runs/pi05_goal_horizon_v2_b8/checkpoint_010000

推理需 model.safetensors（约16GB）、config.json、norm_stats.json、dataset_contract.json、READY.json。training_state.pt（约25GB）仅续训需要。适配包不包含模型权重或全量数据。

包内有 scripts、冻结 vendor/lerobot、实际 tokenizer、checkpoint_metadata、r1pro.yaml、真实输入 fixture、环境版本清单、来源说明和 SHA256SUMS。PaliGemma tokenizer 随包提供，无需另下载其约11GB基础模型权重。

在新机器上执行，若未配置 huoshan 别名则替换为可访问的 SSH 地址：

    mkdir -p ~/behavior_harness
    cd ~/behavior_harness
    scp huoshan:/cs/kyy/behavior_ap_experiment/handoff/pi05_harness_20261017.tar.gz .
    tar -xzf pi05_harness_20261017.tar.gz
    cd pi05_harness_20261017
    sha256sum -c SHA256SUMS
    mkdir -p checkpoint
    rsync -av --progress \
      --include='/model.safetensors' --include='/config.json' \
      --include='/norm_stats.json' --include='/dataset_contract.json' \
      --include='/READY.json' --exclude='*' \
      huoshan:/cs/kyy/behavior_ap_experiment/runs/pi05_goal_horizon_v2_b8/checkpoint_010000/ \
      checkpoint/

包内校验值不包含单独传输的模型文件。在源端和目标端分别执行 sha256sum model.safetensors 并比较。不要加载正在写入的检查点；仅复制 checkpoint_metadata 不会得到权重。

## 环境与代码

源环境为 Python 3.12、torch 2.7.1+cu118、torchvision 0.22.1+cu118、transformers 5.5.4、numpy 2.2.6、OpenCV 4.13.0、safetensors 0.8.0、huggingface-hub 1.22.0、pandas 2.3.3、pyarrow 24.0.0、accelerate 1.14.0、einops 0.8.2。environment_versions.json 是环境快照，不是跨平台可直接安装的 lockfile。根据目标驱动选择兼容环境，记录与源版本的差异。

在适配包目录设置：

    export PYTHONPATH="$PWD/vendor:$PWD/scripts"
    export HF_HUB_OFFLINE=1
    export OMP_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=4
    python -c 'import lerobot; print(lerobot.__file__)'

路径必须指向包内 vendor/lerobot。不要自行替换成另一个 LeRobot commit、openpi 或通用 processor。当前是完整 π0.5 权重，不是 LoRA adapter。

包内执行器仅将 tokenizer 改为随包定位，并允许 tokenizer_path 参数。planner_contract 的覆盖列表改为包内当前训练分布。训练源码与权重未修改；复制时遗漏的悬空链接是其他机器人/策略的说明文档。

参考执行方式是 float32 参数 + CUDA bfloat16 autocast，cache_enabled=False。不要默认把整套权重转换到 BF16 后仍有相同数值。16GB只是权重体积，推理还需额外显存，应实测目标 GPU。VLA 推理无需 RTX；OmniGibson 图像仿真另需合适 RTX 环境。

## 输入输出总览

    当前三路RGB + 当前原始61D状态 + 一条定性语言
      → Pi05Executor
      → 32×23 原始R1Pro控制器输入
      → harness逐步执行前8步
      → 新观测和新指令

n_obs_steps=1，只输入当前观测。视频及未来动作用于审核/监督，不得作为未来图像输入。当前阶段可用人工指令替代 planner，例如：

    向右转并向前移动，观察并接近收音机。

建议共享 renderer 输出与训练一致的模板。单独“找到收音机”不等于当前训练输入分布。支持11个原子标签和31种组合，见 reports/sampling_distribution.json。转向指底盘转向，独立转腰没有有效示范；不输出精确角度和距离。

## 三路图像契约

| 顺序 | 相机 | 模型输入键 |
|---|---|---|
| 0 | 头部 ZED | observation.images.base_0_rgb |
| 1 | 左腕 RealSense | observation.images.left_wrist_0_rgb |
| 2 | 右腕 RealSense | observation.images.right_wrist_0_rgb |

base_0_rgb 实际对应头部相机，不是另一台底盘摄像机。官方扁平观测键：

    robot_r1::robot_r1:zed_link:Camera:0::rgb
    robot_r1::robot_r1:left_realsense_link:Camera:0::rgb
    robot_r1::robot_r1:right_realsense_link:Camera:0::rgb

参考实现从 PNG 读取 BGR，用 cv2.resize 默认插值直接缩放到224×224，再转换RGB，组成 float32 [1,3,224,224]，除255得到[0,1]。不做额外letterbox、裁剪、翻转或增强。模型内部再映射至[-1,1]，harness不要提前重复处理。

三路应同步且完整，不得静默用黑图代替。RGBA应去除alpha。直接传内存RGB时不要重复进行BGR转换。

## 状态映射与归一化

入口要求有限 float32，shape=(61,)，沿用 r1pro.yaml 中 proprio_obs 的顺序。冒号区间左闭右开。

| 原始下标 | 内容 |
|---|---|
| 0:3 | base_qvel |
| 3:10 | 左臂7维关节位置 |
| 10:17 | 左臂7维关节速度 |
| 17:20 | 左末端位置 |
| 20:24 | 左末端四元数 |
| 24:26 | 左夹爪两指位置 |
| 26:28 | 左夹爪两指速度 |
| 28:35 | 右臂7维关节位置 |
| 35:42 | 右臂7维关节速度 |
| 42:45 | 右末端位置 |
| 45:49 | 右末端四元数 |
| 49:51 | 右夹爪两指位置 |
| 51:53 | 右夹爪两指速度 |
| 53:57 | 躯干4维关节位置 |
| 57:61 | 躯干4维关节速度 |

新环境需复现官方字段坐标系、单位和关节排列，不能仅凭61维相同就认定兼容。尤其不要将任意世界系速度直接冒充 base_qvel。

按以下顺序选25维，不能排序：

    [0,1,2,53,54,55,56,3,4,5,6,7,8,9,24,25,28,29,30,31,32,33,34,49,50]

即底盘3 + 躯干4 + 左臂7 + 左夹爪2 + 右臂7 + 右夹爪2。使用当前检查点 norm_stats.json 中 state.q01/q99：

    state_norm = 2 * (state25 - q01) / maximum(q99 - q01, 1e-6) - 1

observation.state 使用未裁剪的 state_norm。仅生成状态token时裁剪到[-1,1]。不要用mean/std，也不要混用action统计。

## 文本与状态 Token

完整字符串格式必须一致，包括换行和空格：

    Task: {instruction}, State: {空格分隔的25个整数};\nAction: 

离散化：

    bins = np.digitize(
        np.clip(state_norm, -1, 1),
        np.linspace(-1, 1, 257)[:-1]
    ) - 1

使用包内tokenizer，add_special_tokens=True、max_length=200、padding='max_length'、truncation=True。模型键和形状：

    observation.language.tokens            int64 [1,200]
    observation.language.attention_mask    bool  [1,200]
    observation.state                      float32 [1,25]

不要把长历史、场景描述或推理过程拼到VLA指令，导致目标和状态token被截断。新增模板先检查token长度。

## Python 加载示例

在适配包目录并设置上述PYTHONPATH后：

    from pathlib import Path
    import json
    from pi05_executor import Pi05Executor

    root = Path.cwd()
    data = json.loads((root / "fixture/input.json").read_text())
    observation = {
        "state": data["state"],
        "image_paths": [str(root / "fixture" / p) for p in data["image_paths"]],
    }
    model = Pi05Executor(root / "checkpoint", device="cuda")
    actions = model.predict(observation, data["instruction"])
    assert actions.shape == (32, 23)

执行器 strict=True 加载，设置eval()、关闭gradient checkpointing、使用inference_mode()，调用 predict_action_chunk。不要混用维护内部动作队列的 select_action。配置 num_inference_steps=10、use_relative_actions=false、chunk_size=32、n_action_steps=8。内部 max_state_dim/max_action_dim=32 是padding上限，不是机器人接口维数。

## 动作解释

模型直接输出归一化动作；执行器已反归一化一次：

    action_raw = (action_norm + 1) / 2 * maximum(q99 - q01, 1e-6) + q01

返回float32 [32,23]，HTTP也返回raw controller inputs，不能再反归一化。不要整体裁剪到[-1,1]，关节绝对位置可能超出此区间。

| 动作下标 | 控制块 | 含义 |
|---|---|---|
| 0:3 | base | 底盘前后、横向、偏航控制输入 |
| 3:7 | trunk | 4维躯干关节位置命令 |
| 7:14 | arm_left | 7维左臂绝对关节位置 |
| 14 | gripper_left | 左夹爪控制器命令 |
| 15:22 | arm_right | 7维右臂绝对关节位置 |
| 22 | gripper_right | 右夹爪控制器命令 |

目标环境须核验具体关节轴、符号、单位与顺序。双臂和trunk采用JointController(position)，use_delta_commands=false。夹爪观测每侧2维，但动作每侧1维，不是单根手指的米制位置。

官方action_normalize=false，但base控制器本身将[-1,1]输入映射到平移±0.75、偏航±1.0的速度范围。harness不可预先再乘一次比例后交给同一个控制器。不能直接把这23维接到另一种机器人。

全零动作不是无动作，可能要求绝对位置关节回零。reached、need_observation和异常处理不能用全零23维伪装停止。保持策略应由目标环境明确定义。

## HTTP 接入

可以模型留在计算GPU上，新机器只运行harness。先指定有足够空闲显存的GPU_ID，不挤占训练任务。在适配包目录执行：

    CUDA_VISIBLE_DEVICES="$GPU_ID" \
      python scripts/vla_http_server.py --checkpoint ./checkpoint --port 18769

服务仅绑定127.0.0.1。新机器建立转发：

    ssh -N -L 18769:127.0.0.1:18769 huoshan

这里是部署步骤，交接时未自动启动常驻服务。模型运行在其他主机时替换转发目的地。

GET /health 返回 ready、checkpoint、state_dim=61、action_dim=23、prediction_steps=32。POST /predict 严格接受 state、images_base64、instruction 三个字段。state是61个实际有限数值；images_base64是按头部/左腕/右腕顺序编码的PNG字节，不带data:image前缀；instruction为非空语言字符串。

返回 actions 为32×23，representation必须是 original_R1Pro_controller_inputs。服务请求上限16MiB，顺序执行，非并发批处理服务。客户端见 scripts/vla_http_client.py。错误应中止并记录，不用旧动作/零动作填补；超时后不可假定物理环境没有执行而盲目重放。

## 时间调度

数据频率30Hz。参考设计每轮预测32步，只执行前8步（约0.267秒），丢弃剩余预测，再观察和生成新动作。一次执行完32步属于不同配置。

模型/网络/上层推理耗时不等于物理控制步长。应明确仿真暂停方式、动作缓冲、超时与重规划频率。改变频率、插值、裁剪、执行步数时记录配置，不能默认保持训练契约。

## 验收命令

在适配包目录完成环境准备后执行：

    python scripts/verify_handoff.py

上面只检查预处理，不占用GPU。权重复制后再执行：

    CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/verify_handoff.py --checkpoint ./checkpoint

后一条会真正加载模型，检查有限32×23输出；不代表模拟器任务成功。

## 验收步骤

1. 校验SHA256，确认权重完整；记录版本。
2. 对fixture/input.json和三路PNG运行预处理。与expected_preprocessing.npz比较：normalized_state和图像张量allclose(atol=1e-6,rtol=1e-6)，state_bins、input_ids、attention_mask逐元素相等；prompt.txt逐字符一致。
3. strict=True加载无missing/unexpected keys；fixture预测得到有限32×23输出。
4. 不同实现比较时固定torch.manual_seed和torch.cuda.manual_seed_all。先比较归一化输出，再比较raw输出。跨GPU不承诺逐位相等；fixture不含“必须匹配的预测动作”，只有预处理参考。
5. 核验R1Pro控制器配置并做小幅单轴检查。优先排查索引错位、左右相机交换、额外归一化及BGR/RGB错误。
6. 固定观测，替换左右方向，检查语言是否影响预测；这只是指令响应诊断，不等同安全物理执行。
7. 接连续循环后，保存当前RGB、原始状态、指令、32步输出、实际提交前8步、时间戳、终止/异常原因和权重版本。

历史上训练预处理和执行器做过契约检查，小步数检查点也通过过真实HTTP模型输出检查。此次可移植包及新机器仍需GPU推理和真实模拟器验收，不能将历史结果视作目标部署已通过。

## 已知限制与接手原则

当前训练421211窗口，验证53977，测试52254，收音机任务留出450。标签由官方技能目标与测得动作事后生成，未逐条验证主动感知意图，部分物体名仍为英文。类别不均衡，采样改善曝光但不能保证泛化。开容器窗口未必覆盖完整开启动作，独立转腰无有效示范。

参考工程在 /cs/kyy/behavior_ap_experiment，原数据在 /cs/datasets/behavior2026_dual_language，设计文档为 docs/planner_design.md，当前范围为 configs/current_scope.json。

先通过输入契约和推理检查，再改harness。不要为消除报错而静默改相机、状态排列、动作控制器或归一化方式。
