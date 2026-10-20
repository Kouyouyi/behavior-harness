from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("psi0")
@dataclass
class Psi0Config(PreTrainedConfig):
    """LeRobot configuration mirroring ``psi.config.model_psi0.Psi0ModelConfig``."""

    # Keep /cs/Psi0 as the single source of truth for model and transforms.
    psi0_source_dir: str = "/cs/Psi0"

    # Canonical Psi0 initialization/checkpoint inputs.
    model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    pretrained_action_header_path: str | None = None
    psi_checkpoint_path: str | None = None
    resnet_store_path: str | None = None

    # Model geometry. These names intentionally match Psi0ModelConfig.
    action_dim: int = 78
    chunk_size: int = 30
    n_action_steps: int = 30
    action_exec_horizon: int = 30
    observation_horizon: int = 1
    odim: int = 46
    img_chunk: int = 1
    n_cams: int = 1
    use_obs: str = "add_token"
    dropout: float = 0.1
    share_cam_features: bool = False
    early_fusion: bool = False
    n_conditions: int | None = 0
    token_fusion: str | None = "concat"
    time_dim: int = 256
    hidden_dim: int = 1536
    num_blocks: int = 6
    dim_feedforward: int = 2048
    nhead: int = 24
    activation: str = "gelu"
    view_feature_dim: int = 2048
    use_film: bool = False
    combined_temb: bool = False
    use_dit: bool = False

    # Flow/DDPM training and inference.
    noise_scheduler: str = "flow"
    train_diffusion_steps: int = 1000
    eval_diffusion_steps: int = 10
    rtc: bool = True
    max_delay: int = 8
    loss_w: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.1])

    # Qwen3-VL controls copied from Psi0.
    tune_vlm: bool = False
    tune_mm_llm: bool = False
    tune_mm_vision: bool = False
    tune_mm_mlp: bool = False
    gradient_checkpointing: bool = True
    lang_backbone_lr: float = 1e-5
    mm_projector_lr: float = 1e-5
    vision_tower_lr: float = 1e-6
    model_max_length: int = 4096
    max_pixels: int = 576 * 28 * 28
    min_pixels: int = 16 * 28 * 28
    attn_implementation: str = "flash_attention_2"
    vlm_dtype: str = "bfloat16"
    # Consumed by LeRobot's Accelerator setup. This matches Psi0's
    # ``--train.mixed_precision=bf16`` profile.
    dtype: str = "bfloat16"

    # Exact HUMANO preprocessing profile used by finetune-humano-psi0.sh.
    image_keys: list[str] = field(default_factory=lambda: ["observation.images.ego_view"])
    state_key: str = OBS_STATE
    task_key: str = "task"
    stats_path: str | None = None
    action_stats_key: str = "action"
    state_stats_key: str = "states"
    normalize_state: bool = True
    action_norm_type: str = "bounds"
    image_resize: tuple[int, int] = (240, 320)
    image_center_crop: tuple[int, int] = (240, 320)
    image_augmentation: bool = True

    # Standalone Psi0 optimizer/scheduler defaults.
    optimizer_lr: float = 1e-3
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1000

    # Standalone Psi0 uses Accelerate bf16 autocast.
    use_amp: bool = True

    @property
    def action_chunk_size(self) -> int:
        """Attribute name consumed by the canonical Psi0 model."""

        return self.chunk_size

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot be larger than chunk_size")
        if self.noise_scheduler not in {"flow", "ddpm"}:
            raise ValueError("noise_scheduler must be 'flow' or 'ddpm'")
        if self.action_norm_type not in {"bounds", "bounds_q99"}:
            raise ValueError("action_norm_type must be 'bounds' or 'bounds_q99'")

    def validate_features(self) -> None:
        if self.input_features is None:
            self.input_features = {}
        if self.output_features is None:
            self.output_features = {}
        if self.state_key not in self.input_features:
            self.input_features[self.state_key] = PolicyFeature(type=FeatureType.STATE, shape=(self.odim,))
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))

        state = self.input_features.get(self.state_key)
        action = self.output_features.get(ACTION)
        if state is not None and state.shape[-1] != self.odim:
            raise ValueError(f"Psi0 expects state dim {self.odim}, got {state.shape}")
        if action is not None and action.shape[-1] != self.action_dim:
            raise ValueError(f"Psi0 expects action dim {self.action_dim}, got {action.shape}")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(name="cosine", num_warmup_steps=self.scheduler_warmup_steps)

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
