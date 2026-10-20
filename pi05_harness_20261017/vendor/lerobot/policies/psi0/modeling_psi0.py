from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import TypedDict, Unpack

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION

from .compat import load_psi_module
from .configuration_psi0 import Psi0Config

logger = logging.getLogger(__name__)


class ActionSelectKwargs(TypedDict, total=False):
    noise: Tensor | None


class Psi0Policy(PreTrainedPolicy):
    """LeRobot adapter around the canonical ``psi.models.psi0.Psi0Model``.

    ``action_header`` and ``vlm_model`` deliberately live at the policy root.
    Consequently the state-dict keys are identical to standalone Psi0:
    ``action_header.*`` and ``vlm_model.*``.
    """

    config_class = Psi0Config
    name = "psi0"

    def __init__(self, config: Psi0Config, **kwargs):
        super().__init__(config)
        config.validate_features()

        core_module = load_psi_module(config.psi0_source_dir, "models.psi0")
        adaptive_module = load_psi_module(config.psi0_source_dir, "utils.adaptive_state_dict")
        self._load_state_dict_adaptive = adaptive_module.load_state_dict_adaptive

        transformers = __import__("transformers", fromlist=["Qwen3VLForConditionalGeneration"])
        dtype = getattr(torch, config.vlm_dtype)
        vlm_model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            config.model_name_or_path,
            attn_implementation=config.attn_implementation,
            dtype=dtype,
        )
        core = core_module.Psi0Model(model_cfg=config, vlm_model=vlm_model)

        # Do not retain `core` as a child module: root names must match Psi0.
        self.action_header = core.action_header
        self.vlm_model = core.vlm_model
        del core

        self._configure_trainability()
        self._load_initial_weights()
        self._noise_scheduler = self._make_noise_scheduler()
        self._action_queue: deque[Tensor] = deque(maxlen=config.n_action_steps)

        if config.action_dim == 7:
            xyz, rpy, gripper = config.loss_w
            loss_w = torch.tensor([xyz] * 3 + [rpy] * 3 + [gripper], dtype=torch.float32)
        else:
            loss_w = torch.full((config.action_dim,), 1.0 / config.action_dim, dtype=torch.float32)
        if not torch.isclose(loss_w.sum(), torch.tensor(1.0)):
            raise ValueError("Psi0 loss weights must sum to one")
        self.register_buffer("_loss_w", loss_w, persistent=False)

    def _configure_trainability(self) -> None:
        for parameter in self.vlm_model.parameters():
            parameter.requires_grad = self.config.tune_vlm
        if self.config.tune_vlm:
            language_model = getattr(getattr(self.vlm_model, "model", None), "language_model", None)
            final_norm = getattr(language_model, "norm", None)
            if final_norm is not None:
                for parameter in final_norm.parameters():
                    parameter.requires_grad = False

    def _load_initial_weights(self) -> None:
        if self.config.pretrained_action_header_path is not None:
            path = Path(self.config.pretrained_action_header_path).expanduser()
            if path.is_dir():
                path = path / "action_header.safetensors"
            if not path.is_file():
                raise FileNotFoundError(f"Psi0 action-head checkpoint does not exist: {path}")
            report = self._load_state_dict_adaptive(self.action_header, load_file(str(path), device="cpu"))
            if not report["loaded"] and not report["resized"]:
                raise RuntimeError(f"No action-header weights could be loaded from {path}")
            logger.info(
                "Loaded Psi0 action head from %s (%d exact, %d resized)",
                path,
                len(report["loaded"]),
                len(report["resized"]),
            )

        if self.config.psi_checkpoint_path is not None:
            path = Path(self.config.psi_checkpoint_path).expanduser()
            if path.is_dir():
                path = path / "model.safetensors"
            if not path.is_file():
                raise FileNotFoundError(f"Psi0 full checkpoint does not exist: {path}")
            state = {
                key.removeprefix("module."): value
                for key, value in load_file(str(path), device="cpu").items()
            }
            unsupported = [key for key in state if not key.startswith(("action_header.", "vlm_model."))]
            if unsupported:
                raise RuntimeError(f"Unsupported Psi0 checkpoint keys: {unsupported[:5]}")

            # Match Psi0Model.from_pretrained: the VLM is strict, while the
            # action head may be resized when action/state dimensions differ.
            vlm_state = {
                key.removeprefix("vlm_model."): value
                for key, value in state.items()
                if key.startswith("vlm_model.")
            }
            embed_key = "model.language_model.embed_tokens.weight"
            if embed_key in vlm_state:
                vlm_state.setdefault("lm_head.weight", vlm_state[embed_key])
                vocab_size = vlm_state[embed_key].shape[0]
                if vocab_size != self.vlm_model.lm_head.weight.shape[0]:
                    self.vlm_model.resize_token_embeddings(
                        vocab_size,
                        pad_to_multiple_of=192,
                        mean_resizing=True,
                    )
            self.vlm_model.load_state_dict(vlm_state, strict=True)

            action_state = {
                key.removeprefix("action_header."): value
                for key, value in state.items()
                if key.startswith("action_header.")
            }
            report = self._load_state_dict_adaptive(self.action_header, action_state)
            if not report["loaded"] and not report["resized"]:
                raise RuntimeError(f"No action-header weights could be loaded from {path}")
            logger.info(
                "Loaded Psi0 checkpoint %s (%d exact action tensors, %d resized)",
                path,
                len(report["loaded"]),
                len(report["resized"]),
            )

    def _make_noise_scheduler(self):
        if self.config.noise_scheduler == "ddpm":
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler

            return DDIMScheduler(
                num_train_timesteps=self.config.train_diffusion_steps,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type="epsilon",
            )
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )

        return FlowMatchEulerDiscreteScheduler(num_train_timesteps=self.config.train_diffusion_steps)

    def get_optim_params(self) -> list[dict]:
        groups: list[dict] = [
            {
                "params": [p for p in self.action_header.parameters() if p.requires_grad],
                "lr": self.config.optimizer_lr,
                "group_name": "action_header",
            }
        ]
        vlm_parameters = [p for p in self.vlm_model.parameters() if p.requires_grad]
        if vlm_parameters:
            groups.append(
                {
                    "params": vlm_parameters,
                    "lr": self.config.lang_backbone_lr,
                    "group_name": "vlm_model",
                }
            )
        return groups

    def reset(self) -> None:
        self._action_queue.clear()

    def _states(self, batch: dict[str, Tensor]) -> Tensor:
        states = batch[self.config.state_key]
        return states.unsqueeze(1) if states.ndim == 2 else states

    def _predict_velocity(
        self,
        batch: dict[str, Tensor],
        action_samples: Tensor,
        timesteps: Tensor,
        *,
        vlm_hidden_states: Tensor | None = None,
        use_vlm_attn_mask: bool = True,
    ) -> Tensor:
        if vlm_hidden_states is None:
            vlm_hidden_states = self.vlm_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[-1]
        return self.action_header(
            hidden_states=None,
            timestep=timesteps,
            joint_attention_kwargs={
                "action_hidden_embeds": action_samples,
                "views": vlm_hidden_states.unsqueeze(1),
                "obs": self._states(batch),
                "traj2ds": batch.get("traj2ds"),
            },
            vlm_attn_mask=batch.get("attention_mask") if use_vlm_attn_mask else None,
            return_dict=True,
        ).action

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        actions = batch[ACTION]
        batch_size, horizon, _ = actions.shape
        prefix_mask = None

        if self.config.noise_scheduler == "ddpm":
            timesteps = torch.randint(
                0,
                self.config.train_diffusion_steps,
                (batch_size,),
                device=actions.device,
            ).long()
            sigmas = None
        else:
            sigmas = torch.rand((batch_size,), device=actions.device)
            timesteps = sigmas * self.config.train_diffusion_steps
            if self.config.rtc:
                delay = torch.randint(
                    0,
                    self.config.max_delay,
                    (batch_size,),
                    device=actions.device,
                    dtype=torch.long,
                )
                prefix_mask = torch.arange(horizon, device=actions.device)[None, :] < delay[:, None]
                sigmas = torch.where(
                    prefix_mask,
                    torch.tensor(0.0, device=actions.device, dtype=sigmas.dtype),
                    sigmas[:, None],
                )
                timesteps = sigmas * self.config.train_diffusion_steps

        noise = torch.randn_like(actions)
        if self.config.noise_scheduler == "ddpm":
            noisy_actions = self._noise_scheduler.add_noise(actions, noise, timesteps)
            target = noise
        else:
            sigma_actions = sigmas
            while sigma_actions.ndim < actions.ndim:
                sigma_actions = sigma_actions.unsqueeze(-1)
            noisy_actions = (1.0 - sigma_actions) * actions + sigma_actions * noise
            target = noise - actions

        prediction = self._predict_velocity(batch, noisy_actions, timesteps)
        element_loss = functional.mse_loss(prediction.float(), target.float(), reduction="none")
        mask = batch.get("actions_mask")
        mask = torch.ones_like(actions) if mask is None else mask.float()
        if prefix_mask is not None:
            mask = mask * (~prefix_mask)[:, :, None].float()
        loss = ((element_loss * mask).sum(1).mean(0) * self._loss_w).sum()
        return loss, {"loss": loss.detach()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        batch_size = self._states(batch).shape[0]
        hidden = self.vlm_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[-1]
        samples = kwargs.get("noise")
        if samples is None:
            samples = torch.randn(
                batch_size,
                self.config.chunk_size,
                self.config.action_dim,
                device=self._states(batch).device,
            )
        self._noise_scheduler.set_timesteps(self.config.eval_diffusion_steps)
        for timestep in self._noise_scheduler.timesteps:
            batched_timestep = timestep.expand(batch_size).to(samples.device)
            prediction = self._predict_velocity(
                batch,
                samples,
                batched_timestep,
                vlm_hidden_states=hidden,
                use_vlm_attn_mask=False,
            )
            samples = self._noise_scheduler.step(
                model_output=prediction,
                timestep=timestep,
                sample=samples,
            ).prev_sample
        return samples.float()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch)
            for index in range(min(self.config.n_action_steps, chunk.shape[1])):
                self._action_queue.append(chunk[:, index])
        return self._action_queue.popleft()
