# Psi0 policy

This adapter intentionally imports the model and image/text transform from
`/cs/Psi0/src/psi`. It does not carry a second copy of Psi0, so architecture,
augmentation and adaptive checkpoint loading cannot drift between repositories.

The LeRobot checkpoint parameter roots are the standalone Psi0 roots:
`action_header.*` and `vlm_model.*`.

## HUMANO training profile

```bash
lerobot-train \
  --seed=292285 \
  --dataset.repo_id=g1_sonic/HUMANO_basic_train_v30_groot_io \
  --dataset.root=/cs/kyy/baselines/datasets/HUMANO_basic_train_v30_groot_io \
  --policy.type=psi0 \
  --policy.device=cuda \
  --policy.psi0_source_dir=/cs/Psi0 \
  --policy.model_name_or_path=/cs/models/psi0/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k \
  --policy.pretrained_action_header_path=/cs/models/psi0/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k \
  --policy.stats_path=/cs/datasets/HUMANO-dataset/train/basic/meta/stats_psi0.json \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.action_dim=78 \
  --policy.odim=46 \
  --policy.rtc=true \
  --policy.max_delay=8 \
  --policy.push_to_hub=false \
  --batch_size=64 \
  --num_workers=8 \
  --steps=30000 \
  --save_freq=1000 \
  --output_dir=/cs/kyy/baselines/outputs/psi0_basic
```

For bit-for-bit comparison, use the package versions from `/cs/Psi0/pyproject.toml`
(PyTorch 2.7.0, Transformers 4.57.0, Diffusers 0.37.0, qwen-vl-utils 0.0.14)
and the same GPU count, seed, sample order and bf16/FlashAttention2 settings.
