# policy/ — Local policy stack (inherited from UniVTAC)

Self-contained baselines that are **trained and deployed locally, in-process**:

- `ACT/` — Action Chunking with Transformers, with/without tactile inputs
- `Ablation/` — ACT ablation variants for modality comparison
- `ViTAL/` — ACT with CLIP-pretrained tactile-vision encoders

Each module has its own data processing (`process_data*.py`), training, and deployment
(`deploy_policy.py`) scripts. They are driven by the upstream entry points at the repo root:

```bash
bash eval_policy.sh    <task> <task_config> <policy_config> <gpu>
bash parallel_eval.sh  <task> <task_config> <policy_config> <gpu> [num_proc] [total]
```

which route through `scripts/eval_policy.py` (loads the policy module in-process).

**This directory is NOT involved in NeoSim's primary evaluation path.** Remote policies
(VTA-style models, Pi0.5, XR0, giga, LingBot-VA) are evaluated over a WebSocket
policy-server protocol via the clients under `eval/` — see `eval/README.md`.
The two paths share only the task environments (`envs/`) and are otherwise independent
(no cross-imports).
