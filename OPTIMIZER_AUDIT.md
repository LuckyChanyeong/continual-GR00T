# AdamW single-step audit

`gr00t_sequential_finetune.py` can record the action-head state and actual optimizer displacement at steps 1,000, 5,000, and 7,500. The feature is disabled by default and does not change ordinary training.

Enable it on the next run with:

```bash
--optimizer-audit
```

The default steps are `1000 5000 7500`. They can be overridden with:

```bash
--optimizer-audit-steps 1000 5000 7500
```

Each task writes:

```text
task_N/optimizer_audit/step_001000/
├── pre_step_action_head.safetensors
├── actual_update.safetensors
└── metadata.json
```

- `pre_step_action_head.safetensors`: original-parameter-dtype action-head state immediately before `optimizer.step()`. Category-specific tensors contain only active embodiment rows.
- `actual_update.safetensors`: `theta_after - theta_before`, FP32 by default.
- `metadata.json`: optimizer step, learning rates, parameter-group mapping, active category rows, update norm, and reconstruction information.

The effective AdamW gradient direction is `-actual_update`; for a parameter group with learning rate `lr`, its scaled value is `-actual_update / lr`.

## Why the AdamW update must be captured during training

A raw gradient can be recomputed at a saved model checkpoint by loading the checkpoint, evaluating a fixed batch and loss, and running backward. It is a function of the current parameters, data batch, and stochastic inputs used for the loss.

The AdamW displacement is not determined by the model parameters and current raw gradient alone. It also uses the optimizer first- and second-moment states, the current learning rate and step, gradient clipping, and decoupled weight decay. A final model checkpoint contains the updated parameter θ_t, but θ_t alone does not uniquely identify the preceding parameter θ_(t-1) or the last displacement Δθ_t = θ_t - θ_(t-1). Therefore the exact update must be captured before and after that optimizer step, unless both adjacent parameter states and all relevant optimizer information were already saved.

The three audit points are needed only to study how the effective update changes during a task. If the research question concerns only the final optimizer step, recording the final step alone is sufficient.

The callback runs at Transformers `on_pre_optimizer_step` and `on_optimizer_step`, so it records the update after backward and gradient clipping and before the scheduler advances. It supports the current single-process training path.

Storage depends on the original parameter dtype. For the current active LIBERO action head, an FP32 tensor collection is about 3.05 GB: FP32 pre-state plus FP32 updates needs about 6.1 GB per audit point; BF16 pre-state plus FP32 updates needs about 4.6 GB. `--optimizer-audit-delta-dtype bfloat16` reduces update storage but lowers update precision. Pre-state now preserves the original parameter dtype, avoiding FP32-to-BF16 snapshot rounding.

No GR00T training was run while adding this feature. Syntax, CLI exposure, and exact delta capture were checked with a toy AdamW step.

## Dense versus sparse recording

Saving every AdamW displacement is required only when the full step-by-step optimizer trajectory must be reconstructed. It is not required to measure the total movement of a task because the sum of all displacements equals the difference between the task-end and task-start parameters.

For the current action-head size, one FP32 displacement is about 3.05 GB. Recording 10,000 steps would require about 30.5 TB per task and about 183 TB for six tasks. A practical design is to save sparse full-vector audits and log scalar quantities such as cosine similarity, dot product, update norm, and old-task loss change more frequently. Computing an old-task gradient at every training step would also add substantial backward-pass cost.

## Interpretation of the gradient-first analysis

The initial comparison cos(g0(theta_after_T0), g0(theta_after_T1)) measures how the Task 0 loss gradient rotates after Task 1 training. This remains a valid diagnostic under AdamW, provided that the same Task 0 evaluation samples and flow-matching noise protocol are used at both checkpoints.

This comparison does not by itself establish that Task 1 updates caused forgetting. A direct local interference test compares the old-task gradient with the actual Task 1 displacement at the same pre-update parameter state. The first-order old-task loss change is g0(theta_t) dot delta_theta_t. A positive value predicts an increase in Task 0 loss; a negative value predicts a decrease.

Raw cross-task gradient conflict, cos(g0(theta_t), g1(theta_t)), directly predicts an SGD update direction but not an AdamW update direction. AdamW introduces moment history, coordinate-wise scaling, clipping, learning-rate state, and decoupled weight decay. Therefore the raw-gradient angle is a useful geometry measurement, while the old-gradient versus actual-displacement measurement is the optimizer-aware mechanism test.
