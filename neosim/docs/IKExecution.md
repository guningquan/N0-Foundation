# EE → joint execution: direct single-shot IK

How the eval clients (and `take_action(..., action_type='ee_ik')`) turn an EE-space
action into joint commands, and why it is a **single-shot IK**, not trajectory planning.

## Why not curobo trajectory planning

`plan_arm`-style trajectory generation chokes the GPU on hard targets (96% util,
~0.01 FPS) and fails often. Policy servers emit dense action chunks (one EE target
per control tick), so there is nothing to "plan" between consecutive targets —
a direct IK solve per tick is both sufficient and orders of magnitude faster.

## Call chain

```
single-arm client          dual-arm client
eval_twam_ee_cl.py          eval_twam_ee_dual_cl.py
  take_action(ee,'ee_ik')    _drive_one(rm, ee)          # inlined mirror of the same logic
        │                       │
        └───────► RobotManager.solve_ik(target_pose)     # envs/robot/robot.py
                    │   auto-fills root_pose + current joints (grippers excluded)
                    └► CuroboPlanner.solve_ik(...)        # envs/robot/curobo_planner.py
                         single-shot IK, seeded (below)
```

- **Single-arm** (`envs/_base_task.py`, `ee_ik` branch): one call = one IK solve →
  `set_arm` → `set_gripper` → one `_step()` → `take_action_cnt += 1`.
- **Dual-arm** (`eval_twam_ee_dual_cl.py::drive_dual`): both arms are solved and set
  **within the same frame**, then a single `_step()` advances the sim — keeping the
  two arms synchronous. `UNIVTAC_TICKS_PER_SLOT` (default 2) repeats this per action
  slot; one dataset row equals 2 sim steps, so the default runs at 2× data cadence
  and `2` matches it exactly. `take_action_cnt` counts ticks, consistent with
  `step_lim` in both modes.

## Solution-branch seeding (anti-jitter)

`UNIVTAC_IK_SEEDCFG=1` (**default on**) seeds the solver with the *current* joint
configuration, so it converges to the nearest solution branch. Without it, null-space
wrist flips make J5/J7 jump up to ~2 rad between frames at an identical EE pose
(visible arm jitter). EE accuracy is unchanged. Set `UNIVTAC_IK_SEEDCFG=0` only for
A/B diagnosis.

## Failure semantics

If IK returns `Fail`, the arm **holds its last commanded configuration**; the gripper
command and the sim step still execute, and the step counter still advances. So an
unreachable target degrades into "arm freezes, time passes" rather than a crash.
Known asymmetry: the single-arm path logs `direct IK failed, holding arm`; the
dual-arm client stays silent by design (two arms × every tick would flood the log).

## Diagnostics

`eval/eval_official_ee.py` is the closest-to-upstream client (open-loop, EE actions
via `take_action('ee')`) — useful to isolate execution-path issues from the
streaming-protocol machinery in `eval_twam_ee_cl.py`.
