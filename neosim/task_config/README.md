# task_config/ — runtime profiles

A profile configures **how a run executes** (episodes, seeds, observations, storage,
GUI); the task itself (assets, success criteria, step limits) lives in `envs/`.
Every entry point takes a profile as its second positional argument:

```bash
python scripts/collect_data.py <task> collect     # collect 100 episodes
python eval/eval_twam_ee_cl.py  <task> demo ...    # eval observations profile
```

## Core profiles (8)

| Profile | Use |
|---|---|
| `demo` | 5 episodes, quick sanity run |
| `demo_one` | 1 episode, fixed seed 0 — deterministic smoke test / video sample (raise `max_seed` to sweep a few seeds) |
| `collect` | 100 episodes, `start_seed: -1` auto-resumes from `suc_map.txt` |
| `gui` | `demo_one` + `render_frequency: 1` (live window) |
| `demo_notac` / `demo_one_notac` | no-tactile variants (5 ep / 1 ep) |
| `repr_demo` | alias of `demo` kept for eval-result path compatibility |
| `fasteval` | reduced storage frequency for fast evaluation |

All core profiles collect `gel_particle` in-loop by default
(see `docs/TactileRepresentations.md` — remove it from `observations.tactile` for leaner storage).

## Force profiles (4)

| Profile | Use |
|---|---|
| `demo_force` | force-field observation set, 100 ep (add `dense_gelpad: true` for the dense gel mesh) |
| `collect_force` | force-set collection |
| `lift_can_force_demo` | minimal-lossless-set example (`vertex_force` based, dense gelpad) |
| `record_force` | video recording with `force_field_img` panel |

## Recording / special (7)

`record_one` (every-frame storage for figures), `record_particle` (gel_particle panel),
`poster_record`, `lc_marker_gp` (marker+depth figure setup, see docs),
`contact` (contact-pair collection), `motion_test` (motion smoke test).
