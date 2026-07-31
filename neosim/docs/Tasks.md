# Tasks

NeoSim ships 20 task implementations: the **12 NeoSim benchmark tasks** of NeoBenchmark (4 single-arm — `pour_ball`, `phone_socket_replug`, `insert_USB`, `grasp_chip` — and 8 dual-arm), plus 8 single-arm tasks inherited from UniVTAC. Each section below documents the goal, the scripted expert's action flow, and the **exact success criteria as implemented** in `envs/<task>.py`, with line references. Units are meters unless noted; tactile depth/indentation in mm; orientation checks are dot-product thresholds (e.g. `> 0.965` ≈ within 15°, `> 0.99` ≈ within 8°).

## Shared evaluation logic

- `BaseTask.check_success` defaults to `False` (`envs/_base_task.py:861-862`); `check_early_stop` defaults to `False` (`envs/_base_task.py:1077-1078`). Tasks override both.
- Timeout: in eval, `take_action` refuses further steps once `take_action_cnt >= cfg.step_lim` (`envs/_base_task.py:1086`); default `step_lim = 300` (`envs/_base_task.py:87`), overridden per task. `check_success()` is evaluated after every action and latches `eval_success` (`envs/_base_task.py:1155-1156`).
- Eval loops abort an episode when `task.check_early_stop()` returns True (e.g. `scripts/eval_policy.py:149`).
- Data collection keeps an episode only if `task.plan_success and task.check_success() and not task.check_early_stop()` (`scripts/collect_data.py:136`).
- `plan_success` is set False on motion-planning failure (`envs/_base_task.py:897,920`) and, when `cfg.keep_contact`, on loss of tactile contact or exceeding `max_save_frames` (`envs/_base_task.py:727-732`).
- Robot tactile constants (gsmini, `envs/robot/robot_cfg.py:99-101`): `tactile_far_plane=34.0` mm, `adaptive_grasp_depth_threshold=27.5` mm, `contact_threshold=(27.5, 28.0)` mm.
- "Held for N consecutive checks" (dual-arm unstack tasks) means N consecutive `check_success` calls, since it is re-evaluated each eval step (`envs/_base_task.py:1155`).


## Single-arm tasks

### lift_bottle
- **Goal**: Rotate a bottle lying against a wall up into a standing (upright) pose next to the wall.
- **Objects/assets**: `Wall.usd` (static, density 1e5), `Bottle.usd` (`envs/lift_bottle.py:17-27`).
- **Action flow**:
  1. pre_move: settle 10 steps; build side grasp pose at bottle bias `[-0.13, 0, -0.015]` with random pitch noise `[-pi/12, 0]`; approach with gripper open (`pre_dis=0.5`) (`envs/lift_bottle.py:34-56`).
  2. Close gripper.
  3. `gripper_rotate` the bottle by 70 deg (4 steps).
  4. If mid-success not reached (`rel x > -0.01` w.r.t. wall target, `envs/lift_bottle.py:71-73`): pitch wrist +pi/6 and translate in x toward the wall target +0.02.
  5. Open gripper to 0.5; delay 30 steps (`envs/lift_bottle.py:58-69`).
- **Success criteria** (`envs/lift_bottle.py:81-84`): with `rel_pose = bottle_pose.rebase(target_pose)` where `target_pose = wall_pose + bias [-0.08, 0, 0]` (`envs/lift_bottle.py:56`):
  - `rel_pose[0] > -0.02` (bottle x at most 0.02 m short of the wall target);
  - `|rel_pose[1]| < 0.1` and `|rel_pose[2]| < 0.001`;
  - `|dot(rel x-axis, [0, 0, 1])| > 0.99` — the bottle's local x axis is vertical (upright), ≈8° tolerance.
- **Failure modes**: `check_early_stop` (`envs/lift_bottle.py:75-79`): after `take_action_cnt > 300`, if the bottle's local x axis is still horizontal (`|dot(rel x-axis, [-1,0,0])| > 0.99`), the episode aborts (bottle never rotated). Shared timeout at `step_lim = 500` (`envs/lift_bottle.py:6`).
- **Tunables**: `step_lim = 500`; `adaptive_grasp_depth_threshold = 27.8` mm (`envs/lift_bottle.py:6-7`).

### lift_can
- **Goal**: Rotate a can lying on the table so it stands upright on its end.
- **Objects/assets**: `Can_d4cm.usd` / `Can_d5cm.usd` / `Can_d6cm.usd` — one diameter (4/5/6 cm) chosen at random per episode (`envs/lift_can.py:61-84`).
- **Action flow**:
  1. pre_move: settle 10; open gripper fully; side grasp at can bias `[-0.065, 0, -0.008]` with random pitch noise in `[-pi/6, -pi/18]`; record in-hand pose (`envs/lift_can.py:86-109`).
  2. Close gripper.
  3. `gripper_rotate` the can by 70 deg (4 steps).
  4. If mid-success not reached (`|dot(can x-axis, z)| > 0.95`, `envs/lift_can.py:119-121`): `gravity_rotate` (loosen grip so gravity pivots the can toward its x-axis-vertical pose).
  5. Open gripper; delay 30 (`envs/lift_can.py:111-117`).
- **Success criteria** (`envs/lift_can.py:139-142`):
  - `can_pose[2] < 0.01` — can center below 0.01 m (resting on the table, not held in air);
  - `|dot(can local x-axis, [0, 0, 1])| > 0.99` — standing on its end, ≈8° tolerance.
- **Failure modes** (`envs/lift_can.py:123-137`): early stop if (a) min tactile gel depth `< 20` mm (squeezed too hard), or (b) in-hand z slip `|Δz| > 0.05` m while the can's local z axis is vertical (`dot > 0.99`) — slid out of grasp. Shared timeout at `step_lim = 600`.
- **Tunables**: `step_lim = 600`; `adaptive_grasp_depth_threshold = 27.75` mm (`envs/lift_can.py:20-21`); `WRIST_TILT_DEG = -30.0`, `WRIST_FOCAL = 1.3` — opt-in fixed wrist-cam variant via env `UNIVTAC_LIFT_CAN_FIXEDCAM=1` (default off = original wrist camera) (`envs/lift_can.py:5-15`).

### insert_HDMI
- **Goal**: Insert a grasped HDMI plug into an HDMI slot on the table.
- **Objects/assets**: `HDMISlot.usd` (density 1e5), `HDMI.usd`; friction 2.5 (`envs/insert_HDMI.py:35-37,59-74`).
- **Action flow**:
  1. pre_move: open gripper 0.5; grasp the HDMI from above (bias `+0.012` z, random pitch `±pi/18`); close; lift 0.02 (`envs/insert_HDMI.py:81-100`).
  2. pre_move: coarse place to `hole_pose + xy noise (±0.005)` (`pre_dis=0.02, dis=0.01`), gripper stays closed; `hole_pose = slot + [0, 0, 0.0128]` (`envs/insert_HDMI.py:102-112`).
  3. Fine place to `hole_pose` (`pre_dis=0.01, dis=0.002`).
  4. Press down 0.005 then 0.002 along gripper local z with rotation-about-z relaxed (`constraint_pose=[1,1,1,1,1,0]`).
  5. Delay 20 (`envs/insert_HDMI.py:114-128`).
- **Success criteria** (`envs/insert_HDMI.py:130-137`): with `prism_pose = HDMI pose rebased to target_pose` where `target_pose = slot + [0, 0, 0.005]`:
  - `np.all(np.abs(prism_pose.p[1:2]) < [0.005, 0.005])` — **as coded, effectively `|y| < 0.005` only** (see Known quirks #1);
  - `prism_pose.p[2] < 0.005` — plug seated within 5 mm of target depth;
  - `ee_pose[2] > 0.145` — gripper did not chase the plug down;
  - `dot(plug local z-axis, [0, 0, 1]) > 0.965` — upright within ≈15°.
- **Failure modes**: no task-specific early stop; shared timeout at `step_lim = 600` (`envs/insert_HDMI.py:30`) and shared `plan_success` failures.
- **Tunables**: `step_lim = 600`, `video_size = (640, 640)`; friction 2.5; tactile panel key via env `UNIVTAC_PANEL` / `UNIVTAC_TAC_KEY` in `get_frame_shot` (`envs/insert_HDMI.py:46`).

### insert_hole
- **Goal**: Insert a test tube into a hole slot whose insertion axis is tilted ±30°, matching the slot's random orientation.
- **Objects/assets**: `TestTubeHoleSlot.usd` (density 1e5), `TestTubeBase.usd` (density 1e5), `TestTube.usd` (density 10) (`envs/insert_hole.py:12-34`).
- **Action flow**:
  1. pre_move: grasp the tube near its top (z bias uniform in `[0.095, 0.10]`); record in-hand pose (`envs/insert_hole.py:42-60`).
  2. pre_move: lift z +0.15; coarse then fine place above `hole_pose = slot + [0, 0, 0.1]`; target orientation tilted `∓pi/6` about y depending on the slot's random 0/pi yaw (`envs/insert_hole.py:62-88`).
  3. Descend z −0.03 (time dilation 0.2).
  4. Corrective x/z move computed from the tilt angle and gripper-to-tube distance.
  5. Push z −0.04 along the tube's own axis; delay 20 (`envs/insert_hole.py:90-102`).
- **Success criteria** (`envs/insert_hole.py:113-124`): with `prism_pose = tube rebased to target_pose` (tilted hole frame):
  - `|x| < 0.01` and `|y| < 0.01` (lateral alignment within 1 cm);
  - `z < -0.04` — inserted at least 0.04 m below the hole reference along the tilted axis;
  - `dot(tube local z-axis, target [0, 0, 1]) > 0.99` — aligned within ≈8°;
  - in-hand slip `< 0.04` m — tube did not slide in the gripper.
- **Failure modes**: early stop when in-hand slip `> 0.04` m (`envs/insert_hole.py:104-111`). Shared timeout at default `step_lim = 300`.
- **Tunables**: grasp height range `[0.095, 0.10]` and tilt `pi/6` inline (`envs/insert_hole.py:45,68-70`).

### insert_tube
- **Goal**: Insert a test tube into a tilted (−30° about y) test-tube slot.
- **Objects/assets**: `TestTubeSlot.usd` (density 1e5), `TestTubeBase.usd` (density 1e5), `TestTube.usd` (density 10) (`envs/insert_tube.py:12-34`).
- **Action flow**:
  1. pre_move: grasp tube near top (z bias uniform `[0.095, 0.10]`); record in-hand pose (`envs/insert_tube.py:41-59`).
  2. pre_move: lift z +0.15; place toward `try_pose = hole_pose + signed xy noise 1–4 mm`, where `hole_pose = slot + [-0.008, 0, 0.077]` rotated `[0, -pi/6, 0]` (`envs/insert_tube.py:61-81`).
  3. `try_forward` 0.02 in 0.01 increments (advance along gripper local z; abort a step if the tube stops moving).
  4. If mid-success not reached (rel xy `< 0.005`, rel z `< -0.02`, `dot > 0.99`, `envs/insert_tube.py:100-103`): re-place exactly onto `hole_pose` and `try_forward` again.
  5. Final push z −0.04 along the tube axis (dilation 0.2); delay 20 (`envs/insert_tube.py:83-98`).
- **Success criteria** (`envs/insert_tube.py:114-122`): with `prism_pose = tube rebased to hole_pose` (tilted frame):
  - `prism_pose.p[:2] < [0.005, 0.005]` — **one-sided as coded, no `abs`** (see Known quirks #2);
  - `z < -0.03` — inserted at least 0.03 m along the tilted axis;
  - `dot(tube local z-axis, hole [0, 0, 1]) > 0.965` — within ≈15°;
  - in-hand slip `< 0.03` m.
- **Failure modes**: early stop when in-hand slip `> 0.03` m (`envs/insert_tube.py:105-112`). Shared timeout at default `step_lim = 300`.
- **Tunables**: placement noise `[0.001, 0.004]` m signed, hole bias `[-0.008, 0, 0.077]`, tilt `-pi/6` inline (`envs/insert_tube.py:62-66`).

### insert_USB
- **Goal**: Peg-in-hole insertion — the gripper starts already holding a USB plug (plug facing down) and inserts it into a slot.
- **Objects/assets**: `USBSlot.usd` (density 1e5; hole 15 × 7.5 mm, depth 11 mm), `USB.usd` (plug 12 × 4.5 mm, protruding 11 mm); friction 2.5; planner ignores the in-hand `usb` (`envs/insert_USB.py:55-64`).
- **Action flow**:
  1. reset: slot xy noise ±0.005; USB teleported into the gripper center (plug down); fingers force-snapped to 0.006 then targeted to 0 (`envs/insert_USB.py:81-96`).
  2. pre_move: settle 10; close gripper; hover the plug tip ~0.06 m above `hole_pose + xy noise ±0.004`, where `hole_pose = slot + [0, 0, SLOT_HEIGHT + USB_TIP]` (`envs/insert_USB.py:99-113`).
  3. Fine align to `hole_pose` (`pre_dis=0.02, dis=0.004`).
  4. Compliant press-down: local z `+0.9*HOLE_DEPTH` then `+0.6*HOLE_DEPTH` with rotation-about-z relaxed (total 1.5× hole depth).
  5. Delay 20 (`envs/insert_USB.py:115-128`).
- **Success criteria** (`envs/insert_USB.py:131-141`): with `usb_pose = USB rebased to hole_pose`:
  - `|x| < 0.005` and `|y| < 0.005`;
  - `usb_pose.p[2] < -0.008` — z = −0.011 is fully seated, −0.008 ≈ 73% depth: must be genuinely inserted, not hovering (criteria fix dated 2026-07-14, `envs/insert_USB.py:133-136`);
  - `dot(USB local z-axis, [0, 0, 1]) > 0.95` — upright within ≈18°.
- **Failure modes**: no task-specific early stop; shared timeout at `step_lim = 600` (`envs/insert_USB.py:50`).
- **Tunables** (`envs/insert_USB.py:21-26`): `SLOT_HEIGHT = 0.013`, `HOLE_DEPTH = 0.011`, `USB_TIP = 0.0315`, `BODY_CENTER = 0.0055`, `GRIP_Z = -BODY_CENTER` `[tune]`.

### pull_out_key
- **Goal**: Rotate a turned key back to neutral, then pull it out of the key slot without dropping it or twisting the slot.
- **Objects/assets**: `KeySlot.usd` (density 1e5), `Key.usd` (density 10); friction 2.0 (`envs/pull_out_key.py:10-30`).
- **Action flow**:
  1. reset: slot xy noise ±0.005 plus random yaw `±pi/4`; key inserted at slot bias `[-0.0025, 0, 0.0785]` and turned by `key_rotation` in uniform `[-pi/2, -pi/4]` (`envs/pull_out_key.py:32-43`).
  2. pre_move: grasp key head (z bias uniform `[-0.015, -0.01]`, `pre_dis=0.08`) (`envs/pull_out_key.py:45-59`).
  3. Rotate about gripper local z by `-key_rotation + over_rotate`, `over_rotate` uniform `[0.09, 0.16]` rad — untwist past neutral.
  4. Rotate back by `-over_rotate + 0.05`.
  5. Pull: z −0.03 in gripper local frame (dilation 0.2); delay 20 (`envs/pull_out_key.py:65-81`).
- **Success criteria** (`envs/pull_out_key.py:98-108`):
  - `key_pose.p[2] > 0.09` — key raised out of the slot;
  - `dot(key local z-axis, [0, 0, 1]) > 0.965` — upright within ≈15°;
  - `slot_x_rotate > 0.99` — the slot was not twisted (its x axis within ≈8° of initial);
  - `|ee z − key z| < 0.14` — key still near the gripper (not dropped).
- **Failure modes** (`envs/pull_out_key.py:83-96`): early stop if `z_dis >= 0.14` (dropped/left behind) or `slot_x_rotate < 0.99` (slot rotated with the key). Shared timeout at default `step_lim = 300`.
- **Tunables**: friction 2.0 and geometry biases inline (`envs/pull_out_key.py:10-12,17`).

### put_bottle_in_shelf
- **Goal**: Grasp a bottle from the table, reorient it upright, and place it on a shelf ledge.
- **Objects/assets**: `Shelf.usd`, `BottleLift.usd`; friction 1.0 (`envs/put_bottle_in_shelf.py:10-29`).
- **Action flow**:
  1. pre_move: grasp the bottle from above (z bias `0.11 + 0.01*rand`, pitch noise `pi/18`); close; place target = shelf + `[-0.2, 0, 0.21]` (`envs/put_bottle_in_shelf.py:40-61`).
  2. Lift z by `0.15 + uniform(0, 0.05)`.
  3. Pitch the wrist by `-pi/3 - 0.2*rand` about y.
  4. `gravity_rotate` the bottle so its local z axis points up.
  5. Place onto `place_target` (`pre_dis=0.01, dis=0.005`, dilation 0.3); open gripper 0.5; delay 20 (`envs/put_bottle_in_shelf.py:63-81`).
- **Success criteria** (`envs/put_bottle_in_shelf.py:92-95`): with `bottle_pose = bottle rebased to place_target`:
  - `|x| < 0.02`, `|y| < 0.1`, `|z| < 0.02`;
  - `dot(bottle local z-axis, [0, 0, 1]) > 0.965` — upright within ≈15°.
- **Failure modes** (`envs/put_bottle_in_shelf.py:83-90`): early stop when min tactile gel depth `< 20` mm (squeezed too hard). Shared timeout at `step_lim = 600`.
- **Tunables**: `step_lim = 600`; `adaptive_grasp_depth_threshold = 27.5` mm (`envs/put_bottle_in_shelf.py:6-7`).

### grasp_classify
- **Goal**: Classify a prism by tactile surface texture and place it on the matching pad — rough prism → orange pad, plain (smooth) prism → green pad.
- **Objects/assets**: `GreenPad.usd`, `OrangePad.usd`, `RoughPrism.usd`, `PlainPrism.usd`; one prism chosen at random per episode (`envs/grasp_classify.py:38-76`); friction 2.5; `use_adaptive_grasp = False`.
- **Action flow**:
  1. pre_move: open gripper 0.5; grasp the prism from above (z bias `0.04 + 0.01*rand`).
  2. pre_move: close to a fixed light grip — `gripper_qpos = uniform(0.0065, 0.0075)/0.039` (`envs/grasp_classify.py:93-94`); lift z +0.05.
  3. Place onto `target_pose = correct pad + [0, 0, 0.015]` (`pre_dis=0, dis=0`, gripper stays closed).
  4. Delay 20 (`envs/grasp_classify.py:99-106`).
- **Success criteria** (`envs/grasp_classify.py:108-111`): with `card_pose = prism rebased to target_pose` (the pad matching its texture):
  - `|x| < 0.02`, `|y| < 0.02`, `|z| < 0.01` — resting on the correct pad;
  - `dot(prism local z-axis, [0, 0, 1]) > 0.965` — flat within ≈15°.
  - Placing on the wrong pad fails implicitly: `target_pose` is defined only over the texture-correct pad (`envs/grasp_classify.py:65-76,97`).
- **Failure modes**: no task-specific early stop; shared timeout at default `step_lim = 300`.
- **Tunables**: `use_adaptive_grasp = False`; grip closure range `0.0065–0.0075 / 0.039` inline (`envs/grasp_classify.py:93`).

### grasp_chip
- **Goal**: Pick up a fragile thin chip and hold it aloft without ever squeezing hard enough to "crush" it. No physical fracture is simulated; tactile indentation is the force proxy.
- **Objects/assets**: `CHIP.usd` (thin curved strip, 82.5 × 18 × 2.4 mm); friction 2.5; planner ignores in-hand `chip` (`envs/grasp_chip.py:60-66,80-88`).
- **Action flow**:
  1. pre_move: settle 10; record start ee height; reset `max_pressure_seen` (`envs/grasp_chip.py:91-95`).
  2. Open fully; descend from above and grasp the chip across its two ends (z bias `GRASP_Z=0.004`, random pitch `±pi/18`).
  3. Adaptive light close (or fixed-depth close if `CHIP_GRIP_DEPTH>0`); record `grasp_ee_z`.
  4. Single direct lift z +`LIFT` (0.08); hold 20 steps. Pressure tracked after every stage (`envs/grasp_chip.py:97-118`).
- **Success criteria** (`envs/grasp_chip.py:125-141`): pressure `p = tactile_far_plane − min(get_min_depth())` in mm (`envs/grasp_chip.py:71`; gsmini far plane 34.0 mm). All three must hold:
  - `held`: current `p > HOLD_PRESSURE` (1.0 mm) — still in the fingers;
  - `not_crushed`: episode-max pressure `<= MAX_PRESSURE` (8.5 mm default, env `CHIP_MAX_PRESSURE`); the max is also accumulated inside `check_success` during eval (`envs/grasp_chip.py:129-133`);
  - `lifted`: `ee_z − grasp_ee_z > LIFT*0.7` = 0.056 m; `grasp_ee_z` is recorded at the scripted grasp (collect) or at the first step with `p > HOLD_PRESSURE` (eval).
- **Failure modes**: `check_early_stop` fires whenever `max_pressure_seen > MAX_PRESSURE` — chip crushed (`envs/grasp_chip.py:121-123`); during collection `_track` also fails `plan_success` on the same condition. Shared timeout at `step_lim = 600`.
- **Tunables** (`envs/grasp_chip.py:23-31`): `CHIP_REST_Z = 0.01` `[tune]`; `GRASP_Z = 0.004` `[tune]`; `LIFT = 0.08`; `MAX_PRESSURE = 8.5` mm `[tune]` (env `CHIP_MAX_PRESSURE`); `HOLD_PRESSURE = 1.0` mm `[tune]`; `GRIP_DEPTH` (env `CHIP_GRIP_DEPTH`, 0 = adaptive light close; >0 = close to given tactile depth).

### pour_ball
- **Goal**: Grasp a cup containing a small ball, carry it above a plate, and tilt so the ball pours out and lands on the plate.
- **Objects/assets**: `CUP.usd` (paper cup, scale 0.93, density 5e3), `BALL.usd` (OD 16 mm, density 2000), `PLATE.usd` (`dish`, kinematic, density 1e4); friction 2.0; planner ignores `{cup, ball, dish}` (`envs/pour_ball.py:86-116`).
- **Action flow**:
  1. pre_move: settle 15; open gripper (`envs/pour_ball.py:186-188`).
  2. `_grasp_cup` (`envs/pour_ball.py:127-157`): lift clearance +0.12; approach the cup's +Y side; descend beside it; insert horizontally along gripper local z; close to `GRIP_CLOSE = 0.60` (loose geometric closure).
  3. `_pour` (`envs/pour_ball.py:159-183`): move gripper directly to the pour prep pose above the plate (`x = plate_x − 0.03`, `z = POUR_Z = 0.26`).
  4. Rotate wrist joint 7 by `POUR_ANGLE = 210°` in 80 interpolation steps.
  5. Dwell 30 steps so the ball settles.
- **Success criteria** (`envs/pour_ball.py:195-215`): all three:
  - `on_plate`: ball-to-plate-center horizontal distance `< PLATE_R − 0.005` = 0.095 m;
  - `low`: ball z `< 0.03` m — resting on the plate;
  - `ball_out`: ball-to-cup-axis horizontal distance `> CUP_TOP_R` ≈ 0.0353 m — actually poured out (criteria fix 2026-07-14: without this, lowering the cup with the ball inside was a false success, `envs/pour_ball.py:201-208`).
- **Failure modes**: no task-specific early stop; shared timeout at `step_lim = 800`; `reset_time_limit = 1200` s for slow UIPC solves (`envs/pour_ball.py:58-60`).
- **Tunables** (`[tune]` blocks, `envs/pour_ball.py:23-53`): geometry (`CUP_SCALE=0.93`, `BALL_R=0.008`, `PLATE_R=0.10`), layout (`PLATE_POS`, `CUP_POS`, `GAP=0.006`), grasp/pour (`GRIP_CLOSE=0.60`, `POUR_Z=0.26`, `POUR_ANGLE=210°`, `POUR_JOINT_STEPS=80`, …). `use_adaptive_grasp = False`.

### phone_socket_replug
- **Goal**: Pull a phone plug out of the orange source base and re-insert it into the dark-gray target base.
- **Objects/assets**: `PHONE_SOCKET_BASE.usd` ×2 (source orange, target dark gray, density 1e5), `PHONE_PLUG.usd` (density 3e3, pins face local −Z); friction 2.0; planner ignores `plug` (`envs/phone_socket_replug.py:66-92`).
- **Action flow**:
  1. reset: source/target xy noise ±0.004/±0.006; plug positioned `INIT_DROP_CLEARANCE = 0.012` above the source hole; pre_move waits 120 steps for it to drop in naturally (`envs/phone_socket_replug.py:85-101,128-129`).
  2. `_grasp_plug_from_source`: open; top grasp at plug + `GRASP_DZ = 0.003`; hover +0.06, drop to grasp pose; close (planner close + 12-step forced direct close) (`envs/phone_socket_replug.py:115-126`).
  3. Unplug: lift z +`UNPLUG_LIFT = 0.10` with rotation-about-z relaxed.
  4. Carry: hover at `target_hole_pose + 0.06`, then align at `target_hole_pose = target base + [0, 0, BASE_HEIGHT + PLUG_TIP]` (`envs/phone_socket_replug.py:95,140-162`).
  5. Insert: local z `+0.65*INSERT_DEPTH` then `+0.35*INSERT_DEPTH` (total 0.011), rotation-about-z relaxed; open; delay 20 (`envs/phone_socket_replug.py:163-176`).
- **Success criteria** (`envs/phone_socket_replug.py:178-191`): both:
  - `inserted_target` (rel to `target_hole_pose`): `|x| < 0.005`, `|y| < 0.005`; `z < -0.008` (negative threshold requires real insertion — criteria fix 2026-07-14: the old `< 0.006` counted hovering as inserted); `dot(plug local z, [0,0,1]) > 0.95` (≈18°);
  - `clear_source` (rel to `source_hole_pose`): `|z| > 0.012` m or horizontal `norm(xy) > 0.02` m — no longer in the source hole.
- **Failure modes**: no task-specific early stop; shared timeout at `step_lim = 900` (`envs/phone_socket_replug.py:63`).
- **Tunables** (`envs/phone_socket_replug.py:19-35`): `BASE_HEIGHT = 0.012`, `HOLE_DEPTH = 0.0115`, `PLUG_TIP = 0.0162`, `GRASP_DZ = 0.003`, `UNPLUG_LIFT = 0.10`, `INSERT_DEPTH = 0.011`, `INIT_DROP_CLEARANCE = 0.012`, source/target positions and xy noise.

---

## Dual-arm tasks

### dual_screw_sleeve
- **Goal**: Arm A grasps a lying sleeve and holds it up at a tilted assembly pose; arm B grasps a screw by its disc and inserts the shaft into the sleeve bore.
- **Objects/assets**: `SLEEVE.usd` (hollow cylinder, length `SLEEVE_H=0.044` m, outer radius 0.018 m, bore along local Z), `SCREW.usd` (thumbtack shape, origin-to-tip `SCREW_TIP=0.0385` m).
- **Action flow**:
  1. Arm A: adaptive top-down grasp of the lying sleeve (tactile depth 28), holds it at `ASSEMBLY_POSE = (0.48, 0.18, 0.22)` with yaw tilt −0.3 rad (bore toward arm B) (lines 45–46, 203–205).
  2. Arm B: adaptive grasp of the screw disc from above (`SCREW_GRASP_DZ=0.033`), lifts +0.15 m (lines 208–210).
  3. Arm B: hovers 0.08 m outside the bore opening along the tilted bore axis, aligns the screw tip outside the opening (lines 218–222).
  4. Arm B: constrained linear insertion along gripper-local Z in two feeds (`0.6*` then `0.5*INSERT_DEPTH`, `INSERT_DEPTH=0.055`), other 5 axes locked (lines 227–230); 20-step settle.
- **Success criteria** (`envs/dual_screw_sleeve.py:234-248`; screw pose in the sleeve frame, bore = sleeve local Z):
  - axis alignment `|dot(screw Z, sleeve Z)| > 0.9`;
  - radial offset `‖rel.p[x,y]‖ < 0.008` m;
  - depth `|rel.p[z]| < SLEEVE_H/2 + 0.02 = 0.042` m.
  - All three ANDed; no release requirement, no hold count — success valid while both arms still hold their objects.
- **Failure modes**: none explicit; `step_lim=1500`. Planners ignore in-hand objects (A: `sleeve`; B: `screw`+`sleeve`).
- **Tunables**: `INSERT_DEPTH=0.055` `[tune]`; layout `[tune]` (`ASSEMBLY_TILT=[0,0,-0.3]`, `ASSEMBLY_POSE`); grasp `[tune]` (`SLEEVE_GRASP_DZ=0.0`, `SCREW_GRASP_DZ=0.033`); friction 2.5.

### dual_bowl_unstack
- **Goal**: Arm A extracts the upper bowl from a nested pair by pinching its rim and places it on a target plate on the +Y side. **Arm B is instantiated but idle for the entire task.**
- **Objects/assets**: 2× `BOWL.usd` (130 mm dia, 39.4 mm tall; lower `bowl_a` density 2e5 ≈ 11 kg, upper `bowl_b` 3e3), `PLATE.usd` target (kinematic) at (0.45, 0.30), `PLATE_R=0.10`.
- **Action flow**:
  1. Arm A: radial rim pinch on the upper bowl's +Y rim (one finger inner wall, one outer; adaptive grasp off, forced joint-level clamp) (lines 180–219, 267–268).
  2. Extracts straight up `EXTRACT_RISE=0.16` m in two constrained feeds (5 axes locked) (lines 273–276).
  3. Moves above the plate (hover `RELEASE_HOVER_Z=0.16`), descends to `PLACE_POSE` + `PLACE_SETTLE_DROP=0.008` (lines 281–291).
  4. Settles 8 steps, opens fully (forced open), 25-step delay (lines 292–295).
- **Success criteria** (`envs/dual_bowl_unstack.py:298-328`) — **all held for `SUCCESS_HOLD_STEPS=8` consecutive checks**:
  - `separated`: bowl-center xy distance > 0.12 m;
  - `bottom_ok`: lower bowl `dot(local Z, world Z) > 0.7`;
  - `b_up`: upper bowl `dot > 0.6`;
  - `on_plate`: bowl_b to plate center xy < `PLATE_R − BOWL_OUTER_R + 0.01 = 0.045` m;
  - `released`: arm A gripper opening > 0.9.
- **Failure modes**: none explicit; a broken condition resets the hold counter. `step_lim=1200`.
- **Tunables**: layout `[tune]` (`STACK_POS`, `PLATE_POSE`); grasp `[tune]` (`RIM_GRASP_DZ`, `EXTRACT_RISE=0.16`, `RIM_DOWN_EXTRA=0.020`, `RELEASE_HOVER_Z=0.16`, `SUCCESS_HOLD_STEPS=8`); nesting `NEST_RISE=0.038`/`NEST_GAP=0.014`; friction 5.0.

### dual_gear_holder
- **Goal**: Place three flat gears onto the three vertical pegs of a gear holder; arm A handles gear_0 and gear_2, arm B handles gear_1.
- **Objects/assets**: `GEAR_HOLDER.usd` (disc base + 3 pegs at 90°/210°/330° on a 0.032 m circle, height 0.0355 m; density 1e5), 3× `GEAR.usd` (thickness 0.016 m, center bore along local Z, density 8e3).
- **Action flow** (each grasp/place returns False and aborts the episode on planning failure):
  1. Arm A: top grasp gear_0 (fixed close `GRIP_CLOSE=0.5`, adaptive off), lift 0.055 m (lines 200–232, 329).
  2. Arm A: place gear_0 on peg 0 — hover at `HOVER_Z` (peg top + half gear + 0.015), constrained straight lower to `RELEASE_Z = SETTLED_Z + 0.009` so the bore threads the peg before release; open; retreat 0.06; cosmetic side-clear (lines 244–308, 332).
  3. Arm B: same cycle for gear_1 on peg 1 (lines 336–340).
  4. Arm A: same cycle for gear_2 on peg 2 (lines 343–347); 5-step delay.
- **Success criteria** (`envs/dual_gear_holder.py:351-374`), for each gear i vs its settled pose (`SETTLED_Z = 0.021` m):
  - `radial < 0.012` m (xy to peg axis); `z_err < 0.018` m; `up > 0.75`;
  - `released`: **arm A only** gripper opening > 0.9 (B rests near ~0.5 after its earlier release);
  - return `all(ok) and released`; no hold count.
- **Failure modes**: explicit — `_play_once` returns False if any grasp/place fails to plan (final side-clear exempt). `step_lim=2400`.
- **Tunables**: `GRASP_Z_BIAS=−0.002`, `PRE_GRASP_UP=0.08`, `LIFT_UP=0.055`, `GRIP_CLOSE=0.5`, `RELEASE_Z` (+0.009), `HOVER_Z` (+0.015), `GEAR_NOISE=0.008`; friction 2.5.

### dual_bowl_place_stack
- **Goal**: Arm A carries bowl_a from the +Y side to a central target; arm B then nests bowl_b on top of it.
- **Objects/assets**: 2× `BOWL_STACK.usd` (130 mm dia × 58 mm tall vertical-wall bowl; `bowl_a` density 1e4, `bowl_b` 5e3). `USE_WELD=False` — held by contact friction only.
- **Action flow**:
  1. Arm A: rim pinch on bowl_a's +Y rim (hover → constrained descent + `RIM_DOWN_EXTRA=0.020` → full close + forced clamp), lift 0.10 (lines 236–273, 337).
  2. Arm A: place at `STACK_TARGET=(0.48, 0, table+BOWL_HALF+0.006)` via 0.12 hover; release, retract 0.12, park (lines 338–347).
  3. Arm B: mirrored rim pinch on bowl_b's −Y rim, lift (line 351).
  4. Arm B: place bowl_b at the fixed nominal `STACK_TOP_TARGET` (+`NEST_RISE=0.048` in z); release without retract; 30-step delay (lines 352–362).
- **Success criteria** (`envs/dual_bowl_place_stack.py:365-407`):
  - `target_err`: bowl_a → STACK_TARGET xy < 0.04 m;
  - `stack_err`: bowl_b → bowl_a xy < 0.03 m;
  - `nested`: `0.025 < dz < 0.072` m (`dz = bowl_b.z − bowl_a.z`);
  - `a_up`, `b_up`: `dot > 0.75` both bowls.
  - Note: `released` (both grippers ≥ 0.65 + no welds) and the 8-step hold counter are computed and stored in metadata but **not part of the final return** (see Known quirks #3).
- **Failure modes**: none explicit; `step_lim=2600`.
- **Tunables**: `NEST_RISE=0.048`, `USE_WELD`, grasp constants (`RIM_GRASP_R=0.0625`, `RIM_DOWN_EXTRA=0.020`, `GRASP_LIFT=0.10`), `PLACE_HOVER=0.12`, `GRIPPER_RELEASE_THRESH=0.65`, `SUCCESS_HOLD_STEPS=8`; friction 5.0.

### dual_cup_place_stack
- **Goal**: Arm A carries cup_a to a central target; arm B nests cup_b on top of cup_a's **actual** resting pose.
- **Objects/assets**: 2× `CUP.usd` thin-wall conical paper cup, scale 0.92 (`CUP_HALF=0.04232`), density 2e5 each. `USE_WELD=False`.
- **Action flow**:
  1. Arm A: staged top grasp across cup_a's rim (staging → vertical descent → 5-axis-locked press-down → close to `TOP_GRASP_CLOSE=0.5`; full close would crush the soft cup) (lines 195–217, 276).
  2. Arm A: place at `STACK_TARGET=(0.48, 0, table+CUP_HALF+0.006)` via 0.11 hover; release, retract, park (lines 277–286).
  3. Arm B: same staged top grasp on cup_b (line 289).
  4. Arm B: place cup_b at cup_a's **actual** xy + `NEST_RISE=0.04784` in z (nominal-target placement caused rim collisions, ~73% failure — lines 290–307); release; 30-step delay.
- **Success criteria** (`envs/dual_cup_place_stack.py:313-356`):
  - `target_err`: cup_a → STACK_TARGET xy < 0.035 m;
  - `stack_err`: cup_b → cup_a xy < 0.025 m;
  - `nested`: `0.010 < dz < 0.055` m;
  - `a_up`, `b_up`: `dot > 0.75` both cups.
  - Note: released/hold-counter machinery logged only, **not in the final return** (Known quirks #3).
- **Failure modes**: none explicit; `step_lim=2400`.
- **Tunables**: `CUP_SCALE=0.92`, `OVERLAP=0.040*scale`, top-grasp constants (`TOP_GRASP_DZ`, `TOP_GRASP_CLOSE=0.5`), `PLACE_HOVER=0.11`, `GRIPPER_RELEASE_THRESH=0.65`, `SUCCESS_HOLD_STEPS=8`, `USE_WELD`; friction 5.0.

### dual_cup_handover_place
- **Goal**: Arm A grasps a cup from above and brings it to a mid-air handover pose; arm B receives it from the side; A releases; B places the cup upright on a target plate on B's side.
- **Objects/assets**: `CUP.usd` (scale 0.92, density 2e5), `PLATE.usd` target (kinematic) at (0.50, −0.26), plus a hidden `planner_anchor` (keeps cuRobo's obstacle world non-empty, lines 183–190). `USE_WELD=False`.
- **Action flow**:
  1. Arm A: staged top grasp (`A_GRASP_CLOSE=0.5`), lift +0.12 constrained (lines 411–417).
  2. Arm A: carry to `HANDOVER_TARGET=(0.50, 0.02, 0.20)` via 0.08 hover (lines 418–424).
  3. Arm B: side-receive along −Y at cup mid-body height, pre-back 0.08, into 0.015 past the cup axis, close to `SIDE_CLOSE_GRIPPER_POS=0.60` (lines 305–328, 428).
  4. Arm A: open fully, pause 10, retract 0.10, park (line 432).
  5. Arm B: place at `PLACE_TARGET=(0.50, −0.26, ≈0.060)` via 0.11 hover; **B does not release**; 30-step delay (lines 436–442).
- **Success criteria** (`envs/dual_cup_handover_place.py:445-476`; final return = `on_plate_upright` only):
  - `target_err`: cup → PLACE_TARGET xy < 0.035 m;
  - `height_err`: |cup z − target z| < 0.025 m;
  - `up`: `dot(cup Z, world Z) > 0.75`.
  - `released` explicitly **not required** ("B keeps holding the cup"); hold machinery logged only.
- **Failure modes**: none explicit; `step_lim=2400`.
- **Tunables**: `HANDOVER_TARGET`; grasp/receive constants (`A_GRASP_CLOSE=0.5`, `SIDE_CLOSE_GRIPPER_POS=0.60`, `B_GRASP_SIDE=0.05`, `A_LIFT_UP=0.12`); hover/park constants; `USE_WELD`; friction 5.0.

### dual_plate_place_stack
- **Goal**: Arm A places plate_a at a central target; arm B stacks plate_b directly on top of plate_a's **actual** pose.
- **Objects/assets**: 2× `PLATE_STACK.usd` (190 mm dia × 40 mm tall vertical-wall tray; `plate_a` density 1e4, `plate_b` 5e3). `USE_WELD=False`.
- **Action flow**:
  1. Arm A: rim pinch on plate_a's +Y rim (`RIM_GRASP_R=0.091`; hover → constrained descent + `RIM_DOWN_EXTRA=0.014` → full close + clamp), lift 0.10 (lines 216–255, 318).
  2. Arm A: place at `STACK_TARGET=(0.50, 0, table+PLATE_HALF_Z)` via 0.10 hover; release, retract 0.10 (plain move — a constrained retreat often fails to plan), park (lines 281–291, 319–328).
  3. Arm B: mirrored rim pinch on plate_b's −Y rim (lines 332–333).
  4. Arm B: place plate_b at plate_a's actual xy + `STACK_DZ=0.028` in z; release with retract; 30-step delay (lines 338–353).
- **Success criteria** (`envs/dual_plate_place_stack.py:356-397`):
  - `target_err`: plate_a → STACK_TARGET xy < 0.04 m;
  - `stack_err`: plate_b → plate_a xy < 0.03 m;
  - `nested`: `0.012 < dz < 0.050` m;
  - `a_up`, `b_up`: `dot > 0.8` both plates.
  - Note: released/hold machinery logged only, **not in the final return** (Known quirks #3).
- **Failure modes**: none explicit; `step_lim=2600`.
- **Tunables**: `STACK_DZ=0.028`, grasp constants (`RIM_GRASP_R=0.091`, `RIM_DOWN_EXTRA=0.014`, `GRASP_LIFT=0.10`), `PLACE_HOVER=0.10`, `GRIPPER_RELEASE_THRESH=0.65`, `SUCCESS_HOLD_STEPS=8`, `USE_WELD`; friction 5.0.

### dual_cup_unstack
- **Goal**: Arm A grasps the upper of two nested paper cups by the rim from above, extracts it vertically, and sets it down on the +Y side; the heavy lower cup stays put by weight. **Arm B is instantiated but idle** (a swapped-role variant is described in comments but not implemented).
- **Objects/assets**: 2× `CUP.usd` (scale 0.92, `CUP_HALF=0.04232`, density 2e5 each); upper cup starts nested `NEST_RISE=0.06624` m (+`NEST_GAP=0.005`) above the lower.
- **Action flow**:
  1. Arm A: two-stage top grasp of the upper cup's rim (`CUP_B_GRASP_DZ=CUP_HALF−0.006`; staging → descent → constrained linear feed → adaptive close to tactile depth 22) (lines 165–194, 223–225).
  2. Lift straight up `EXTRACT_RISE=0.13` m, 5 axes locked (lines 230–231).
  3. Move to 0.10 above `PLACE_POSE=(0.45, 0.22, 0.050)`, descend −0.10, settle 8 (lines 235–240).
  4. Open fully (release), 25-step delay (lines 241–243).
- **Success criteria** (`envs/dual_cup_unstack.py:246-277`) — **all held for `SUCCESS_HOLD_STEPS=8` consecutive checks**:
  - `separated`: cup-center xy distance > 0.12 m;
  - `bottom_ok`: lower cup `dot > 0.7`;
  - `b_up`: upper cup `dot > 0.7`;
  - `placed`: cup_b center z < `TABLE_TOP + CUP_HALF + 0.035 = 0.08132` m (set down, not held aloft);
  - `released`: arm A gripper opening > 0.9.
- **Failure modes**: none explicit; a broken condition resets the hold counter. `step_lim=1500`.
- **Tunables**: layout `[tune]` (`STACK_POS`, `PLACE_POSE`; `HOLD_POSE` defined but unused), grasp `[tune]` (`CUP_B_GRASP_DZ`, `EXTRACT_RISE=0.13`, `NEST_GAP=0.005`, `SUCCESS_HOLD_STEPS=8`; `CUP_A_GRASP_DZ` defined but unused), `CUP_SCALE=0.92`/`OVERLAP=0.020*scale`; friction 5.0.
