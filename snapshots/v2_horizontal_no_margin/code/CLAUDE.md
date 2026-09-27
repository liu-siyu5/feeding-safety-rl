# Project: safety mechanisms for robot-assisted feeding (RL)

Research code for a paper comparing four safety mechanisms on a bite-transfer
task in RCareWorld / Unity. Read this before writing or changing any code in
this directory; several platform behaviours here are counter-intuitive and
have already cost days of debugging.

Environment: conda env `rcareworld`, Python 3.10.
Always `conda activate rcareworld` first. If the prompt shows `(.venv)` in
front, run `deactivate` — a stray venv from another project shadows the conda
environment and `pyrcareworld` will not import.

---

## Verified platform facts

These were all measured on this exact setup. Do not re-derive them, and do not
write code that assumes otherwise.

### Scene objects

| object | id | notes |
|---|---|---|
| robot (Kinova Gen3) | 315893 | 7 revolute joints |
| mouth | 9999 | world position `[0.2604, 0.8994, -0.5484]`, static (0.00 mm drift over 600 frames) |
| spoon | 114514 | kinematic rigid body |
| human (SMPL-X) | 2333 | `HumanArticulationAttr` is a **stub class I wrote**; it exposes no skeleton data |
| food | 19024 | not used by the current task |

The human's skeleton (`names`, `positions`) comes back empty. Do not try to
read `head`, `neck`, `jaw` or eye positions from it.

### Joint control

Velocity control only works after releasing the position drive:

```python
robot.SetJointStiffness([0.0] * 7)
robot.SetJointDamping([1000.0] * 7)
robot.SetJointVelocity(cmd)          # cmd in command units
```

Without this the default position drive (stiffness 10000) fights the velocity
command and the joints oscillate wildly — measured velocity swung from +50 to
−44 deg/s under a constant command of 5.

Calibration: `measured deg/s ≈ 11.3 × command`. Linear for `|cmd| ≤ 5`;
saturates above that (cmd 10 gave only 66 deg/s instead of 113).

**But velocity control does not move the joints as commanded** (measured
2026-09-25 at the start pose, one joint commanded at a time): the commanded
joint moves 8–290 % of the commanded amount and the others are dragged by up
to 26°; joints 5 and 6 barely respond to their own commands (commanding joint 6
mostly turns joint 4). C4's prediction `dp = J·q̇·dt` does not hold under it.

The environment therefore defaults to **position-target actuation**: the
commanded joint speed (same 11.3 deg/s per unit, `|cmd| ≤ 5`) is integrated into
a position target tracked by the stiff position drive below. Measured: 94–112 %
of the commanded motion, ≤ 2.4° on the other joints at full speed, no drift at
zero command. The target may lead the measured joints by at most 10°
(anti-windup). `FeedingTaskEnv(actuation="velocity")` restores the old mode;
all runs before 2026-09-25 used it.

Because the arm keeps heading for the target, the lead is motion still to come:
~2 cm at the tip while moving, up to ~5 cm after the spoon has been held up at
the lips. The arm is also fast and has momentum: the tip can move 7–8 cm in one
control step, and it bounces off the face by up to 8 cm. C4 therefore predicts
from lead + joint velocity × dt + this step's increment
(`SafetyContext.target_lead_deg`, `joint_vel_deg`) and lets the tip close at
most half of its remaining margin to the 10 N distance per step
(`C4PredictiveShield.barrier_rate = 0.5`, a discrete-time barrier function).
Measured 2026-09-26 on 20 fixed starts: increment only, 7 of 20 episodes went
past 10 N; with the lead 0 of 20 for that policy, but a policy retrained with it
rushed in (15 cm in ~5 steps) and went past in 6 of 20; the velocity term alone
did not help (6 of 20); velocity + rate 0.5 gave 1 of 20 (policy not retrained).
The user chose the rate ("slower the closer") over a global speed limit, which
would also change C1–C3.

**C4 scales a violating action, it does not project it.** The minimum-norm
projection onto the constraint removes only the motion toward the mouth and
keeps the sideways part, so the tip slid along the limit, 3–6 cm off the centre
line: a C4 policy trained with it never succeeded in 100k steps, and even the
trained C1 policy (9/20 successes on its own, 11/20 over 10 N) got 0/20 under
it. Scaling the whole action (`s·a`, same direction, slower; a radial back-off
only when stopping is not enough) gave that policy 16/20 (rate 1) or 8/20
(rate 0.5), none over 10 N (2026-09-26). `train_short.py --barrier-rate` sets
the rate for a run. Retrained with scaling (100k steps, `models/short/v10`), on
the 20 fixed starts: rate 0.5 → 7 successes, 0 over 10 N, the other 13 timed
out at the mouth 1.2–2.1 cm off the centre line; rate 1 → 11 successes, 2 over
10 N (10.5, 11.1 N), 7 timed out; C1 on the same starts 9 successes, 11 over
10 N.

**`SetJointPositionDirectly` keeps the joint velocities.** Teleport + 30 ticks
left the arm up to 3° off and still moving at 66 deg/s, so each episode
inherited the previous one. The reset now brakes first (stiffness 0, damping
1000, 10 ticks), then teleports, then holds with the position drive: start
pose identical to ~0.005° whatever came before.

Reproducibility: with that reset, replaying a recorded **action sequence**
gives the same trajectory (≤ 0.03°, time scale 2 or 10, GUI or headless).
Re-running a **policy** from the same seed does not: an under-trained policy
amplifies the ~0.005° start difference to tens of degrees within ~5 steps. To
show a specific episode, record its actions and replay them open-loop.

Position control (for setting a pose directly):

```python
robot.SetJointStiffness([10000.0] * 7)
robot.SetJointDamping([100.0] * 7)
robot.SetJointPosition(q_deg)
env.step(60)                          # let it settle
```

**`IKTargetDoMove` does not work on this robot.** It returns without error and
the arm does not move. Use `SetJointPosition` or `SetJointVelocity` only.

### Joint limits

Joints 1, 3, 5: `±138.1`, `±152.4`, `±127.8` degrees.
Joints 0, 2, 4, 6: the platform reports `[0, 0]`, which means **unlimited
continuous rotation**, not locked. Treating `[0, 0]` as a real limit caused
every episode to terminate on step 1. Use a large sentinel (1e6) or skip the
check for those joints.

### Timing

- Physics tick: 0.02 s (50 Hz)
- Control interval: 3 ticks = 0.06 s (~16.7 Hz). 0.05 s is not an integer
  multiple of the tick, so 20 Hz is not achievable.
- `env.SetTimeScale(10)` runs the simulator at 10× real time. Verified:
  trajectories under a fixed action sequence are identical (0.00 mm deviation)
  at scales 1, 2, 5, 10. Throughput goes from 4 to ~120 env steps/s.
- PPO runs on **CPU**. With a 2×64 MLP the GPU is slower and SB3 warns about it.
  Runs in parallel are launched with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`
  (since 2026-09-26) so their torch threads do not oversubscribe the 8 cores.
- Each simulator instance binds one port, `port + 1` (pyrcareworld
  `base_env.py`); if it is taken it tries `+256`. Adjacent `port` values are fine.
- SB3 `model.save(name)` adds `.zip` only if `name` has no suffix, and a name with
  a dot (`C2_lam0.05_seed0`, `C4_rate0.5_...`) has one. Both training scripts
  therefore save to an explicit `<tag>.zip`.

### Ports

Each process must bind its own port or the second one dies with
`OSError: [Errno 98] Address already in use`:

```python
FeedingEnv(graphics=False, port=5100 + offset)
```

`train_eval.py` already derives a unique port per (condition, lambda, seed).

### Kinematics

Use the analytic Jacobian from the URDF, never finite differences.

```python
from kinematics import Gen3Kinematics
kin = Gen3Kinematics()                  # loads gen3_kinematics_only.urdf
p_tip = kin.fk_tip_unity(q_deg)         # spoon tip (bowl), Unity frame, m
J_tip = kin.jacobian_unity(q_deg)       # 3 x 7, spoon tip, metres per radian
```

- `gen3_kinematics_only.urdf` is `GEN3_URDF_V12.urdf` with all `<mesh>` tags
  replaced by boxes. The original fails to load (`PackageNotFoundError: meshes`).
  Kinematics are unaffected.
- The URDF's `@GraspPoint_Link` at `SE3(0, 0, 0.15)` from the end-effector is
  the **gripper fingertip centre**, i.e. what the simulator reports as
  `grasp_point_position`. It is **not** the spoon tip. The spoon tip is the
  **front of the bowl**, 22.3 cm further along the tool axis:
  `kinematics.SPOON_TIP_IN_GRASP = (0, 0.0093, 0.2227)` m in that link's frame,
  taken from the bowl Box Collider the user fitted to the spoon mesh in Unity
  (bowl centre at 17.5 cm). An earlier visual estimate of 13 cm was 9 cm short,
  which is why a "closest attempt" showed the spoon in the nose.
  `fk_tip_unity` / `jacobian_unity` return the tip, `fk_grasp_unity` the grasp
  point. A raw `rb.jacob0(q)[:3]` is the grasp point's Jacobian, not the tip's.
- The URDF -> Unity mapping `R = [[0,-1,0],[0,0,1],[1,0,0]]` is a reflection
  (det −1): do rotations in the URDF frame, map only positions and linear
  velocities.
- The frame mapping was validated against the simulator's own readout over
  several poses: max discrepancy 1.2 mm.
- Analytic tip Jacobian: ~0.07 ms (0.012 ms for `jacob0` alone; the tip offset
  needs one FK). Finite differencing costs 8.4 s per control step in this
  simulator and the estimates varied by a factor of 8 with settling time. Do
  not use it.

**IK caveat:** `rb.ikine_LM` returns solutions that the simulator often cannot
reach — the arm collides with the wheelchair or the user's thigh and stalls
20+ degrees short on joint 1. Always verify an IK solution by commanding it and
reading back `grasp_point_position`, rather than trusting the solver.

### Collisions

The arm is mounted on the wheelchair's right armrest. Reaching the mouth
requires passing near the user's thigh and forearm. Naive IK solutions get
blocked there. This configuration puts the **grasp point** 0.0137 m from the
mouth:

```python
q = [38.3, -33.4, -99.0, -45.0, -1.4, -40.5, 22.2]   # q_reach
```

It is **not** a feeding pose: the spoon passes through the side of the head and
the bowl ends 12.5 cm from the mouth, inside the head. The joint-space
interpolation from home to it has tracking error ≤ 2.4°, but
`env.GetCurrentCollisionPairs()` reports the arm touching the human (ids 2322 /
2333) from f = 0.70 onward; f ≤ 0.60 is contact-free.

A pose that does put the bowl at the mouth — from the front, tool axis along
−Z, concave side up — was verified 2026-09-25 (bowl 2.5 cm from the mouth,
0.0° tracking error; reached from `0.6 * q_reach` via a pre-approach 10 cm
further out along +Z):

```python
q_front = [45.3, 15.1, -78.1, -46.6, 36.3, -61.3, 52.9]
```

The gripper reports contact with the human (id 2322) there; where is not yet
known.

**Face collisions (build `FeedingBuild/FeedingPlayer_v2.x86_64`, the default
since 2026-09-25).** In the v1 build the spoon passed through the head in every
configuration tried: its 26 MeshColliders reference a VHACD mesh asset that is
missing from the RCareCommon repo, and the scene overrides the spoon Rigidbody
to kinematic. The user rebuilt the scene (Unity project `~/research/RCareUnity`,
scene `Assets/RCareCommon/Example Scenes/Feeding.unity`):
- spoon: two Box Colliders on `sphere4_extract6_default` (bowl, handle), the
  child Rigidbody removed, layer `Spoon`; the collision matrix lets `Spoon`
  collide only with `Human`;
- visible human (`SMPLX_male`, layer `Human`): `Col_Head` (capsule r 8.9 cm,
  covers skull, nose and upper jaw), `Col_LeftCheek` / `Col_RightCheek`,
  `Col_Jaw`, `Col_Neck`; the mouth is left open between them;
- `Mouth` (9999) and `SpoonForceSensor` (8888) spheres are triggers, so they do
  not block the spoon (the sensor therefore always reads 0);
- the invisible body `man` (misplaced colliders) is disabled;
- the display camera (`RCareWorld/Main Camera`, from `RCareWorld.prefab`) only
  renders layers 0–4 and 16–20 by default, so its Culling Mask was extended to
  include `Human` and `Spoon`. Without that the person and the spoon collide
  but are invisible in the player window.
`FeedingEnv.attach_spoon()` makes the spoon dynamic and fixes it to the end
effector. Measured: along the centre line the bowl front reaches the mouth
point; the nose stops it 2.8 cm in front at 2.5 cm up, the chin 2.0 cm in front
at 3.4 cm down. Under contact the FK tip matches the simulated spoon to < 1 mm
(the arm is stopped, the spoon does not bend); during fast free motion
(> 90 deg/s) they can differ by up to 5 cm, recovering on reset.
`GetCurrentCollisionPairs` reports spoon contacts as `kinova_gen3`.

To make a Rigidbody follow the arm dynamically: `Link(315893, 8)` first, then
`SetKinematic(False)`. The other order moves the spoon ~41 cm away the moment
it becomes dynamic.

`check_reachability_v2.py` is not a reliable reachability test: it reads the
tip 15 ticks after each jump, before the arm has arrived, so results depend on
the previous sample (the same q scored 0.143 m and then 1.058 m). Its
UNREACHABLE verdict for the bowl contradicts the pose above.

---

## Task and reward

Contact is modelled with a penalty model, `F = k·d`, where `d` is the
penetration depth of the spoon tip into a sphere of radius `R = 0.04 m`
centred on the mouth. `k = 500 N/m`. No damping.

Thresholds: comfort `F_c = 6 N`, danger `F_d = 10 N` (the latter follows
Assistive Gym). Success requires the tip within `3 cm`, at which point the
force is 5.0 N, i.e. inside the safe band — success and violation never
coincide. The environment asserts this at start-up.

Feeding-pose constraints (added 2026-09-25): success also requires the tool
axis within 30° of −Z (straight into the face), the concave normal within 30°
of world up, and the bowl **in front of the mouth within 1.5 cm of its centre
line** (the line through the mouth along +Z). Without the last rule the first
learned "success" had the bowl 2.1 cm below the mouth, touching the lower lip
from underneath. All three limits are our own choice (1.5 cm agreed with the
user).

Spoon pitch (added 2026-09-26 at the user's request): the bowl must be level or
tipped down, never higher than the gripper, or the food slides toward the
handle. Pitch = elevation of the tool axis (gripper → bowl, + = bowl higher);
success requires `−20° ≤ pitch ≤ +2°` (`MAX_PITCH_DOWN_DEG`, `MAX_PITCH_UP_DEG`).
Before the rule the start pose was level but episodes ended at +4° on average
and up to +19° after pressing on the lips; the 30° tilt limit did not catch it.

**Hold phase** (added 2026-09-26, the user's choice): arriving in the feeding
pose (all the conditions above) is not enough; the spoon must stay in it for
`HOLD_STEPS = 8` consecutive control steps (0.48 s, the person takes the food),
leaving resets the count, and success = hold completed with the force never
above `F_d`. The episode ends when the hold is complete. Before, the episode
ended on arrival, so a violation on arrival was never followed by a step in
which the reactive shield could act: the trained C1 policy under C3 gave results
identical to C1, C3 never intervening. Metrics: `success`, `held` (ignoring
safety), `reached` (arrived at least once), `hold_best`.

The spoon must also **never pass through the head**. This is now enforced
physically by the face colliders (see Collisions). A Python keep-out box used
before (guessed adult-head sizes) was removed: its dimensions, combined with the
wrong 13 cm tip, let the spoon into the nose. Only the front of the bowl fits
through the mouth opening (bowl 8.5 cm wide, mouth ~5 cm), as intended.

Reward: `r = −w_d · dist − w_c · off_centre − w_o · e_o − w_p · e_p + hold bonus`,
`w_d = 1.0`, `w_c = 10.0`, `w_o = 0.5`, `w_p = 2.0`, `w_s = 10.0`, with
`e_o = ((1 − cos approach) + (1 − cos tilt)) / 2`, `e_p` the pitch outside its
limits (rad) and `off_centre` the tip's distance from the centre line (m).
The hold bonus pays `w_s / HOLD_STEPS` each time the consecutive hold passes its
best so far in the episode, so it totals at most `w_s`; moving in and out of the
pose earns nothing extra. (A per-step bonus while in the pose would make
lingering without completing the hold worth far more than completing it — the
same trap as the known failure mode below.)
The observation (27-D) adds the tool axis and concave normal to the former 21.

Start pose (default `start="pre_feed"`, 2026-09-25): bowl front 15 cm in front
of the mouth on the centre line, tool axis −Z, concave up
(`Q_PRE_FEED = [13.7, 28.7, −57.7, −49.0, 37.1, −79.1, 51.5]`), ±3° joint
noise; reached by teleport so episodes are independent (12/12 resets
contact-free, spoon attached to < 1 mm). The learned part is the final bite
transfer; reaching the pre-feed pose is left to a planner. A scripted
controller driving the joints to `q_front` succeeded 6/6 in an earlier v2 build,
but in the current one (enlarged `Col_Head`, bowl-front-only mouth opening) it
reaches < 3 cm in only 1 of 6 starts (2026-09-26): it pushes into the cheek and
slides sideways. It is no longer a feasibility check; trained policies do reach.
`start="path"` restores the earlier start (`0.5 × q_front`, ±8°).

**Known failure mode, do not reintroduce:** an earlier version terminated the
episode on joint-limit violation. With a per-step negative reward and no
penalty on that termination, ending the episode immediately dominated
completing it; within 3×10⁴ steps the policy learned to drive a joint into its
limit on step 1 (episode length 200 → 1, return −156 → −1.24). Joints are now
clamped rather than terminating. Any new termination condition must either
carry a penalty or be unreachable by a degenerate policy.

---

## Known open problems

Limitations kept on purpose for the current paper, and what the planned
extended version should change, are in **`FUTURE_WORK.md`** — read it before
redesigning the task, the force model or the grid. In particular the force is a
virtual penalty model around the mouth point (R = 4 cm), not the physical
spoon–face contact force; the user decided (2026-09-26) to keep it for this
paper and document it.

1. **No orientation constraint.** Success is `dist < 0.03` only, so the
   optimiser finds solutions that approach from behind the ear and hold the
   spoon sideways. **Addressed 2026-09-25:** success now also needs the
   approach and tilt limits (see Task and reward). References: approach from
   world +Z; concave side up means spoon local +y ≈ world +Y. The spoon is
   rigid in the grasp frame, so its axes follow from FK
   (`Gen3Kinematics.spoon_pose_unity`, 0.06° vs the simulator).

2. **Face direction** — measured 2026-09-25 with `calibrate_pose.py` by
   watching the Unity window: the face points along world **+Z**; from the
   default camera, screen right is world −X. The skeleton is still unreadable;
   do not derive this from the human model.

3. **Training plateaus far from the mouth.** After 1e6 steps, min distance over
   an episode was 0.2551 m and success 0%, no better than random search. That
   run measured distance to the grasp point, not the bowl, and used velocity
   actuation (see Joint control). **Resolved 2026-09-25:** with the bowl-front
   tip, pre-feed start, physical face colliders and observation normalisation
   (VecNormalize in `train_short.py`), 100k-step PPO reaches 50 % (C1) and 75 %
   (C4) deterministic success over 20 episodes; without normalisation the same
   setup stayed ~15 cm out (0 %). C1 often overshoots past 2 cm (> 10 N). C4's
   overshoots came from gaps in its prediction (target lead, momentum) and from
   allowing the tip to reach the limit in one step; addressed 2026-09-26 (see
   Joint control), together with scaling instead of projecting the action.
   `train_eval.py` (paper runs) normalises with the same settings
   (`VECNORM_KWARGS`) since 2026-09-26; its earlier models are in `models/old/`
   and do not fit the current 27-D observation.

---

## Paper grid (started 2026-09-26 02:19)

The user asked for a grid that finishes in about 8 hours, so the planned
35 runs × 1M steps (`train_eval.py plan`, ~1 day) were cut to **21 runs: C1, C3,
C4 and C2 at λ ∈ {0.005, 0.05, 0.5, 5}, seeds 0–2, 400k steps each**, each
evaluated over 100 deterministic episodes; C4 first. 400k because C4 learns
slowest with the hold phase (still at ~4 cm from the mouth after 100k steps);
3 seeds is the minimum for mean ± std.

- Job list `results/logs/jobs.txt`, runner `results/logs/run_grid.sh` (7 jobs in
  parallel via `xargs -P 7`, `OMP_NUM_THREADS=1`, unbuffered logs), started with
  `setsid nohup` so it survives the Claude session. Process group id in
  `results/logs/grid.pgid`; stop everything with `kill -- -$(cat results/logs/grid.pgid)`.
- Progress: `results/logs/progress.txt` (start/done per run with exit code);
  per-run logs `results/logs/train_<tag>.log`, `eval_<tag>.log`; the runner
  writes `results/logs/report.txt` (`train_eval.py report`) at the end.
- Models `models/<tag>.zip` + `<tag>_vecnormalize.pkl`, evaluations
  `results/<tag>.json`.

Finished 2026-09-26 08:36, all 21 runs exit 0. Success per seed (0/1/2):
C1 0/0/0 % (holds, but 72–75 % of steps over 10 N); C2 λ0.005 0/0/0, λ0.05
1/0/2, λ0.5 0/74/83 (seed 0 never approaches), λ5 0/0/0 (never holds);
C3 59/88/87 (retraction is disruptive, so the policy learns to stay under
10 N itself; shield active 1–3 % of steps); C4 7/98/100 (shield active ~77 %).
C4 seed 0 parks at 2.1–2.2 cm with the bowl front against the upper lip: small
outward commands are blocked by the lip and the bowl slides 1–2 mm further in
per step, past the 10 N distance, while a full retreat pulls it out at once
(seen in the GUI by the user). C4's linear one-step prediction does not model
this sliding contact.

**Retreats are horizontal since 2026-09-26 11:16** (the user's rule: a feeder
withdraws the spoon straight out, wherever it is, or the food spills). The env
passes `retreat_dir = FACE_FORWARD`; C4 measures its margin as the depth in
front of the mouth along it (conservative, since distance ≥ depth; skipped when
the tip is more than R off the centre line) and backs off exactly along it
(`J⁺`); C3 retracts along it. The grid's C3 and C4 runs were archived to
`models/grid_v1_radial` / `results/grid_v1_radial` and are being rerun (6 jobs,
`results/logs/jobs_horizontal.txt`); C1 and C2 are unaffected and kept.

---

## Files

| file | role |
|---|---|
| `feeding_env.py` | thin wrapper over `RCareWorld`; `get_robot/get_mouth/get_food/get_person`, `attach_spoon`; default build `FeedingPlayer_v2` |
| `feeding_task_env.py` | Gym env: observation, action, reward, contact model, metrics |
| `safety.py` | the four mechanisms C1–C4 |
| `kinematics.py` | URDF FK and Jacobian of the spoon tip (bowl) in the Unity frame |
| `calibrate_pose.py` | visual calibration of face direction / spoon axes (`--only 1 3 6 --hold-scale 4 --time-scale 2` to watch) |
| `train_eval.py` | training, evaluation, run planning, reporting; normalises like `train_short.py` (stats `models/<tag>_vecnormalize.pkl`, required by `eval`) |
| `FUTURE_WORK.md` | limitations of the current paper and plans for the extended version |
| `train_short.py` | short sanity-check training; writes `models/short/`, `results/short/` only; normalises observations and rewards (VecNormalize, stats saved as `<tag>_vecnormalize.pkl`) unless `--no-normalize` |
| `gen3_kinematics_only.urdf` | mesh-stripped URDF |
| `FeedingBuild/FeedingPlayer.x86_64` | packaged Unity scene |

The window title of the packaged build says `DressingPlayer`. That is a
leftover Product Name in the Unity project, not a wrong build.

---

## Conventions

- Comments and identifiers in English.
- Units: positions and distances in metres, forces in newtons, joint angles and
  actions in degrees. State the unit in the name or a comment when it is not
  obvious.
- Verify any claim about the simulator by measuring it, not by reasoning from
  the API docs — several documented methods do not behave as described.
