# Project archive: safety mechanisms for robot-assisted feeding

Purpose: a durable, chronological record of the work, decisions, options
considered (including everything not adopted), data and reasoning behind this
project, for storage and as a reference for the planned rebuild/extension of
the paper (see `FUTURE_WORK.md` and memory `project-extended-paper-plan`).
This complements rather than replaces the other project docs:

- `CLAUDE.md` — reference: verified platform facts, current code behaviour,
  file table. Read this to work on the code.
- `FUTURE_WORK.md` — forward-looking: known limitations and deferred design
  choices for the extended version.
- **This file** — backward-looking: what was tried, in what order, why each
  choice was made or rejected, and the numbers behind each decision.
- The verbatim session transcripts (this machine,
  `~/.claude/projects/-home-sophia-research-my-feeding-project/*.jsonl`) are
  the authoritative source for anything compressed or summarised below.

Everything here reflects the state as of 2026-09-26, end of day.

---

## 1. Project goal

Research code for a paper comparing four safety mechanisms on a bite-transfer
task, simulated in RCareWorld/Unity with a Kinova Gen3 arm and an SMPL-X human
model on a wheelchair. The four conditions, compared under identical
environment, contact model, learning algorithm and evaluation protocol:

- **C1 — unconstrained.** Base reward, no filtering. Lower bound.
- **C2 — reward penalty.** Reward minus `λ · max(0, F − F_c)`, swept over
  several λ. Produces a success-rate/peak-force trade-off curve.
- **C3 — reactive shield.** Once measured force exceeds a trigger, replace the
  action with a retraction until force drops. Post-collision.
- **C4 — predictive shield (the paper's proposed method).** Predict the force
  the current action would produce and project/scale it to stay safe.
  Pre-collision.

The comparison is only meaningful if C1–C4 share literally everything except
the mechanism — same task, same reward shape apart from the mechanism-specific
term, same PPO hyperparameters, same seeds, same evaluation protocol. This
constraint shaped several decisions below (e.g. why C2/C3 could not be left
un-retrained after a shared change, why seeds cannot be dropped selectively).

---

## 2. Chronology and decisions

### 2.1 Calibration (face direction, spoon axes)

The human skeleton (SMPL-X `HumanArticulationAttr`) returns no usable joint
data on this platform stub, so face direction and the spoon's local axes could
not be derived analytically and had to be measured visually.

- Built `calibrate_pose.py`: positions the spoon at fixed offsets from the
  mouth along ±X/±Y/±Z and displays each pose for visual confirmation, plus a
  first stop showing the spoon's own axes.
- **Result:** face points along world **+Z**; from the default camera, screen
  right is world **−X**; the spoon's concave side is local **+y** (equal to
  the grasp frame's +y to within 0.05°).
- Explicit constraint honoured throughout: never derive face direction from
  the SMPL-X skeleton or bone frames — it must be measured, because the
  skeleton data is not trustworthy on this platform.

### 2.2 Spoon tip definition — a real bug and its fix

- **First (wrong) estimate:** the spoon tip (the part that should enter the
  mouth) was guessed visually at 13 cm from the gripper's `GraspPoint_Link`.
  This is 9 cm short of the truth.
- **Symptom:** with this wrong tip, a "closest attempt" pose showed the spoon
  in the nose — the arm was placing the *true* bowl front (22.3 cm out) well
  past where the 13 cm model thought the tip was.
- **Fix:** the user fitted a Box Collider to the spoon bowl mesh in Unity;
  from its geometry, `SPOON_TIP_IN_GRASP = (0, 0.0093, 0.2227)` m (bowl front
  at 22.3 cm along the tool axis, 0.93 cm toward the concave side; bowl centre
  at 17.5 cm). This is the definition used everywhere since
  (`kinematics.Gen3Kinematics`, tip Jacobian, contact model, reward).
- **Lesson kept in CLAUDE.md:** never estimate a physical offset visually when
  a measurable source (a fitted collider, in this case) is available.

### 2.3 Actuation: velocity control abandoned

- Velocity control (`SetJointVelocity` after releasing the position drive) was
  measured to be badly cross-coupled: commanding one joint moved it 8–290 % of
  the commanded amount and dragged others by up to 26°; joint 6 commands
  mostly moved joint 4. A predictive shield's kinematic model (`dp = J·q̇·dt`)
  does not hold under this.
- **Adopted:** position-target actuation — integrate the commanded joint speed
  into a position target tracked by a stiff PD drive (stiffness 10000/damping
  100), with an anti-windup cap (target may lead the joints by at most 10°).
  Measured 94–112 % of commanded motion, ≤2.4° cross-talk at full speed.
- `actuation="velocity"` is kept in the code only to reproduce pre-2026-09-25
  runs; it is not used for the paper.
- This lead (target ahead of the joints) later turned out to be the root
  cause of one of C4's failure modes (§2.7).

### 2.4 Reset and reproducibility

- `SetJointPositionDirectly` (teleport) **keeps the joint velocities.**
  Teleporting without braking first left the arm up to 3° off and still moving
  at 66°/s, so each episode silently inherited the end-state of the previous
  one.
- **Fix:** reset now brakes (stiffness 0, damping 1000, 10 ticks), teleports,
  then holds with the position drive (30 ticks) before returning control.
- Closed-loop **policy** replays from a saved seed are not reproducible (an
  under-trained policy amplifies a ~0.005° residual difference to tens of
  degrees within ~5 steps). Open-loop **action-sequence** replays are exact
  (≤0.03° drift, verified across time scales 1×, 2×, 5×, 10× and
  headless/GUI). **Every GUI demonstration in this project therefore records
  actions from a headless run and replays them open-loop in the GUI** — never
  a live policy.

### 2.5 Physical face collisions (the biggest Unity-side effort)

**Why it was needed:** without face colliders, the spoon passes straight
through the head — there is no way to keep it out of the eyes/nose/skull
without physical geometry, and no way to give a reactive or predictive shield
anything real to be measured against.

**Option considered and rejected: a Python keep-out box.** A geometric
exclusion region with guessed adult-head dimensions was implemented first
(before physical colliders existed) to keep the tip out of the head in the
reward/termination logic. Combined with the wrong 13 cm tip estimate (§2.2),
its guessed dimensions let the spoon reach the nose anyway. It was removed
outright once physical colliders existed; kept only as a historical note, not
as a fallback.

**Adopted: physical colliders built by the user in Unity, Claude teaching the
steps.** Explicit, repeated user correction during this phase: *"没让你改，
我让你教我怎么再unity里面改"* ("I didn't ask you to change it — teach me how
to change it in Unity myself"). From that point on, all Unity scene edits were
made by the user; Claude's role was to explain what to change and why, verify
the result from the simulator side, and never touch the `.unity`/`.prefab`
files directly. This is recorded as a standing preference
(`feedback-ask-before-changing-code`).

Iterations on the Unity side (each round: user edits, screenshots, Claude
verifies in the simulator):

1. **v1 build — spoon never collided.** Root cause found: the spoon prefab's
   26 MeshColliders reference a VHACD collision mesh asset with GUID
   `cc9e0542` that is missing from the `RCareCommon` repo, and the scene
   overrides the spoon's Rigidbody to kinematic. Since the mesh reference is
   silently broken, Unity reports no error — the spoon simply never collides.
2. **v2 build — box colliders instead of mesh colliders.** Spoon: two Box
   Colliders (bowl, handle) fitted to the mesh, child Rigidbody removed, new
   layer `Spoon`; collision matrix set so `Spoon` only collides with `Human`.
   User raised a real concern about capsule colliders during this phase —
   *"不管是柱状还是球体都有半径r，那就肯定会覆盖多余部分"* (a capsule/sphere's
   radius necessarily over-covers some volume) — which shaped the choice to
   use tighter Box Colliders for the spoon and to accept some
   over-/under-coverage on the face capsules as a known simplification
   (documented, not hidden).
3. **Face colliders:** `Col_Head` (capsule), `Col_LeftCheek`/`Col_RightCheek`,
   `Col_Jaw`, `Col_Neck`, deliberately leaving a mouth-sized opening. The user
   later enlarged `Col_Head` to also cover the nose and upper jaw ("我直接用
   head的col覆盖了鼻子和上腭") and then narrowed the opening so that only the
   front portion of the bowl — not the whole spoon head — can pass through
   ("改成只让勺子能进取的部分进去（部分勺子尖）"), matching a human mouth's
   actual opening.
4. **A caught mistake, not adopted as reasoned:** a pasted critique flagged
   three problems with an earlier build attempt: (1) building to a
   `FeedingBuild_v2/` folder would not be picked up by the Python side unless
   the default executable path was updated; (2) deleting the spoon's
   Rigidbody would break `FeedingEnv.attach_spoon()`, which needs a Rigidbody
   present (kinematic) to `Link()` before making it dynamic; (3) making
   `Mouth`/`SpoonForceSensor` triggers was the right call, but the reason
   originally given for it was wrong. All three were fixed/corrected before
   proceeding. Kept here because it shows a real error that was caught rather
   than shipped.
5. **`man` (an invisible duplicate body with misplaced colliders)** was found
   and disabled — otherwise it silently duplicated collisions.
6. **Camera culling-mask gotcha:** `RCareWorld/Main Camera`
   (`RCareWorld.prefab`) only renders layers 0–4 and 16–20 by default. The new
   `Human`/`Spoon` layers collided correctly (verified via
   `GetCurrentCollisionPairs`) but were invisible in the player window until
   the user added them to the camera's Culling Mask override (983839). This
   caused a GUI demo to look like "啥也没有" (nothing there) even though the
   physics was correct — worth remembering as a class of bug (physics correct,
   rendering wrong) distinct from an actual simulation error.
7. **`attach_spoon()` ordering matters:** `Link(robot_id, link_index)` must be
   called *before* `SetKinematic(False)`. The reverse order moves the spoon
   ~41 cm away the instant it goes dynamic.

**Measured result (v2 build):** along the centre line the bowl front reaches
the mouth point; the nose stops it 2.8 cm short at 2.5 cm up, the chin 2.0 cm
short at 3.4 cm down. Under contact the analytic FK matches the simulated
spoon to <1 mm; during fast free motion (>90°/s) they can differ by up to 5 cm,
recovering on reset.

**Option raised and rejected: move the `Mouth` point in front of the lips
instead of inside the mouth.** The user considered this for anatomical
realism, then declined — *"还是按照现在的方式吧，不用挪嘴"* — kept the mouth
point where it was. Recorded as a real fork for the extension: mouth-point
placement (inside vs. just in front) interacts with the force-ball radius
question in §2.10 and should probably be revisited together with it.

### 2.6 Task constraints beyond "close enough"

**Orientation.** Distance-only success let the optimiser approach from behind
the ear holding the spoon sideways. Added: tool axis within 30° of straight
into the face, concave normal within 30° of world up.

**Centre line.** The first "success" under distance+orientation alone had the
bowl 2.1 cm below the mouth, touching the lower lip from underneath. Added: the
tip must be within 1.5 cm of the mouth's centre line (agreed with the user at
1.5 cm, not derived from any external standard).

**Pitch (added later, at the user's request, §2.4 below in time but placed
here topically).** The user noticed the bowl was often tipped nose-up at the
end of an episode ("front-high-back-low", food would spill toward the
handle). Measured: the start pose is level, but episodes ended at +4° on
average and up to +19° after pressing on the lips — the existing 30° tilt
limit does not distinguish pitch direction from roll, so it did not catch
this. Added `MAX_PITCH_UP_DEG = 2°`, `MAX_PITCH_DOWN_DEG = 20°`, and a reward
term `W_PITCH = 2.0` penalising pitch outside that band (radians).

**A structural lesson carried forward from before this session and
re-affirmed:** an early version terminated the episode on a joint-limit
violation, with a per-step negative reward and no penalty on that
termination. Ending the episode immediately dominated completing it: within
3×10⁴ steps the policy learned to drive a joint into its limit on step 1
(episode length 200→1, return −156→−1.24). Fix: joints are clamped, never
terminate the episode on their own. **Rule applied since:** any new
termination condition must either carry an explicit penalty or be
structurally unreachable by a degenerate policy. This rule directly shaped the
hold-phase reward design in §2.8 (equal-instalment bonus rather than a flat
arrival bonus, specifically to avoid rewarding "linger without completing"
more than "complete").

### 2.7 The predictive shield (C4) — full design history

This was the most-iterated piece of the project; every step below changed
`safety.py`'s `C4PredictiveShield`, and every change was unit-checked against
an independently derived closed-form reference (`np.allclose` over hundreds–
thousands of random trials) before being trusted, and cross-checked between
the "scratchpad" experimental version and the actual in-repo mechanism (exact
match required) before any conclusion from a scratchpad test was accepted.

**v0 — increment-only prediction.** `dp = J(q)·(a·vel_scale·dt)`; project the
action onto the half-space keeping predicted distance ≥ `d_min`. This is the
originally planned method.

- **Failure found:** position-target actuation creates a *lead* — the target
  is ahead of the joints, growing when the spoon is blocked (up to ~5 cm at
  the tip after being held at the lips). v0 cannot see this pending motion:
  when the block released, the tip jumped several cm in one step, in 7 of 20
  fixed-start episodes exceeding 10 N (up to 11.8 N).

**v1 — add the target lead.** `dp = J·(lead + a·vel_scale·dt)`
(`SafetyContext.target_lead_deg`). Fixed the 7/20 failures for that policy (0
of 20), but a *policy retrained* with v1 learned to rush in (15 cm in ~5
steps), and the arm's momentum alone then carried the tip past the limit in 6
of 20 episodes.

**v2 — add joint velocity.** `dp = J·(lead + qd·dt + a·vel_scale·dt)`
(`SafetyContext.joint_vel_deg`). Velocity alone did not help (still 6 of 20).

**v3 — barrier rate (discrete-time control barrier function).** Instead of
allowing the tip to reach `d_min` in one step, require it to close by at most
a fraction `barrier_rate` of its remaining margin per step:
`h_next ≥ (1 − rate)·h`, `h = dist − d_min`. `rate = 1` reduces to the
original one-step-to-the-limit behaviour; smaller rates leave more margin the
closer the tip gets, without restricting it far from the limit. Velocity term
+ `rate = 0.5` together gave 1 of 20 (best combination found using a policy
*not* retrained with it).

**v4 — projection vs. scaling (a real bug, found by asking "is 68% success a
training problem or a method problem?").** The original *projection* onto the
linear constraint (`a + Δ·n`) only removes the action's *closing* component
and leaves the perpendicular component untouched. Diagnosed directly: taking
an already-competent C1 policy (9/20 baseline successes on its own, 11/20 over
10 N unshielded) and wrapping it with C4-projection gave **0 of 20 successes**
— the shield pushed the tip 3–6 cm sideways off the centre line instead of
slowing it down (a car steered along a wall, still hitting it, rather than
braked). Replacing the correction with **uniform scaling of the whole action
vector** (same direction, slower — braking rather than steering), with a
minimal radial back-off only when even a zero action would violate the
constraint, gave the same C1 policy 16 of 20 successes at `rate = 1`, 8 of 20
at `rate = 0.5`, **none** over 10 N in either case. This became the adopted
correction mechanism, and the projection-vs-scaling distinction became a
documented part of the method (`safety.py` docstring), not just an internal
bug fix — the paper's ablation table (§2.11) reports it explicitly.

**v5 — retreat direction: radial vs. horizontal (a second real bug, found in
the full paper-scale grid, not just diagnostics).** In the first full grid one
C4 seed (seed 0) still failed structurally: 7 % success, 19.5 % of steps over
10 N. Diagnosed via a purpose-built trace (`c4_violation_diag.py`) plus a
targeted physical test (`creep_test.py` — freeze the arm vs. resume the
recorded actions): the policy parked with the bowl against the *upper lip*.
"Away from the mouth" there was defined as away from the mouth *point* (a
point inside the mouth) — which, from that position, points forward **and
up**, into the lip. The lip blocks the upward component; the joints lag the
target by up to 14°; the net motion still creeps 1–2 mm deeper per step. A
**full** retreat (large magnitude) pulls the spoon out at once — it is not
stuck or wedged, only small/marginal retreat commands in that direction fail.
(An earlier, since-corrected hypothesis told to the user was that the spoon
was physically wedged between the head and jaw colliders and being pushed in
by the geometry itself; the freeze test disproved this — with the arm frozen,
the tip does not move at all. The correction was recorded explicitly, not
silently dropped.)

The user's stated principle resolved this cleanly and generally: *a human
feeder always withdraws the spoon straight out, horizontally, wherever it is —
otherwise the food spills.* Both shields were changed to retreat exactly
along the face's forward axis (`SafetyContext.retreat_dir`); C4's margin is
now measured as horizontal depth in front of the mouth rather than
straight-line distance to the mouth point (skipped when the tip is more than
the mouth radius off the centre line, since then the force is zero anyway;
conservative, since straight-line distance ≥ horizontal depth). Deployment-
only test on the existing (un-retrained) seed-0 policy: 3 %→97 % success, 0 %
over 10 N. An intermediate variant whose back-off was not *exactly* horizontal
(moved along `J·Jᵀ·u` rather than exactly along `u`) regressed seeds 1 and 2
(97 %/100 %→60 %/73 %) — this is why the actual fix computes the minimal joint
motion via `J⁺(b·u)` rather than the earlier `(b/‖n‖²)·n`, to guarantee the
tip moves *exactly* along the retreat direction, not merely reduces the
violation.

All three C4 seeds were then retrained with the horizontal rule (not only the
failing seed 0 — the mechanism must be identical across all seeds of a
condition for the comparison to mean anything). Result: 84/51/99 %. Seed 1
regressed slightly: it now parks exactly on the horizontal limit, and
overshoots of 0.0–0.3 N (a fraction of a millimetre of prediction error) cost
it several episodes.

**v6 — robust margin (the last change, and the one initially contested).**
The user's first reaction to "give the shield a margin" was that this is
*unsuitable for a research paper* — a fair concern, since an arbitrary
after-the-fact fudge factor chosen to make one's own method win would be
exactly the kind of result-shaping a reviewer should reject. This was resolved
by reframing and constraining the margin so it is not that:

- It is **derived from measured data**, not chosen to fit the desired
  outcome: `prediction_error.py` measured C4's one-step prediction error
  in-contact over 639 steps within 3 cm of the mouth across 3 policies × 30
  episodes. The tip moved at most 0.71 mm deeper than predicted (99th
  percentile 0.49 mm; median error was in the *safe* direction, −0.14 to
  −0.35 mm). `SHIELD_MARGIN_N = 0.7 N` = twice the largest observed error
  (0.71 mm × 0.5 N/mm), a conventional safety factor.
- It is applied **identically to both shields**: C3's trigger moved from
  `F > F_d` to `F > F_d − margin`; C4's `d_min` is computed from
  `F_d − margin` instead of `F_d`. A margin that only helped C4 would be
  suspect; measured on existing models it did *not* help C3 on average
  (74.0 % vs. 77.3 % without it) — consistent with C3's failures being a
  structural lag (it reacts only after the threshold, a margin does not
  change that), not a small prediction error (which is what a margin
  compensates for). This asymmetric effect is itself evidence the margin is
  not simply "handicapping the baseline."
- **The paper is committed to reporting a sensitivity analysis** over the
  margin value (candidates: 0, 0.35, 0.7, 1.4 N — not yet run; see §3, open
  work), so the choice of 0.7 N is not presented as unquestionable.
- The precedent cited to the user for why this is standard rather than
  special pleading: robust control barrier functions, constraint tightening /
  tube MPC, and speed-and-separation monitoring standards (ISO/TS 15066) all
  reserve an explicit margin for unmodelled error in exactly this way.

Retrained C3+C4 with the margin (3 seeds each, 400k steps): **C4 100/100/100 %
success, 0 % of steps over 10 N** (peak force 9.17±0.19 N, impulse
4.18±0.08 N·s); **C3 31/94/97 %** (mean 74.0±30.4 %, i.e. *not* reliably
improved — one seed got worse).

**Rejected alternative, explicitly: dropping the bad seed.** When C4 seed 0
first failed, the user suggested simply excluding it. This was declined, with
the reasoning made explicit: excluding a run because of its *result* rather
than a pre-registered, condition-blind criterion (e.g. "the process crashed")
is seed-selection / cherry-picking, a known integrity problem in RL
reproducibility research, and would undermine the paper's central claim the
moment a reviewer asked how many seeds were run and discarded. The offered
alternatives — fix the root cause and rerun all seeds (what was actually
done), run more seeds so one outlier matters less, use robust statistics fixed
in advance (median, IQM, bootstrap CIs), or simply report the failure and
discuss it — are the ones a paper can defend.

**Deliberately not pursued in this project (left for the extension, "option
B" in `FUTURE_WORK.md`):** improving C4's prediction itself (e.g. multi-step
look-ahead, a contact-aware/learned correction term) so that no margin is
needed at all. The state immediately before the margin was added is preserved
specifically so this can be attempted from that exact base:
`snapshots/v2_horizontal_no_margin/` (full code copy) plus
`models/grid_v2_horizontal/`, `results/grid_v2_horizontal/` (that grid's C3/C4
outputs, 3 seeds each, with a README).

### 2.8 The reactive shield (C3) and why it almost never triggered

Early on, C3 wrapped around an already-good C1 policy behaved *identically* to
C1 alone — it never intervened once in 20 episodes. Cause: the episode used
to terminate the instant the feeding pose was reached (`dist < 3 cm`), and the
overshoot past 10 N happened in the *same* control step that satisfied
"reached." There was structurally no subsequent step in which a reactive
mechanism could act.

**Options considered:**

- **A — leave it and document the limitation** ("a reactive shield cannot act
  before the episode ends"). Defensible, but risks looking like an unfair
  comparison rather than a genuine finding.
- **B — add a hold phase (adopted).** Success now requires holding the
  feeding pose for `HOLD_STEPS = 8` consecutive control steps (0.48 s,
  standing in for "the person takes the food"); leaving the pose resets the
  streak; the episode ends only when the hold completes. This gives C3
  repeated opportunities per episode to react. The reward for this was
  designed specifically to avoid re-triggering the joint-limit exploit lesson
  from §2.6: instead of one flat bonus for arriving, `W_SUCCESS` is paid in
  `HOLD_STEPS` equal instalments, one each time the running hold streak passes
  its personal best *so far in the episode* — so leaving and re-entering earns
  nothing extra, and the total is capped at `W_SUCCESS` regardless of how long
  the policy lingers.
- **C — model food transfer physically / a different termination signal.**
  Raised only implicitly as "what a fuller simulation would need"; not
  designed or attempted. Noted in `FUTURE_WORK.md` (no simulated withdrawal
  phase, no food object in use).

**Effect once B was in place:** C3 became meaningfully better than initially
expected — 74–78 % mean success across the different design iterations,
because its retraction is abrupt enough that policies learn to avoid
triggering it and keep the force under the danger line largely on their own
(intervention rate only ~1–5 % of steps, versus C4's ~90 %). This narrows the
gap between C3 and C4 in a way that is worth discussing in the paper rather
than hiding.

Also required, as a side effect of adding the hold phase: `train_eval.py`'s
`report()` aggregator crashed / silently produced nonsense because
`results/` also contains non-evaluation JSON files (e.g.
`calibrate_pose.json`, `diag_approach.json`, and its own `tradeoff.json`
output). Fixed with a guard that only aggregates files whose `summary` block
actually contains `condition`.

### 2.9 Observation/reward normalisation

Raw 27-D observations mix joint angles (tens of degrees) with a tip-to-mouth
vector (centimetres); on raw inputs, 100k-step PPO stayed ~15 cm from the
mouth (0 % success) — no better than random search, and indistinguishable
from "the task is unlearnable." Adding `VecNormalize` (observation + reward
normalisation) let the same setup reach 50 % (C1) / 75 % (C4) over 20
episodes at the same step count. This was nearly missed as a possible
non-issue with the task design itself, since the symptom (flat, near-zero
progress) looks identical to a genuinely broken reward or environment.
Approved by the user as a fallback ("可以的") specifically framed as "try this
before concluding the task itself is unlearnable" — it worked, and was then
folded into both `train_short.py` (already default) and `train_eval.py`
(added later, once it became clear the paper-scale script had been left
without it — an oversight caught by review, not by a failed run).

### 2.10 The contact/force model — a limitation kept intentionally, not fixed

Contact force is a **penalty model**, not the physical engine's contact force:
`F = k·max(0, R − dist)`, `R = 4 cm` (a force ball around the mouth *point*),
`k = 500 N/m`; comfort `F_c = 6 N`, danger `F_d = 10 N` (the latter following
Assistive Gym; `F_c` is the project's own choice). This was designed before
physical face colliders existed — at that point nothing could measure a real
collision force, so a virtual proxy was necessary. Once physical colliders
were added (§2.5), this proxy and the real geometry drifted apart: the lips
sit ~2–3 cm in front of the mouth point, so the virtual force starts being
"felt" 1–2 cm before the spoon can physically reach the lips, while the real
force of the spoon pressing on the lips (e.g. the seed-0 upper-lip contact,
§2.7) is not measured at all — the dedicated `SpoonForceSensor` object is
currently a *trigger*, and trigger colliders always read zero force.

The user asked directly why the radius is 4 cm and whether this was simply a
mistake. It is not a mistake so much as a design that was never revisited
after the geometry changed underneath it. **Options given:**

- **A — keep it, document it (adopted for this paper).** State plainly in the
  paper that force is a modelled proxy, not the physical contact force, and
  that it is decoupled from the actual lip/jaw geometry. No reruns needed.
- **B — refit the force ball to the lip geometry** (e.g. R ≈ 2.5–3 cm) and
  redefine the success/danger distances around it. Requires re-deriving the
  thresholds (the "success and danger never coincide" assertion the
  environment checks at start-up would need to be re-verified) and rerunning
  every condition — roughly a full day of compute at this project's current
  scale.
- **C — use the physical engine's contact force directly.** Most realistic;
  requires first determining whether RCareWorld/Unity can report contact
  impulses for a non-trigger collider (the current sensor would need to change
  from a trigger to a real collider, which would then also *block* the spoon
  physically rather than only measuring it — a further design question). Most
  work, not started.

**Decision:** keep A for the current paper; B/C are recorded in
`FUTURE_WORK.md` as the first item for the planned rebuild, since the user
stated this paper will later be substantially rebuilt and extended for
another venue.

Also noted, and deliberately not (yet) revisited: the mouth point's placement
inside vs. just in front of the mouth (§2.5) interacts with this same
question and should likely be decided together with it in the rebuild.

### 2.11 Experimental protocol for the paper grid

**Original plan:** 5 seeds × 7 conditions (C1, C3, C4, C2 at 4 λ) × 1,000,000
steps = 35 runs, ≈1 day at 7-way parallel on this machine.

**Constraint that forced a reduction:** the user asked for a grid that could
finish in about 8 hours. Given the measured throughput (≈65–80 env steps/s per
process, 8 logical cores) and that C4 learns slowest under the hold phase
(still ~4 cm from the mouth after 100k steps in the short run), the grid was
cut to **21 runs: C1, C3, C4, and C2 at λ ∈ {0.005, 0.05, 0.5, 5}, 3 seeds
each, 400,000 steps, 100 evaluation episodes**, C4 scheduled first so a
learnability problem would surface early rather than at the end of an 8-hour
run. This reduction is recorded as a deliberate, reasoned trade-off, not an
oversight — and flagged in `FUTURE_WORK.md` as something the extension should
restore (≥5 seeds, ideally 10; a finer λ grid, since between 0.05 and 5 there
is currently only one informative point at 0.5; robust statistics fixed before
the runs).

**Infrastructure built for this:** a flat job list (one shell command per
run, each writing its own train+eval logs and appending a timestamped
start/done line to a shared progress file) executed with
`xargs -P <n> -I{} bash -c '{}'`, launched detached from the interactive
session via `setsid nohup ... &` (so it survives the Claude Code session
ending) with its process-group id saved to a file for clean shutdown
(`kill -- -$(cat grid.pgid)`). Re-run three times total as the design changed
(original grid; horizontal-retreat rerun of C3/C4 only; margin rerun of C3/C4
only) — each time C1/C2 were left untouched since neither uses
`retreat_dir`/margin, so only the two changed conditions needed retraining,
saving roughly two-thirds of the compute each time.

**Every intermediate state was archived, never deleted**, each with a short
README explaining what it is and its headline numbers:

| location | what it is |
|---|---|
| `models/old/`, `results/old/` | pre-normalisation paper models (superseded by §2.9) |
| `models/grid_v1_radial/`, `results/grid_v1_radial/` | first full grid's C3/C4 (radial retreat, no margin); includes a snapshot of `feeding_task_env.py` as it was |
| `snapshots/v2_horizontal_no_margin/` + `models/grid_v2_horizontal/`, `results/grid_v2_horizontal/` | full code snapshot + C3/C4 outputs, horizontal retreat, before the margin — the intended base for future "option B" work |
| `models/`, `results/` (current) | final: C1, C2 from the first grid; C3, C4 from the margin rerun |
| `results/logs/` | job lists (`jobs.txt`, `jobs_horizontal.txt`, `jobs_margin.txt`), the runner script, progress log, per-run train/eval logs, and `report.txt` snapshots (`report_margin_0.7N.txt` is the final one) |

### 2.12 Final results

100 deterministic evaluation episodes per run, 3 seeds, mean ± sample std
(full numbers: `results/paper/numbers.json`; tables:
`results/paper/main_table.{md,tex}`, `ablation_table.{md,tex}`; figure:
`results/paper/tradeoff.{pdf,png}`; full prose draft:
`results/paper/results_section_draft.md`):

| condition | success % (per seed) | peak force (N) | steps > 10 N (%) | impulse (N·s) | intervention (%) |
|---|---|---|---|---|---|
| C1 | 0 (0/0/0) | 13.91 ± 0.39 | 73.1 ± 1.7 | 6.25 ± 0.21 | 0 |
| C2 λ=0.005 | 0 (0/0/0) | 13.59 ± 0.77 | 74.1 ± 2.0 | 6.17 ± 0.38 | 0 |
| C2 λ=0.05 | 1 (1/0/2) | 12.26 ± 0.96 | 64.6 ± 9.5 | 5.38 ± 0.15 | 0 |
| C2 λ=0.5 | 52.3 (0/74/83) | 5.27 ± 4.13 | 0.91 ± 1.00 | 7.32 ± 6.30 | 0 |
| C2 λ=5 | 0 (0/0/0) | 1.59 ± 2.74 | 0 | 2.82 ± 4.88 | 0 |
| C3 | 74.0 (31/94/97) | 9.57 ± 1.08 | 1.05 ± 1.37 | 9.23 ± 8.15 | 3.3 ± 1.7 |
| **C4** | **100 (100/100/100)** | **9.17 ± 0.24** | **0** | **4.18 ± 0.08** | 90.3 ± 1.8 |

C4 lies outside the success/force trade-off curve traced by C2's λ sweep (no
λ reaches C4's success rate at a force below the danger threshold), succeeds
in every evaluated episode, and has the lowest impulse among the conditions
that succeed at all — the central claim the paper argues for. The caveats the
draft results section states plainly: only 3 seeds; C4 relies on the shield
for ~90 % of steps rather than learning to be gentle on its own; the margin's
sensitivity has not yet been swept; the force model is a proxy (§2.10).

---

## 3. Everything not adopted — a reference list for the extension

Collected in one place for convenience; each is described in full where it
first came up above.

1. **Python keep-out box** for face avoidance (§2.5) — replaced by physical
   colliders; kept only as a historical note.
2. **Moving the `Mouth` point in front of the lips** rather than inside the
   mouth (§2.5) — declined by the user; worth revisiting together with the
   force-model radius (§2.10).
3. **C4 projection correction** (minimum-norm projection onto the linear
   constraint) (§2.7) — replaced by uniform scaling after it was shown to push
   the tip sideways instead of slowing it; the projection formula is preserved
   in the ablation table as a deliberate comparison point.
4. **Radial retreat direction** (away from the mouth point) for both shields
   (§2.7) — replaced by horizontal retreat (the user's rule: a feeder always
   withdraws level, or food spills); the radial-retreat grid is fully
   preserved in `models|results/grid_v1_radial`.
5. **An intermediate near-horizontal back-off** (`(b/‖n‖²)·n`, close to but
   not exactly horizontal) (§2.7) — regressed two seeds; replaced by an exact
   `J⁺(b·u)` horizontal back-off.
6. **Global action speed limit** (slow down C1–C4 equally) as an alternative
   to C4's per-step barrier rate (§2.7) — mentioned as an option, not tested;
   rejected in reasoning because it would change all four conditions rather
   than isolating C4's behaviour.
7. **Dropping/excluding C4's failing seed 0** (§2.7) — explicitly rejected as
   seed-selection bias; the root cause was fixed instead and all seeds rerun.
8. **A safety margin judged "unsuitable for a paper" in its first framing**
   (§2.7) — not rejected outright; reframed as a data-derived, symmetric,
   sensitivity-tested robust margin and adopted on that basis.
9. **Refitting the virtual force-ball radius to the real lip geometry**, and
   **using the physics engine's real contact force** (§2.10) — both deferred
   to the planned rebuild; kept as options B and C in `FUTURE_WORK.md`.
10. **Modelling the withdrawal phase and real food** (§2.8, §2.10) — not
    designed; noted as scope the current task leaves out entirely.
11. **Improving C4's prediction itself (multi-step look-ahead / contact-aware
    model) instead of a margin** ("option B") — deliberately deferred, with a
    clean starting snapshot preserved specifically so it can be attempted
    without the margin already baked in (§2.7 v6, `snapshots/v2_horizontal_no_margin`).
12. **Running the full 5-seed × 1M-step × 7-condition grid** (§2.11) —
    deferred for time; the reduced 21-run/3-seed/400k-step grid was run
    instead, with the full grid recorded as the target for more seeds later.

---

## 4. Working methods and preferences that shaped this project

(Cross-referenced in the memory files; repeated here since they materially
affected how the work above was done, not just how it was communicated.)

- **Verify, don't assume, about the simulator.** Nearly every "verified
  platform fact" in `CLAUDE.md` exists because something plausible-sounding
  (a documented API, an intuitive default) turned out not to hold when
  actually measured — velocity control, `SetJointPositionDirectly`,
  `IKTargetDoMove`, joint limits reported as `[0, 0]`, camera culling masks.
- **Never trust a scratchpad finding until it is reproduced by the real
  code.** Every shield-formula change was (a) checked against an
  independently-derived closed-form reference over many random trials, and
  (b) checked that the actual in-repo mechanism, run through the simulator,
  matched a standalone test script's result exactly, before the conclusion
  was used to justify a change.
- **Never delete a superseded result — snapshot it with a short README.**
  Applied at every major pivot (radial→horizontal retreat, no-margin→margin),
  specifically because the user stated an intent to revisit earlier states
  for future work.
- **Propose code changes and wait for approval; teach Unity edits instead of
  making them.** Established after an unrequested edit was corrected; applied
  for the remainder of the project.
- **GUI demonstrations:** always open-loop (recorded actions, never a live
  policy), always at a slow, explicitly-announced time scale with the pause
  schedule stated before the window opens, long holds at the start and end
  poses.
- **Plain, short, action-first Chinese in conversation**; technical project
  documents (`CLAUDE.md`, `FUTURE_WORK.md`, this file, code comments) in
  English, matching the codebase's existing convention.
