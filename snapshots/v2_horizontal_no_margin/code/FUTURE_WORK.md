# Limitations and future work

Recorded 2026-09-26 for the planned rebuild and extension of this paper for
another venue. The current paper keeps the design below as is (the user's
decision); each item says what is simplified, what was measured, and what the
extended version could do instead. Details and platform facts are in CLAUDE.md.

---

## 1. Contact force is virtual, not physical

- **What the paper measures.** "Force" is a penalty model on the spoon tip's
  distance to the mouth point (a point *inside* the mouth):
  `F = k · max(0, R − dist)`, `R = 4 cm`, `k = 500 N/m`. The 4 cm is the radius
  of an invisible force ball around that point, not the size of the mouth.
  Force starts at 4 cm, is 5 N at 3 cm (success line) and 10 N at 2 cm (danger
  line). All safety metrics (peak force, violation rates, impulse, success)
  come from this model.
- **Why it no longer matches the scene.** The model was chosen before the face
  had colliders (the spoon then passed through the head). Since the v2 build
  the face is physical: the lips sit roughly 2–3 cm in front of the mouth
  point, so the virtual force starts 1–2 cm before the spoon touches the lips,
  while the real force of the spoon pressing on the lips is not counted at all
  (`SpoonForceSensor` is a trigger and always reads 0).
- **Options for the extension.**
  - B: fit the force ball to the lip geometry (e.g. R ≈ 2.5–3 cm) and redefine
    the success and danger distances; all conditions rerun.
  - C: use the physical contact force between spoon and face colliders. First
    find out whether RCareWorld/Unity can report contact impulses for the
    spoon (the sensor would need to be a non-trigger collider or the force
    read from the physics engine). Most realistic, most work.

## 2. C4 failure mode: sliding contact at the upper lip

- Paper grid C4 seed 0 (400k steps): 7 % success, 19 % of steps over 10 N; seeds
  1 and 2: 98 % and 100 %.
- Measured cause: the policy parks at 2.1–2.2 cm with the bowl front against the
  upper lip. C4 (and the policy) treat "away" as away from the mouth *point*,
  which there means forward **and up**, into the lip. The lip blocks the upward
  part, the joints end up to 14° from their targets, and the bowl slides 1–2 mm
  further in per step. With the arm frozen the spoon does not move (not stuck,
  no creep on its own); a full retreat pulls it out at once. C4's kinematic
  one-step prediction cannot foresee motion that the contact deflects.
- **Adopted for the current paper (2026-09-26 11:16): horizontal retreat.**
  A feeder withdraws the spoon straight out, level, or the food spills (the
  user's rule: horizontal wherever the spoon is). C4 now measures its margin as
  the depth in front of the mouth along the face axis and backs off exactly
  horizontally (`SafetyContext.retreat_dir`); C3 retracts horizontally. Tested
  first on the grid-v1 policies (trained with the old rule, 30 episodes):
  C4 seed 0 3 % → 97 % success, 0 % of steps over 10 N; with an earlier
  variant whose back-off was not exactly horizontal, seeds 1 and 2 dropped to
  60 % and 73 %. C3 and C4 (3 seeds each) are being retrained with it; the
  grid-v1 C3/C4 runs are archived in `models/grid_v1_radial`,
  `results/grid_v1_radial` (with a code snapshot). All C4 seeds use the new
  rule; seed 0 is not treated differently.
- **Not adopted (yet): safety margin.** C4 would stop at 2.3 cm (≈ 8.5 N)
  instead of 2.0 cm; for a fair comparison C3 should then retract from 8.5 N as
  well. Keep in mind if the retrained C4 still shows sliding-contact violations.

## 3. What the task leaves out

- **No withdrawal phase.** The episode ends after the 0.48 s hold at the mouth;
  taking the spoon back out (horizontally, level) is not simulated.
- **No food.** Spilling is only proxied by the pose limits (bowl pitch between
  −20° and +2°, tilt ≤ 30°); the food object in the scene is unused.
- **Rigid face.** Head, cheeks, jaw and neck are rigid capsules; the mouth
  opening only admits the front of the bowl. A soft mouth would deform instead
  of deflecting the spoon.
- **Coarse spoon collider.** The bowl is a box 8.5 × 3.3 × 9.6 cm, much thicker
  than a real spoon bowl (~1 cm).
- **References measured by eye.** Face direction and spoon axes were calibrated
  visually (calibrate_pose.py); the SMPL-X skeleton cannot be read.

## 4. Experimental protocol

- **Reduced grid.** To finish in ~8 hours the paper grid used 3 seeds × 400k
  steps for C1, C3, C4 and C2 at λ ∈ {0.005, 0.05, 0.5, 5} (21 runs). The planned
  grid was 5 seeds × 1M steps (35 runs, ~1 day at 7 in parallel on this 4-core
  machine). For the extension: ≥ 5 seeds (10 preferred), a finer λ grid in the
  informative range (e.g. 0.1, 0.2, 1, 2; between 0.05 and 5 there is only one
  point now), and robust statistics (interquartile mean, bootstrap confidence
  intervals) fixed before the runs.
- **Results of the reduced grid** (success per seed 0/1/2): C1 0/0/0 %;
  C2 λ0.005 0/0/0, λ0.05 1/0/2, λ0.5 0/74/83, λ5 0/0/0; C3 59/88/87; C4 7/98/100.
  Report: results/logs/report.txt.
- **Observations worth discussing.** C3 does better than expected: its
  retraction is abrupt, so the policy learns not to trigger it and keeps the
  force under 10 N itself. C4 intervenes in ~77 % of steps: the policy leaves
  the slowing down to the shield. C2 at λ = 0.5 varies strongly with the seed
  (one seed never approaches), the usual sensitivity of penalty methods.
- **No seed selection.** Runs may only be excluded by a criterion fixed in
  advance and applied to all conditions (e.g. a crashed run), never because of
  their result.
