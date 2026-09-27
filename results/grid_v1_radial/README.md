# Paper grid v1 — C3 and C4 with "away from the mouth point"

C3 and C4 results of the first paper grid (2026-09-26 02:19–08:36, 3 seeds,
400k steps, 100 evaluation episodes), archived when both were switched to a
horizontal retreat. C1 and C2 from the same grid stay in models/ and results/
(they do not use the retreat direction).

Code: feeding_task_env.py here is the version these runs used. safety.py was
the current one with `SafetyContext.retreat_dir = None` (retreat away from the
mouth point), except that C4's back-off was `clip((b / ||n||^2) * n)` instead of
`clip(J^+ (b u) / (vel_scale dt))`.

Success per seed 0/1/2: C3 59/88/87 %, C4 7/98/100 %. C4 seed 0 failure analysed
in FUTURE_WORK.md, section 2.
