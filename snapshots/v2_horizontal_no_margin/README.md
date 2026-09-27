# Snapshot: horizontal retreat, no safety margin (2026-09-26 13:06)

State of the project after the C3/C4 rerun with horizontal retreat and before
the shield margin was added. Kept as the base for improving C4's prediction
itself ("option B") without a margin.

- code/: every .py file, CLAUDE.md, FUTURE_WORK.md and the URDF as they were.
  To work from this version, copy code/*.py back into the project root (after
  saving whatever is there).
- logs_scripts/: grid runner, job lists, progress log and the report.
- Models: models/grid_v2_horizontal/ (C3, C4 seeds 0-2).
  Results and logs: results/grid_v2_horizontal/.
  C1 and C2 models/results in models/ and results/ belong to both versions.

Results (100 evaluation episodes per seed, success 0/1/2):
C3 56/87/89 %, C4 84/51/99 %; C1, C2 as in results/logs/report.txt.
