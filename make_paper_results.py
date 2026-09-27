"""Tables and the trade-off figure for the paper's results section.

Reads the evaluation files written by train_eval.py (100 deterministic episodes
per run) and writes
    results/paper/main_table.{md,tex}      mean ± sample std over seeds
    results/paper/ablation_table.{md,tex}  C3/C4 across the shield versions
    results/paper/tradeoff.{pdf,png}       success rate vs peak force
    results/paper/numbers.json             every number used in the text

Main comparison: C1 and C2 from the first grid (results/), C3 and C4 from the
rerun with horizontal retreat and the 0.7 N shield margin (results/). Ablation:
results/grid_v1_radial (retreat away from the mouth point, no margin),
results/grid_v2_horizontal (horizontal retreat, no margin), results/ (both).

Usage: python make_paper_results.py
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feeding_task_env import CONTROL_DT

OUT = "results/paper"
SEEDS = (0, 1, 2)
MAIN = [("C1", "C1"), ("C2 (λ=0.005)", "C2_lam0.005"), ("C2 (λ=0.05)", "C2_lam0.05"),
        ("C2 (λ=0.5)", "C2_lam0.5"), ("C2 (λ=5)", "C2_lam5"), ("C3", "C3"), ("C4", "C4")]
ABLATION = [("away from mouth point, no margin", "results/grid_v1_radial"),
            ("horizontal, no margin", "results/grid_v2_horizontal"),
            ("horizontal, 0.7 N margin", "results")]


def load_run(directory, tag, seed):
    with open(os.path.join(directory, f"{tag}_seed{seed}.json")) as fh:
        return json.load(fh)


def run_stats(run):
    """Per-run metrics (one evaluation of 100 episodes)."""
    s, eps = run["summary"], run["episodes"]
    times = [e["time_to_success"] * CONTROL_DT for e in eps
             if e["success"] > 0 and e["time_to_success"] > 0]
    return {
        "success": 100 * s["success"],
        "held": 100 * s["held"],
        "peak_force": s["peak_force"],
        "danger_rate": 100 * s["danger_violation_rate"],
        "comfort_rate": 100 * s["comfort_violation_rate"],
        "impulse": s["force_impulse"],
        "intervention": 100 * s["shield_rate"],
        "time_to_success": float(np.mean(times)) if times else float("nan"),
    }


def aggregate(directory, tag):
    per_seed = [run_stats(load_run(directory, tag, s)) for s in SEEDS]
    out = {"per_seed": per_seed}
    for k in per_seed[0]:
        vals = np.array([p[k] for p in per_seed], dtype=float)
        finite = vals[np.isfinite(vals)]
        out[k] = (float(finite.mean()) if finite.size else float("nan"),
                  float(finite.std(ddof=1)) if finite.size > 1 else float("nan"))
    return out


def fmt(ms, digits=1):
    m, s = ms
    if not np.isfinite(m):
        return "–"
    return f"{m:.{digits}f} ± {s:.{digits}f}" if np.isfinite(s) else f"{m:.{digits}f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    main_rows = {name: aggregate("results", tag) for name, tag in MAIN}
    abl_rows = {(label, c): aggregate(d, c) for label, d in ABLATION for c in ("C3", "C4")}

    # ---------------- main table
    cols = [("Success (%)", "success", 1), ("Hold completed (%)", "held", 1),
            ("Peak force (N)", "peak_force", 2), ("Steps > 10 N (%)", "danger_rate", 2),
            ("Impulse (N·s)", "impulse", 2), ("Intervention (%)", "intervention", 1),
            ("Time to success (s)", "time_to_success", 2)]
    md = ["| Condition | " + " | ".join(c[0] for c in cols) + " | Success per seed (%) |",
          "|---" * (len(cols) + 2) + "|"]
    tex = [r"\begin{tabular}{l" + "c" * (len(cols) + 1) + "}", r"\toprule",
           "Condition & " + " & ".join(c[0].replace("%", r"\%") for c in cols) + r" & Per seed (\%) \\",
           r"\midrule"]
    for name, _ in MAIN:
        r = main_rows[name]
        cells = [fmt(r[k], d) for _, k, d in cols]
        seeds = "/".join(f"{p['success']:.0f}" for p in r["per_seed"])
        md.append(f"| {name} | " + " | ".join(cells) + f" | {seeds} |")
        tex_name = name.replace("λ", r"$\lambda$")
        tex.append(f"{tex_name} & " + " & ".join(c.replace("±", r"$\pm$") for c in cells) + f" & {seeds} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/main_table.md", "w").write("\n".join(md) + "\n")
    open(f"{OUT}/main_table.tex", "w").write("\n".join(tex) + "\n")

    # ---------------- ablation table
    md = ["| Shield version | C3 success (%) | C3 steps > 10 N (%) | C4 success (%) | C4 steps > 10 N (%) |",
          "|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{lcccc}", r"\toprule",
           r"Shield version & C3 success (\%) & C3 steps $>$10\,N (\%) & C4 success (\%) & C4 steps $>$10\,N (\%) \\",
           r"\midrule"]
    for label, _ in ABLATION:
        c3, c4 = abl_rows[(label, "C3")], abl_rows[(label, "C4")]
        s3 = "/".join(f"{p['success']:.0f}" for p in c3["per_seed"])
        s4 = "/".join(f"{p['success']:.0f}" for p in c4["per_seed"])
        cells = [f"{fmt(c3['success'])} ({s3})", fmt(c3["danger_rate"], 2),
                 f"{fmt(c4['success'])} ({s4})", fmt(c4["danger_rate"], 2)]
        md.append(f"| {label} | " + " | ".join(cells) + " |")
        tex.append(f"{label} & " + " & ".join(c.replace("±", r"$\pm$") for c in cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/ablation_table.md", "w").write("\n".join(md) + "\n")
    open(f"{OUT}/ablation_table.tex", "w").write("\n".join(tex) + "\n")

    # ---------------- trade-off figure
    # With 3 seeds a mean ± std bar runs past 0 and 100 %, so each seed is drawn
    # as a small point and the mean as a large marker.
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    c2 = [n for n, _ in MAIN if n.startswith("C2")]
    xs = [main_rows[n]["success"][0] for n in c2]
    ys = [main_rows[n]["peak_force"][0] for n in c2]
    ax.plot(xs, ys, "-", color="0.6", zorder=1)
    styles = {n: ("s", "0.35", 7) for n in c2}
    styles.update({"C1": ("o", "tab:red", 7), "C3": ("^", "tab:orange", 8), "C4": ("*", "tab:blue", 13)})
    for n, _ in MAIN:
        r, (marker, colour, size) = main_rows[n], styles[n]
        ax.scatter([p["success"] for p in r["per_seed"]], [p["peak_force"] for p in r["per_seed"]],
                   marker=marker, color=colour, s=size * 2.5, alpha=0.35, linewidths=0, zorder=2)
        ax.plot(r["success"][0], r["peak_force"][0], marker, color=colour, markersize=size, zorder=3,
                label=n if not n.startswith("C2") else None)
        if n.startswith("C2"):
            ax.annotate(n.replace("C2 ", ""), (r["success"][0], r["peak_force"][0]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8, color="0.3")
    ax.plot([], [], "s-", color="0.35", label="C2 (λ sweep)")
    ax.plot([], [], "o", color="0.5", alpha=0.35, markersize=5, label="single seeds")
    ax.axhline(10.0, ls="--", lw=1, color="k")
    ax.text(1, 10.15, "danger threshold $F_d$ = 10 N", fontsize=8)
    ax.set_xlabel("Success rate (%)")
    ax.set_ylabel("Peak contact force (N)")
    ax.set_xlim(-5, 105)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/tradeoff.pdf")
    fig.savefig(f"{OUT}/tradeoff.png", dpi=200)

    numbers = {"main": {n: main_rows[n] for n, _ in MAIN},
               "ablation": {f"{label} | {c}": abl_rows[(label, c)] for (label, c) in abl_rows}}
    json.dump(numbers, open(f"{OUT}/numbers.json", "w"), indent=2)
    print(open(f"{OUT}/main_table.md").read())
    print(open(f"{OUT}/ablation_table.md").read())


if __name__ == "__main__":
    main()
