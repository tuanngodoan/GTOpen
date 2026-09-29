#!/usr/bin/env python3
"""Does the deep-SRP region actually say small suited connectors realise worse than the shipped
table claims? This prints the one number the refit decision waits on.

Why it exists (2026-09-27): `cache/realization_fit.json` is a **class** table (`balanced_share`
reads `class_base` only, `preflop/mod.rs:544-549`), and the N8 refit of it is neither GO nor
NO-GO. The blocker is not the fitter, it is the data: the 9 SRP spots at 50/75/100 bb (SPR
8.6-17.7, the region where a suited connector's implied odds live) never solved on a 12 GB card
(the first one ran 57 min, 11 960/12 288 MiB, 0/40 boards). What *is* solved is SRP 40 bb
(SPR 6.8) plus 12 three-bet pots, and those two sets say different things:

| set | 54s / ATo |
|---|---|
| shipped table | 1.30 |
| SRP 40 bb only (SPR 6.8) | 0.78-0.85 |
| three-bet pots, the 3/4 with a 54s observation | 1.31-1.34 |

The three-bet aggregate is **not** evidence about 54s/65s: those hands barely enter a 3-bet range,
so its 0.788/0.705 numbers are the fitter's prior curve (`LAMN=40`), not measurement. That is why
this probe prints `reach` next to every `r_obs` — a number with no reach behind it is the prior.

Decision rule (the thresholds are the ones recorded in
`plans/journals/2026-09-27-kho-preflop-balanced-va-huong-di-grind.md`):

* `54s / ATo < 1.0` at SPR 17.7 -> **CLOSE** the refit branch. N8 does not let small suited
  connectors realise at any measured depth, so the shipped table is not the thing to change; the
  separate question of the `btn_must_open` clause's source stays open, with evidence.
* `54s / ATo >= 1.3` -> **GO**. The class x SPR form has something to capture; the other 8 spots
  and the solver patch are worth their GPU hours.
* in between -> **UNDECIDED**, one spot is not a table.

The run needs a rented card (>= 24 GB) and one spot, 10 boards: `--runbook` prints it. Local runs
work on whatever is already solved, which is how this script was checked against the 15 spots in
`m5_out_n8/`.

    python3 m5_spots/probe_deep_srp.py --out-dir m5_out_n8
    python3 m5_spots/probe_deep_srp.py --out-dir m5_out_deep --runbook
"""

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The decisive spot: SRP, BTN opens 2.5x, BB calls, 100 bb -> SPR 17.7.
DECISIVE = "n8_100__btn_open_bb_call"
LABELS = ["54s", "65s", "76s", "ATo"]

CLOSE_BELOW = 1.0
GO_AT = 1.3


def spot_spr(path):
    """SPR of one exported spot: `effective_stack / starting_pot` of its tree."""
    try:
        with open(path, encoding="utf-8") as f:
            tree = json.load(f)["tree"]
        return tree["effective_stack"] / tree["starting_pot"]
    except (OSError, ValueError, KeyError):
        return None


def load_obs(path):
    """`[(label, r_obs, reach)]` of the `obs` rows — `board` rows are the solver's per-board line."""
    out = []
    try:
        fh = open(path, encoding="utf-8")
    except OSError:
        return out
    with fh:
        for line in fh:
            if '"obs"' not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("type") == "obs":
                out.append((r["label"], float(r["r_obs"]), float(r["reach"])))
    return out


def reach_weighted(rows, label):
    """`(r_obs, reach_sum, n)` of one hand class, weighted by reach — `None` r_obs when unobserved.

    Reach is the weight the fitter itself uses (`fit_phase_c5.py`: `acc[k][0] += eq * reach`), and
    it is what separates a measurement from the prior: a class with no reach behind it is not
    measured at all.
    """
    sel = [(r, w) for lab, r, w in rows if lab == label]
    if not sel:
        return None, 0.0, 0
    reach = sum(w for _, w in sel)
    if reach <= 0:
        return None, 0.0, len(sel)
    return sum(r * w for r, w in sel) / reach, reach, len(sel)


def fmt(x, nd=3):
    return "  -  " if x is None else f"{x:.{nd}f}"


def report(out_dir, labels, decisive):
    spots = sorted(glob.glob(os.path.join(out_dir, "spots", "*.json")))
    if not spots:
        sys.exit(f"no spots in {out_dir}/spots — export them first (phase_b.py --stage spots)")
    printed, decisive_ratio = 0, None
    for path in spots:
        name = os.path.basename(path)[:-5]
        rows = load_obs(os.path.join(out_dir, f"obs_{name}.jsonl"))
        if not rows:
            continue
        printed += 1
        spr = spot_spr(path)
        with open(os.path.join(out_dir, f"obs_{name}.jsonl"), encoding="utf-8") as fh:
            boards = sum(1 for line in fh if '"board"' in line)
        print(f"\n== {name} ==  SPR {fmt(spr, 2)}  boards {boards}  obs {len(rows)}")
        for lab in labels:
            r, reach, n = reach_weighted(rows, lab)
            print(f"   {lab:<4} r_obs {fmt(r)}   reach {reach:8.1f}   n {n}")
        base, _, _ = reach_weighted(rows, "ATo")
        for lab in labels:
            if lab == "ATo":
                continue
            r, _, _ = reach_weighted(rows, lab)
            ratio = None if (r is None or base in (None, 0.0)) else r / base
            if name == decisive:
                decisive_ratio = ratio
            print(f"   {lab} / ATo  {fmt(ratio, 2)}")
    print(f"\n{printed} spot(s) with observations in {out_dir}")
    return decisive_ratio


def verdict(ratio, decisive):
    print(f"\ndecisive spot {decisive}: 54s / ATo = {fmt(ratio, 2)}")
    if ratio is None:
        print("  NOT SOLVED — the decisive region (SRP 50/75/100 bb) is still empty. "
              "Nothing here decides the refit; run the one spot with `--runbook`.")
    elif ratio < CLOSE_BELOW:
        print(f"  CLOSE the refit branch (< {CLOSE_BELOW}): N8 does not let small suited "
              "connectors realise at any measured depth, so the class table is not the thing to "
              "change. The `btn_must_open` source question stays open, with evidence.")
    elif ratio >= GO_AT:
        print(f"  GO (>= {GO_AT}): the class x SPR form has something to capture — solve the "
              "other 8 spots and patch the solver.")
    else:
        print(f"  UNDECIDED (between {CLOSE_BELOW} and {GO_AT}): one spot is not a table.")


def runbook(out_dir):
    print(f"""# Rented card (>= 24 GB VRAM), Linux. One spot, 10 boards.

# 1. build (the spot file is already exported; only the solver is needed)
git clone <GTOpen> && cd GTOpen
cargo build --release -p solve-cli          # CUDA build; nvrtc on PATH/LD_LIBRARY_PATH
printf '%s\\n' $(cut -d' ' -f1 m5_spots/flops100.txt | head -10) > m5_spots/flops10.txt

# 2. one spot only: a fresh out-dir with that single spot, so `stage_run` cannot pick up the
#    9 SRP spots that thrashed a 12 GB card, and 10 boards keep it inside an hour
mkdir -p {out_dir}/spots
cp m5_out_n8/spots/{DECISIVE}.json {out_dir}/spots/
python3 m5_spots/phase_b.py --stage run --out-dir {out_dir} \\
    --flops m5_spots/flops10.txt --iters 400 --target 0.3

# 3. read the number and the decision rule
python3 m5_spots/probe_deep_srp.py --out-dir {out_dir}

# 4. what to do with the answer: probe_deep_srp.py prints CLOSE / GO / UNDECIDED.
#    Keep the obs file and the manifest: they are the evidence for whichever branch.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "m5_out_n8"),
                    help="where spots/ and obs_*.jsonl live (default: m5_out_n8)")
    ap.add_argument("--labels", default=",".join(LABELS))
    ap.add_argument("--decisive", default=DECISIVE)
    ap.add_argument("--runbook", action="store_true",
                    help="print the rented-GPU commands for the one decisive spot and exit")
    a = ap.parse_args()
    out_dir = os.path.abspath(a.out_dir)
    if a.runbook:
        runbook(a.out_dir)
        return
    labels = [x for x in a.labels.split(",") if x]
    verdict(report(out_dir, labels, a.decisive), a.decisive)


if __name__ == "__main__":
    main()
