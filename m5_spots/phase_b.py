#!/usr/bin/env python3
"""M5 Phase B driver: preflop lab solves -> HU flop exports -> realization runs.

Runs against a local gto-server (started automatically if :3737 is quiet).
Stages:
  spots  solve each game config in the lab (GPU when built with it), walk
         the study lines, export HU flop spots into m5_out/spots/*.json
  run    solve-cli realization for every spot x the flop subset, one output
         file per spot (resumable: finished spots are skipped), then concat
         into m5_out/realization_obs.jsonl and write m5_out/DONE
  all    both

  --pilot   NL10 game only, 2 spots, first 10 flops, loose targets
  --mini    tiny 3-max game, 2 flops - local smoke test of the plumbing
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "m5_out")
API = "http://127.0.0.1:3737"

# ---------------------------------------------------------------- games ----

P8 = ["UTG", "UTG1", "MP", "HJ", "CO", "BTN", "SB", "BB"]
P6 = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
P3 = ["BTN", "SB", "BB"]

def pf(positions, stack, opens, mults, max_raises, limp, allin, rake_pct, rake_cap):
    posts = [0.0] * (len(positions) - 2) + [0.5, 1.0]
    return {
        "positions": positions, "stack": stack, "posts": posts, "ante": 0.0,
        "limp": limp, "open_raises": opens, "raise_mults": mults,
        "max_raises": max_raises, "add_allin": allin, "allin_threshold": 0.85,
        "rake_pct": rake_pct, "rake_cap": rake_cap,
        "no_flop_no_drop": True, "realization": "static",
    }

# line steps: F=fold, C=call, K=check, ("R", to)=open to amount, RM=min re-raise
F, C, K, RM = ("fold",), ("call",), ("check",), ("raise_min",)
def R(to):
    return ("raise", to)

def eight_max_lines(open_to, limped):
    lines = {
        "utg_open_btn_call": [R(open_to), F, F, F, F, C, F, F],
        "utg_open_bb_call":  [R(open_to), F, F, F, F, F, F, C],
        "co_open_bb_call":   [F, F, F, F, R(open_to), F, F, C],
        "utg_open_btn_3bet": [R(open_to), F, F, F, F, RM, F, F, C],
    }
    if limped:
        lines["sb_limp_bb_check"] = [F, F, F, F, F, F, C, K]
    return lines

GAMES = [
    ("g22_open5",  pf(P8, 150, [5.0],  [2.0, 3.5], 2, True, True, 10.0, 11.0),
     eight_max_lines(5.0, True)),
    ("g22_open75", pf(P8, 150, [7.5],  [2.0, 3.5], 2, True, True, 10.0, 11.0),
     eight_max_lines(7.5, False)),
    ("g22_open10", pf(P8, 150, [10.0], [2.0, 3.5], 2, True, True, 10.0, 11.0),
     eight_max_lines(10.0, False)),
    ("g25_150",    pf(P8, 150, [3.0, 4.0], [2.5, 4.0], 2, True, True, 10.0, 9.0),
     eight_max_lines(3.0, True)),
    ("g25_200",    pf(P8, 200, [3.0, 4.0], [2.5, 4.0], 2, True, True, 10.0, 5.0),
     eight_max_lines(3.0, False)),
    ("nl10_6max",  pf(P6, 100, [2.5], [3.0], 3, True, False, 5.0, 3.0), {
        "co_open_bb_call":  [F, F, R(2.5), F, F, C],
        "btn_open_bb_call": [F, F, F, R(2.5), F, C],
        "btn_open_sb_3bet": [F, F, F, R(2.5), RM, F, C],
        "sb_limp_bb_check": [F, F, F, F, C, K],
    }),
]

MINI = [("mini_3max", pf(P3, 40, [2.5], [3.0], 2, True, True, 0.0, 0.0), {
    "btn_open_bb_call": [R(2.5), F, C],
    "sb_limp_bb_check": [F, C, K],
})]

# Round 2 (fit v5): dense 3-bet-pot coverage across positions/directions +
# Matthew's 4-max 7.5x game, so class x (role, pot-type) standardization
# cells have real support and the model's domain covers the game he plays.
# Lab solves run under realization "calibrated" (current shipped table), so
# exported ranges reflect the model's own play — the first half-step of the
# fixed-point refit loop (TODO: M5 round-2).
P4 = ["CO", "BTN", "SB", "BB"]

def pf_cal(*args, **kwargs):
    cfg = pf(*args, **kwargs)
    cfg["realization"] = "calibrated"
    return cfg

GAMES_R2 = [
    ("g22_open75", pf_cal(P8, 150, [7.5], [2.0, 3.5], 2, True, True, 10.0, 11.0), {
        "co_open_btn_3bet": [F, F, F, F, R(7.5), RM, F, F, C],
        "utg_open_sb_3bet": [R(7.5), F, F, F, F, F, RM, F, C],
        "btn_open_bb_3bet": [F, F, F, F, F, R(7.5), F, RM, C],
        "btn_open_bb_call": [F, F, F, F, F, R(7.5), F, C],
    }),
    ("g25_150", pf_cal(P8, 150, [3.0, 4.0], [2.5, 4.0], 2, True, True, 10.0, 9.0), {
        "co_open_btn_3bet": [F, F, F, F, R(3.0), RM, F, F, C],
        "btn_open_bb_3bet": [F, F, F, F, F, R(3.0), F, RM, C],
        "btn_open_bb_call": [F, F, F, F, F, R(3.0), F, C],
    }),
    ("g4max_75", pf_cal(P4, 150, [7.5], [3.0], 2, True, True, 10.0, 11.0), {
        "co_open_bb_call":  [R(7.5), F, F, C],
        "co_open_btn_call": [R(7.5), C, F, F],
        "co_open_btn_3bet": [R(7.5), RM, F, F, C],
        "co_open_bb_3bet":  [R(7.5), F, F, RM, C],
        "btn_open_sb_3bet": [F, R(7.5), RM, F, C],
        "sb_open_bb_3bet":  [F, F, R(7.5), RM, C],
    }),
    ("nl10_6max", pf_cal(P6, 100, [2.5], [3.0], 3, True, False, 5.0, 3.0), {
        "co_open_btn_3bet": [F, F, R(2.5), RM, F, F, C],
        "btn_open_bb_3bet": [F, F, F, R(2.5), F, RM, C],
    }),
]

# Fixed-point iteration (M5 round-2 close-out): restage EVERY line —
# round-1 games + round-2 additions — with labs solved under the SHIPPED
# calibrated table, so the training mix converges to the mix the model
# produces. Known-bad lines excluded up front: the two SPR-74 limped
# spots (cost/insight ratio) and the vs-10x BB defend (0 observations —
# the range barely exists under an 11bb cap).
FP_DROP = {
    ("g22_open5", "sb_limp_bb_check"),
    ("g25_150", "sb_limp_bb_check"),
    ("g22_open10", "utg_open_bb_call"),
}

def games_fixedpoint():
    merged = {}
    for gname, cfg, lines in GAMES + GAMES_R2:
        cfg = dict(cfg, realization="calibrated")
        if gname not in merged:
            merged[gname] = (cfg, {})
        merged[gname][1].update(
            {l: s for l, s in lines.items() if (gname, l) not in FP_DROP})
    return [(g, c, l) for g, (c, l) in merged.items()]

# ---------------------------------------------- N8-shaped dataset (--n8) ----
# The store under test is 6-max N8: 40-100 bb, 5 % rake capped at 10 bb, one
# open size per variant (2-4x), N8's fold-out rule (rake only with >= 2
# voluntary players) and the 2.5x 4-bet ladder. Labs run under `balanced` — the
# payoff the store is generated with — so the observations describe the model
# that will consume them, instead of the shipped `calibrated` table talking to
# itself (the fixed-point loop). Lines are the standard 6-max shapes
# `fit_phase_c5.spot_axes` parses, all open to 2.5x so the exported ranges
# match the bucket under test; 12 of the 24 spots are 3-bet pots (guard:
# MIN_3BP_SPOTS = 8).
N8_OPENS = [2.5]          # ONE open size in the lab tree: `balanced_share`
                          # reads `class_base` only, so the leaf has no size
                          # axis to fit — and 3 sizes in one tree measured
                          # ~11.1/12.2 GB VRAM at 55 s/iter on this 3060.
N8_MULTS = [3.0, 4.5]
N8_LINES = {
    "utg_open_bb_call": [R(2.5), F, F, F, F, C],
    "co_open_bb_call":  [F, F, R(2.5), F, F, C],
    "btn_open_bb_call": [F, F, F, R(2.5), F, C],
    "co_open_btn_3bet": [F, F, R(2.5), RM, F, F, C],
    "btn_open_sb_3bet": [F, F, F, R(2.5), RM, F, C],
    "btn_open_bb_3bet": [F, F, F, R(2.5), F, RM, C],
}

def n8_pf(stack):
    cfg = pf(P6, stack, N8_OPENS, N8_MULTS, 4, True, False, 5.0, 10.0)
    cfg.update(realization="balanced", no_flop_no_drop=False,
               fold_win_min_voluntary=2, fourbet_mults=[2.5])
    return cfg

GAMES_N8 = [(f"n8_{d}", n8_pf(d), dict(N8_LINES)) for d in (40, 50, 75, 100)]

# Postflop menu per Matthew: 30/75 + all-in on the flop, single size after
# (raises 3x, raise cap 2). The full two-size-every-street menu explodes to
# 100GB+ arenas on wide-range NL10 spots (pilot 2026-07-07: 8.2M nodes) —
# calibration needs realistic play, not exhaustive sizing.
def street3075():
    return {"bet": [{"PotPct": 30}, {"PotPct": 75}], "raise": [{"PrevMult": 3.0}], "donk": []}

def street75():
    return {"bet": [{"PotPct": 75}], "raise": [{"PrevMult": 3.0}], "donk": []}

def spot_config(ex, mini=False):
    # mini: one small size, no all-in — plumbing test on laptop-class RAM
    if mini:
        mini_st = lambda: {"bet": [{"PotPct": 33}], "raise": [{"PrevMult": 3.0}], "donk": []}
        streets = [mini_st(), mini_st(), mini_st()]
    else:
        streets = [street3075(), street75(), street75()]
    return {
        "board": "AhKs2d",  # overwritten per board by solve-cli
        "range_oop": ex["range_oop"], "range_ip": ex["range_ip"],
        "tree": {
            "starting_pot": ex["pot_bb"], "effective_stack": ex["eff_stack_bb"],
            # exports carry PERCENT (10.0); TreeConfig wants a FRACTION (0.10)
            "rake_pct": ex.get("rake_pct", 0.0) / 100.0, "rake_cap": ex.get("rake_cap", 0.0),
            "oop": [dict(s) for s in streets],
            "ip": [dict(s) for s in streets],
            "allin_threshold": 0.85, "add_allin": not mini, "max_raises": 2,
        },
    }

# ------------------------------------------------------------- plumbing ----

def req(method, path, body=None, timeout=1800, retries=3):
    for attempt in range(retries):
        try:
            r = urllib.request.Request(
                API + path,
                data=json.dumps(body).encode() if body is not None else None,
                headers={"Content-Type": "application/json"}, method=method)
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"  api retry {path}: {exc}", flush=True)
            time.sleep(5)

def ensure_server():
    try:
        req("GET", "/api/preflop/status", timeout=5, retries=1)
        return None
    except Exception:
        pass
    binpath = os.path.join(ROOT, "target/release/gto-server")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    proc = subprocess.Popen([binpath], cwd=ROOT, env=env,
                            stdout=open(os.path.join(OUT, "server.log"), "ab"),
                            stderr=subprocess.STDOUT)
    for _ in range(60):
        time.sleep(1)
        try:
            req("GET", "/api/preflop/status", timeout=5, retries=1)
            print("server up", flush=True)
            return proc
        except Exception:
            continue
    raise RuntimeError("gto-server did not come up")

def solve_lab(cfg, iters, check, target):
    req("POST", "/api/preflop/spot", cfg)
    req("POST", "/api/preflop/solve",
        {"iterations": iters, "check_every": check, "target_gap": target})
    t0 = time.time()
    while True:
        time.sleep(5)
        st = req("GET", "/api/preflop/status")
        if st["state"] != "running":
            return st, time.time() - t0

def find_action(actions, step):
    kind = step[0]
    if kind == "raise":
        want = step[1]
        for i, a in enumerate(actions):
            amt = a.get("to", a.get("amount", 0)) or 0
            if a["kind"] in ("raise", "jam") and abs(amt - want) < 0.01:
                return i
        raise RuntimeError(f"no raise-to-{want} in {[a['label'] for a in actions]}")
    if kind == "raise_min":
        for i, a in enumerate(actions):
            if a["kind"] == "raise":
                return i  # actions are size-ordered; first = smallest
        raise RuntimeError(f"no raise in {[a['label'] for a in actions]}")
    for i, a in enumerate(actions):
        if a["kind"] == kind:
            return i
    raise RuntimeError(f"no {kind} in {[a['label'] for a in actions]}")

def walk(steps):
    path = []
    for step in steps:
        node = req("POST", "/api/preflop/node", {"path": path})
        path.append(find_action(node["actions"], step))
    return path

def stage_spots(games, iters, check, target, mini=False):
    os.makedirs(os.path.join(OUT, "spots"), exist_ok=True)
    manifest = {}
    mpath = os.path.join(OUT, "manifest.json")
    if os.path.exists(mpath):
        # `--only` runs compose: the 4-game stage is split one game per
        # invocation (a single call outlives a one-hour job clamp and writes
        # the manifest only at the end), so merge what is already on disk.
        manifest = json.load(open(mpath))
    for gname, cfg, lines in games:
        print(f"=== {gname}: solving lab game", flush=True)
        st, secs = solve_lab(cfg, iters, check, target)
        print(f"    iter {st['iteration']} gap {st.get('gap_total', -1):.4f} "
              f"[{secs:.0f}s] gpu={st.get('gpu')}", flush=True)
        for lname, steps in lines.items():
            sname = f"{gname}__{lname}"
            try:
                path = walk(steps)
                ex = req("POST", "/api/preflop/export", {"path": path})
            except Exception as exc:
                print(f"    {sname}: EXPORT FAILED: {exc}", flush=True)
                continue
            spot = spot_config(ex, mini=mini)
            fname = os.path.join(OUT, "spots", sname + ".json")
            with open(fname, "w") as f:
                json.dump(spot, f, indent=1)
            manifest[sname] = {
                "game": gname, "line": lname, "oop": ex["oop_pos"], "ip": ex["ip_pos"],
                "pot_bb": ex["pot_bb"], "eff_stack_bb": ex["eff_stack_bb"],
                "spr": ex["eff_stack_bb"] / max(ex["pot_bb"], 1e-9),
                "lab_iterations": st["iteration"], "lab_gap": st.get("gap_total"),
            }
            print(f"    {sname}: pot {ex['pot_bb']:.1f} spr "
                  f"{manifest[sname]['spr']:.1f} ({ex['oop_pos']} vs {ex['ip_pos']})",
                  flush=True)
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"spots stage done: {len(manifest)} spots", flush=True)

def stage_run(flops_file, iters, target):
    # Cheapest first — 3-bet pots, then shallow to deep. A 3-bet-pot spot
    # solves in seconds per board, while a wide BB-defend SRP spot at 100 bb
    # measured 35 s/board and 8.5 GB peak on the 3060; a one-hour job clamp
    # would otherwise throw that unfinished spot away on every retry.
    def cost(s):
        try:
            depth = int(s.split("n8_")[1].split("__")[0])
        except (IndexError, ValueError):
            depth = 0
        return ("_3bet" not in s, depth, s)
    spots = sorted(os.listdir(os.path.join(OUT, "spots")), key=cost)
    cli = os.path.join(ROOT, "target/release/solve-cli"
                       + (".exe" if os.name == "nt" else ""))
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    # Windows: cudarc resolves `nvrtc64_*.dll` by bare name through
    # LoadLibrary, so its bin directory has to be on PATH (run-stack.py:485
    # does the same before starting the server). Without it, solve-cli panics
    # with "Unable to dynamically load the nvrtc shared library".
    nvrtc_bin = os.path.join(ROOT, ".cuda-nvrtc", "nvidia", "cuda_nvrtc", "bin")
    if os.path.isdir(nvrtc_bin):
        env["PATH"] = nvrtc_bin + os.pathsep + env.get("PATH", "")
    # wide-range spots (limped pots, NL10 defends) cost ~10x per flop; 40
    # flops x ~265 classes still out-samples 100 x ~42 on tight spots
    flops40 = os.path.join(ROOT, "m5_spots/flops40.txt")
    if not os.path.exists(cli):
        raise SystemExit(f"{cli} not built — run `cargo build --release` first")
    if not os.path.exists(flops40) or os.path.getsize(flops40) == 0:
        # write to a temp path and rename on success: a failed generation
        # (CLI missing/crashing) used to leave an EMPTY flops40.txt behind,
        # and the next run then solved every wide spot against no boards,
        # printed "zero observations" and marked it DONE for good
        tmp = flops40 + ".tmp"
        with open(tmp, "w") as f:
            subprocess.check_call([cli, "flops", "40"], stdout=f)
        if os.path.getsize(tmp) == 0:
            os.remove(tmp)
            raise SystemExit("solve-cli flops 40 produced no boards")
        os.replace(tmp, flops40)
    total_t0 = time.time()
    for i, s in enumerate(spots):
        name = s[:-5]
        obs = os.path.join(OUT, f"obs_{name}.jsonl")
        mark = obs + ".done"
        if os.path.exists(mark):
            print(f"[{i+1}/{len(spots)}] {name}: already done, skipping", flush=True)
            continue
        if os.path.exists(obs):
            os.remove(obs)  # partial from a crash: redo cleanly
        # 3-bet pots have narrow ranges regardless of game — never "wide"
        wide = (("limp" in name) or ("nl10" in name)) and "_3bet" not in name
        f_file = flops40 if wide else flops_file
        if not os.path.exists(f_file) or os.path.getsize(f_file) == 0:
            raise SystemExit(f"board list {f_file} is missing or empty")
        f_target = max(target, 0.45) if wide else target
        tag = " (wide: 40 flops, 0.45%)" if wide else ""
        print(f"[{i+1}/{len(spots)}] {name}{tag}", flush=True)
        rc = subprocess.call(
            [cli, "realization", os.path.join(OUT, "spots", s), f_file,
             str(iters), str(f_target), obs],
            cwd=ROOT, env=env)
        if rc != 0:
            print(f"    FAILED rc={rc}", flush=True)
            continue
        n_obs = sum(1 for l in open(obs) if '"obs"' in l)
        if n_obs == 0:
            # legitimate under calibrated labs: the line itself can be
            # off-equilibrium (e.g. flat-call vs 7.5x) -> zero-mass export.
            # Loud, not fatal: the spot contributes nothing to the fit.
            print("    ZERO OBSERVATIONS - line has ~0 reach under this "
                  "lab model (off-equilibrium)", flush=True)
        open(mark, "w").write("ok\n")
    with open(os.path.join(OUT, "realization_obs.jsonl"), "w") as out:
        for s in spots:
            name = s[:-5]
            p = os.path.join(OUT, f"obs_{name}.jsonl")
            if os.path.exists(p + ".done"):
                out.write(open(p).read())
    open(os.path.join(OUT, "DONE"), "w").write(
        f"finished in {time.time() - total_t0:.0f}s\n")
    print(f"run stage done -> {OUT}/realization_obs.jsonl", flush=True)

def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["spots", "run", "all"], default="all")
    ap.add_argument("--flops", default=os.path.join(ROOT, "m5_spots/flops100.txt"))
    ap.add_argument("--out-dir", default=OUT,
                    help="where spots/obs land (default: m5_out)")
    ap.add_argument("--only", default=None,
                    help="restrict the run to one game by name (e.g. n8_40)")
    ap.add_argument("--n8", action="store_true",
                    help="N8-shaped 6-max dataset: 40/50/75/100 bb, 2-3x opens, "
                         "rake 5 %/10 bb, labs under `balanced`")
    ap.add_argument("--iters", type=int, default=900)
    ap.add_argument("--target", type=float, default=0.3)
    ap.add_argument("--lab-iters", type=int, default=1500)
    ap.add_argument("--lab-target", type=float, default=0.008)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--round2", action="store_true",
                    help="fit-v5 data round: 3-bet pots + 4-max game, "
                         "labs solved under the shipped calibrated table")
    ap.add_argument("--fixedpoint", action="store_true",
                    help="restage ALL lines (round-1 + round-2) with labs "
                         "under the shipped calibrated table")
    args = ap.parse_args()

    OUT = os.path.abspath(args.out_dir)
    os.makedirs(OUT, exist_ok=True)
    games = GAMES
    if args.n8:
        if args.round2 or args.fixedpoint:
            raise SystemExit("--n8 is its own dataset; drop --round2/--fixedpoint")
        games = GAMES_N8
    if args.only:
        games = [g for g in games if g[0] == args.only]
        if not games:
            raise SystemExit(f"no game named {args.only!r} in this dataset")
    if args.round2:
        games = GAMES_R2
    if args.fixedpoint:
        games = games_fixedpoint()
    if args.mini:
        games = MINI
        args.lab_iters, args.lab_target = 300, 0.02
        args.iters, args.target = 300, 0.6
        flops = os.path.join(OUT, "flops_mini.txt")
        open(flops, "w").write("Ks7h2d\nAc6c2d\n")
        args.flops = flops
    elif args.pilot:
        # tight-range 8-max spots: they fit a 24GB card; the wide NL10
        # ranges are H100 material
        games = [g for g in GAMES if g[0] == "g22_open5"]
        games = [(g[0], g[1], {k: g[2][k] for k in ("utg_open_btn_call", "utg_open_bb_call")}) for g in games]
        args.lab_iters, args.lab_target = 800, 0.012
        args.iters, args.target = 600, 0.4
        flops = os.path.join(OUT, "flops_pilot.txt")
        subset = open(args.flops).read().split()[:10]
        open(flops, "w").write("\n".join(subset) + "\n")
        args.flops = flops

    proc = ensure_server() if args.stage in ("spots", "all") else None
    try:
        if args.stage in ("spots", "all"):
            stage_spots(games, args.lab_iters, 100, args.lab_target, mini=args.mini)
        if args.stage in ("run", "all"):
            stage_run(args.flops, args.iters, args.target)
    finally:
        if proc is not None:
            proc.terminate()

if __name__ == "__main__":
    main()
