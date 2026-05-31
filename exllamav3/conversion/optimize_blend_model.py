import argparse
import torch
from .. import Config, Model
import json
import yaml
from .compile import compile_model
from ..loader.safetensors import VariantSafetensorsCollection

col_default = "[0m"
col_red = "[31;1m"
col_yellow = "[33;1m"
col_blue = "[34;1m"
col_green = "[32;1m"
col_purple = "[35;1m"
col_cyan = "[36;1m"
col_white = "[37;1m"

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 200)

parser = argparse.ArgumentParser(
    description = "Blend three quants (low/mid/high bpw) into one model using a vote between two measurement.json files."
)
parser.add_argument("-ml", "--measurement_low", type = str, default = None,
                    help = "Low measurement.json (base = low bpw, alt = mid bpw)")
parser.add_argument("-mh", "--measurement_high", type = str, default = None,
                    help = "High measurement.json (base = mid bpw, alt = high bpw)")
parser.add_argument("-o", "--out_dir", type = str, default = None, help = "Output directory for blended model")
parser.add_argument("-y", "--out_yaml", type = str, default = None,
                    help = "Where to write the recompile override YAML, default: <out_dir>/blend.yml")
parser.add_argument("-N", "--promote_pct", type = float, default = 15.0,
                    help = "Percent of total groups to promote to high bpw, default: 15")
parser.add_argument("-M", "--neg_weight", type = float, default = 2.0,
                    help = "Negative-vote weight in conflicts; a high vote wins only if |pos| >= M*|neg|, default: 2")
parser.add_argument("--normalize", action = "store_true",
                    help = "Normalize each file's deltas by its base_kld before the M comparison (removes cross-file scale bias)")
parser.add_argument("--low_threshold", type = float, default = 0.0,
                    help = "Minimum |dkld_low| for a 'better at low' (negative) vote to count, default: 0 (off)")
parser.add_argument("-ss", "--shard_size", type = int, default = 8192, help = "Max shard size in MB, default: 8192")
parser.add_argument("--low_dir", type = str, default = None, help = "Override low-bpw model dir (default: from measurement)")
parser.add_argument("--mid_dir", type = str, default = None, help = "Override mid-bpw model dir (default: from measurement)")
parser.add_argument("--high_dir", type = str, default = None, help = "Override high-bpw model dir (default: from measurement)")


def prepare(args) -> (dict, dict, bool, str):
    if not args.measurement_low:
        return None, None, False, "Please specify --measurement_low"
    if not args.measurement_high:
        return None, None, False, "Please specify --measurement_high"
    if not args.out_dir:
        return None, None, False, "Please specify --out_dir"
    if args.promote_pct < 0 or args.promote_pct > 100:
        return None, None, False, "--promote_pct must be in [0, 100]"
    if args.neg_weight <= 0:
        return None, None, False, "--neg_weight must be > 0"

    in_args = {
        "measurement_low": args.measurement_low,
        "measurement_high": args.measurement_high,
        "out_dir": args.out_dir,
        "out_yaml": args.out_yaml,
        "promote_pct": args.promote_pct,
        "neg_weight": args.neg_weight,
        "normalize": args.normalize,
        "low_threshold": args.low_threshold,
        "shard_size": args.shard_size,
        "low_dir": args.low_dir,
        "mid_dir": args.mid_dir,
        "high_dir": args.high_dir,
    }

    print(f"    Low measurement:  {in_args['measurement_low']}")
    print(f"    High measurement: {in_args['measurement_high']}")
    print(f"    Output directory: {in_args['out_dir']}")
    print(f"    Promote (N):      {in_args['promote_pct']:.1f}% of groups -> high bpw")
    print(f"    Neg weight (M):   {in_args['neg_weight']:.2f}")
    print(f"    Normalize:        {in_args['normalize']}")
    print(f"    Low threshold:    {in_args['low_threshold']}")

    return in_args, {}, True, None


def load_measurement(path):
    with open(path, "r", encoding = "utf8") as f:
        return json.load(f)


def validate(meas_low, meas_high) -> (bool, str):
    # Architecture must match
    if meas_low.get("arch_string") != meas_high.get("arch_string"):
        return False, (f"arch_string mismatch: low='{meas_low.get('arch_string')}' "
                       f"high='{meas_high.get('arch_string')}'")

    # Exactly one alt per file (single candidate level)
    for name, m in (("low", meas_low), ("high", meas_high)):
        if len(m.get("alts", [])) != 1:
            return False, f"{name} measurement must contain exactly one alt (found {len(m.get('alts', []))})"
        if any(len(g["candidates"]) != 1 for g in m["groups"]):
            return False, f"{name} measurement groups must each have exactly one candidate"

    # Group layout must align 1:1
    gl, gh = meas_low["groups"], meas_high["groups"]
    if len(gl) != len(gh):
        return False, f"group count mismatch: low={len(gl)} high={len(gh)}"
    for i, (a, b) in enumerate(zip(gl, gh)):
        if a["layers"] != b["layers"]:
            return False, f"group {i} layers differ between low and high measurements"

    return True, None


def vote(meas_low, meas_high, promote_pct, neg_weight, normalize, low_threshold):
    """
    Returns (cells, stats). cells[g] < 0 -> low, == 0 -> mid, > 0 -> high.
    """
    groups = meas_low["groups"]
    n = len(groups)
    dk_low = [g["candidates"][0]["dkld"] for g in meas_low["groups"]]
    dk_high = [g["candidates"][0]["dkld"] for g in meas_high["groups"]]

    norm_low = meas_low.get("base_kld", 1.0) or 1.0
    norm_high = meas_high.get("base_kld", 1.0) or 1.0

    cells = [0.0] * n

    # Pass 1: groups that are better at LOW than mid -> dkld_low > 0 (raising bpw hurt)
    n_neg = 0
    for g in range(n):
        if dk_low[g] > low_threshold:
            cells[g] = -dk_low[g]
            n_neg += 1

    # Pass 2: groups that improve at HIGH (dkld_high < 0); take top N% of *all* groups by |dkld_high|
    improvers = sorted((g for g in range(n) if dk_high[g] < 0), key = lambda g: dk_high[g])  # most negative first
    num_promote = round(promote_pct / 100.0 * n)
    selected = improvers[:num_promote]

    n_high = 0
    n_conflict = 0
    n_conflict_high = 0
    for g in selected:
        pos = abs(dk_high[g])
        if cells[g] == 0.0:
            cells[g] = pos
            n_high += 1
        else:
            # Conflict: group wants low (negative) but also a top high-improver
            n_conflict += 1
            neg = -cells[g]
            if normalize:
                pos_cmp, neg_cmp = pos / norm_high, neg / norm_low
            else:
                pos_cmp, neg_cmp = pos, neg
            if pos_cmp >= neg_weight * neg_cmp:
                cells[g] = pos       # high wins
                n_neg -= 1
                n_high += 1
                n_conflict_high += 1
            # else: keep negative (stays low)

    n_mid = sum(1 for c in cells if c == 0.0)
    stats = {
        "n": n,
        "low": sum(1 for c in cells if c < 0),
        "mid": n_mid,
        "high": sum(1 for c in cells if c > 0),
        "num_promote": num_promote,
        "improvers": len(improvers),
        "conflicts": n_conflict,
        "conflicts_to_high": n_conflict_high,
    }
    return cells, stats


def build_overrides(cells, groups, low_id, high_id):
    overrides = []
    for c, g in zip(cells, groups):
        if c < 0:
            src = low_id
        elif c > 0:
            src = high_id
        else:
            continue  # mid = base, no override
        for key in g["layers"]:
            overrides.append({"key": key + ".*", "source": src})
    return overrides


@torch.inference_mode()
def main(args, job_state):

    torch.set_grad_enabled(False)

    meas_low = load_measurement(args["measurement_low"])
    meas_high = load_measurement(args["measurement_high"])

    ok, err = validate(meas_low, meas_high)
    if not ok:
        print(f" !! {col_red}Error: {err}{col_default}")
        return

    # Resolve the three model directories (defaults come from the measurement files)
    low_dir = args["low_dir"] or meas_low["base"]["dir"]
    mid_dir = args["mid_dir"] or meas_low["alts"][0]["dir"]
    high_dir = args["high_dir"] or meas_high["alts"][0]["dir"]
    mid_dir_high = meas_high["base"]["dir"]
    if not args["mid_dir"] and mid_dir != mid_dir_high:
        print(f" !! {col_yellow}Warning: mid model differs between files "
              f"(low.alt='{mid_dir}', high.base='{mid_dir_high}'). Using '{mid_dir}'.{col_default}")

    print(f" -- Low  (base):  {low_dir}  ({meas_low['base']['bpw']:.3f} bpw)")
    print(f" -- Mid  (base):  {mid_dir}  ({meas_low['alts'][0]['bpw']:.3f} bpw)")
    print(f" -- High (alt):   {high_dir}  ({meas_high['alts'][0]['bpw']:.3f} bpw)")

    # Vote
    cells, stats = vote(
        meas_low, meas_high,
        args["promote_pct"], args["neg_weight"], args["normalize"], args["low_threshold"],
    )
    print(" -- Vote results:")
    print(f"    Groups total:        {stats['n']}")
    print(f"    -> low  ({col_green}better at low{col_default}):   {stats['low']}")
    print(f"    -> mid  (default):           {stats['mid']}")
    print(f"    -> high ({col_purple}promoted{col_default}):        {stats['high']}")
    print(f"    Promote budget (N):  {stats['num_promote']}  (of {stats['improvers']} improvers)")
    print(f"    Conflicts:           {stats['conflicts']}  ({stats['conflicts_to_high']} resolved to high)")

    # Build override spec
    groups = meas_low["groups"]
    overrides = build_overrides(cells, groups, "low", "high")
    spec = {
        "sources": [
            {"id": "low", "model_dir": low_dir},
            {"id": "high", "model_dir": high_dir},
        ],
        "overrides": overrides,
    }

    import os
    out_yaml = args["out_yaml"] or os.path.join(args["out_dir"], "blend.yml")
    os.makedirs(args["out_dir"], exist_ok = True)
    with open(out_yaml, "w", encoding = "utf8") as f:
        yaml.safe_dump(spec, f, sort_keys = False)
    print(f" -- Wrote override spec: {out_yaml}")

    # Auto-compile: mid is the structural base, overrides pulled from low/high
    config = Config.from_directory(mid_dir)
    model = Model.from_config(config)
    print(f" -- Loaded model config")
    print(f"    Architecture: {config.architecture}")

    sources = {"low": low_dir, "high": high_dir}
    collections = {}
    for o in overrides:
        collections.setdefault(o["source"], []).append(o["key"])
    if collections:
        from ..loader.safetensors import SafetensorsCollection
        vstc = VariantSafetensorsCollection(config.stc)
        for src, keys in collections.items():
            vstc.add_stc(keys, SafetensorsCollection(sources[src]))
        config.stc = vstc

    bpw_layer, bpw_head, vram_bits = model.get_storage_info()
    bpw_layer = round(bpw_layer, 2)
    bpw_head = round(bpw_head)
    print(f" -- New estimated model bitrate: {bpw_layer:.2f} bpw / {bpw_head:.2f} bpw (head)")

    compile_args = {
        "bits": bpw_layer,
        "final_bits": bpw_layer,
        "head_bits": bpw_head,
        "in_dir": mid_dir,
        "out_dir": args["out_dir"],
        "shard_size": args["shard_size"],
        "model_stc": True,
    }
    compile_model(compile_args, model, config, None)
    print(" -- Done")
