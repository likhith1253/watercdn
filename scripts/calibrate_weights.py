"""Apply the weight-normalisation rule once and write the frozen weights file."""
from __future__ import annotations

import argparse

import yaml

from tle.config import load_config, problem_spec, provenance, repo_path
from tle.scenarios.space import base_scenario
from tle.hydraulics.weights import normalise_weights


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base_net3.yaml")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261107)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["objective"].pop("weights", None)
    out_file = repo_path(cfg["objective"]["weights_file"])
    sc = base_scenario(cfg)
    res = normalise_weights(problem_spec(cfg), sc, args.n, args.seed)
    res["base_scenario"] = sc.to_dict()
    res["seed"] = args.seed
    res["provenance"] = provenance(cfg)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(yaml.safe_dump(res, sort_keys=False))
    print(yaml.safe_dump(res, sort_keys=False))


if __name__ == "__main__":
    main()
