#!/usr/bin/env python3
"""Ensure the dense duration bank configured as data.reuse_bank_dir exists.

Core experiment YAMLs contain only their six active durations.  When the
superset bank is absent, this tool derives the 0.20--3.00 s, 0.01 s grid from
that same core physics config, generates it once, and otherwise exits without
simulation.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.banks.power_grid import bank_is_complete, generate_physical_bank
from src.config import SBOEDConfig, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = dict(cfg.raw.get("data") or {})
    reuse = data.get("reuse_bank_dir")
    if not reuse:
        print("[full-bank] no data.reuse_bank_dir; nothing to ensure")
        return
    root = ROOT
    output = Path(str(reuse))
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if bank_is_complete(output):
        print(f"[full-bank] complete -> {output}; skipping generation")
        return

    raw = copy.deepcopy(cfg.raw)
    full_data = dict(raw.get("data") or {})
    full_data["dataset_dir"] = str(output)
    full_data.pop("reuse_bank_dir", None)
    full_data.pop("mocu_dataset_dir", None)
    full_data["generate_if_missing"] = True
    raw["data"] = full_data
    swing = dict(raw.get("swing_equation") or {})
    swing["probe_durations"] = [round(0.20 + 0.01 * i, 2) for i in range(281)]
    raw["swing_equation"] = swing
    full_cfg = SBOEDConfig(raw=raw, config_path=Path(args.config).resolve())
    print(
        f"[full-bank] missing -> {output}; generating 281 durations "
        "from 0.20 to 3.00 s"
    )
    generate_physical_bank(full_cfg, project_root=root, smoke=args.smoke, force=False)


if __name__ == "__main__":
    main()
