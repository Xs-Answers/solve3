#!/usr/bin/env python3
"""
运行 Taguchi L18 实验矩阵，并为每次实验生成合并参数表。

输出内容：
- 每个 run 的参数 JSON、合并 CSV（可选）
- L18 总览 JSON + CSV（包含水平与数值对应）

用法示例：
  python run_l18.py
  python run_l18.py --end-year 2150 --output-dir l18_outputs
  python run_l18.py --base-config params_template.json --no-run
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
import csv
from pathlib import Path
from typing import Dict, List

import build_parameters


# L18 水平映射表（A-G）
LEVELS = {
    "A": {"L1": 0.6, "L2": 1.0, "L3": 1.2},   # k_heavy
    "B": {"L1": 0.15, "L2": 0.25, "L3": 0.35},  # k_shuttle
    "C": {"L1": 395.0, "L2": 400.0, "L3": 405.0},  # IspVmax
    "D": {"L1": 340.0, "L2": 350.0, "L3": 355.0},  # IspSLmax
    "E": {"L1": 0.04, "L2": 0.05, "L3": 0.07},  # k_I
    "F": {"L1": 100.0, "L2": 125.0, "L3": 150.0},  # x_2050
    "G": {"L1": 0.03, "L2": 0.05, "L3": 0.08},  # g_r
}

# L18 标准矩阵（含 H 因子：地表段 Isp 口径）
L18_RUNS = [
    {"A": "L1", "B": "L1", "C": "L1", "D": "L1", "E": "L1", "F": "L1", "G": "L1", "H": 1},
    {"A": "L1", "B": "L2", "C": "L2", "D": "L2", "E": "L2", "F": "L2", "G": "L2", "H": 1},
    {"A": "L1", "B": "L3", "C": "L3", "D": "L3", "E": "L3", "F": "L3", "G": "L3", "H": 1},
    {"A": "L2", "B": "L1", "C": "L1", "D": "L2", "E": "L2", "F": "L3", "G": "L3", "H": 1},
    {"A": "L2", "B": "L2", "C": "L2", "D": "L3", "E": "L3", "F": "L1", "G": "L1", "H": 1},
    {"A": "L2", "B": "L3", "C": "L3", "D": "L1", "E": "L1", "F": "L2", "G": "L2", "H": 1},
    {"A": "L3", "B": "L1", "C": "L2", "D": "L1", "E": "L3", "F": "L2", "G": "L3", "H": 1},
    {"A": "L3", "B": "L2", "C": "L3", "D": "L2", "E": "L1", "F": "L3", "G": "L1", "H": 1},
    {"A": "L3", "B": "L3", "C": "L1", "D": "L3", "E": "L2", "F": "L1", "G": "L2", "H": 1},
    {"A": "L1", "B": "L1", "C": "L3", "D": "L3", "E": "L2", "F": "L2", "G": "L1", "H": 2},
    {"A": "L1", "B": "L2", "C": "L1", "D": "L1", "E": "L3", "F": "L3", "G": "L2", "H": 2},
    {"A": "L1", "B": "L3", "C": "L2", "D": "L2", "E": "L1", "F": "L1", "G": "L3", "H": 2},
    {"A": "L2", "B": "L1", "C": "L2", "D": "L3", "E": "L1", "F": "L1", "G": "L2", "H": 2},
    {"A": "L2", "B": "L2", "C": "L3", "D": "L1", "E": "L2", "F": "L2", "G": "L3", "H": 2},
    {"A": "L2", "B": "L3", "C": "L1", "D": "L2", "E": "L3", "F": "L3", "G": "L1", "H": 2},
    {"A": "L3", "B": "L1", "C": "L3", "D": "L2", "E": "L3", "F": "L1", "G": "L2", "H": 2},
    {"A": "L3", "B": "L2", "C": "L1", "D": "L3", "E": "L1", "F": "L2", "G": "L3", "H": 2},
    {"A": "L3", "B": "L3", "C": "L2", "D": "L1", "E": "L2", "F": "L3", "G": "L1", "H": 2},
]


def _load_base_config(path: Path) -> Dict[str, float]:
    # 读取基础参数模板（用于被 L18 覆盖）
    if not path.exists():
        raise FileNotFoundError(f"Base config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _merge_params(base: Dict[str, float], run: Dict[str, str]) -> Dict[str, float]:
    # 将 L18 水平映射为具体数值，覆盖基础参数
    params = dict(base)
    params["k_heavy"] = LEVELS["A"][run["A"]]
    params["k_shuttle"] = LEVELS["B"][run["B"]]
    params["isp_vmax_s"] = LEVELS["C"][run["C"]]
    params["isp_slmax_s"] = LEVELS["D"][run["D"]]
    params["k_I"] = LEVELS["E"][run["E"]]
    params["x_2050_ton"] = LEVELS["F"][run["F"]]
    params["g_r"] = LEVELS["G"][run["G"]]
    params["surface_isp"] = "SL" if run["H"] == 1 else "V"
    return params


def _parse_args() -> argparse.Namespace:
    # 命令行参数
    parser = argparse.ArgumentParser(description="Run L18 sensitivity batch.")
    parser.add_argument("--input", default="launch_capacity_adjusted_2025_2100.csv")
    parser.add_argument("--output-dir", default="l18_outputs")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--launch-col", default=None)
    parser.add_argument("--rounding", choices=["round", "floor", "ceil"], default="round")
    parser.add_argument("--base-config", default="params_template.json")
    parser.add_argument("--no-run", action="store_true", help="Only write configs; do not run build_parameters.")
    return parser.parse_args()


def main() -> int:
    # 主流程：生成每次 run 的参数文件与输出，并汇总
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 基础参数载入与校验
    base = _load_base_config(Path(args.base_config))
    if set(base.keys()) - set(asdict(build_parameters.ModelParams()).keys()):
        raise ValueError("Base config contains unknown keys.")

    summary_rows: List[Dict[str, str]] = []
    summary_csv_rows: List[Dict[str, str]] = []
    for idx, run in enumerate(L18_RUNS, start=1):
        # 生成该次 run 的参数
        params = _merge_params(base, run)
        config_path = output_dir / f"run_{idx:02d}_params.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

        output_csv = output_dir / f"run_{idx:02d}_combined.csv"
        meta_json = output_dir / f"run_{idx:02d}_meta.json"

        # 需要时才执行 build_parameters（默认会执行）
        if not args.no_run:
            argv = [
                "--input",
                args.input,
                "--output",
                str(output_csv),
                "--config",
                str(config_path),
                "--rounding",
                args.rounding,
                "--write-meta",
                str(meta_json),
            ]
            if args.start_year is not None:
                argv.extend(["--start-year", str(args.start_year)])
            if args.end_year is not None:
                argv.extend(["--end-year", str(args.end_year)])
            if args.launch_col:
                argv.extend(["--launch-col", args.launch_col])
            build_parameters.main(argv)

        # JSON 汇总（保持和之前一致的结构）
        summary_rows.append(
            {
                "run_id": f"{idx:02d}",
                "A_k_heavy": run["A"],
                "B_k_shuttle": run["B"],
                "C_isp_vmax": run["C"],
                "D_isp_slmax": run["D"],
                "E_k_I": run["E"],
                "F_x_2050": run["F"],
                "G_g_r": run["G"],
                "H_surface_isp": "SL" if run["H"] == 1 else "V",
                "output_csv": str(output_csv),
                "config_json": str(config_path),
            }
        )

        # CSV 汇总（包含水平与数值，便于表格查看）
        summary_csv_rows.append(
            {
                "run_id": f"{idx:02d}",
                "A_level": run["A"],
                "B_level": run["B"],
                "C_level": run["C"],
                "D_level": run["D"],
                "E_level": run["E"],
                "F_level": run["F"],
                "G_level": run["G"],
                "H_surface_isp": "SL" if run["H"] == 1 else "V",
                "k_heavy": f"{params['k_heavy']:.6f}",
                "k_shuttle": f"{params['k_shuttle']:.6f}",
                "isp_vmax_s": f"{params['isp_vmax_s']:.6f}",
                "isp_slmax_s": f"{params['isp_slmax_s']:.6f}",
                "k_I": f"{params['k_I']:.6f}",
                "x_2050_ton": f"{params['x_2050_ton']:.6f}",
                "g_r": f"{params['g_r']:.6f}",
                "output_csv": str(output_csv),
                "config_json": str(config_path),
            }
        )

    summary_path = output_dir / "l18_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    summary_csv_path = output_dir / "l18_summary.csv"
    with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary_csv_rows[0].keys()) if summary_csv_rows else [],
        )
        if summary_csv_rows:
            writer.writeheader()
            writer.writerows(summary_csv_rows)

    print(f"Wrote {len(L18_RUNS)} configs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
