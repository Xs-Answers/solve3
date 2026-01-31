#!/usr/bin/env python3
"""
聚合 L18 批量结果的关键指标。

输入：
- l18_summary.json（由 run_l18.py 生成）

输出：
- l18_metrics.csv：每个 run 的汇总指标
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _to_float(value: str, field: str) -> float:
    # 字符串转浮点，并带字段名提示
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing numeric field: {field}")
    return float(value)


def _read_summary(path: Path) -> List[Dict[str, str]]:
    # 读取 run 列表
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    # 读取合并参数表
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _aggregate_one_run(rows: List[Dict[str, str]]) -> Dict[str, float]:
    # 基于合并表计算汇总指标（发射次数、载荷能力、平均推进剂比等）
    total_launches = 0.0
    total_launches_int = 0.0
    total_payload = 0.0
    w_prop1 = 0.0
    w_prop2 = 0.0
    w_prop3 = 0.0
    w_dv_surface = 0.0
    w_dv_geo = 0.0
    extrapolated_launches = 0.0

    years: List[int] = []
    for row in rows:
        year = int(float(row["year"]))
        years.append(year)

        launches = _to_float(row["launches_eff_model"], "launches_eff_model")
        launches_int = _to_float(row["launches_eff_int_model"], "launches_eff_int_model")
        payload = _to_float(row["payload_per_launch_ton"], "payload_per_launch_ton")

        prop1 = _to_float(row["prop_ratio_mode1"], "prop_ratio_mode1")
        prop2 = _to_float(row["prop_ratio_mode2"], "prop_ratio_mode2")
        prop3 = _to_float(row["prop_ratio_mode3"], "prop_ratio_mode3")

        dv_surface = _to_float(row["dv_surface_to_moon_kmps"], "dv_surface_to_moon_kmps")
        dv_geo = _to_float(row["dv_geo_to_moon_kmps"], "dv_geo_to_moon_kmps")

        total_launches += launches
        total_launches_int += launches_int
        total_payload += launches * payload

        w_prop1 += launches * prop1
        w_prop2 += launches * prop2
        w_prop3 += launches * prop3
        w_dv_surface += launches * dv_surface
        w_dv_geo += launches * dv_geo

        if row.get("is_extrapolated", "0") == "1":
            extrapolated_launches += launches

    if total_launches > 0:
        avg_prop1 = w_prop1 / total_launches
        avg_prop2 = w_prop2 / total_launches
        avg_prop3 = w_prop3 / total_launches
        avg_dv_surface = w_dv_surface / total_launches
        avg_dv_geo = w_dv_geo / total_launches
        extrapolated_ratio = extrapolated_launches / total_launches
    else:
        avg_prop1 = avg_prop2 = avg_prop3 = 0.0
        avg_dv_surface = avg_dv_geo = 0.0
        extrapolated_ratio = 0.0

    year_min = min(years) if years else 0
    year_max = max(years) if years else 0

    return {
        "year_min": year_min,
        "year_max": year_max,
        "total_launches": total_launches,
        "total_launches_int": total_launches_int,
        "total_payload_capacity_ton": total_payload,
        "avg_prop_ratio_mode1": avg_prop1,
        "avg_prop_ratio_mode2": avg_prop2,
        "avg_prop_ratio_mode3": avg_prop3,
        "avg_dv_surface_to_moon_kmps": avg_dv_surface,
        "avg_dv_geo_to_moon_kmps": avg_dv_geo,
        "extrapolated_launch_ratio": extrapolated_ratio,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate L18 results into summary metrics.")
    parser.add_argument("--summary", default="l18_outputs/l18_summary.json")
    parser.add_argument("--output", default="l18_outputs/l18_metrics.csv")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary_path = Path(args.summary)
    output_path = Path(args.output)

    runs = _read_summary(summary_path)
    if not runs:
        raise ValueError("Empty summary list.")

    rows_out: List[Dict[str, str]] = []
    for run in runs:
        output_csv = Path(run["output_csv"])
        data_rows = _read_csv_rows(output_csv)
        metrics = _aggregate_one_run(data_rows)

        rows_out.append(
            {
                "run_id": run["run_id"],
                "output_csv": run["output_csv"],
                "year_min": str(metrics["year_min"]),
                "year_max": str(metrics["year_max"]),
                "total_launches": f"{metrics['total_launches']:.6f}",
                "total_launches_int": f"{metrics['total_launches_int']:.0f}",
                "total_payload_capacity_ton": f"{metrics['total_payload_capacity_ton']:.6f}",
                "avg_prop_ratio_mode1": f"{metrics['avg_prop_ratio_mode1']:.8f}",
                "avg_prop_ratio_mode2": f"{metrics['avg_prop_ratio_mode2']:.8f}",
                "avg_prop_ratio_mode3": f"{metrics['avg_prop_ratio_mode3']:.8f}",
                "avg_dv_surface_to_moon_kmps": f"{metrics['avg_dv_surface_to_moon_kmps']:.6f}",
                "avg_dv_geo_to_moon_kmps": f"{metrics['avg_dv_geo_to_moon_kmps']:.6f}",
                "extrapolated_launch_ratio": f"{metrics['extrapolated_launch_ratio']:.6f}",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
