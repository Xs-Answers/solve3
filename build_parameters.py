#!/usr/bin/env python3
"""
构建 MILP/MCMC 使用的“参数合并总表”。

核心思路：
- 读取发射能力 CSV（2025-2100），并与模型假设合并；
- 按站点 Logistic 规律将数据外推到指定年份（可超过 2100）；
- 输出带全量中间量与参数标识的 CSV，便于后续 MILP / MCMC 复用。

用法示例：
  python build_parameters.py --input launch_capacity_adjusted_2025_2100.csv --output combined_parameters.csv
  python build_parameters.py --end-year 2150 --surface-isp SL --x-2050 125 --g-r 0.05
  python build_parameters.py --config params.json --output combined_parameters.csv --write-meta combined_parameters.meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# 发射场经纬度与规范名称（从“假设”文件抽取的固定表）
SITE_COORDS = {
    "FL_US": {"name": "Florida_Cape_Canaveral_KSC", "lat": 28.5, "lon": -80.6},
    "TX_US": {"name": "Texas_Starbase", "lat": 26.0, "lon": -97.2},
    "CA_US": {"name": "California_Vandenberg", "lat": 34.7, "lon": -120.6},
    "VA_US": {"name": "Virginia_Wallops", "lat": 37.9, "lon": -75.5},
    "AK_US": {"name": "Alaska_Kodiak", "lat": 57.4, "lon": -152.3},
    "GF_FR": {"name": "French_Guiana_Kourou", "lat": 5.2, "lon": -52.8},
    "KZ_KAZ": {"name": "Kazakhstan_Baikonur", "lat": 45.9, "lon": 63.3},
    "IN_IND": {"name": "India_SDSC_Sriharikota", "lat": 13.7, "lon": 80.2},
    "CN_CHN": {"name": "China_Taiyuan", "lat": 38.8, "lon": 111.6},
    "NZ_NZL": {"name": "New_Zealand_Mahia", "lat": -39.1, "lon": 177.9},
}


@dataclass(frozen=True)
class ModelParams:
    # Δv 基准与地球自转增益
    dv_base_kmps: float = 14.917
    dv_geo_kmps: float = 4.693
    v_eq_kmps: float = 0.465
    # 火箭方程常数
    g0_mps2: float = 9.81
    # 比冲演化参数（饱和型增长）
    isp_v0_s: float = 380.0
    isp_vmax_s: float = 400.0
    isp_sl0_s: float = 330.0
    isp_slmax_s: float = 350.0
    k_I: float = 0.05
    isp_ref_year: int = 2026
    clamp_isp: bool = True
    # 结构系数（dry/pay）
    k_heavy: float = 1.0
    k_shuttle: float = 0.25
    # 载荷能力与增长
    x_2050_ton: float = 125.0
    g_r: float = 0.05
    # 地表段 Isp 口径选择："SL" 或 "V"
    surface_isp: str = "SL"


def _to_float(value: str, field: str) -> float:
    # 将字符串安全转换为 float，并给出字段级错误信息
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing numeric field: {field}")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {field}: {value}") from exc


def _to_int(value: str, field: str) -> int:
    # 将字符串安全转换为 int（允许 CSV 中的浮点型年份）
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing integer field: {field}")
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(f"Invalid int for {field}: {value}") from exc


def _round_launches(value: float, method: str) -> int:
    # 发射次数的整数化方式：四舍五入/向下取整/向上取整
    if method == "round":
        return int(round(value))
    if method == "floor":
        return int(math.floor(value))
    if method == "ceil":
        return int(math.ceil(value))
    raise ValueError(f"Unknown rounding method: {method}")


def _isp_value(t: int, isp0: float, isp_max: float, k_I: float, ref_year: int, clamp: bool) -> float:
    # 饱和型比冲增长：Isp(t)=Isp_max-(Isp_max-Isp0)*exp(-k_I*(t-ref))
    if clamp and t <= ref_year:
        return isp0
    dt = t - ref_year
    return isp_max - (isp_max - isp0) * math.exp(-k_I * dt)


def _payload_per_launch(t: int, x_2050: float, g_r: float) -> float:
    # 载荷能力按年指数增长（以 2050 为基准）
    return x_2050 * ((1.0 + g_r) ** (t - 2050))


def _dv_rot_kmps(lat_deg: float, v_eq_kmps: float) -> float:
    # 地球自转增益：v_eq * cos(lat)
    return v_eq_kmps * math.cos(math.radians(lat_deg))


def _mass_ratio(dv_kmps: float, isp_s: float, g0: float) -> float:
    # 火箭方程：MR=exp(Δv/(g0*Isp))
    return math.exp((dv_kmps * 1000.0) / (g0 * isp_s))


def _prop_ratio(mr: float, k_struct: float) -> float:
    # 推进剂比：(MR-1)*(1+k)
    return (mr - 1.0) * (1.0 + k_struct)


def _logistic_step(n: float, r: float, k: float) -> float:
    # 离散 Logistic 递推：N_{t+1}=N_t+r*N_t*(1-N_t/K)
    if k <= 0 or r <= 0:
        raise ValueError(f"Invalid logistic parameters: r={r}, K={k}")
    if n > k:
        n = k
    return n + r * n * (1.0 - n / k)


def _read_csv(path: Path) -> Tuple[List[dict], List[str]]:
    # 读取 CSV 到字典列表，同时返回表头
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows, list(rows[0].keys())


def _select_launch_col(headers: List[str], override: Optional[str]) -> str:
    # 自动选择“发射次数”列，允许用户强制覆盖
    if override:
        if override not in headers:
            raise ValueError(f"launch column not found: {override}")
        return override
    for cand in ("launches_eff_int", "launches_eff", "launches"):
        if cand in headers:
            return cand
    raise ValueError("Could not find launch count column (expected launches_eff_int/launches_eff/launches)")


def _group_by_site(rows: List[dict], site_field: str) -> Dict[str, List[dict]]:
    # 按站点分组，并按年份排序
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        site_id = row.get(site_field)
        if not site_id:
            raise ValueError("Missing site_id")
        grouped.setdefault(site_id, []).append(row)
    for site_id in grouped:
        grouped[site_id].sort(key=lambda r: _to_int(r["year"], "year"))
    return grouped


def _build_year_index(rows: List[dict], site_field: str) -> Dict[Tuple[str, int], dict]:
    # 构建 (site_id, year) 的快速索引
    index: Dict[Tuple[str, int], dict] = {}
    for row in rows:
        site_id = row.get(site_field)
        year = _to_int(row["year"], "year")
        index[(site_id, year)] = row
    return index


def _extend_rows(
    grouped: Dict[str, List[dict]],
    launch_col: str,
    end_year: int,
) -> List[dict]:
    # 对每个站点使用 Logistic 规则外推到 end_year
    extended: List[dict] = []
    for site_id, rows in grouped.items():
        last = rows[-1]
        last_year = _to_int(last["year"], "year")
        extended.extend(rows)
        if end_year <= last_year:
            continue
        r = _to_float(last.get("r_post2050", ""), "r_post2050")
        k = _to_float(last.get("K_carrying", ""), "K_carrying")
        current = _to_float(last[launch_col], launch_col)
        for year in range(last_year + 1, end_year + 1):
            current = _logistic_step(current, r, k)
            new_row = dict(last)
            new_row["year"] = str(year)
            new_row[launch_col] = f"{current:.6f}"
            new_row["model"] = "logistic_extrapolated"
            new_row["_extrapolated"] = "1"
            extended.append(new_row)
    return extended


def _compute_output_row(
    row: dict,
    params: ModelParams,
    launch_col: str,
    rounding: str,
    site_field: str,
) -> dict:
    # 将单行 CSV 记录转换为“合并后的完整输出行”
    site_id = row.get(site_field)
    if site_id not in SITE_COORDS:
        raise ValueError(f"Unknown site_id in coords map: {site_id}")
    year = _to_int(row["year"], "year")

    # 站点地理信息
    coord = SITE_COORDS[site_id]
    lat = coord["lat"]
    lon = coord["lon"]

    # 当年比冲
    isp_v = _isp_value(year, params.isp_v0_s, params.isp_vmax_s, params.k_I, params.isp_ref_year, params.clamp_isp)
    isp_sl = _isp_value(year, params.isp_sl0_s, params.isp_slmax_s, params.k_I, params.isp_ref_year, params.clamp_isp)

    # Δv 与自转增益
    dv_rot = _dv_rot_kmps(lat, params.v_eq_kmps)
    dv_surface = params.dv_base_kmps - dv_rot
    dv_geo = params.dv_geo_kmps

    # 地表段 Isp 口径选择
    surface_isp_choice = params.surface_isp.upper()
    if surface_isp_choice not in ("SL", "V"):
        raise ValueError("surface_isp must be SL or V")
    isp_surface = isp_sl if surface_isp_choice == "SL" else isp_v

    # 质量比与推进剂比
    mr_surface = _mass_ratio(dv_surface, isp_surface, params.g0_mps2)
    mr_geo = _mass_ratio(dv_geo, isp_v, params.g0_mps2)

    prop_surface = _prop_ratio(mr_surface, params.k_heavy)
    prop_geo = _prop_ratio(mr_geo, params.k_shuttle)

    # 单次可交付载荷（吨）
    payload_ton = _payload_per_launch(year, params.x_2050_ton, params.g_r)

    # 发射次数（浮点->整数）
    launch_value = _to_float(row.get(launch_col, ""), launch_col)
    launch_int = _round_launches(launch_value, rounding)

    # 合并原始字段与派生字段
    out = dict(row)
    out.update(
        {
            "site_name_full": coord["name"],
            "lat_deg": f"{lat:.4f}",
            "lon_deg": f"{lon:.4f}",
            "launches_eff_model": f"{launch_value:.6f}",
            "launches_eff_int_model": str(launch_int),
            "is_extrapolated": row.get("_extrapolated", "0"),
            "dv_rot_kmps": f"{dv_rot:.6f}",
            "dv_surface_to_moon_kmps": f"{dv_surface:.6f}",
            "dv_geo_to_moon_kmps": f"{dv_geo:.6f}",
            "isp_v_s": f"{isp_v:.6f}",
            "isp_sl_s": f"{isp_sl:.6f}",
            "isp_surface_choice": surface_isp_choice,
            "mr_surface_to_moon": f"{mr_surface:.8f}",
            "mr_geo_to_moon": f"{mr_geo:.8f}",
            "prop_ratio_surface": f"{prop_surface:.8f}",
            "prop_ratio_geo": f"{prop_geo:.8f}",
            "prop_ratio_mode1": f"{(2.0 * prop_geo):.8f}",
            "prop_ratio_mode2": f"{(prop_surface + prop_geo):.8f}",
            "prop_ratio_mode3": f"{(2.0 * prop_surface):.8f}",
            "payload_per_launch_ton": f"{payload_ton:.6f}",
            "k_heavy": f"{params.k_heavy:.6f}",
            "k_shuttle": f"{params.k_shuttle:.6f}",
            "x_2050_ton": f"{params.x_2050_ton:.6f}",
            "g_r": f"{params.g_r:.6f}",
            "k_I": f"{params.k_I:.6f}",
            "isp_v0_s": f"{params.isp_v0_s:.6f}",
            "isp_vmax_s": f"{params.isp_vmax_s:.6f}",
            "isp_sl0_s": f"{params.isp_sl0_s:.6f}",
            "isp_slmax_s": f"{params.isp_slmax_s:.6f}",
            "g0_mps2": f"{params.g0_mps2:.6f}",
            "dv_base_kmps": f"{params.dv_base_kmps:.6f}",
            "v_eq_kmps": f"{params.v_eq_kmps:.6f}",
        }
    )
    return out


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    # 写出 CSV（保持字段顺序）
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_params_from_json(path: Path, defaults: ModelParams) -> ModelParams:
    # 从 JSON 配置覆盖默认参数（禁止未知字段）
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    allowed = set(asdict(defaults).keys())
    unknown = set(data.keys()) - allowed
    if unknown:
        raise ValueError(f"Unknown parameter keys in config: {sorted(unknown)}")
    return replace(defaults, **data)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Build combined parameter table for MILP/MCMC.")
    parser.add_argument("--input", default="launch_capacity_adjusted_2025_2100.csv", help="Input CSV file")
    parser.add_argument("--output", default="combined_parameters.csv", help="Output CSV file")
    parser.add_argument("--start-year", type=int, default=None, help="Start year (default: CSV min year)")
    parser.add_argument("--end-year", type=int, default=None, help="End year (default: CSV max year + 50)")
    parser.add_argument("--launch-col", default=None, help="Override launch count column name")
    parser.add_argument("--rounding", choices=["round", "floor", "ceil"], default="round", help="Rounding for integer launches")
    parser.add_argument("--config", default=None, help="JSON file with parameter overrides")
    parser.add_argument("--write-meta", default=None, help="Write run metadata to JSON")

    # 参数覆盖项（全是可选）
    parser.add_argument("--surface-isp", choices=["SL", "V"], default=None, help="Surface segment Isp choice")
    parser.add_argument("--k-heavy", type=float, default=None)
    parser.add_argument("--k-shuttle", type=float, default=None)
    parser.add_argument("--x-2050", type=float, default=None)
    parser.add_argument("--g-r", type=float, default=None)
    parser.add_argument("--isp-v0", type=float, default=None)
    parser.add_argument("--isp-vmax", type=float, default=None)
    parser.add_argument("--isp-sl0", type=float, default=None)
    parser.add_argument("--isp-slmax", type=float, default=None)
    parser.add_argument("--k-I", type=float, default=None)
    parser.add_argument("--dv-base", type=float, default=None)
    parser.add_argument("--dv-geo", type=float, default=None)
    parser.add_argument("--v-eq", type=float, default=None)
    parser.add_argument("--clamp-isp", action="store_true", help="Clamp Isp at ref year for t<=ref_year")
    parser.add_argument("--no-clamp-isp", action="store_true", help="Disable Isp clamping")
    return parser.parse_args(argv)


def _apply_overrides(params: ModelParams, args: argparse.Namespace) -> ModelParams:
    # 将 CLI 覆盖项应用到参数对象
    updates = {}
    if args.surface_isp:
        updates["surface_isp"] = args.surface_isp
    if args.k_heavy is not None:
        updates["k_heavy"] = args.k_heavy
    if args.k_shuttle is not None:
        updates["k_shuttle"] = args.k_shuttle
    if args.x_2050 is not None:
        updates["x_2050_ton"] = args.x_2050
    if args.g_r is not None:
        updates["g_r"] = args.g_r
    if args.isp_v0 is not None:
        updates["isp_v0_s"] = args.isp_v0
    if args.isp_vmax is not None:
        updates["isp_vmax_s"] = args.isp_vmax
    if args.isp_sl0 is not None:
        updates["isp_sl0_s"] = args.isp_sl0
    if args.isp_slmax is not None:
        updates["isp_slmax_s"] = args.isp_slmax
    if args.k_I is not None:
        updates["k_I"] = args.k_I
    if args.dv_base is not None:
        updates["dv_base_kmps"] = args.dv_base
    if args.dv_geo is not None:
        updates["dv_geo_kmps"] = args.dv_geo
    if args.v_eq is not None:
        updates["v_eq_kmps"] = args.v_eq
    if args.clamp_isp and args.no_clamp_isp:
        raise ValueError("Cannot set both --clamp-isp and --no-clamp-isp")
    if args.clamp_isp:
        updates["clamp_isp"] = True
    if args.no_clamp_isp:
        updates["clamp_isp"] = False
    return replace(params, **updates)


def main(argv: Optional[List[str]] = None) -> int:
    # 主流程：读入 -> 外推 -> 计算派生列 -> 写出
    args = _parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    # 读取原始 CSV
    rows, headers = _read_csv(input_path)
    launch_col = _select_launch_col(headers, args.launch_col)
    site_field = "site_id"
    year_min = min(_to_int(r["year"], "year") for r in rows)
    year_max = max(_to_int(r["year"], "year") for r in rows)

    # 时间范围校验
    start_year = args.start_year if args.start_year is not None else year_min
    end_year = args.end_year if args.end_year is not None else (year_max + 50)
    if start_year < year_min:
        raise ValueError(f"start_year {start_year} < CSV min year {year_min}")
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")

    # 载入参数（默认 + JSON + CLI）
    params = ModelParams()
    if args.config:
        params = _load_params_from_json(Path(args.config), params)
    params = _apply_overrides(params, args)

    # 分站点外推，构建索引
    grouped = _group_by_site(rows, site_field)
    extended_rows = _extend_rows(grouped, launch_col, end_year)
    index = _build_year_index(extended_rows, site_field)

    # 逐年逐站生成合并输出行
    output_rows: List[dict] = []
    for site_id in sorted(grouped.keys()):
        for year in range(start_year, end_year + 1):
            row = index.get((site_id, year))
            if row is None:
                raise ValueError(f"Missing row for {site_id} in year {year}")
            output_rows.append(_compute_output_row(row, params, launch_col, args.rounding, site_field))

    # 追加派生字段到输出表头
    extra_fields = [
        "site_name_full",
        "lat_deg",
        "lon_deg",
        "launches_eff_model",
        "launches_eff_int_model",
        "is_extrapolated",
        "dv_rot_kmps",
        "dv_surface_to_moon_kmps",
        "dv_geo_to_moon_kmps",
        "isp_v_s",
        "isp_sl_s",
        "isp_surface_choice",
        "mr_surface_to_moon",
        "mr_geo_to_moon",
        "prop_ratio_surface",
        "prop_ratio_geo",
        "prop_ratio_mode1",
        "prop_ratio_mode2",
        "prop_ratio_mode3",
        "payload_per_launch_ton",
        "k_heavy",
        "k_shuttle",
        "x_2050_ton",
        "g_r",
        "k_I",
        "isp_v0_s",
        "isp_vmax_s",
        "isp_sl0_s",
        "isp_slmax_s",
        "g0_mps2",
        "dv_base_kmps",
        "v_eq_kmps",
    ]
    fieldnames = headers + [f for f in extra_fields if f not in headers]
    _write_csv(output_path, output_rows, fieldnames)

    # 可选写出元信息
    if args.write_meta:
        meta = {
            "input": str(input_path),
            "output": str(output_path),
            "start_year": start_year,
            "end_year": end_year,
            "launch_col": launch_col,
            "rounding": args.rounding,
            "params": asdict(params),
        }
        with Path(args.write_meta).open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print(f"Wrote {len(output_rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
