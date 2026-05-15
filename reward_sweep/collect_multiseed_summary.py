from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DETAIL_COLUMNS = [
    "sweep",
    "seed",
    "equation",
    "reward",
    "controller",
    "policy_hidden_dim",
    "agent_update_interval",
    "adam_steps",
    "lbfgs_steps",
    "rank_pre_relative_l2",
    "rank_post_relative_l2",
    "rank_post_equal_loss",
    "pre_relative_l2",
    "post_relative_l2",
    "pre_l2_vs_fixed_pct",
    "post_l2_vs_fixed_pct",
    "pre_equal_loss",
    "post_equal_loss",
    "pre_equal_vs_fixed_pct",
    "post_equal_vs_fixed_pct",
    "lbfgs_relative_l2_improvement_pct",
    "lbfgs_equal_loss_improvement_pct",
    "final_weights",
    "lbfgs_closure_calls_total",
    "history_path",
]

SUMMARY_COLUMNS = [
    "reward",
    "n",
    "seeds",
    "mean_pre_relative_l2",
    "median_pre_relative_l2",
    "std_pre_relative_l2",
    "mean_post_relative_l2",
    "median_post_relative_l2",
    "std_post_relative_l2",
    "mean_pre_l2_vs_fixed_pct",
    "median_pre_l2_vs_fixed_pct",
    "win_rate_pre_l2_vs_fixed",
    "mean_post_l2_vs_fixed_pct",
    "median_post_l2_vs_fixed_pct",
    "win_rate_post_l2_vs_fixed",
    "mean_rank_post_relative_l2",
    "median_rank_post_relative_l2",
    "std_rank_post_relative_l2",
    "best_rank_post_relative_l2",
    "worst_rank_post_relative_l2",
    "mean_rank_pre_relative_l2",
    "median_rank_pre_relative_l2",
    "std_rank_pre_relative_l2",
    "best_rank_pre_relative_l2",
    "worst_rank_pre_relative_l2",
    "mean_rank_post_equal_loss",
    "median_rank_post_equal_loss",
    "std_rank_post_equal_loss",
    "best_rank_post_equal_loss",
    "worst_rank_post_equal_loss",
    "mean_lbfgs_relative_l2_improvement_pct",
    "median_lbfgs_relative_l2_improvement_pct",
    "mean_pre_equal_loss",
    "median_pre_equal_loss",
    "mean_post_equal_loss",
    "median_post_equal_loss",
    "mean_post_equal_vs_fixed_pct",
    "median_post_equal_vs_fixed_pct",
    "win_rate_post_equal_vs_fixed",
    "mean_lbfgs_closure_calls_total",
    "median_lbfgs_closure_calls_total",
    "mean_final_w_pde",
    "mean_final_w_ic",
    "mean_final_w_bc",
]

MARKDOWN_COLUMNS = [
    "reward",
    "n",
    "median_post_relative_l2",
    "mean_post_l2_vs_fixed_pct",
    "win_rate_post_l2_vs_fixed",
    "median_rank_post_relative_l2",
    "worst_rank_post_relative_l2",
    "median_pre_relative_l2",
    "mean_pre_l2_vs_fixed_pct",
    "win_rate_pre_l2_vs_fixed",
    "mean_final_w_pde",
    "mean_final_w_ic",
    "mean_final_w_bc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate reward sweep results across multiple seeds/runs."
    )
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help=(
            "Sweep folders or parent folders. A sweep folder contains reward_sweep_config.json "
            "and equation subfolders."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/multiseed_summary"),
        help="Output directory for CSV/JSON/Markdown tables.",
    )
    parser.add_argument(
        "--equation",
        help="Equation folder to aggregate. Default: aggregate all equations found.",
    )
    parser.add_argument(
        "--rewards",
        help="Optional comma-separated reward subset. Include fixed explicitly if needed.",
    )
    parser.add_argument(
        "--pre-index",
        type=int,
        help="Zero-based history index for pre-LBFGS point. Default: last row before L-BFGS.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="How many summary rows to print.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def finite_value_at_or_before(values: Any, index: int | None) -> float | None:
    if not isinstance(values, list) or index is None:
        return None
    index = min(index, len(values) - 1)
    for idx in range(index, -1, -1):
        value = as_float(values[idx])
        if value is not None:
            return value
    return None


def last_finite_value(values: Any) -> float | None:
    if not isinstance(values, list):
        return None
    return finite_value_at_or_before(values, len(values) - 1)


def last_finite_index(values: Any) -> int | None:
    if not isinstance(values, list):
        return None
    for idx in range(len(values) - 1, -1, -1):
        if as_float(values[idx]) is not None:
            return idx
    return None


def first_lbfgs_index(history: dict[str, Any]) -> int | None:
    phases = history.get("optimizer_phase")
    if not isinstance(phases, list):
        phases = history.get("phase")
    if not isinstance(phases, list):
        return None
    for idx, phase in enumerate(phases):
        if str(phase).lower() == "lbfgs":
            return idx
    return None


def choose_pre_index(history: dict[str, Any], override: int | None) -> int | None:
    if override is not None:
        return max(0, min(override, max(history_length(history) - 1, 0)))
    first_lbfgs = first_lbfgs_index(history)
    if first_lbfgs is not None:
        return max(0, first_lbfgs - 1)
    for key in ("relative_l2", "equal_weight_total", "weighted_total"):
        idx = last_finite_index(history.get(key))
        if idx is not None:
            return idx
    return None


def history_length(history: dict[str, Any]) -> int:
    return max((len(value) for value in history.values() if isinstance(value, list)), default=0)


def reward_name(method_name: str, history: dict[str, Any]) -> str:
    if method_name == "fixed":
        return "fixed"
    if method_name.startswith("agent_"):
        return method_name[len("agent_") :]
    controller = history.get("controller")
    if isinstance(controller, str) and "_" in controller:
        return controller.split("_", 1)[1]
    return method_name


def relative_gap_pct(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or reference == 0.0:
        return None
    return 100.0 * (value / reference - 1.0)


def improvement_pct(pre: float | None, post: float | None) -> float | None:
    if pre is None or post is None or pre == 0.0:
        return None
    return 100.0 * (pre - post) / pre


def closure_calls_total(history: dict[str, Any]) -> int | None:
    calls = history.get("lbfgs_closure_calls")
    if not isinstance(calls, list):
        return None
    phases = history.get("optimizer_phase")
    if not isinstance(phases, list):
        phases = history.get("phase")
    total = 0
    found = False
    for idx, value in enumerate(calls):
        if isinstance(phases, list) and idx < len(phases) and str(phases[idx]).lower() != "lbfgs":
            continue
        numeric = as_float(value)
        if numeric is None:
            continue
        total += int(numeric)
        found = True
    return total if found else None


def count_lbfgs_steps(history: dict[str, Any]) -> int:
    phases = history.get("optimizer_phase")
    if not isinstance(phases, list):
        phases = history.get("phase")
    if not isinstance(phases, list):
        return 0
    return sum(1 for phase in phases if str(phase).lower() == "lbfgs")


def final_list_value(values: Any) -> Any:
    if not isinstance(values, list):
        return None
    for value in reversed(values):
        if value is not None:
            return value
    return None


def find_sweep_roots(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        candidates: list[Path]
        if (root / "reward_sweep_config.json").is_file():
            candidates = [root]
        else:
            candidates = sorted(path.parent for path in root.glob("*/reward_sweep_config.json"))
            candidates.extend(sorted(path.parent for path in root.glob("*/*/reward_sweep_config.json")))
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                found.append(candidate)
    return found


def equation_roots(sweep_root: Path, equation: str | None) -> list[Path]:
    if equation is not None:
        candidate = sweep_root / equation
        return [candidate] if candidate.is_dir() else []
    return sorted(
        path
        for path in sweep_root.iterdir()
        if path.is_dir() and list(path.glob("*/history.json"))
    )


def run_config(sweep_root: Path) -> dict[str, Any]:
    path = sweep_root / "reward_sweep_config.json"
    if not path.is_file():
        return {}
    return load_json(path)


def collect_equation_rows(
    *,
    sweep_root: Path,
    equation_root: Path,
    cfg: dict[str, Any],
    pre_index: int | None,
) -> list[dict[str, Any]]:
    histories = sorted(
        path
        for path in equation_root.glob("*/history.json")
        if path.parent.name != "reward_sweep"
    )
    rows: list[dict[str, Any]] = []
    for history_path in histories:
        history = load_json(history_path)
        pre_idx = choose_pre_index(history, pre_index)
        reward = reward_name(history_path.parent.name, history)
        training = cfg.get("training", {})
        agent_cfg = cfg.get("controller_params", {}).get("tiny_loss_weight", {})
        row = {
            "sweep": sweep_root.name,
            "seed": cfg.get("seed"),
            "equation": equation_root.name,
            "reward": reward,
            "controller": history.get("controller"),
            "policy_hidden_dim": agent_cfg.get("policy_hidden_dim"),
            "agent_update_interval": training.get("agent_update_interval"),
            "adam_steps": training.get("adam_steps"),
            "lbfgs_steps": count_lbfgs_steps(history),
            "pre_relative_l2": finite_value_at_or_before(history.get("relative_l2"), pre_idx),
            "post_relative_l2": last_finite_value(history.get("relative_l2")),
            "pre_equal_loss": finite_value_at_or_before(history.get("equal_weight_total"), pre_idx),
            "post_equal_loss": last_finite_value(history.get("equal_weight_total")),
            "pre_weighted_loss": finite_value_at_or_before(history.get("weighted_total"), pre_idx),
            "post_weighted_loss": last_finite_value(history.get("weighted_total")),
            "lbfgs_closure_calls_total": closure_calls_total(history),
            "final_weights": final_list_value(history.get("weights")),
            "history_path": str(history_path),
        }
        row["lbfgs_relative_l2_improvement_pct"] = improvement_pct(
            row["pre_relative_l2"],
            row["post_relative_l2"],
        )
        row["lbfgs_equal_loss_improvement_pct"] = improvement_pct(
            row["pre_equal_loss"],
            row["post_equal_loss"],
        )
        rows.append(row)

    fixed = next((row for row in rows if row["reward"] == "fixed"), None)
    if fixed is not None:
        for row in rows:
            row["pre_l2_vs_fixed_pct"] = relative_gap_pct(
                row.get("pre_relative_l2"),
                fixed.get("pre_relative_l2"),
            )
            row["post_l2_vs_fixed_pct"] = relative_gap_pct(
                row.get("post_relative_l2"),
                fixed.get("post_relative_l2"),
            )
            row["pre_equal_vs_fixed_pct"] = relative_gap_pct(
                row.get("pre_equal_loss"),
                fixed.get("pre_equal_loss"),
            )
            row["post_equal_vs_fixed_pct"] = relative_gap_pct(
                row.get("post_equal_loss"),
                fixed.get("post_equal_loss"),
            )
    return rows


def add_run_ranks(rows: list[dict[str, Any]]) -> None:
    rank_specs = [
        ("pre_relative_l2", "rank_pre_relative_l2"),
        ("post_relative_l2", "rank_post_relative_l2"),
        ("post_equal_loss", "rank_post_equal_loss"),
    ]
    for metric, rank_key in rank_specs:
        ranked = sorted(
            [row for row in rows if as_float(row.get(metric)) is not None],
            key=lambda row: (float(row[metric]), row["reward"]),
        )
        for rank, row in enumerate(ranked, start=1):
            row[rank_key] = rank


def filtered_rows(rows: list[dict[str, Any]], rewards: str | None) -> list[dict[str, Any]]:
    if not rewards:
        return rows
    selected = {item.strip() for item in rewards.split(",") if item.strip()}
    return [row for row in rows if row["reward"] in selected]


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = as_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def median(values: list[float]) -> float | None:
    return None if not values else float(np_median(values))


def std(values: list[float]) -> float | None:
    if len(values) <= 1:
        return 0.0 if values else None
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def best(values: list[float]) -> float | None:
    return None if not values else min(values)


def worst(values: list[float]) -> float | None:
    return None if not values else max(values)


def np_median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def win_rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = finite_values(rows, key)
    if not values:
        return None
    return sum(1 for value in values if value < 0.0) / len(values)


def mean_weight(rows: list[dict[str, Any]], index: int) -> float | None:
    values = []
    for row in rows:
        weights = row.get("final_weights")
        if isinstance(weights, list) and len(weights) > index:
            value = as_float(weights[index])
            if value is not None:
                values.append(value)
    return mean(values)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["reward"]].append(row)

    summaries: list[dict[str, Any]] = []
    for reward, group in grouped.items():
        seeds = sorted({str(row.get("seed")) for row in group})
        summary = {
            "reward": reward,
            "n": len(group),
            "seeds": ",".join(seeds),
        }
        for key in (
            "pre_relative_l2",
            "post_relative_l2",
            "pre_l2_vs_fixed_pct",
            "post_l2_vs_fixed_pct",
            "rank_pre_relative_l2",
            "rank_post_relative_l2",
            "rank_post_equal_loss",
            "lbfgs_relative_l2_improvement_pct",
            "pre_equal_loss",
            "post_equal_loss",
            "post_equal_vs_fixed_pct",
            "lbfgs_closure_calls_total",
        ):
            values = finite_values(group, key)
            summary[f"mean_{key}"] = mean(values)
            summary[f"median_{key}"] = median(values)
            if key in {"pre_relative_l2", "post_relative_l2"}:
                summary[f"std_{key}"] = std(values)
            if key in {
                "rank_pre_relative_l2",
                "rank_post_relative_l2",
                "rank_post_equal_loss",
            }:
                summary[f"std_{key}"] = std(values)
                summary[f"best_{key}"] = best(values)
                summary[f"worst_{key}"] = worst(values)
        summary["win_rate_pre_l2_vs_fixed"] = win_rate(group, "pre_l2_vs_fixed_pct")
        summary["win_rate_post_l2_vs_fixed"] = win_rate(group, "post_l2_vs_fixed_pct")
        summary["win_rate_post_equal_vs_fixed"] = win_rate(group, "post_equal_vs_fixed_pct")
        summary["mean_final_w_pde"] = mean_weight(group, 0)
        summary["mean_final_w_ic"] = mean_weight(group, 1)
        summary["mean_final_w_bc"] = mean_weight(group, 2)
        summaries.append(summary)

    return sorted(
        summaries,
        key=lambda row: (
            row.get("median_post_relative_l2") is None,
            row.get("median_post_relative_l2", math.inf),
            row["reward"],
        ),
    )


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: csv_value(row.get(column)) for column in columns})


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if abs(value) < 1e-3 or abs(value) >= 1e4:
            return f"{value:.4e}"
        return f"{value:.6f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(markdown_value(row.get(column)) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_table(rows, columns), encoding="utf-8")


def main() -> None:
    args = parse_args()
    sweep_roots = find_sweep_roots(args.roots)
    if not sweep_roots:
        raise SystemExit("No reward_sweep_config.json files found")

    all_rows: list[dict[str, Any]] = []
    for sweep_root in sweep_roots:
        cfg = run_config(sweep_root)
        for equation_root in equation_roots(sweep_root, args.equation):
            equation_rows = collect_equation_rows(
                sweep_root=sweep_root,
                equation_root=equation_root,
                cfg=cfg,
                pre_index=args.pre_index,
            )
            add_run_ranks(equation_rows)
            all_rows.extend(equation_rows)

    all_rows = filtered_rows(all_rows, args.rewards)
    if not all_rows:
        raise SystemExit("No rows collected")

    summaries = summarize(all_rows)
    out_dir = args.out
    write_csv(out_dir / "multiseed_details.csv", all_rows, DETAIL_COLUMNS)
    write_json(out_dir / "multiseed_details.json", all_rows)
    write_csv(out_dir / "multiseed_summary.csv", summaries, SUMMARY_COLUMNS)
    write_json(out_dir / "multiseed_summary.json", summaries)
    write_markdown(out_dir / "multiseed_summary.md", summaries, MARKDOWN_COLUMNS)

    print(f"Sweeps: {len(sweep_roots)}")
    print(f"Rows: {len(all_rows)}")
    print(f"Output: {out_dir}")
    print(markdown_table(summaries[: args.top], MARKDOWN_COLUMNS), end="")


if __name__ == "__main__":
    main()
