from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


PRIMARY_COLUMNS = [
    "equation",
    "run",
    "reward",
    "controller",
    "pre_step",
    "lbfgs_start_step",
    "post_step",
    "rank_pre_relative_l2",
    "rank_post_relative_l2",
    "rank_post_equal_loss",
    "rank_shift_relative_l2",
    "pre_relative_l2",
    "post_relative_l2",
    "lbfgs_relative_l2_improvement_pct",
    "pre_l2_vs_fixed_pct",
    "post_l2_vs_fixed_pct",
    "pre_equal_loss",
    "post_equal_loss",
    "lbfgs_equal_loss_improvement_pct",
    "pre_weighted_loss",
    "post_weighted_loss",
    "lbfgs_weighted_loss_improvement_pct",
    "final_agent_reward",
    "final_agent_sigma",
    "final_weights",
    "lbfgs_steps",
    "lbfgs_closure_calls_total",
    "history_path",
]

MARKDOWN_COLUMNS = [
    "rank_post_relative_l2",
    "reward",
    "pre_relative_l2",
    "post_relative_l2",
    "lbfgs_relative_l2_improvement_pct",
    "post_l2_vs_fixed_pct",
    "pre_equal_loss",
    "post_equal_loss",
    "lbfgs_equal_loss_improvement_pct",
    "rank_post_equal_loss",
    "final_weights",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect pre-LBFGS vs post-LBFGS metrics from reward sweep history.json files."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help=(
            "Sweep folder. Can be an equation folder like .../burgers or a parent "
            "folder that contains equation folders."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        help=(
            "Output directory. Default: <equation>/reward_sweep/lbfgs_pre_post "
            "for one equation, or <root>/lbfgs_pre_post for multiple equations."
        ),
    )
    parser.add_argument(
        "--pre-index",
        type=int,
        help=(
            "Zero-based history index to use as the pre-LBFGS point. By default "
            "the script uses the last row before optimizer_phase becomes 'lbfgs'."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="How many rows to print to stdout after writing files.",
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


def count_lbfgs_steps(history: dict[str, Any]) -> int:
    phases = history.get("optimizer_phase")
    if not isinstance(phases, list):
        phases = history.get("phase")
    if not isinstance(phases, list):
        return 0
    return sum(1 for phase in phases if str(phase).lower() == "lbfgs")


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


def improvement_pct(pre: float | None, post: float | None) -> float | None:
    if pre is None or post is None or pre == 0.0:
        return None
    return 100.0 * (pre - post) / pre


def relative_gap_pct(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or reference == 0.0:
        return None
    return 100.0 * (value / reference - 1.0)


def history_length(history: dict[str, Any]) -> int:
    lengths = [
        len(value)
        for value in history.values()
        if isinstance(value, list)
    ]
    return max(lengths, default=0)


def choose_pre_index(history: dict[str, Any], override: int | None) -> int | None:
    if override is not None:
        length = history_length(history)
        if length == 0:
            return None
        return max(0, min(override, length - 1))
    first_lbfgs = first_lbfgs_index(history)
    if first_lbfgs is not None:
        return max(0, first_lbfgs - 1)
    fallback_keys = ("relative_l2", "equal_weight_total", "weighted_total")
    for key in fallback_keys:
        idx = last_finite_index(history.get(key))
        if idx is not None:
            return idx
    return None


def choose_post_index(history: dict[str, Any]) -> int | None:
    fallback_keys = ("relative_l2", "equal_weight_total", "weighted_total")
    for key in fallback_keys:
        idx = last_finite_index(history.get(key))
        if idx is not None:
            return idx
    return None


def final_list_value(values: Any) -> Any:
    if not isinstance(values, list):
        return None
    for value in reversed(values):
        if value is not None:
            return value
    return None


def reward_name(method_name: str, history: dict[str, Any]) -> str:
    if method_name == "fixed":
        return "fixed"
    if method_name.startswith("agent_"):
        return method_name[len("agent_") :]
    controller = history.get("controller")
    if isinstance(controller, str) and "_" in controller:
        return controller.split("_", 1)[1]
    return method_name


def sweep_roots(root: Path) -> list[Path]:
    root = root.resolve()
    if (root / "history.json").is_file():
        return [root.parent]
    if root.name == "reward_sweep" and list(root.parent.glob("*/history.json")):
        return [root.parent]
    direct_histories = sorted(root.glob("*/history.json"))
    if direct_histories:
        return [root]
    candidates = sorted({path.parent.parent for path in root.glob("*/*/history.json")})
    return candidates


def collect_row(equation_root: Path, history_path: Path, pre_override: int | None) -> dict[str, Any]:
    history = load_json(history_path)
    run_name = history_path.parent.name
    pre_idx = choose_pre_index(history, pre_override)
    post_idx = choose_post_index(history)
    first_lbfgs = first_lbfgs_index(history)

    row: dict[str, Any] = {
        "equation": equation_root.name,
        "run": run_name,
        "reward": reward_name(run_name, history),
        "controller": history.get("controller"),
        "pre_step": None if pre_idx is None else pre_idx + 1,
        "lbfgs_start_step": None if first_lbfgs is None else first_lbfgs + 1,
        "post_step": None if post_idx is None else post_idx + 1,
        "pre_relative_l2": finite_value_at_or_before(history.get("relative_l2"), pre_idx),
        "post_relative_l2": last_finite_value(history.get("relative_l2")),
        "pre_equal_loss": finite_value_at_or_before(history.get("equal_weight_total"), pre_idx),
        "post_equal_loss": last_finite_value(history.get("equal_weight_total")),
        "pre_weighted_loss": finite_value_at_or_before(history.get("weighted_total"), pre_idx),
        "post_weighted_loss": last_finite_value(history.get("weighted_total")),
        "final_agent_reward": last_finite_value(history.get("agent_reward")),
        "final_agent_sigma": last_finite_value(history.get("agent_sigma")),
        "final_weights": final_list_value(history.get("weights")),
        "lbfgs_steps": count_lbfgs_steps(history),
        "lbfgs_closure_calls_total": closure_calls_total(history),
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
    row["lbfgs_weighted_loss_improvement_pct"] = improvement_pct(
        row["pre_weighted_loss"],
        row["post_weighted_loss"],
    )

    components = history.get("components")
    component_names = history.get("component_names")
    if isinstance(components, dict) and isinstance(component_names, list):
        for name in component_names:
            if not isinstance(name, str):
                continue
            values = components.get(name)
            row[f"pre_{name}_loss"] = finite_value_at_or_before(values, pre_idx)
            row[f"post_{name}_loss"] = last_finite_value(values)
            row[f"lbfgs_{name}_loss_improvement_pct"] = improvement_pct(
                row[f"pre_{name}_loss"],
                row[f"post_{name}_loss"],
            )
    return row


def add_ranks(rows: list[dict[str, Any]]) -> None:
    rank_specs = [
        ("pre_relative_l2", "rank_pre_relative_l2"),
        ("post_relative_l2", "rank_post_relative_l2"),
        ("post_equal_loss", "rank_post_equal_loss"),
    ]
    for metric, rank_key in rank_specs:
        ranked = sorted(
            [row for row in rows if row.get(metric) is not None],
            key=lambda row: (row[metric], row["reward"]),
        )
        for rank, row in enumerate(ranked, start=1):
            row[rank_key] = rank
    for row in rows:
        pre_rank = row.get("rank_pre_relative_l2")
        post_rank = row.get("rank_post_relative_l2")
        row["rank_shift_relative_l2"] = (
            None if pre_rank is None or post_rank is None else pre_rank - post_rank
        )

    fixed = next((row for row in rows if row["reward"] == "fixed"), None)
    if fixed is None:
        return
    fixed_pre_l2 = fixed.get("pre_relative_l2")
    fixed_post_l2 = fixed.get("post_relative_l2")
    for row in rows:
        row["pre_l2_vs_fixed_pct"] = relative_gap_pct(row.get("pre_relative_l2"), fixed_pre_l2)
        row["post_l2_vs_fixed_pct"] = relative_gap_pct(row.get("post_relative_l2"), fixed_post_l2)


def all_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns = list(PRIMARY_COLUMNS)
    dynamic_columns = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in columns
        }
    )
    return columns + dynamic_columns


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
    with path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)


def markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if abs(value) < 1e-3 or abs(value) >= 1e4:
            return f"{value:.4e}"
        return f"{value:.6f}"
    if isinstance(value, list):
        compact = []
        for item in value:
            numeric = as_float(item)
            compact.append(f"{numeric:.3f}" if numeric is not None else str(item))
        return "[" + ", ".join(compact) + "]"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(markdown_value(row.get(column)) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    sorted_rows = sort_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_table(sorted_rows, MARKDOWN_COLUMNS), encoding="utf-8")


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            row.get("post_relative_l2") is None,
            row.get("post_relative_l2", math.inf),
            row["reward"],
        ),
    )


def collect_equation(equation_root: Path, pre_override: int | None) -> list[dict[str, Any]]:
    histories = sorted(equation_root.glob("*/history.json"))
    rows = [
        collect_row(equation_root, history_path, pre_override)
        for history_path in histories
        if history_path.parent.name != "reward_sweep"
    ]
    add_ranks(rows)
    return sort_rows(rows)


def default_out(root: Path, equations: list[Path]) -> Path:
    if len(equations) == 1:
        return equations[0] / "reward_sweep" / "lbfgs_pre_post"
    return root.resolve() / "lbfgs_pre_post"


def print_preview(rows: list[dict[str, Any]], top: int) -> None:
    if top <= 0:
        return
    preview = rows[:top]
    print(markdown_table(preview, MARKDOWN_COLUMNS), end="")


def main() -> None:
    args = parse_args()
    equations = sweep_roots(args.root)
    if not equations:
        raise SystemExit(f"No history.json files found under {args.root}")

    out_dir = args.out.resolve() if args.out is not None else default_out(args.root, equations)
    all_rows: list[dict[str, Any]] = []

    for equation in equations:
        rows = collect_equation(equation, args.pre_index)
        if not rows:
            continue
        all_rows.extend(rows)
        stem = equation.name
        columns = all_columns(rows)
        write_csv(out_dir / f"{stem}_lbfgs_pre_post.csv", rows, columns)
        write_json(out_dir / f"{stem}_lbfgs_pre_post.json", rows)
        write_markdown(out_dir / f"{stem}_lbfgs_pre_post.md", rows)

    if not all_rows:
        raise SystemExit(f"No usable history.json files found under {args.root}")

    if len(equations) > 1:
        add_ranks(all_rows)
        all_rows = sort_rows(all_rows)
        columns = all_columns(all_rows)
        write_csv(out_dir / "all_lbfgs_pre_post.csv", all_rows, columns)
        write_json(out_dir / "all_lbfgs_pre_post.json", all_rows)
        write_markdown(out_dir / "all_lbfgs_pre_post.md", all_rows)

    print(f"Wrote analysis to: {out_dir}")
    print_preview(all_rows, args.top)


if __name__ == "__main__":
    main()
