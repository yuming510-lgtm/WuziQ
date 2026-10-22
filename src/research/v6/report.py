"""V6 route, pairwise, compute-cost, and color analysis."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import BUDGET_LABELS, FAMILIES
from .schedule import build_schedule
from .state import derive_state


def model_catalog() -> list[str]:
    """Return all 22 ABCD and 35 E model-budget identities."""
    return ["A@natural"] + [
        f"{family}@{budget}"
        for family in FAMILIES[1:]
        for budget in BUDGET_LABELS
    ]


def _model_id(family: str, versions: dict[str, str]) -> str:
    return f"{family}@{versions[family]}"


def _wilson(score: float, games: int, z: float = 1.959963984540054) -> list[float] | None:
    if games <= 0:
        return None
    p = score / games
    z2 = z * z
    den = 1.0 + z2 / games
    center = (p + z2 / (2.0 * games)) / den
    half = z * ((p * (1.0 - p) / games + z2 / (4.0 * games * games)) ** 0.5) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize_pair(group: dict[str, Any], games: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a complete two-game paired color swap; draws count half."""
    if len(games) != 2:
        raise ValueError("V6 pair summaries require exactly two completed games")
    family_a, family_b = map(str, group["pair"])
    model_a = _model_id(family_a, group["model_versions"])
    model_b = _model_id(family_b, group["model_versions"])
    wins = {model_a: 0, model_b: 0}
    losses = {model_a: 0, model_b: 0}
    draws = 0
    points = {model_a: 0.0, model_b: 0.0}
    color_points = {
        model_a: {"black": 0.0, "white": 0.0},
        model_b: {"black": 0.0, "white": 0.0},
    }
    color_games = {
        model_a: {"black": 0, "white": 0},
        model_b: {"black": 0, "white": 0},
    }
    for game in games:
        winner = game.get("winner_model")
        if winner is None:
            draws += 1
            game_points = {model_a: 0.5, model_b: 0.5}
        elif winner == family_a:
            wins[model_a] += 1
            losses[model_b] += 1
            game_points = {model_a: 1.0, model_b: 0.0}
        elif winner == family_b:
            wins[model_b] += 1
            losses[model_a] += 1
            game_points = {model_a: 0.0, model_b: 1.0}
        else:
            raise ValueError(f"Unexpected winner family {winner!r} for {group['group_id']}")
        points[model_a] += game_points[model_a]
        points[model_b] += game_points[model_b]
        black = str(game["black_model"])
        white = str(game["white_model"])
        for model, family, color in ((model_a, family_a, "black"), (model_b, family_b, "black")):
            if black == family:
                color_points[model][color] += game_points[model]
                color_games[model][color] += 1
        for model, family in ((model_a, family_a), (model_b, family_b)):
            if white == family:
                color_points[model]["white"] += game_points[model]
                color_games[model]["white"] += 1
    score_rate = {model: points[model] / 2.0 for model in (model_a, model_b)}
    color = {
        model: {
            side: {
                "points": color_points[model][side],
                "games": color_games[model][side],
                "score_rate": (
                    color_points[model][side] / color_games[model][side]
                    if color_games[model][side] else None
                ),
            }
            for side in ("black", "white")
        }
        for model in (model_a, model_b)
    }
    return {
        "group_id": str(group["group_id"]),
        "model_a": model_a,
        "model_b": model_b,
        "games": 2,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "points": points,
        "score_rate": score_rate,
        "paired_wilson_95_ci": {
            model: _wilson(points[model], 2) for model in (model_a, model_b)
        },
        "color": color,
    }


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _model_usage(records: list[dict[str, Any]], schedule_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        group = schedule_by_id[str(record["group_id"])]
        versions = group["model_versions"]
        for move in record["moves"]:
            family = move.get("model")
            if family not in versions:
                continue
            family = str(family)
            identity = _model_id(family, versions)
            for field in ("elapsed_seconds", "nodes", "completed_depth", "tree_depth", "cutoffs", "simulations"):
                values[identity][field].append(float(move.get(field, 0)))
            values[identity]["timeouts"].append(float(bool(move.get("timed_out"))))
            values[identity]["proofs"].append(float(move.get("proof_status") == "proven"))
    rows = []
    for model in model_catalog():
        bucket = values.get(model, {})
        row: dict[str, Any] = {"model": model, "decisions": len(bucket.get("elapsed_seconds", []))}
        for key in ("elapsed_seconds", "nodes", "completed_depth", "tree_depth", "cutoffs", "simulations", "timeouts", "proofs"):
            sequence = bucket.get(key, [])
            row[f"mean_{key}"] = sum(sequence) / len(sequence) if sequence else None
            row[f"total_{key}"] = sum(sequence) if sequence else 0
        row["max_completed_depth"] = max(bucket.get("completed_depth", []), default=0)
        row["max_tree_depth"] = max(bucket.get("tree_depth", []), default=0)
        rows.append(row)
    return rows


def build_report(project_root: str | Path, results_dir: str | Path) -> dict[str, Any]:
    """Build validated V6 CSV/JSON/Markdown and 57-version heatmaps."""
    root = Path(project_root).resolve()
    results = Path(results_dir)
    results = results if results.is_absolute() else root / results
    config = _read_json(results / "config.json")
    schedule, routes = build_schedule(config)
    schedule_by_id = {str(group["group_id"]): group for group in schedule}
    records = [json.loads(line) for line in (results / "9x9_games.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    state = derive_state(records, schedule, routes)
    if not state["complete"]:
        raise ValueError(f"Cannot generate final report with {len(state['pending_group_ids'])} pending groups")

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_group[str(row["group_id"])].append(row)
    catalog = model_catalog()
    pair_rows: list[dict[str, Any]] = []
    score_matrix: dict[str, dict[str, float | None]] = {m: {n: None for n in catalog} for m in catalog}
    status_matrix: dict[str, dict[str, str]] = {m: {n: "same_family_not_compared" for n in catalog} for m in catalog}
    black_matrix: dict[str, dict[str, float | None]] = {m: {n: None for n in catalog} for m in catalog}
    white_matrix: dict[str, dict[str, float | None]] = {m: {n: None for n in catalog} for m in catalog}
    model_games: dict[str, list[float]] = defaultdict(list)
    for a in catalog:
        status_matrix[a][a] = "self"
    status_counts: dict[str, int] = defaultdict(int)
    for group in schedule:
        group_id = str(group["group_id"])
        rows = by_group.get(group_id, [])
        group_state = state["groups"][group_id]
        a = _model_id(group["pair"][0], group["model_versions"])
        b = _model_id(group["pair"][1], group["model_versions"])
        row: dict[str, Any] = {
            "group_id": group_id,
            "model_a": a,
            "model_b": b,
            "route_ids": ";".join(group.get("route_ids", [])),
            "opening_seed": group["opening_seed"],
            "status": group_state["status"],
            "source_type": group_state.get("source_type", ""),
            "observed_games": len(rows),
            "score_rate_a": None,
            "score_rate_b": None,
            "wins_a": None,
            "draws": None,
            "losses_a": None,
        }
        matrix_status = "inferred" if str(group_state["status"]).startswith("inferred") else (
            "unrun" if not rows else "partial_unscored"
        )
        if len(rows) == 2:
            pair = summarize_pair(group, rows)
            source_types = {str(record.get("source_type", "new")) for record in rows}
            source_type = "reused_v5_1" if source_types == {"reused_v5_1"} else "new"
            matrix_status = "observed_reused" if source_type == "reused_v5_1" else "observed_new"
            row.update({
                "status": "played",
                "source_type": source_type,
                "score_rate_a": pair["score_rate"][a],
                "score_rate_b": pair["score_rate"][b],
                "wins_a": pair["wins"][a],
                "draws": pair["draws"],
                "losses_a": pair["losses"][a],
                "paired_ci95_a_low": pair["paired_wilson_95_ci"][a][0],
                "paired_ci95_a_high": pair["paired_wilson_95_ci"][a][1],
                "paired_ci95_b_low": pair["paired_wilson_95_ci"][b][0],
                "paired_ci95_b_high": pair["paired_wilson_95_ci"][b][1],
            })
            score_matrix[a][b] = pair["score_rate"][a]
            score_matrix[b][a] = pair["score_rate"][b]
            status_matrix[a][b] = status_matrix[b][a] = matrix_status
            for game in rows:
                winner = game.get("winner_model")
                points_a = 0.5 if winner is None else float(winner == group["pair"][0])
                points_b = 1.0 - points_a
                model_games[a].append(points_a)
                model_games[b].append(points_b)
                black_family = game["black_model"]
                black_identity = a if black_family == group["pair"][0] else b
                white_identity = b if black_identity == a else a
                black_matrix[black_identity][white_identity] = points_a if black_identity == a else points_b
                white_matrix[white_identity][black_identity] = points_a if white_identity == a else points_b
            for model, opponent in ((a, b), (b, a)):
                side_points = {
                    "black": [
                        (0.5 if game.get("winner_model") is None else float(game.get("winner_model") == group["pair"][0 if model == a else 1]))
                        for game in rows if game["black_model"] == group["pair"][0 if model == a else 1]
                    ],
                    "white": [
                        (0.5 if game.get("winner_model") is None else float(game.get("winner_model") == group["pair"][0 if model == a else 1]))
                        for game in rows if game["white_model"] == group["pair"][0 if model == a else 1]
                    ],
                }
                if side_points["black"]:
                    black_matrix[model][opponent] = sum(side_points["black"]) / len(side_points["black"])
                if side_points["white"]:
                    white_matrix[model][opponent] = sum(side_points["white"]) / len(side_points["white"])
            row["games_data"] = json.dumps([r["game_id"] for r in rows])
        else:
            row["reason"] = group_state.get("status")
            row["inference"] = json.dumps(group_state.get("inferences", {}), ensure_ascii=False)
            status_matrix[a][b] = status_matrix[b][a] = matrix_status
        status_counts[matrix_status] += 1
        pair_rows.append(row)

    usage = _model_usage(records, schedule_by_id)
    model_rows = []
    for model in catalog:
        points = model_games.get(model, [])
        model_rows.append({
            "model": model,
            "games": len(points),
            "wins": sum(value == 1.0 for value in points),
            "draws": sum(value == 0.5 for value in points),
            "losses": sum(value == 0.0 for value in points),
            "points": sum(points),
            "score_rate": sum(points) / len(points) if points else None,
            "score_ci95_low": (_wilson(sum(points), len(points)) or [None, None])[0],
            "score_ci95_high": (_wilson(sum(points), len(points)) or [None, None])[1],
        })
    ranked = sorted((r for r in model_rows if r["games"]), key=lambda r: (-r["score_rate"], -r["games"], r["model"]))
    family_points: dict[str, list[float]] = defaultdict(list)
    for identity, points in model_games.items():
        family_points[identity.split("@", 1)[0]].extend(points)
    family_ranked = []
    for family, points in family_points.items():
        family_versions = BUDGET_LABELS if family != "A" else ("natural",)
        family_ranked.append({
            "family": family,
            "games": len(points),
            "wins": sum(value == 1.0 for value in points),
            "draws": sum(value == 0.5 for value in points),
            "losses": sum(value == 0.0 for value in points),
            "points": sum(points),
            "score_rate": sum(points) / len(points) if points else None,
            "score_ci95": _wilson(sum(points), len(points)),
            "versions_with_games": sum(bool(model_games.get(f"{family}@{budget}", [])) for budget in family_versions),
        })
    family_ranked.sort(key=lambda row: (-row["score_rate"], -row["games"], row["family"]))
    routes_out = []
    for route_id, route in state["routes"].items():
        routes_out.append({
            "route_id": route_id,
            "challenger": _model_id(route["challenger_family"], {route["challenger_family"]: route["challenger_version"]}),
            "challenger_family": route["challenger_family"],
            "challenger_version": route["challenger_version"],
            "opponent_family": route["opponent_family"],
            "status": route["status"],
            "highest_passed_budget": route["highest_passed_budget"],
            "highest_actual_passed_budget": route["highest_actual_passed_budget"],
            "failed_budget": route["failed_budget"],
            "inferred_passed_levels": route["inferred_passed_levels"],
            "levels": json.dumps(route["levels"], ensure_ascii=False),
        })

    reused_games = sum(str(record.get("source_type")) == "reused_v5_1" for record in records)
    new_games = len(records) - reused_games
    usage_by_model = {row["model"]: row for row in usage}
    summary = {
        "experiment_id": config["experiment_id"],
        "board_size": 9,
        "models": len(catalog),
        "abcd_model_versions": 22,
        "e_model_versions": 35,
        "maximum_groups": len(schedule),
        "completed_groups": len(records) // 2,
        "new_games": new_games,
        "reused_games": reused_games,
        "actual_games": len(records),
        "inferred_groups": state["inferred_groups"],
        "group_status_counts": dict(sorted(status_counts.items())),
        "ranking_exploratory": ranked,
        "family_ranking_exploratory": family_ranked,
        "model_usage": usage_by_model,
        "routes": routes_out,
        "pairwise": pair_rows,
        "score_matrix": score_matrix,
        "status_matrix": status_matrix,
        "black_score_matrix": black_matrix,
        "white_score_matrix": white_matrix,
    }
    results.mkdir(parents=True, exist_ok=True)
    (results / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(results / "pairwise_results.csv", pair_rows)
    _write_csv(results / "routes.csv", routes_out)
    _write_csv(results / "model_usage.csv", usage)
    _write_csv(results / "model_summary.csv", model_rows)
    for name, matrix in (("score_matrix", score_matrix), ("status_matrix", status_matrix), ("black_score_matrix", black_matrix), ("white_score_matrix", white_matrix)):
        _write_csv(results / f"{name}.csv", [{"model": model, **matrix[model]} for model in catalog], ["model", *catalog])

    from .validation import validate_records
    replay = validate_records(records, schedule, require_complete_groups=True, verify_proofs=True)
    (results / "validation.json").write_text(json.dumps({"valid": True, "replay": replay, "route_state_complete": state["complete"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    progress = _read_json(results / "progress.json", {})
    tactical = _read_json(root / "research/v6/validation/tactical_suite.json", {})
    ablation = _read_json(root / "research/v6/validation/e5_ablation.json", {})
    _figures(results.parent / "figures", catalog, score_matrix, status_matrix, black_matrix, white_matrix, usage)
    report = _render_markdown(summary, progress, tactical, ablation, replay)
    report_path = root / "research/v6/REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    return {"report_path": str(report_path), "summary_path": str(results / "summary.json"), "validation": replay}


def _figures(directory: Path, catalog: list[str], scores: dict[str, dict[str, float | None]], statuses: dict[str, dict[str, str]], black: dict[str, dict[str, float | None]], white: dict[str, dict[str, float | None]], usage: list[dict[str, Any]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, matrix in (("score", scores), ("black_score", black), ("white_score", white)):
        values = np.array([[np.nan if matrix[row][col] is None else matrix[row][col] for col in catalog] for row in catalog], dtype=float)
        fig, ax = plt.subplots(figsize=(18, 16), layout="constrained")
        masked = np.ma.masked_invalid(values)
        image = ax.imshow(masked, vmin=0, vmax=1, cmap="RdYlGn", interpolation="nearest")
        ax.set_xticks(range(len(catalog)), catalog, rotation=90, fontsize=5)
        ax.set_yticks(range(len(catalog)), catalog, fontsize=5)
        ax.set_title({"score": "V6 全部57版本实测得分率", "black_score": "V6 黑方得分率热力图", "white_score": "V6 白方得分率热力图"}[name])
        ax.set_xlabel("对手版本")
        ax.set_ylabel("行棋模型版本")
        fig.colorbar(image, ax=ax, label="得分率；空白表示未实测或不适用")
        path = directory / f"v6_{name}_heatmap.svg"
        fig.savefig(path, format="svg")
        plt.close(fig)
        paths.append(path.name)
    # Actual and inferred boundaries are deliberately shown as separate series.
    route_labels = ["50ms", "200ms", "1s", "5s", "10s", "20s", "50s"]
    data = {family: [sum(r["model"] == f"{family}@{label}" and r["games"] > 0 for r in usage) for label in route_labels] for family in ("B", "C", "D", "E1", "E2", "E3", "E4", "E5")}
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    x = np.arange(len(route_labels))
    for family, counts in data.items():
        ax.plot(x, counts, marker="o", label=family)
    ax.set_xticks(x, route_labels)
    ax.set_ylabel("该预算实际完成的决策数")
    ax.set_title("V6 各搜索版本实际计算量覆盖")
    ax.grid(alpha=0.25)
    ax.legend(ncol=4)
    path = directory / "v6_budget_usage.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    paths.append(path.name)
    return paths


def _render_markdown(summary: dict[str, Any], progress: dict[str, Any], tactical: dict[str, Any], ablation: dict[str, Any], replay: dict[str, Any]) -> str:
    def pct(value: float | None) -> str:
        return "—" if value is None else f"{100*value:.1f}%"
    ranked = summary["ranking_exploratory"]
    family_ranked = summary["family_ranking_exploratory"]
    rank_lines = ["| 排名 | 模型预算版本 | 对局数 | 胜–平–负 | 得分率（95% Wilson） | 平均每步秒数 | 平均节点 | 平均深度 |"]
    rank_lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for index, model in enumerate(ranked, start=1):
        usage = summary["model_usage"].get(model["model"], {})
        rank_lines.append(f"| {index} | {model['model']} | {model['games']} | {model['wins']}–{model['draws']}–{model['losses']} | {pct(model['score_rate'])} [{pct(model['score_ci95_low'])}, {pct(model['score_ci95_high'])}] | {usage.get('mean_elapsed_seconds') or 0:.3f} | {usage.get('mean_nodes') or 0:.1f} | {usage.get('mean_completed_depth') or 0:.2f} |")
    route_lines = ["| 挑战者版本 | 对手家族 | 最高实测通过档 | 失败档 | inferred pass数 | 路线状态 |", "|---|---|---|---|---:|---|"]
    for route in summary["routes"]:
        route_lines.append(f"| {route['challenger']} | {route['opponent_family']} | {route['highest_actual_passed_budget'] or '无'} | {route['failed_budget'] or '无'} | {route['inferred_passed_levels']} | {route['status']} |")
    family_lines = ["| 探索性排名 | 家族 | 实际游戏样本 | 得分率（95% Wilson） | 覆盖预算版本数 |", "|---:|---|---:|---:|---:|"]
    for index, item in enumerate(family_ranked, 1):
        ci = item["score_ci95"] or [None, None]
        family_lines.append(f"| {index} | {item['family']} | {item['games']} | {pct(item['score_rate'])} [{pct(ci[0])}, {pct(ci[1])}] | {item['versions_with_games']} |")
    hours = progress.get("elapsed_wall_seconds_total", 0) / 3600.0
    return "\n".join([
        "# V6 E家族高级棋类搜索算法开发与实战评测", "",
        "## 摘要", "",
        f"本研究在9×9五子棋中新增E1–E5五类非学习型棋手。共有{summary['actual_games']}局完整且通过回放检查的实际对局，其中V5.1精确复用{summary['reused_games']}局，本轮新增{summary['new_games']}局；赛程最高上限为2856局。另有{summary['inferred_groups']}组由预注册的更高预算单调性假设推断，未当作实测胜负。累计记录耗时约{hours:.2f}小时。", "",
        "V6模型及预算版本的排序按实际出现的对局平均得分率计算，样本覆盖不同且赛制为逐档早停，因此仅为探索性描述，不能视为完整循环赛或统计显著性排名。A为自然耗时版本；B/C/D/E1–E5各有7档，合计22个ABCD版本和35个E版本。", "",
        "## 算法与正确性验收", "",
        "- E1 VCF：对连续冲四强制获胜路线进行OR/AND搜索；不能证明时调用统一A后备。",
        "- E2 VCT：在VCF基础上纳入活三和连续威胁，防守节点保留全部合法关键回应。",
        "- E3 Threat-Space Search：用威胁增益、成本、防守冲突及依赖关系组织搜索；启发式排序不裁掉合法防守。",
        "- E4 Proof-Number Search：以攻方OR/守方AND证明树搜索必胜路线，证明在落子前独立回放证书。",
        "- E5：C上的PVS、置换表、Zobrist哈希、排序、战术延伸和根节点即时胜/必防；E5组件消融单独保存。",
        f"固定战术题集包括{tactical.get('positions', 0)}个局面、{tactical.get('runs', 0)}次决策；{tactical.get('scored_cases', 0)}个可评分oracle项目中正确{tactical.get('correct_cases', 0)}个，错误证明{tactical.get('invalid_proofs', 0)}，证书独立核验{tactical.get('proof_claims', 0)-tactical.get('invalid_proofs', 0)}条；合法落子{tactical.get('illegal_moves', 0)}、棋盘恢复失败{tactical.get('board_restore_failures', 0)}。超时{tactical.get('timeouts', 0)}次均作为受限搜索记载。",
        f"E5消融包含{len(ablation.get('rows', []))}次运行；结果保留于validation/e5_ablation.json与CSV，不把尚未完成深度的结果混入完成深度比较。", "",
        "## 实验方法与复现", "",
        "- 固定V6配置、代码SHA-256、12个开局种子、9×9规则与唯一对局ID；每组同开局换色两局。A自然决策，其他版本按50ms至50s七档截止时间。",
        "- 每完整对局先回放，再逐局写入JSONL；每组结束后原子更新路线状态。断点续跑按唯一ID去重。复用条件只有V5.1中条件完全一致且完整成对的对局。",
        "- 双局得分为胜1、平0.5、负0；路线累计积分由wins + draws/2计算。95%区间使用每局分数作为分数成功的Wilson描述区间，n=2样本很小。",
        "- 只有实测和精确复用的完整配对写入数值矩阵；推断和未运行版本在状态矩阵中单独标记，数值留空。",
        "- 无神经网络或强化学习训练，无训练指标或checkpoint。Codex模型档位和推理强度无法由本运行环境独立读取，故不作验证性声明。", "",
        "## 探索性算法家族排序", "", *family_lines, "",
        "这是按所有实际测试版本汇总的描述性分数，参赛预算覆盖不相同，因此只作导航，不作为家族能力的平衡估计。", "",
        "## 探索性版本排序", "", *rank_lines, "",
        "得分率区间跨越很宽，且各预算版本的参赛对手数因早停而不同。平均得分率不能消除赛程覆盖偏差；请结合逐组矩阵和挑战路线阅读。", "",
        "## 挑战路线", "", *route_lines, "",
        "路线的最高实测档与最高推断档分别记录；推断通过不表示实战胜利。失败只终止该路线，不影响其他版本。", "",
        "## 颜色与对局矩阵", "",
        "- `figures/v6_score_heatmap.svg`：57×57全部模型版本的实际配对得分率。",
        "- `figures/v6_black_score_heatmap.svg` 与 `v6_white_score_heatmap.svg`：按该模型执黑/执白分组的实际得分率；空白表示没有该条件的实测对局。",
        "- `results/score_matrix.csv`、`status_matrix.csv`、`black_score_matrix.csv`、`white_score_matrix.csv`同时区分实际新跑、V5.1复用、推断和未测试。", "",
        "## 关键结论与限制", "",
        "E1–E4的证明输出只表示搜索范围内找到并经独立验证的胜利证书；没有证书表示预算或搜索边界内未找到，不表示不存在必胜路线。E3的威胁排序同样不等于数学证明。E5组件的独立收益应由相同局面、相同深度的消融条目比较；在不同节点数/深度下的墙钟结果同时受截止时间影响。",
        "逐档挑战假设对手更高预算不会更弱，这只用于控制实验规模。传统搜索的时间分配、开局和对手策略会造成非单调观测；任何推断行都没有当作实测。两局配对可以显示方向性，但不足以稳定估计棋力差距。", "",
        "## 原始成果", "",
        "完整记录：`results/9x9_games.jsonl`；进度/来源：`progress.json`、`reuse_manifest.json`、`routes.csv`；统计：`summary.json`、`pairwise_results.csv`、矩阵CSV、`model_usage.csv`；验收：`validation.json`；战术题与E5消融见`validation/`目录。", "",
        f"验收回放：{replay.get('games', 0)}局、{replay.get('replayed_moves', 0)}手；赛程最终状态：完整；活动耗时约{hours:.2f}小时。",
        ""
    ])
