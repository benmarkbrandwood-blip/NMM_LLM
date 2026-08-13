"""Read-only local web report for the retained-v3/v4 passivity diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.retained_passivity_diagnostic import (  # noqa: E402
    EXPECTED_CANDIDATES,
    EXPECTED_GAMES,
    load_game_ledger,
    summarize_diagnostic_records,
)
from learned_ai.training.run_contract import canonical_sha256  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def _empty_candidate() -> dict[str, Any]:
    return {
        "games": 0,
        "horizon_120": {
            "survived": 0,
            "rules_terminal_on_or_before": 0,
            "survival_rate": None,
        },
        "termination_classes": {},
        "outcome_reasons": {},
        "lengths": {
            "support": 0,
            "min_total_logical_plies": None,
            "median_total_logical_plies": None,
            "p90_total_logical_plies": None,
            "max_total_logical_plies": None,
            "mean_total_logical_plies": None,
        },
        "malom_at_ply_120_candidate_perspective": {
            "snapshot_support": 0,
            "queryable": 0,
            "unqueryable": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "history_aware": False,
        },
        "candidate_malom_moves": {
            "candidate_turns": 0,
            "queryable_turns": 0,
            "unqueryable_turns": 0,
            "preserving_turns": 0,
            "one_step_downgrade_turns": 0,
            "two_step_downgrade_turns": 0,
            "query_coverage": None,
            "preserving_rate_given_queryable": None,
            "downgrade_rate_given_queryable": None,
        },
        "eventual_rules_wdl": {
            "support": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "score_rate": None,
            "safety_cap_excluded": 0,
            "strength_claim_allowed": False,
        },
    }


def _empty_report(spec: dict[str, Any]) -> dict[str, Any]:
    primary = {
        "support": 0,
        "mean": None,
        "sample_standard_deviation": None,
        "standard_error": None,
        "interval": [None, None],
        "half_width": None,
        "decision": "pending",
        "maximum_half_width": 0.10,
        "precision_adequate": False,
        "interpretation": "fixed-corpus engineering interval, not population inference",
    }
    return {
        "schema_version": "nmm.retained-passivity-diagnostic-result.v1",
        "diagnostic_id": spec["diagnostic_id"],
        "spec_identity": spec["spec_identity"],
        "status": "running",
        "completed_games": 0,
        "expected_games": EXPECTED_GAMES,
        "ledger_tail_record_sha256": None,
        "by_candidate": {
            candidate_id: _empty_candidate() for candidate_id in EXPECTED_CANDIDATES
        },
        "paired": {
            "matched_units_complete": 0,
            "matched_units_expected": 128,
            "primary_horizon_survival_v4_minus_v3": primary,
            "restricted_length_v4_minus_v3": {
                "support": 0,
                "mean": None,
                "sample_standard_deviation": None,
                "standard_error": None,
                "interval": [None, None],
                "half_width": None,
            },
            "per_game_preserving_rate_v4_minus_v3": {
                "support": 0,
                "mean": None,
                "sample_standard_deviation": None,
                "standard_error": None,
                "interval": [None, None],
                "half_width": None,
            },
        },
        "by_candidate_color": {},
        "by_source_stratum": {},
        "claim_boundary": {
            "development_corpus_reused": True,
            "playing_strength_claim": False,
            "refresh_causal_claim": False,
            "promotion_or_publication": False,
            "malom_is_history_aware": False,
            "safety_cap_is_draw": False,
        },
        "result_identity": None,
    }


def _start_level_score_precision(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize paired score differences after clustering both colours by start."""
    by_colour_unit: dict[tuple[str, str], dict[str, float]] = {}
    for record in records:
        if record.get("termination_class") != "rules_terminal":
            continue
        score = record.get("candidate_score")
        if score is None:
            continue
        key = (str(record["source_core_id"]), str(record["candidate_color"]))
        by_colour_unit.setdefault(key, {})[str(record["candidate_id"])] = float(score)

    by_start: dict[str, dict[str, float]] = {}
    matched_colour_units = 0
    for (source_core_id, candidate_color), candidates in by_colour_unit.items():
        if set(candidates) != set(EXPECTED_CANDIDATES):
            continue
        matched_colour_units += 1
        by_start.setdefault(source_core_id, {})[candidate_color] = (
            candidates[EXPECTED_CANDIDATES[1]]
            - candidates[EXPECTED_CANDIDATES[0]]
        )

    start_differences = [
        (colours["W"] + colours["B"]) / 2.0
        for _, colours in sorted(by_start.items())
        if set(colours) == {"W", "B"}
    ]
    support = len(start_differences)
    mean = sum(start_differences) / support if support else None
    if support == 0:
        deviation = error = half_width = None
        interval: list[float | None] = [None, None]
    else:
        deviation = statistics.stdev(start_differences) if support > 1 else 0.0
        error = deviation / math.sqrt(support)
        half_width = 1.96 * error
        interval = [mean - half_width, mean + half_width]

    distribution = Counter(start_differences)
    fixed_width_budgets = []
    if deviation is not None:
        for target_half_width in (0.02, 0.015, 0.01):
            starts = math.ceil((1.96 * deviation / target_half_width) ** 2)
            fixed_width_budgets.append(
                {
                    "target_half_width": target_half_width,
                    "starts": starts,
                    "games": starts * 4,
                }
            )
    return {
        "start_units_complete": support,
        "start_units_expected": 64,
        "matched_colour_units_rules_terminal": matched_colour_units,
        "mean": mean,
        "sample_standard_deviation": deviation,
        "standard_error": error,
        "interval": interval,
        "half_width": half_width,
        "distribution": {
            str(value): distribution[value] for value in sorted(distribution)
        },
        "fixed_width_budgets": fixed_width_budgets,
        "interpretation": (
            "fixed reused-development-corpus engineering interval at the start "
            "cluster level; not population inference, equivalence, or a playing-"
            "strength claim"
        ),
        "strength_claim_allowed": False,
    }


def build_payload(output_root: str | Path) -> dict[str, Any]:
    """Build current progress and independently recomputed process metrics."""
    root = Path(output_root).resolve()
    spec_path = root / "spec.json"
    if not spec_path.is_file():
        return {
            "available": False,
            "status": "not_started",
            "message": "诊断尚未启动；没有伪造或预填结果。",
            "expected_games": EXPECTED_GAMES,
        }
    spec = _read_json(spec_path)
    ledger = root / "games.jsonl"
    if ledger.is_file():
        records, tail = load_game_ledger(spec, ledger)
        report = summarize_diagnostic_records(spec, records, tail)
    else:
        records = []
        report = _empty_report(spec)
    progress = _read_json(root / "progress.json") if (root / "progress.json").is_file() else {}
    status = (
        "completed"
        if (root / "completion.json").is_file()
        else "failed"
        if (root / "failure.json").is_file()
        else "running"
    )
    report["status"] = status
    safe_progress_path = root / "safe-progress-report.json"
    safe_progress = None
    if safe_progress_path.is_file():
        safe_progress = _read_json(safe_progress_path)
        safe_body = {
            key: value
            for key, value in safe_progress.items()
            if key != "result_identity"
        }
        if canonical_sha256(safe_body) != safe_progress.get("result_identity"):
            raise ValueError("safe-progress report identity differs")
        safe_source = safe_progress.get("source")
        if (
            safe_progress.get("schema_version")
            != "nmm.retained-safe-progress-audit-result.v1"
            or not isinstance(safe_source, Mapping)
            or safe_source.get("diagnostic_id") != spec.get("diagnostic_id")
            or safe_source.get("spec_identity") != spec.get("spec_identity")
            or safe_source.get("result_identity") != report.get("result_identity")
        ):
            raise ValueError("safe-progress source binding differs")
    oracle_order_path = root / "oracle-order-report.json"
    oracle_order = None
    if oracle_order_path.is_file():
        oracle_order = _read_json(oracle_order_path)
        oracle_body = {
            key: value
            for key, value in oracle_order.items()
            if key != "result_identity"
        }
        if canonical_sha256(oracle_body) != oracle_order.get("result_identity"):
            raise ValueError("oracle-order report identity differs")
        oracle_source = oracle_order.get("source")
        if (
            oracle_order.get("schema_version")
            != "nmm.retained-oracle-order-audit-result.v1"
            or not isinstance(oracle_source, Mapping)
            or safe_progress is None
            or oracle_source.get("diagnostic_id") != spec.get("diagnostic_id")
            or oracle_source.get("spec_identity") != spec.get("spec_identity")
            or oracle_source.get("result_identity") != report.get("result_identity")
            or oracle_source.get("safe_progress_result_identity")
            != safe_progress.get("result_identity")
        ):
            raise ValueError("oracle-order source binding differs")
    return {
        "available": True,
        "status": status,
        "report": report,
        "progress": {
            "completed_games": int(progress.get("completed_games") or report["completed_games"]),
            "expected_games": EXPECTED_GAMES,
            "current_game_ordinal": progress.get("current_game_ordinal"),
            "current_stage": progress.get("current_stage"),
            "current_stage_ply": progress.get("current_stage_ply"),
            "active_seconds": progress.get("active_seconds"),
        },
        "identities": {
            "diagnostic_id": spec["diagnostic_id"],
            "spec_identity": spec["spec_identity"],
            "plan_identity": spec["plan"]["identity"],
            "implementation_commit": spec["implementation"]["commit"],
        },
        "safe_progress": safe_progress,
        "oracle_order": oracle_order,
        "score_precision_by_start": _start_level_score_precision(records),
    }


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NMM_LLM · v3/v4 被动性诊断</title>
<style>
:root{color-scheme:dark;--bg:#06111f;--panel:#0b1b2d;--panel2:#10243a;--line:#24415d;--text:#edf6ff;--muted:#96adc2;--cyan:#52c7ef;--amber:#f2b84b;--pink:#e98bb4;--green:#6dd7a0;--red:#ff7c8a}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(160deg,#06111f,#081522 52%,#06101b);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text)}main{max-width:1320px;margin:auto;padding:26px}h1{font-size:23px;margin:0 0 4px}.sub{color:var(--muted);margin-bottom:18px}.notice{border:1px solid var(--amber);background:#2b2112;color:#ffe3a2;padding:12px 14px;border-radius:8px;margin:14px 0}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.card,.panel{border:1px solid var(--line);background:linear-gradient(160deg,var(--panel2),var(--panel));border-radius:7px;padding:14px}.card .k{color:var(--muted);font-size:12px}.card .v{font-size:25px;font-weight:700;margin:3px 0}.card .d{color:var(--muted);font-size:12px}.panel{margin-top:12px}h2{font-size:16px;margin:0 0 10px}.help{color:var(--muted);font-size:12px;margin:-5px 0 12px}.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}.bars{display:grid;gap:8px}.barrow{display:grid;grid-template-columns:170px 1fr 72px;gap:10px;align-items:center}.track{height:10px;background:#071320;border-radius:20px;overflow:hidden}.fill{height:100%;background:var(--cyan)}.fill.v4{background:var(--amber)}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:8px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:500}.badge{display:inline-block;border:1px solid var(--line);padding:2px 7px;border-radius:10px;color:var(--muted)}.ok{color:var(--green)}.bad{color:var(--red)}.pending{color:var(--amber)}details{border-top:1px solid var(--line);padding-top:10px;margin-top:10px}summary{cursor:pointer;color:var(--cyan)}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:#b9ddf0;word-break:break-all}.empty{max-width:760px;margin:20vh auto;text-align:center}.empty .panel{padding:30px}@media(max-width:900px){.grid{grid-template-columns:1fr 1fr}.two{grid-template-columns:1fr}.barrow{grid-template-columns:120px 1fr 62px}}@media(max-width:520px){main{padding:14px}.grid{grid-template-columns:1fr}}
</style>
</head>
<body><main id="app"><div class="empty"><div class="panel"><h1>正在读取诊断证据…</h1></div></div></main>
<script>
const C={v3:'retained-v3-refresh50',v4:'retained-v4-no-refresh'};
const pct=v=>v==null?'—':(100*v).toFixed(1)+'%';
const pp=v=>v==null?'—':`${v>0?'+':''}${(100*v).toFixed(2)}pp`;
const num=(v,d=2)=>v==null?'—':Number(v).toFixed(d);
const intval=v=>v==null?'—':Math.round(v).toLocaleString('zh-CN');
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const card=(k,v,d)=>`<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
function bar(label,value,klass=''){const w=value==null?0:Math.max(0,Math.min(100,value*100));return `<div class="barrow"><span>${label}</span><div class="track"><div class="fill ${klass}" style="width:${w}%"></div></div><b>${pct(value)}</b></div>`}
function decisionText(d){return ({pending:'等待完整配对',inconclusive:'区间跨 0：不确定',inconclusive_precision:'精度不足：不确定',v4_higher_120_ply_survival:'v4 的 120 手存活率更高',v3_higher_120_ply_survival:'v3 的 120 手存活率更高'})[d]||d}
function safeDecision(d){return ({pending:'等待完整审计',insufficient_safe_capture_opportunities:'安全吃子机会不足',inconclusive_precision:'精度不足',inconclusive:'未发现方向性差异',v4_higher_missed_safe_capture_share:'v4 每候选回合错过保值吃子机会的比例更高',v3_higher_missed_safe_capture_share:'v3 每候选回合错过保值吃子机会的比例更高'})[d]||d}
function safeProgress(s){if(!s)return `<div class="panel"><h2>安全推进机会审计</h2><p class="help">零游戏审计尚未生成。网页不会根据原始吃子率推测“是否有保值吃子机会”。</p></div>`;const a=s.by_candidate[C.v3],b=s.by_candidate[C.v4],x=a.all_candidate_turns,y=b.all_candidate_turns,x120=a.after_ply_120_candidate_turns,y120=b.after_ply_120_candidate_turns,p=s.paired.primary_missed_safe_capture_share_v4_minus_v3,iv=p.interval||[null,null];return `<div class="panel"><h2>安全推进机会（零游戏复算）</h2><p class="help">“安全吃子机会”只表示至少一个完整合法吃子动作保持当前 Malom 粗 W/D/L，并会重置无吃子计数；它不是最佳着、棋力标签或 refresh 因果证据。主指标是每局错过机会的候选回合数 / 该局全部候选回合，因此同时包含机会暴露与选择结果；机会内选择率另列。</p><table><thead><tr><th>候选</th><th>候选回合</th><th>实际吃子率</th><th>保值吃子机会</th><th>机会内选择率</th><th>错过/候选回合</th><th>120 后错过</th></tr></thead><tbody><tr><td>v3</td><td>${intval(x.candidate_turns)}</td><td>${pct(x.chosen_capture_rate)}</td><td>${intval(x.safe_capture_opportunity_turns)}</td><td>${pct(x.safe_capture_selection_rate_given_opportunity)}</td><td>${pct(x.missed_safe_capture_share_per_candidate_turn)}</td><td>${x120.missed_safe_capture_turns} / ${x120.safe_capture_opportunity_turns}</td></tr><tr><td>v4</td><td>${intval(y.candidate_turns)}</td><td>${pct(y.chosen_capture_rate)}</td><td>${intval(y.safe_capture_opportunity_turns)}</td><td>${pct(y.safe_capture_selection_rate_given_opportunity)}</td><td>${pct(y.missed_safe_capture_share_per_candidate_turn)}</td><td>${y120.missed_safe_capture_turns} / ${y120.safe_capture_opportunity_turns}</td></tr></tbody></table><p class="help"><b>${esc(safeDecision(p.decision))}</b>；配对差 v4−v3 ${pct(p.mean)}，工程区间 ${pct(iv[0])} … ${pct(iv[1])}。分母和 query coverage 保留在报告中。</p></div><div class="panel"><h2>棋盘重访（次级）</h2><p class="help">棋盘重访仅检查固定前缀末局面及其后的审计后缀中，本地 BoardState FEN 是否已出现；“可避免”要求存在另一个 Malom W/D/L 保值且在该范围内未见过的合法后继。它不包含前缀内部局面，不等于严格三次重复，也不能替代完整历史裁判。</p><table><thead><tr><th>候选</th><th>重访率</th><th>可避免重访/候选回合</th><th>120 后重访率</th><th>120 后可避免</th></tr></thead><tbody><tr><td>v3</td><td>${pct(x.chosen_board_revisit_rate)}</td><td>${pct(x.avoidable_board_revisit_share_per_candidate_turn)}</td><td>${pct(x120.chosen_board_revisit_rate)}</td><td>${intval(x120.avoidable_board_revisit_turns)}</td></tr><tr><td>v4</td><td>${pct(y.chosen_board_revisit_rate)}</td><td>${pct(y.avoidable_board_revisit_share_per_candidate_turn)}</td><td>${pct(y120.chosen_board_revisit_rate)}</td><td>${intval(y120.avoidable_board_revisit_turns)}</td></tr></tbody></table></div>`}
function oracleDecision(d){return ({pending:'等待完整审计',insufficient_full_order_coverage:'完整排序覆盖不足',insufficient_ordering_opportunities:'排序选择机会不足',inconclusive_precision:'精度不足',inconclusive:'未发现方向性排序差异',v4_higher_full_order_regret:'v4 的完整 Malom 序位后悔更高',v3_higher_full_order_regret:'v3 的完整 Malom 序位后悔更高'})[d]||d}
function oracleOrder(s){if(!s)return `<div class="panel"><h2>完整 Malom 排序对齐</h2><p class="help">零游戏完整排序审计尚未生成。粗 W/D/L 保值不能回答候选是否选择了同一保值集合中的 Malom 最优等级。</p></div>`;const a=s.by_candidate[C.v3],b=s.by_candidate[C.v4],x=a.all_candidate_turns,y=b.all_candidate_turns,x120=a.after_ply_120_candidate_turns,y120=b.after_ply_120_candidate_turns,p=s.paired.primary_mean_normalised_ordinal_regret_v4_minus_v3,iv=p.interval||[null,null];return `<div class="panel"><h2>完整 Malom 排序对齐（零游戏复算）</h2><p class="help">只在粗 W/D/L 保值集合内比较完整 Malom 值。归一化序位后悔 0 表示选择最高排序等级，1 表示选择最低等级；同值/强制回合计 0。它是序位量，不是步数、胜率、严格规则活性或棋力分数。</p><table><thead><tr><th>候选</th><th>候选回合</th><th>可排序覆盖</th><th>有等级选择的回合</th><th>机会内选最高等级</th><th>平均序位后悔</th><th>120 后序位后悔</th></tr></thead><tbody><tr><td>v3</td><td>${intval(x.candidate_turns)}</td><td>${pct(x.within_wdl_orderable_coverage_per_candidate_turn)}</td><td>${intval(x.full_order_choice_opportunity_turns)}</td><td>${pct(x.chosen_full_order_best_rate_given_opportunity)}</td><td>${pct(x.mean_normalised_ordinal_regret_given_orderable)}</td><td>${pct(x120.mean_normalised_ordinal_regret_given_orderable)}</td></tr><tr><td>v4</td><td>${intval(y.candidate_turns)}</td><td>${pct(y.within_wdl_orderable_coverage_per_candidate_turn)}</td><td>${intval(y.full_order_choice_opportunity_turns)}</td><td>${pct(y.chosen_full_order_best_rate_given_opportunity)}</td><td>${pct(y.mean_normalised_ordinal_regret_given_orderable)}</td><td>${pct(y120.mean_normalised_ordinal_regret_given_orderable)}</td></tr></tbody></table><p class="help"><b>${esc(oracleDecision(p.decision))}</b>；配对差 v4−v3 ${pct(p.mean)}，工程区间 ${pct(iv[0])} … ${pct(iv[1])}。完整比较器是位置型 ultra-strong 排序；不含三次重复或无吃子历史，不能据此声称“更主动”或“更强”。</p></div>`}
function scorePrecision(s){if(!s||!s.start_units_complete)return `<div class="panel"><h2>起点层配对得分精度</h2><p class="help">等待同一起点的黑白两色都形成完整 v3/v4 规则终局配对。安全上限局不会被当作和棋补入。</p></div>`;const iv=s.interval||[null,null],dist=Object.entries(s.distribution||{}).sort((a,b)=>Number(a[0])-Number(b[0])).map(([value,count])=>`<tr><td>${pp(Number(value))}</td><td>${intval(count)}</td></tr>`).join(''),budget=(s.fixed_width_budgets||[]).map(x=>`<tr><td>${pp(x.target_half_width)}</td><td>${intval(x.starts)}</td><td>${intval(x.games)}</td></tr>`).join('');return `<div class="panel"><h2>起点层配对得分精度（零新对局）</h2><p class="help">每个独立起点先把候选执白与执黑的两个 v4−v3 得分差平均，再在起点之间计算工程区间，避免把共享同一开局的两种颜色误当成独立样本。得分为胜 1、和 0.5、负 0；本语料已用于开发，区间不是总体推断、等效证明或棋力结论。</p><div class="grid">${card('完整起点',`${s.start_units_complete} / ${s.start_units_expected}`,`${s.matched_colour_units_rules_terminal} 个规则终局颜色单元`)}${card('起点层均值 v4 − v3',pp(s.mean),`工程区间 ${pp(iv[0])} … ${pp(iv[1])}`)}${card('起点层标准差',pp(s.sample_standard_deviation),`区间半宽 ${pp(s.half_width)}`)}${card('声明边界','仅规划依据','不授权新对局；不能声称等效或更强')}</div><div class="two"><div><h2>起点差值分布</h2><table><thead><tr><th>两色平均得分差</th><th>起点数</th></tr></thead><tbody>${dist}</tbody></table></div><div><h2>固定半宽预算（语料内估计）</h2><table><thead><tr><th>目标半宽</th><th>起点</th><th>总局数</th></tr></thead><tbody>${budget}</tbody></table><p class="help">每个新起点需要两候选 × 黑白交换，共 4 局。预算只复用本开发语料的起点层标准差；正式 held-out 仍须预先冻结目标半宽、语料和资源上限。</p></div></div></div>`}
function render(p){const app=document.getElementById('app');if(!p.available){app.innerHTML=`<div class="empty"><div class="panel"><h1>v3/v4 被动性诊断</h1><p class="sub">${esc(p.message)}</p><div class="notice">这不是错误：没有精确授权前，网页只显示“未启动”，不会放置空结果或推测值。</div></div></div>`;return}
const r=p.report,v3=r.by_candidate[C.v3],v4=r.by_candidate[C.v4],pr=r.paired.primary_horizon_survival_v4_minus_v3,iv=pr.interval||[null,null],active=p.progress.active_seconds;
app.innerHTML=`<h1>NMM_LLM · retained-v3 / no-refresh-v4 被动性诊断</h1><div class="sub">${esc(p.identities.diagnostic_id)} · <span class="badge">${esc(p.status)}</span></div>
<div class="notice"><b>开发诊断，不是棋力评测。</b> 语料已使用；结果只能描述两个命名 final route 在固定语料上的过程差异，不能归因为 refresh、不能晋级或发布模型。</div>
<div class="grid">${card('完成进度',`${p.progress.completed_games} / ${p.progress.expected_games}`,p.progress.current_stage?`game ${Number(p.progress.current_game_ordinal)+1} · ${p.progress.current_stage} ply ${p.progress.current_stage_ply}`:'当前无在途对局')}${card('完整匹配单元',`${r.paired.matched_units_complete} / ${r.paired.matched_units_expected}`,'同一起点、同一候选颜色，v3/v4 各一局')}${card('主差值 v4 − v3',pct(pr.mean),iv[0]==null?'等待配对':`工程区间 ${pct(iv[0])} … ${pct(iv[1])}`)}${card('主判决',decisionText(pr.decision),`区间半宽 ${pct(pr.half_width)}；上限 10.0%`)}${card('v3: 120 手仍在进行',pct(v3.horizon_120.survival_rate),`${v3.horizon_120.survived} / ${v3.games} 局`)}${card('v4: 120 手仍在进行',pct(v4.horizon_120.survival_rate),`${v4.horizon_120.survived} / ${v4.games} 局`)}${card('活动用时',active==null?'—':num(active/60,1)+' min','只计 evaluator active time；上限 2 h')}${card('报告状态',esc(r.status),r.result_identity?`result ${esc(r.result_identity.slice(0,12))}`:'实时从账本复算')}</div>
<div class="panel"><h2>120 手 horizon survival</h2><p class="help">严格裁判在总第 120 个完整逻辑回合之后仍未终局。它不是和棋，也不预判这局最终结果；恰好对应训练日志中被 120 手上限截断的过程事件。</p><div class="bars">${bar('retained-v3',v3.horizon_120.survival_rate)}${bar('no-refresh-v4',v4.horizon_120.survival_rate,'v4')}</div></div>
${safeProgress(p.safe_progress)}
${oracleOrder(p.oracle_order)}
<div class="two"><div class="panel"><h2>对局长度</h2><p class="help">总逻辑手数包含固定 12 手前缀。1,536 post-prefix 是安全上限；命中时记 incomplete，不算和棋。</p><table><thead><tr><th>候选</th><th>支持</th><th>均值</th><th>中位</th><th>P90</th><th>最大</th></tr></thead><tbody><tr><td>v3</td><td>${v3.lengths.support}</td><td>${num(v3.lengths.mean_total_logical_plies,1)}</td><td>${num(v3.lengths.median_total_logical_plies,0)}</td><td>${num(v3.lengths.p90_total_logical_plies,0)}</td><td>${intval(v3.lengths.max_total_logical_plies)}</td></tr><tr><td>v4</td><td>${v4.lengths.support}</td><td>${num(v4.lengths.mean_total_logical_plies,1)}</td><td>${num(v4.lengths.median_total_logical_plies,0)}</td><td>${num(v4.lengths.p90_total_logical_plies,0)}</td><td>${intval(v4.lengths.max_total_logical_plies)}</td></tr></tbody></table></div>
<div class="panel"><h2>Malom move 过程</h2><p class="help">保值率的分母仅是可查询的候选回合；query coverage 单独显示。Malom 不含重复/无吃子历史，因此不能代替严格裁判。</p><table><thead><tr><th>候选</th><th>候选回合</th><th>覆盖率</th><th>保值率*</th><th>降级率*</th></tr></thead><tbody><tr><td>v3</td><td>${intval(v3.candidate_malom_moves.candidate_turns)}</td><td>${pct(v3.candidate_malom_moves.query_coverage)}</td><td>${pct(v3.candidate_malom_moves.preserving_rate_given_queryable)}</td><td>${pct(v3.candidate_malom_moves.downgrade_rate_given_queryable)}</td></tr><tr><td>v4</td><td>${intval(v4.candidate_malom_moves.candidate_turns)}</td><td>${pct(v4.candidate_malom_moves.query_coverage)}</td><td>${pct(v4.candidate_malom_moves.preserving_rate_given_queryable)}</td><td>${pct(v4.candidate_malom_moves.downgrade_rate_given_queryable)}</td></tr></tbody></table></div></div>
<div class="two"><div class="panel"><h2>第 120 手 Malom 理论 W/D/L</h2><p class="help">只对仍在进行且可查询的快照，从候选视角投影；这是 history-free 理论棋盘值，不是实际终局裁定。</p><table><thead><tr><th>候选</th><th>快照</th><th>可查询</th><th>W</th><th>D</th><th>L</th></tr></thead><tbody>${[v3,v4].map((x,i)=>`<tr><td>${i?'v4':'v3'}</td><td>${x.malom_at_ply_120_candidate_perspective.snapshot_support}</td><td>${x.malom_at_ply_120_candidate_perspective.queryable}</td><td>${x.malom_at_ply_120_candidate_perspective.wins}</td><td>${x.malom_at_ply_120_candidate_perspective.draws}</td><td>${x.malom_at_ply_120_candidate_perspective.losses}</td></tr>`).join('')}</tbody></table></div>
<div class="panel"><h2>最终严格规则 W/D/L（次要）</h2><p class="help">仅含规则终局；安全上限局剔除并单列。语料已使用且方案未按稀有胜负做功效设计，因此这些数字没有棋力、晋级或因果含义。</p><table><thead><tr><th>候选</th><th>支持</th><th>胜</th><th>和</th><th>负</th><th>incomplete</th></tr></thead><tbody>${[v3,v4].map((x,i)=>`<tr><td>${i?'v4':'v3'}</td><td>${x.eventual_rules_wdl.support}</td><td>${x.eventual_rules_wdl.wins}</td><td>${x.eventual_rules_wdl.draws}</td><td>${x.eventual_rules_wdl.losses}</td><td>${x.eventual_rules_wdl.safety_cap_excluded}</td></tr>`).join('')}</tbody></table></div></div>
${scorePrecision(p.score_precision_by_start)}
<div class="panel"><h2>身份与解释边界</h2><details><summary>为什么这个网页不能回答“no-refresh 更强吗？”</summary><p class="help">v3/v4 的训练 seed、源码、冻结目标年龄和累计 SpecialistDB 都不同；这里又复用了已经查看过的起点。任何差异都只能描述两个 final route，不能归因 refresh，也不是独立 held-out strength evidence。</p></details><code>plan ${esc(p.identities.plan_identity)} · spec ${esc(p.identities.spec_identity)} · source ${esc(p.identities.implementation_commit)}</code></div>`}
async function tick(){try{const r=await fetch('/api/diagnostic',{cache:'no-store'});if(!r.ok)throw new Error(await r.text());render(await r.json())}catch(e){document.getElementById('app').innerHTML=`<div class="empty"><div class="panel"><h1>网页读取失败</h1><p class="bad">${esc(e.message)}</p></div></div>`}}
tick();setInterval(tick,3000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    output_root: Path

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        route = urlsplit(self.path).path
        try:
            if route == "/":
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", HTML.encode())
                return
            if route == "/api/diagnostic":
                body = json.dumps(
                    build_payload(self.output_root),
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
                self._send(HTTPStatus.OK, "application/json; charset=utf-8", body)
                return
            if route == "/healthz":
                self._send(HTTPStatus.OK, "text/plain; charset=utf-8", b"ok\n")
                return
        except (OSError, ValueError, KeyError) as exc:
            message = json.dumps(
                {"error": type(exc).__name__, "message": str(exc)},
                separators=(",", ":"),
            ).encode()
            self._send(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "application/json; charset=utf-8",
                message,
            )
            return
        self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=str(
            _ROOT
            / "learned_ai"
            / "checkpoints"
            / "evaluation"
            / "sanmill-retained-v3-v4-passivity-diagnostic-v1"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_root = Path(args.output_root).resolve()
    handler = type("DiagnosticHandler", (Handler,), {"output_root": output_root})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"retained passivity report: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
