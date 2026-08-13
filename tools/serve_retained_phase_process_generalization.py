"""Read-only local web report for the retained phase-process confirmation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.retained_phase_process_generalization import (  # noqa: E402
    EXPECTED_GAMES,
    EXPECTED_STARTS,
    load_game_ledger,
    summarize_records,
)
from learned_ai.training.run_contract import canonical_sha256  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def _validate_spec_identity(spec: dict[str, Any]) -> None:
    identity = spec.get("spec_identity")
    body = {key: value for key, value in spec.items() if key != "spec_identity"}
    if not isinstance(identity, str) or canonical_sha256(body) != identity:
        raise ValueError("phase-process spec identity differs")


def _fixed_width_budgets(primary: dict[str, Any]) -> list[dict[str, Any]]:
    deviation = primary.get("sample_standard_deviation")
    if deviation is None:
        return []
    return [
        {
            "target_half_width": width,
            "starts": max(1, math.ceil((1.96 * float(deviation) / width) ** 2)),
            "games": max(1, math.ceil((1.96 * float(deviation) / width) ** 2))
            * 4,
            "planning_only": True,
        }
        for width in (0.10, 0.075, 0.05)
    ]


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
            "expected_starts": EXPECTED_STARTS,
        }

    spec = _read_json(spec_path)
    _validate_spec_identity(spec)
    ledger_path = root / "games.jsonl"
    if ledger_path.is_file():
        records, tail = load_game_ledger(spec, ledger_path)
    else:
        records, tail = [], None
    report = summarize_records(spec, records, tail)
    progress_path = root / "progress.json"
    progress = _read_json(progress_path) if progress_path.is_file() else {}
    status = (
        "completed"
        if (root / "completion.json").is_file()
        else "failed"
        if (root / "failure.json").is_file()
        else "running"
    )
    report["status"] = status
    primary = report["paired"][
        "primary_start_clustered_108_ply_survival_v4_minus_v3"
    ]
    return {
        "available": True,
        "status": status,
        "report": report,
        "progress": {
            "completed_games": int(
                progress.get("completed_games") or report["completed_games"]
            ),
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
            "corpus_identity": spec["corpus"]["identity"],
        },
        "precision": {
            "start_clustered_primary": primary,
            "fixed_width_budgets": _fixed_width_budgets(primary),
        },
    }


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NMM_LLM · v3/v4 阶段过程确认</title>
<style>
:root{color-scheme:dark;--bg:#06111f;--panel:#0b1b2d;--panel2:#10243a;--line:#24415d;--text:#edf6ff;--muted:#96adc2;--cyan:#52c7ef;--amber:#f2b84b;--green:#6dd7a0;--red:#ff7c8a}*{box-sizing:border-box}body{margin:0;background:linear-gradient(160deg,#06111f,#081522 52%,#06101b);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text)}main{max-width:1380px;margin:auto;padding:26px}h1{font-size:23px;margin:0 0 4px}h2{font-size:16px;margin:0 0 10px}.sub,.help{color:var(--muted)}.sub{margin-bottom:18px}.help{font-size:12px;margin:-5px 0 12px}.notice{border:1px solid var(--amber);background:#2b2112;color:#ffe3a2;padding:12px 14px;border-radius:8px;margin:14px 0}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.card,.panel{border:1px solid var(--line);background:linear-gradient(160deg,var(--panel2),var(--panel));border-radius:7px;padding:14px}.card .k{color:var(--muted);font-size:12px}.card .v{font-size:24px;font-weight:700;margin:3px 0}.card .d{color:var(--muted);font-size:12px}.panel{margin-top:12px}.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}.bars{display:grid;gap:8px}.barrow{display:grid;grid-template-columns:190px 1fr 72px;gap:10px;align-items:center}.track{height:10px;background:#071320;border-radius:20px;overflow:hidden}.fill{height:100%;background:var(--cyan)}.fill.v4{background:var(--amber)}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:8px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:500}.badge{display:inline-block;border:1px solid var(--line);padding:2px 7px;border-radius:10px;color:var(--muted)}details{border-top:1px solid var(--line);padding-top:10px;margin-top:10px}summary{cursor:pointer;color:var(--cyan)}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:#b9ddf0;word-break:break-all}.empty{max-width:760px;margin:20vh auto;text-align:center}.empty .panel{padding:30px}.bad{color:var(--red)}@media(max-width:900px){.grid{grid-template-columns:1fr 1fr}.two{grid-template-columns:1fr}.barrow{grid-template-columns:130px 1fr 62px}}@media(max-width:520px){main{padding:14px}.grid{grid-template-columns:1fr}}
</style>
</head>
<body><main id="app"><div class="empty"><div class="panel"><h1>正在读取阶段过程证据…</h1></div></div></main>
<script>
const C={v3:'retained-v3-refresh50',v4:'retained-v4-no-refresh'};
const pct=v=>v==null?'—':(100*v).toFixed(1)+'%';
const pp=v=>v==null?'—':`${v>0?'+':''}${(100*v).toFixed(2)}pp`;
const num=(v,d=1)=>v==null?'—':Number(v).toFixed(d);
const integer=v=>v==null?'—':Math.round(v).toLocaleString('zh-CN');
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const card=(k,v,d)=>`<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
function bar(label,value,klass=''){const w=value==null?0:Math.max(0,Math.min(100,value*100));return `<div class="barrow"><span>${label}</span><div class="track"><div class="fill ${klass}" style="width:${w}%"></div></div><b>${pct(value)}</b></div>`}
function decisionText(value){return ({pending:'等待 39 个完整起点',inconclusive:'区间跨 0：不确定',inconclusive_precision:'半宽超门：精度不足',v4_higher_108_post_start_ply_survival:'v4 的相对窗口存活率更高',v3_higher_108_post_start_ply_survival:'v3 的相对窗口存活率更高'})[value]||value}
function phaseRows(phases){return ['placement','movement','flying'].map(phase=>{const row=phases[phase]||{},a=row[C.v3],b=row[C.v4];return `<tr><td>${phase}</td><td>${a?integer(a.games):'—'}</td><td>${a?pct(a.horizon_108_post_start.survival_rate):'—'}</td><td>${b?integer(b.games):'—'}</td><td>${b?pct(b.horizon_108_post_start.survival_rate):'—'}</td></tr>`}).join('')}
function processRows(a,b){const rows=[['起点 no-capture','start_no_capture'],['窗口 no-capture','horizon_no_capture'],['窗口 − 起点 no-capture','horizon_minus_start_no_capture'],['终局 no-capture','final_no_capture'],['起点当前重复计数','start_repetition_current'],['窗口当前重复计数','horizon_repetition_current'],['终局当前重复计数','final_repetition_current']];return rows.map(([label,key])=>{const x=a.history_process[key],y=b.history_process[key];return `<tr><td>${label}</td><td>${integer(x.support)}</td><td>${num(x.mean)}</td><td>${integer(y.support)}</td><td>${num(y.mean)}</td></tr>`}).join('')}
function reasonRows(a,b){const keys=[...new Set([...Object.keys(a.outcome_reasons||{}),...Object.keys(b.outcome_reasons||{})])].sort();if(!keys.length)return '<tr><td>暂无规则终局</td><td>0</td><td>0</td></tr>';return keys.map(key=>`<tr><td>${esc(key)}</td><td>${integer(a.outcome_reasons[key]||0)}</td><td>${integer(b.outcome_reasons[key]||0)}</td></tr>`).join('')}
function precisionBlock(x){if(!x.start_clustered_primary.support)return `<div class="panel"><h2>起点聚类精度</h2><p class="help">等待同一起点的候选执白、执黑两个颜色单元都形成完整 v3/v4 配对。</p></div>`;const p=x.start_clustered_primary,iv=p.interval||[null,null],dist=Object.entries(p.distribution||{}).sort((a,b)=>Number(a[0])-Number(b[0])).map(([v,n])=>`<tr><td>${pp(Number(v))}</td><td>${integer(n)}</td></tr>`).join(''),budgets=x.fixed_width_budgets.map(row=>`<tr><td>${pp(row.target_half_width)}</td><td>${integer(row.starts)}</td><td>${integer(row.games)}</td></tr>`).join('');return `<div class="panel"><h2>起点聚类精度与差值分布</h2><p class="help">先在每个起点内平均候选执白、执黑的两个差值，再跨独立起点计算工程区间；颜色单元不能当成独立样本。预算只是用已观测标准差做的固定半宽说明，不会自动扩展本次 39 起点合同。</p><div class="two"><table><thead><tr><th>两色平均差</th><th>起点</th></tr></thead><tbody>${dist}</tbody></table><table><thead><tr><th>目标半宽</th><th>估计起点</th><th>估计局数</th></tr></thead><tbody>${budgets}</tbody></table></div></div>`}
function render(payload){const app=document.getElementById('app');if(!payload.available){app.innerHTML=`<div class="empty"><div class="panel"><h1>v3/v4 阶段过程确认</h1><p class="sub">${esc(payload.message)}</p><div class="notice">没有精确计划和授权时，网页只显示“未启动”，不会预填或推测结果。</div></div></div>`;return}const r=payload.report,a=r.by_candidate[C.v3],b=r.by_candidate[C.v4],p=r.paired.primary_start_clustered_108_ply_survival_v4_minus_v3,iv=p.interval||[null,null],active=payload.progress.active_seconds;
app.innerHTML=`<h1>NMM_LLM · retained-v3 / no-refresh-v4 阶段过程确认</h1><div class="sub">${esc(payload.identities.diagnostic_id)} · <span class="badge">${esc(payload.status)}</span></div><div class="notice"><b>固定项目可见语料的过程确认，不是 held-out 棋力评测。</b> 结果不能归因 refresh，不能用于晋级、发布或释放。</div>
<div class="grid">${card('完成进度',`${payload.progress.completed_games} / ${payload.progress.expected_games}`,payload.progress.current_stage?`game ${Number(payload.progress.current_game_ordinal)+1} · ${payload.progress.current_stage} ply ${payload.progress.current_stage_ply}`:'当前无在途对局')}${card('完整起点',`${r.paired.start_units_complete} / ${r.paired.start_units_expected}`,`${r.paired.matched_colour_units_complete} / ${r.paired.matched_colour_units_expected} 个颜色配对`)}${card('主差值 v4 − v3',pp(p.mean),iv[0]==null?'等待完整起点':`工程区间 ${pp(iv[0])} … ${pp(iv[1])}`)}${card('主判决',decisionText(p.decision),`半宽 ${pp(p.half_width)}；门限 10.00pp`)}${card('v3: 相对 108 手仍在进行',pct(a.horizon_108_post_start.survival_rate),`${a.horizon_108_post_start.survived} / ${a.games} 局`)}${card('v4: 相对 108 手仍在进行',pct(b.horizon_108_post_start.survival_rate),`${b.horizon_108_post_start.survived} / ${b.games} 局`)}${card('活动用时',active==null?'—':num(active/60)+' min','只计 evaluator active time；上限 2 h')}${card('报告身份',r.result_identity?esc(r.result_identity.slice(0,12)):'—','实时从规范账本独立复算')}</div>
<div class="panel"><h2>相对 108 手 continuation survival</h2><p class="help">从每个冻结历史起点再走 108 个完整逻辑手后，严格裁判仍未终局。它不是和棋、不是胜率，也不预测最终结果；不同起点的绝对手数不同。</p><div class="bars">${bar('retained-v3 refresh-50',a.horizon_108_post_start.survival_rate)}${bar('retained-v4 no-refresh',b.horizon_108_post_start.survival_rate,'v4')}</div></div>
${precisionBlock(payload.precision)}
<div class="panel"><h2>按起始阶段分层</h2><p class="help">分母是各阶段已经完成的候选对局数；placement / movement / flying 的固定支持分别来自 18 / 14 / 7 个起点。</p><table><thead><tr><th>阶段</th><th>v3 局数</th><th>v3 存活率</th><th>v4 局数</th><th>v4 存活率</th></tr></thead><tbody>${phaseRows(r.by_phase)}</tbody></table></div>
<div class="two"><div class="panel"><h2>无吃子与重复过程</h2><p class="help">每一行都显示自己的支持数；窗口行只含到达相对 108 手的局。严格无吃子/三次重复历史由 Sanmill 裁判持有，不能由 Malom 棋盘值替代。</p><table><thead><tr><th>指标</th><th>v3 n</th><th>v3 均值</th><th>v4 n</th><th>v4 均值</th></tr></thead><tbody>${processRows(a,b)}</tbody></table></div><div class="panel"><h2>规则终止原因</h2><p class="help">1,536 post-start 是故障安全 cap；命中时记 incomplete，绝不转成和棋。</p><table><thead><tr><th>原因</th><th>v3</th><th>v4</th></tr></thead><tbody>${reasonRows(a,b)}</tbody></table></div></div>
<div class="two"><div class="panel"><h2>长度（post-start）</h2><table><thead><tr><th>候选</th><th>支持</th><th>均值</th><th>中位</th><th>P90</th><th>最大</th></tr></thead><tbody>${[[a,'v3'],[b,'v4']].map(([x,label])=>`<tr><td>${label}</td><td>${x.lengths.post_start.support}</td><td>${num(x.lengths.post_start.mean)}</td><td>${num(x.lengths.post_start.median,0)}</td><td>${num(x.lengths.p90_post_start,0)}</td><td>${integer(x.lengths.post_start.max)}</td></tr>`).join('')}</tbody></table></div><div class="panel"><h2>Malom 候选动作过程</h2><p class="help">query coverage 的分母是候选回合；保值/降级率的分母仅是可查询候选回合。Malom 是 history-free 位置理论值。</p><table><thead><tr><th>候选</th><th>候选回合</th><th>覆盖率</th><th>保值率*</th><th>降级率*</th></tr></thead><tbody>${[[a,'v3'],[b,'v4']].map(([x,label])=>`<tr><td>${label}</td><td>${integer(x.candidate_malom_moves.candidate_turns)}</td><td>${pct(x.candidate_malom_moves.query_coverage)}</td><td>${pct(x.candidate_malom_moves.preserving_rate_given_queryable)}</td><td>${pct(x.candidate_malom_moves.downgrade_rate_given_queryable)}</td></tr>`).join('')}</tbody></table></div></div>
<div class="two"><div class="panel"><h2>相对窗口 Malom 理论 W/D/L</h2><p class="help">仅含到达窗口的快照，从候选视角投影；它不携带三次重复与无吃子历史，不是严格终局裁定。</p><table><thead><tr><th>候选</th><th>快照</th><th>可查</th><th>W</th><th>D</th><th>L</th></tr></thead><tbody>${[[a,'v3'],[b,'v4']].map(([x,label])=>{const m=x.malom_at_horizon_candidate_perspective;return `<tr><td>${label}</td><td>${m.snapshot_support}</td><td>${m.queryable}</td><td>${m.wins}</td><td>${m.draws}</td><td>${m.losses}</td></tr>`}).join('')}</tbody></table></div><div class="panel"><h2>最终严格规则 W/D/L（描述性）</h2><p class="help">只含规则终局，cap 单列剔除。该端点未按稀有胜负设计功效，不能产生 held-out 棋力、等效或晋级主张。</p><table><thead><tr><th>候选</th><th>支持</th><th>胜</th><th>和</th><th>负</th><th>cap</th></tr></thead><tbody>${[[a,'v3'],[b,'v4']].map(([x,label])=>{const w=x.eventual_rules_wdl;return `<tr><td>${label}</td><td>${w.support}</td><td>${w.wins}</td><td>${w.draws}</td><td>${w.losses}</td><td>${w.safety_cap_excluded}</td></tr>`}).join('')}</tbody></table></div></div>
<div class="panel"><h2>指标帮助与身份边界</h2><details open><summary>这些数能回答什么？</summary><p class="help">只能回答两个命名 final route 在这 39 个固定、项目已可见、起点处对两候选数据库均无 D4 命中的阶段历史上，过程指标是否复现。存活率方向本身不表示棋力；W/D/L 是次要描述；差异不能归因 target refresh。</p></details><details><summary>安全吃子与完整排序指标在哪里？</summary><p class="help">它们必须从完整逐手账本做身份绑定的零新对局复算，并显示各自机会分母与 query coverage。在该复算产物存在前，网页不会从普通吃子率或粗 W/D/L 猜测结果。</p></details><code>plan ${esc(payload.identities.plan_identity)} · spec ${esc(payload.identities.spec_identity)} · corpus ${esc(payload.identities.corpus_identity)} · source ${esc(payload.identities.implementation_commit)}</code></div>`}
async function tick(){try{const response=await fetch('/api/diagnostic',{cache:'no-store'});if(!response.ok)throw new Error(await response.text());render(await response.json())}catch(error){document.getElementById('app').innerHTML=`<div class="empty"><div class="panel"><h1>网页读取失败</h1><p class="bad">${esc(error.message)}</p></div></div>`}}
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
                self._send(
                    HTTPStatus.OK,
                    "text/html; charset=utf-8",
                    HTML.encode("utf-8"),
                )
                return
            if route == "/api/diagnostic":
                body = json.dumps(
                    build_payload(self.output_root),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                self._send(
                    HTTPStatus.OK,
                    "application/json; charset=utf-8",
                    body,
                )
                return
            self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found")
        except Exception as exc:
            self._send(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "text/plain; charset=utf-8",
                str(exc).encode("utf-8"),
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    return parser


def main() -> int:
    args = _parser().parse_args()
    handler = type("PhaseProcessHandler", (Handler,), {"output_root": args.output_root})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
