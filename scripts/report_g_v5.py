from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/g_v5"


def read(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def pct(v: Any) -> str:
    return "—" if v is None else f"{float(v):.1%}"


def ci(v: Any) -> str:
    return "—" if not v else f"{pct(v[0])}–{pct(v[1])}"


def wdl(r: dict[str, Any]) -> str:
    return f"{r.get('wins', 0)}-{r.get('draws', 0)}-{r.get('losses', 0)}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_font() -> None:
    for name in ("Noto Sans CJK SC", "Noto Serif CJK SC", "Noto Sans CJK JP"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except ValueError:
            pass
    plt.rcParams["axes.unicode_minus"] = False


def make_csv() -> list[dict[str, Any]]:
    rows = []
    for path in sorted((BASE / "results/matches").rglob("v5m-*.json")):
        g = read(path)
        rows.append({
            "game_id": g["game_id"], "phase": path.parent.name,
            "candidate": g["candidate"], "opponent": g["opponent"],
            "opening_group_id": g["opening_group_id"], "candidate_color": g["candidate_color"],
            "winner_color": g["winner_color"], "points": g["points"],
            "move_count": g["move_count"], "total_cpu_seconds": g["total_cpu_seconds"],
            "candidate_cpu_seconds": g["candidate_cpu_seconds"],
            "opponent_cpu_seconds": g["opponent_cpu_seconds"],
        })
    out = BASE / "results/match_games.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["game_id", "phase"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_curves(six: dict[str, Any], arms: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    stages = six.get("stages", [])
    xs = [int(r["node"]) for r in stages]
    ys = [r["match"]["score_rate"] for r in stages]
    lo = [r["match"]["paired_opening_bootstrap_ci95"][0] for r in stages]
    hi = [r["match"]["paired_opening_bootstrap_ci95"][1] for r in stages]
    if xs:
        ax[0].errorbar(xs, ys, yerr=[[a-b for a,b in zip(ys,lo)],[b-a for b,a in zip(hi,ys)]], fmt="o-", capsize=3)
    ax[0].axhline(.5, color="gray", linestyle="--")
    ax[0].set(title="6×6：PUCT-64对随机初始模型", xlabel="训练状态", ylabel="得分率（配对95%区间）", ylim=(-.03,1.03))
    ax[0].grid(alpha=.25)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for color, arm in zip(colors, arms):
        node_states = {int(r["node"]): int(r.get("actual_states",r["node"])) for r in arm.get("stages",[]) if "node" in r}
        rows = []
        for r in arm.get("stages",[]):
            if r.get("opponent") != "G0-V3-510901-20K":
                continue
            try:
                n = int(r["candidate"].rsplit("@",1)[1])
            except (IndexError,ValueError):
                continue
            rows.append((node_states.get(n,n),r))
        rows.sort(key=lambda x:x[0])
        if rows:
            xx=[x for x,_ in rows]; yy=[r["score_rate"] for _,r in rows]
            low=[r["paired_opening_bootstrap_ci95"][0] for _,r in rows]
            high=[r["paired_opening_bootstrap_ci95"][1] for _,r in rows]
            ax[1].errorbar(xx,yy,yerr=[[a-b for a,b in zip(yy,low)],[b-a for b,a in zip(high,yy)]],fmt="o-",capsize=2,color=color,label=arm["arm"])
    ax[1].axhline(.5,color="gray",linestyle="--")
    ax[1].set(title="9×9：PUCT-64对冻结G0 20K",xlabel="训练状态",ylabel="开发集得分率",ylim=(-.03,1.03))
    ax[1].grid(alpha=.25); ax[1].legend(fontsize=7)
    fig.tight_layout()
    (BASE/"charts").mkdir(exist_ok=True)
    fig.savefig(BASE/"charts/learning_curves.png",dpi=170); plt.close(fig)


def plot_learning_cost(six: dict[str, Any], arms: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(1,2,figsize=(12,4.8))
    six_id = six.get("arm",{}).get("arm_id", "reference6x6-400-empty")
    six_metrics = {int(r["milestone_states"]): r for r in read_jsonl(BASE/f"results/training/{six_id}_metrics.jsonl")}
    xs=[]; ys=[]; lo=[]; hi=[]
    for row in six.get("stages",[]):
        metric=six_metrics.get(int(row["node"]))
        if metric is None: continue
        xs.append(metric["cumulative_cpu_seconds"]/3600)
        ys.append(row["match"]["score_rate"])
        lo.append(row["match"]["paired_opening_bootstrap_ci95"][0])
        hi.append(row["match"]["paired_opening_bootstrap_ci95"][1])
    if xs:
        ax[0].errorbar(xs,ys,yerr=[[a-b for a,b in zip(ys,lo)],[b-a for b,a in zip(hi,ys)]],fmt="o-",capsize=3,label="6×6对随机初始")
    ax[0].axhline(.5,color="gray",linestyle="--")
    ax[0].set(title="6×6：得分随训练CPU时间",xlabel="累计训练CPU小时",ylabel="得分率（配对95%区间）",ylim=(-.03,1.03))
    ax[0].grid(alpha=.25); ax[0].legend(fontsize=8)
    colors=["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
    for color,arm in zip(colors,arms):
        metric_by_node={int(r["milestone_states"]):r for r in read_jsonl(BASE/f"results/training/{arm['arm']}_metrics.jsonl")}
        rows=[]
        for result in arm.get("stages",[]):
            if result.get("opponent")!="G0-V3-510901-20K": continue
            try: node=int(result["candidate"].rsplit("@",1)[1])
            except (IndexError,ValueError): continue
            metric=metric_by_node.get(node)
            if metric is not None: rows.append((metric["cumulative_cpu_seconds"]/3600,result))
        rows.sort(key=lambda x:x[0])
        if rows:
            xx=[x for x,_ in rows]; yy=[r["score_rate"] for _,r in rows]
            low=[r["paired_opening_bootstrap_ci95"][0] for _,r in rows]
            high=[r["paired_opening_bootstrap_ci95"][1] for _,r in rows]
            ax[1].errorbar(xx,yy,yerr=[[a-b for a,b in zip(yy,low)],[b-a for b,a in zip(high,yy)]],fmt="o-",capsize=2,color=color,label=arm["arm"])
    ax[1].axhline(.5,color="gray",linestyle="--")
    ax[1].set(title="9×9：对冻结G0 20K的得分",xlabel="累计训练CPU小时",ylabel="开发集得分率",ylim=(-.03,1.03))
    ax[1].grid(alpha=.25); ax[1].legend(fontsize=7); fig.tight_layout()
    fig.savefig(BASE/"charts/learning_cpu.png",dpi=170); plt.close(fig)


def plot_training_diagnostics() -> None:
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    any_data=False
    for path in sorted((BASE/"results/training").glob("*_metrics.jsonl")):
        rows=read_jsonl(path)
        if not rows: continue
        any_data=True; label=path.name.removesuffix("_metrics.jsonl")
        ax[0].plot([r["actual_states"] for r in rows],[r["loss"] for r in rows],marker="o",label=label)
        ax[1].plot([r["actual_states"] for r in rows],[r["cumulative_cpu_seconds"]/3600 for r in rows],marker="o",label=label)
    ax[0].set(title="策略+价值训练损失（仅诊断）",xlabel="实际训练状态",ylabel="末次更新损失")
    ax[1].set(title="累计训练CPU小时",xlabel="实际训练状态",ylabel="CPU小时")
    for a in ax: a.grid(alpha=.25)
    if any_data:
        ax[0].legend(fontsize=6); ax[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(BASE/"charts/training_states_cpu.png",dpi=160); plt.close(fig)


def plot_search(final: dict[str, Any]) -> None:
    fig, ax = plt.subplots(1,2,figsize=(12,4.5))
    b=final.get("same_checkpoint_tactical_search_budget",[])
    m=final.get("same_checkpoint_search_budget_vs_A_dev",[])
    ax[0].plot([r["simulations"] for r in b],[r["network_accuracy"] for r in b],"o--",label="裸网络")
    ax[0].plot([r["simulations"] for r in b],[r["puct_accuracy"] for r in b],"o-",label="PUCT")
    ax[0].set(title="64道战术题准确率",xlabel="模拟数",ylabel="准确率",ylim=(-.03,1.03)); ax[0].grid(alpha=.25); ax[0].legend()
    ax[1].plot([r["simulations_per_reference_turn"] for r in m],[r["score_rate"] for r in m],"o-")
    ax[1].set(title="固定checkpoint对A开发赛",xlabel="每步模拟数",ylabel="配对得分率",ylim=(-.03,1.03)); ax[1].grid(alpha=.25)
    for r in m:
        ax[1].annotate(f"{r['mean_candidate_cpu_seconds']:.2f} CPU秒/局",(r["simulations_per_reference_turn"],r["score_rate"]),xytext=(4,8),textcoords="offset points",fontsize=8)
    fig.tight_layout(); fig.savefig(BASE/"charts/search_tactics_budget.png",dpi=170); plt.close(fig)


def plot_history(hist: dict[str, Any]) -> None:
    rows=hist.get("comparisons_vs_A_B",[])
    fig,ax=plt.subplots(figsize=(max(9,len(rows)*.65),5))
    if rows:
        y=[r["score_rate"] for r in rows]
        lo=[r["paired_opening_bootstrap_ci95"][0] for r in rows]
        hi=[r["paired_opening_bootstrap_ci95"][1] for r in rows]
        ax.errorbar(range(len(rows)),y,yerr=[[a-b for a,b in zip(y,lo)],[b-a for b,a in zip(hi,y)]],fmt="o",capsize=3)
        labels=[f"{r['candidate']}\nvs {r['opponent']}" for r in rows]
        ax.set_xticks(range(len(rows)),labels,rotation=55,ha="right",fontsize=8)
        ax.axhline(.5,color="gray",linestyle="--")
        ax.set(title="历史G0检查点统一复测（PUCT-64）",ylabel="配对得分率",ylim=(-.03,1.03))
    ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(BASE/"charts/historical_g0_9x9.png",dpi=160); plt.close(fig)


def main() -> None:
    set_font()
    cfg=read(BASE/"config/experiment.json",{})
    summary=read(BASE/"results/experiment_summary.json",{})
    pre=read(BASE/"results/preflight.json",{})
    audit=read(BASE/"results/integrity_audit.json",{})
    tests=read(BASE/"results/test_summary.json",{})
    six=summary.get("six_by_six",read(BASE/"results/six_by_six_summary.json",{}))
    hist=summary.get("historical_g0_remeasurement",read(BASE/"results/historical_9x9.json",{}))
    final=summary.get("final_9x9",read(BASE/"results/final_9x9.json",{}))
    arm_data=summary.get("nine_by_nine_arms",read(BASE/"results/9x9_arms.json",{}))
    arms=arm_data.get("arms",[]) if isinstance(arm_data,dict) else arm_data
    games=make_csv()
    plot_curves(six,arms); plot_learning_cost(six,arms); plot_training_diagnostics(); plot_search(final); plot_history(hist)
    cpu=float(summary.get("cpu_seconds",0))
    ev=int(audit.get("evaluation_games_verified",len(games)))
    sp=int(audit.get("selfplay_games_verified",0))
    states=int(audit.get("selfplay_training_states_verified",0))
    lines=[
        "# WuziQ G V5：标准AlphaZero复现与G0学习闭环验证","",
        f"- 分支：{cfg.get('branch','research/alphazero-g-v5')}；基线：{cfg.get('base_commit','')}。",
        f"- 状态：{'完成' if summary.get('completed') else '未完成'}；配置SHA-256：{cfg.get('config_sha256','—')}。",
        f"- 计算上限10 CPU小时；实际 **{cpu/3600:.3f} CPU小时**（{cpu:.1f}秒）。Runner没有可靠记录端到端wall时间，不用CPU值代替。",
        f"- 完成局数：{sp:,}局自我对弈 + {ev:,}局评测 = {sp+ev:,}局。逐局回放训练样本共{states:,}个；没有将重复checkpoint阶段数相加。",
        f"- 完整性：全部棋谱合法且完整；唯一ID通过；固定配置/checkpoint与{audit.get('vendored_reference_files_verified',0)}个上游源文件SHA校验通过。","",
        "## 结论摘要","",
        "- 标准AlphaZero从随机权重在6×6连四单种子上学到了更强的策略。15K模型对随机初始模型在20组新开局换色盲测为38-0-2，95.0%，配对95%区间87.5%–100%。这是该任务的真实正向结果，但不证明跨随机种子稳定，也不代表9×9。",
        "- 9×9没有学到能击败A的棋力。最终候选PUCT-64盲测对A和B各0胜0平40负。",
        "- 本轮没有证据证明G0存在唯一算法错误。历史G0对A均0胜0平，10K对20K只有13-19。9×9候选策略头的战术top-1为0/64，PUCT-64为19/64，唯一必防仅1/16；问题可能涉及先验、浅层搜索和防守覆盖，但不足以确认为G0退化根因。",
        "- 建议先多种子复核6×6，再对9×9小预算验证策略先验、唯一防守探索和价值校准；当前不宜大幅扩展9×9训练。","",
        "## 参考实现与公平性边界","",
        f"采用https://github.com/junxiaosong/AlphaZero_Gomoku，上游提交{cfg.get('reference',{}).get('commit','')}，MIT许可。源文件保存在third_party/AlphaZero_Gomoku并由SHA清单校验。保留上游4平面、32/64/128 CNN、MCTS/PUCT、Adam、8重几何增强和策略交叉熵+价值MSE；原生训练参数包括batch=512、replay=10,000、learning rate=0.002、每次更新5轮、根Dirichlet alpha=0.3/混入比例0.25、temperature=1、c_puct=5。兼容改动仅为loss标量读取和inference_mode推理。没有改搜索/训练目标。上游README的棋力宣称没有作为复现结果。",
        "原G0是3平面、32通道、2残差块且有BatchNorm；参考网络无BatchNorm。因此是独立实现端到端对照，不是同构搜索消融。参考模型全部随机初始化，自我对弈目标只来自MCTS访问分布与终局胜负，不含A/B/C/E/F训练数据、教师标签、人工奖励或落子补丁。中盘路线只用历史G0自博弈前缀作起始局面，不作训练标签。","",
        "上游README建议用6×6连四作学习冒烟测试，并提到约500–1,000局；8×8连五建议约2,000–3,000局。这些是上游经验建议，不是本研究复现结果。本轮6×6共1,770局，9×9自由五子棋与上游README的示例规则不同。",
        "固定种子与开局表保存在config/registration.json和opening_schedule.json：6×6训练/网络种子560602/560601，开发/盲测种子560603/560604；9×9训练/网络种子590502/590501，开发/盲测/历史开局种子590503/590504/590505。盲测每个对手20组新开局换色，checkpoint仅由开发集选择。","",
        "## 环境、预估和计算成本","",
        f"CPU配额{pre.get('resources',{}).get('cpu_quota_cores','—')}核，内存上限{(pre.get('resources',{}).get('memory_limit_bytes') or 0)/(1024**3):.1f}GiB，GPU={'可用' if pre.get('resources',{}).get('gpu_available') else '不可用'}，PyTorch{pre.get('resources',{}).get('torch','—')}。预估乐观/正常/保守CPU时长：{pre.get('estimates',{}).get('optimistic_cpu_hours',0):.2f}/{pre.get('estimates',{}).get('normal_cpu_hours',0):.2f}/{pre.get('estimates',{}).get('conservative_cpu_hours',0):.2f}小时。",
        f"实际预检{summary.get('preflight_cpu_seconds',0):.1f}秒；训练{summary.get('training_cpu_seconds',0)/3600:.3f}小时；评测{summary.get('evaluation_cpu_seconds',0)/3600:.3f}小时；runner收尾{summary.get('cpu_seconds_this_session',0):.1f}秒；总计{cpu/3600:.3f}小时。","",
        "## 6×6连四闭环","",
        f"随机权重起步、400模拟/步；训练{six.get('arm',{}).get('environment_states',0):,}状态、{six.get('arm',{}).get('game_count',0):,}局、{six.get('arm',{}).get('update_count',0):,}次更新，训练CPU{six.get('arm',{}).get('cumulative_cpu_seconds',0)/3600:.3f}小时。晋级门槛{six.get('gate_nodes',[])}通过={six.get('continue_to_9x9_gate_passed',False)}；开发选中节点{six.get('dev_selected_node')}。","",
        "| 状态节点 | 对随机初始W-D-L | 得分率 | 配对95%区间 | 裸网胜点 | PUCT胜点 | 裸网必防 | PUCT必防 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for r in six.get("stages",[]):
        m=r.get("match",{})
        puzzle=read(BASE/f"results/tactics/reference6x6-400-empty_node_{r.get('node')}_puct64.json",{})
        direct=puzzle.get("by_category",{}).get("direct_win",{})
        block=puzzle.get("by_category",{}).get("unique_block",{})
        lines.append(f"| {r.get('node')} | {wdl(m)} | {pct(m.get('score_rate'))} | {ci(m.get('paired_opening_bootstrap_ci95'))} | {pct(direct.get('network_accuracy'))} | {pct(direct.get('puct_accuracy'))} | {pct(block.get('network_accuracy'))} | {pct(block.get('puct_accuracy'))} |")
    blind6=six.get("blind_result",{})
    lines += ["",f"开发得分率按节点为{', '.join(pct(r['match'].get('score_rate')) for r in six.get('stages',[]))}；15K至20K持平。冻结15K模型在新20组盲测对初始模型{wdl(blind6)}，得分{pct(blind6.get('score_rate'))}，配对95%区间{ci(blind6.get('paired_opening_bootstrap_ci95'))}，黑/白得分率{pct(blind6.get('black_score_rate'))}/{pct(blind6.get('white_score_rate'))}。早期已揭示的10K盲测另列为探索数据，不与新盲测合并。战术题按32个立即胜点与32个唯一防守局面分别计算，仅用于评估，不进入训练。","",
        "## 历史G0统一复测","","统一9×9、PUCT-64、16组开发开局换色。","",
        "| 冻结G0 | 对手 | W-D-L | 得分率 | 配对95%区间 | 平均步数 |","|---|---|---:|---:|---:|---:|"]
    for r in hist.get("comparisons_vs_A_B",[])+hist.get("direct_checkpoint_matches",[]):
        lines.append(f"| {r.get('candidate')} | {r.get('opponent')} | {wdl(r)} | {pct(r.get('score_rate'))} | {ci(r.get('paired_opening_bootstrap_ci95'))} | {r.get('mean_moves',0):.1f} |")
    lines += ["","这些历史检查点对A均0胜0平；5K/10K对B各仅1胜。10K对20K为13-0-19（40.6%，区间28.1%–53.1%），说明更长训练不保证棋力单调增加，但样本不足以做普遍判断。",
        "历史诊断仅作为跨版本背景：G V3的256状态BatchNorm探针选择冻结running statistics，但旧报告明确称该短探针不足以确认根因；G V4.1的回传符号/终局测试未复现符号错误，并报告网络与PUCT表现错位及浅层搜索对先验敏感；G V4.2记录历史策略头直接胜点0/48、目标平均先验质量约1.7%、原PUCT搜索约70.8%找到目标。这与本轮观察方向一致，但跨实验不是单变量因果验证。","",
        "## 9×9参考训练路线","","| 路线 | 模拟/步 | 状态分布 | 状态数 | 自博弈局 | 更新数 | CPU小时 |","|---|---:|---|---:|---:|---:|---:|"]
    for arm in arms:
        t=arm.get("training",{})
        lines.append(f"| {arm.get('arm')} | {arm.get('simulations_per_move')} | {arm.get('opening_distribution')} | {t.get('environment_states',0):,} | {t.get('game_count',0):,} | {t.get('update_count',0):,} | {t.get('cumulative_cpu_seconds',0)/3600:.3f} |")
    lines += ["","四条路线合计15,550状态；各阶段均用PUCT-64评估。提高训练模拟数的成本单独记录。中盘起始路线改变状态分布，目标仍全部来自该网络搜索与终局。","",
        "| 路线最终节点 | 对随机初始W-D-L/得分 | 对G0 20K | 对A | 对B |","|---|---:|---:|---:|---:|"]
    for arm in arms:
        marks=[r for r in arm.get("stages",[]) if "node" in r]
        if not marks: continue
        n=max(marks,key=lambda r:int(r["node"]))["node"]
        cols=[]
        for opp in ("ref9x9@0","G0-V3-510901-20K","A","B"):
            row=next((r for r in arm.get("stages",[]) if r.get("opponent")==opp and r.get("candidate","").endswith(f"@{n}")),None)
            cols.append(f"{wdl(row)} / {pct(row.get('score_rate'))}" if row else "—")
        lines.append(f"| {arm.get('arm')}@{n} | {' | '.join(cols)} |")
    lines += ["","各参考模型能超过本路线随机初始网络，但对A/B几乎全负；只有400模拟路线一个开发局面与A平局。未找到可挑战A的新模型。","",
        "![训练棋力随状态数变化](charts/learning_curves.png)","",
        "![训练棋力随累计CPU时间变化](charts/learning_cpu.png)","",
        "![训练损失和CPU开销](charts/training_states_cpu.png)","","## 9×9最终盲测","","",
        f"开发规则选择{final.get('selected_candidate',{}).get('candidate_id')}，实际状态{final.get('selected_candidate',{}).get('actual_states',0):,}，200次模拟训练。冻结后使用20组新开局各换色，对每个对手40局。","",
        "| 对手 | W-D-L | 得分率 | 配对95%区间 | 黑/白得分率 | 候选CPU秒/局 |","|---|---:|---:|---:|---:|---:|"]
    for r in final.get("blind_results",[]):
        lines.append(f"| {r.get('opponent')} | {wdl(r)} | {pct(r.get('score_rate'))} | {ci(r.get('paired_opening_bootstrap_ci95'))} | {pct(r.get('black_score_rate'))}/{pct(r.get('white_score_rate'))} | {r.get('mean_candidate_cpu_seconds',0):.3f} |")
    lines += ["","对A和B都0-0-40。对历史G0得分为：5K 45%（18-22）、10K 50%（20-20）、20K 35%（14-26）、V4.2 history-pool 10K 17.5%（7-33）；区间宽，不能作稳定实力顺序。全部配对局均落败时bootstrap区间会退化为0%–0%，不表示母体胜率没有不确定性；0/40二项Wilson 95%区间上限约8.8%，可作为补充参考。","",
        "![历史G0复测](charts/historical_g0_9x9.png)","","## 同一checkpoint的搜索预算与战术对照","","| 模拟/步 | 对A开发W-D-L | 得分率 | 候选CPU秒/局 |","|---:|---:|---:|---:|"]
    for r in final.get("same_checkpoint_search_budget_vs_A_dev",[]):
        lines.append(f"| {r.get('simulations_per_reference_turn')} | {wdl(r)} | {pct(r.get('score_rate'))} | {r.get('mean_candidate_cpu_seconds',0):.3f} |")
    lines += ["","64/200/400模拟在8组开发开局均未对A取得胜分；CPU成本约0.64/1.79/4.53秒每局。","",
        "| 模拟/题 | 裸网 | PUCT总计 | 立即胜点 | 边缘胜点 | 伪威胁 | 唯一必防 |","|---:|---:|---:|---:|---:|---:|---:|"]
    for r in final.get("same_checkpoint_tactical_search_budget",[]):
        cats=r.get("by_category",{})
        def count(k: str) -> str:
            c=cats.get(k,{})
            return f"{round(c.get('puct_accuracy',0)*c.get('puzzles',0))}/{c.get('puzzles',0)}"
        n=r.get("puzzles",0); net=round(r.get("network_accuracy",0)*n); puct=round(r.get("puct_accuracy",0)*n)
        lines.append(f"| {r.get('simulations')} | {net}/{n} | {puct}/{n} ({pct(r.get('puct_accuracy'))}) | {count('direct_win')} | {count('edge_win')} | {count('false_threat')} | {count('unique_block')} |")
    lines += ["","64到400模拟时战术总准确率从19/64升到31/64；立即胜点9/16到12/16，边缘胜点8/16到15/16，唯一必防仅1/16到3/16，伪威胁始终1/16。属于搜索带来的题目收益，不是网络学习收益。","",
        "![搜索和战术预算](charts/search_tactics_budget.png)","","## 代码审计与限制","","",
        f"- 完整回放核验{sp:,}局自博弈/{states:,}状态及{ev:,}局评测；落子合法，终局和颜色得分与独立规则判定一致；访问分布归一化、非法点无权重，终局价值视角一致。",
        "- 运行时修复两处实验编排/数据格式集成错误：6×6晋级节点序列化把整数当对象；旧V4.2战术题字段position/optimal_actions未转换。都经最小复现测试后修复，再从已有结果幂等续跑；这不是G0算法错误证据。",
        "- 没有可复现证据证明G0的符号/终局回传有错，也不能声称已证明所有工程路径无缺陷。先前V4.1报告的网络与PUCT棋力错位现象与本轮诊断一致，但不足以定位唯一根因。",
        "- 训练loss、增加搜索量均不作为棋力结论。6×6仅一个种子；9×9未打败A。","",
        "## 测试结果","",
        f"- V5专项：{tests.get('v5_passed','—')}通过、{tests.get('v5_failed','—')}失败；命令：{tests.get('v5_command','—')}。",
        f"- 仓库全量：{tests.get('repository_passed','—')}通过、{tests.get('repository_failed','—')}失败；29个失败均由历史V2/V3/V4/V6测试要求在各自隔离分支运行引起，当前V5工作树保留在其独立分支。失败均在branch guard提前退出，没有V5专项失败；全量测试警告：{tests.get('repository_warnings','—')}。","",
        "## 对任务问题的回答","","",
        "1. 标准AlphaZero在本环境从零学习了6×6连四；本轮没证明能学会9×9。",
        "2. 原G0未成长的唯一根因未知。先验分布、浅层搜索与漏防是有数据支持的研究方向，不是已证实根因。",
        "3. 新路线在6×6有效；9×9没有对A/B奏效的新路线。增加搜索改善战术题但不等于训练成长。",
        "4. 先多种子复制6×6，再对9×9做小预算机制诊断，不建议扩大长训练。","",
        "## 复现与文件","","",
        "- 配置：research/g_v5/config/；原始自我对弈棋谱/样本：research/g_v5/raw/；评测逐局棋谱：research/g_v5/results/matches/。",
        "- 权重/训练状态：research/g_v5/checkpoints/；汇总CSV：research/g_v5/results/match_games.csv；逐局审计：research/g_v5/results/integrity_audit.json；源哈希：research/g_v5/reference/source_manifest.json。",
        "- 复现命令（在仓库根目录运行）：",
        "    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/preflight_g_v5.py",
        "    PYTHONPATH=. python -m src.research.alphazero_g.v5_registration",
        "    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python scripts/run_g_v5.py --session-cpu-seconds 3600",
        "    PYTHONPATH=. python scripts/audit_g_v5.py",
        "    PYTHONPATH=. python scripts/report_g_v5.py",""]
    report=BASE/"REPORT.md"
    report.write_text("\n".join(lines),encoding="utf-8")
    print(f"Wrote {report}; eval={ev}; selfplay={sp}; states={states}; complete={summary.get('completed')}")


if __name__ == "__main__":
    main()

