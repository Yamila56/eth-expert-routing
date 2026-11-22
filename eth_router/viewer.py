"""Export a self-contained, minute-by-minute expert replay."""
import argparse
import base64
import gzip
import json
from pathlib import Path

import numpy as np


PAGE = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ETH 专家决策回放</title>
<style>
:root{color-scheme:dark}body{font:15px system-ui,sans-serif;background:#0d1620;color:#ecf2f6;margin:0}
main{max-width:1030px;margin:auto;padding:28px}h1{font-size:28px}h2{font-size:18px}
section{background:#162332;border:1px solid #334656;border-radius:12px;padding:18px;margin:16px 0}
.controls,.metrics{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
input[type=range],canvas{width:100%;display:block;margin:14px 0}canvas{height:185px;cursor:crosshair}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:9px;border-bottom:1px solid #334656}
th:first-child,td:first-child{text-align:left}small,p{color:#b4c6d5}.accent{color:#64dac0}
button,input{font:inherit}button{padding:4px 10px}#error{color:#ffae9c}
@media(max-width:700px){main{padding:14px}section{overflow:auto}th,td{padding:6px;white-space:nowrap}}
</style>
<main>
<h1>ETH 专家决策回放</h1>
<p>选择一分钟，查看专家证据、策略估值与模拟成交。</p>
<p id="error" role="alert"></p>
<section>
<h2>累计净损益 · USDT</h2>
<canvas id="curve" aria-label="累计净损益随时间变化，点击定位"></canvas>
<input id="cursor" type="range" min="0" max="0" value="0" aria-label="回放分钟">
<div class="controls">
<button id="prev" aria-label="上一分钟">←</button><button id="next" aria-label="下一分钟">→</button>
<label>分钟 <input id="minute" type="number" min="1" value="1" size="8"></label>
<button id="switch">下一次换仓</button><button id="disagree" hidden>下一次专家分歧</button><strong id="time" class="accent"></strong>
</div>
</section>
<section><h2>仓位决策</h2><div class="metrics" id="decision"></div><p id="explanation"></p></section>
<section><h2>专家证据</h2>
<label id="window-label" hidden>评价窗口 <select id="outcome-window"></select></label>
<table><thead><tr id="expert-header"><th>专家</th><th>建议</th><th>适用度</th><th>近期净损益 · bp</th><th>换仓频率</th><th>权重</th></tr></thead>
<tbody id="experts"></tbody></table></section>
</main>
<script>
const el = id => document.getElementById(id);
const num = (x, digits=3) => Number(x).toFixed(digits);
const legacyNames = ["趋势 1","趋势 2","趋势 3","波动 1","波动 2","波动 3"];
const holding = n => n===1 ? "持有" : "空仓";
async function start(){
  const blob = Uint8Array.from(atob("__PAYLOAD__"), c=>c.charCodeAt(0));
  const stream = new Blob([blob]).stream().pipeThrough(new DecompressionStream("gzip"));
  const d = await new Response(stream).json(), n=d.action.length;
  const horizonReplay=Boolean(d.half_lives);
  const names=horizonReplay ? d.half_lives.map(h=>h+" 分钟") : legacyNames;
  const windows=d.outcome_windows;
  const evidenceWidth=windows ? 2+3*windows.length : 5;
  const picker=el("outcome-window");
  if(windows){
    el("window-label").hidden=false;
    for(const [index,minutes] of windows.entries()){
      const option=document.createElement("option");
      option.value=index;option.textContent=minutes+" 分钟";picker.append(option);
    }
    picker.value=windows.indexOf(30)>=0 ? windows.indexOf(30) : 0;
  }
  if(horizonReplay){
    el("disagree").hidden=false;
    el("expert-header").replaceChildren(...["折扣半衰期","建议","标准化价值差","近期净损益 · bp",
      "换仓频率","建议平均得失 · bp","建议胜率","已结算样本"].map(label=>{
        const th=document.createElement("th");th.textContent=label;return th;
      }));
  }
  let i=0, cumulative=0, min=0, max=0;
  const equity=d.reward.map(x=>{cumulative+=x;min=Math.min(min,cumulative);max=Math.max(max,cumulative);return cumulative});
  el("cursor").max=n-1; el("minute").max=n;
  function draw(){
    const c=el("curve"), width=c.clientWidth||600, height=c.clientHeight;
    c.width=width; c.height=height;
    const ctx=c.getContext("2d"), y=v=>height-10-(v-min)/(max-min||1)*(height-20);
    ctx.strokeStyle="#405569";ctx.beginPath();ctx.moveTo(0,y(0));ctx.lineTo(width,y(0));ctx.stroke();
    ctx.strokeStyle="#64dac0";ctx.beginPath();
    for(let x=0;x<width;x++){
      const v=y(equity[Math.round(x/Math.max(1,width-1)*(n-1))]);
      if(x===0)ctx.moveTo(x,v);else ctx.lineTo(x,v);
    }
    ctx.stroke(); const p=i/Math.max(1,n-1)*(width-1);
    ctx.strokeStyle="#ffd28b";ctx.beginPath();ctx.moveTo(p,0);ctx.lineTo(p,height);ctx.stroke();
  }
  function select(value){
    i=Math.max(0,Math.min(n-1,Math.trunc(value)||0));
    el("cursor").value=i;el("minute").value=i+1;el("time").textContent=d.time[i];
    const previous=d.previous_action[i], actual=d.action[i], q=horizonReplay ? d.action_q[i] : d.mixed_q[i];
    const proposed=Number(q[1]>q[0]), edge=q[1-previous]-q[previous];
    const fields=[["上一仓位",holding(previous)],["最终仓位",holding(actual)],["换仓优势",num(edge,5)]];
    if(horizonReplay) {
      const unit=windows ? " · bp" : "";
      fields.push(["空仓价值"+unit,num(q[0],5)],["持仓价值"+unit,num(q[1],5)]);
      if(windows){
        const prior=d.market_q[i], priorEdge=prior[1-previous]-prior[previous];
        fields.push(["市场基线换仓优势 · bp",num(priorEdge,5)],["专家修正 · bp",num(edge-priorEdge,5)],
                    ["模拟成交",d.execution_time[i]]);
      }
    }
    else fields.push(["加权建议",holding(proposed)],["换仓门槛",num(d.threshold[i],5)]);
    fields.push(["手续费",num(d.fee[i],4)],["累计净损益",num(equity[i],4)]);
    el("decision").replaceChildren(...fields.map(([label,value])=>{
      const v=document.createElement("span");v.textContent=label+" "+value;return v;
    }));
    el("explanation").textContent=horizonReplay ? "" : proposed!==actual ? "换仓门槛使账户维持原仓位。" :
      actual!==previous ? "加权建议通过换仓门槛，账户改变仓位。" : "加权建议维持当前仓位。";
    const outcome=windows ? 2+3*Number(picker.value) : 2;
    el("experts").replaceChildren(...names.map((name,j)=>{
      const e=d.evidence[i].slice(j*evidenceWidth,(j+1)*evidenceWidth), v=d.votes[i][j];
      const cells=horizonReplay ?
        [name,holding(v),num(d.margins[i][j],2),num(e[0],2),num(e[1]*100,2)+"%",
         e[outcome+2] ? num(e[outcome],2) : "待结算",
         e[outcome+2] ? num(e[outcome+1]*100,1)+"%" : "待结算",e[outcome+2]] :
        [name,v===0?"持平":v>0?"持有":"空仓",e[4] ? num(e[3]*100,1)+"%" : "历史不足",
                   num(e[0],2),num(e[1]*100,2)+"%",num(d.weights[i][j]*100,2)+"%"];
      const tr=document.createElement("tr");
      for(const value of cells){const td=document.createElement("td");td.textContent=value;tr.append(td)}
      return tr;
    }));
    draw();
  }
  picker.onchange=()=>select(i);
  el("cursor").oninput=e=>select(+e.target.value);
  el("minute").onchange=e=>select(+e.target.value-1);
  el("prev").onclick=()=>select(i-1);el("next").onclick=()=>select(i+1);
  el("switch").onclick=()=>{const j=d.action.findIndex((a,k)=>k>i&&a!==d.previous_action[k]);if(j>=0)select(j)};
  el("disagree").onclick=()=>{
    const j=d.votes.findIndex((votes,k)=>k>i&&votes.some(v=>v!==votes[0]));
    if(j>=0)select(j);
  };
  el("curve").onclick=e=>select(Math.round(e.offsetX/el("curve").clientWidth*(n-1)));
  window.addEventListener("resize",draw);
  select(0);
}
start().catch(e=>{el("error").textContent="无法加载回放："+e.message});
</script></html>
"""


def export(trace_path, output):
    with np.load(trace_path) as trace:
        evidence = trace["evidence"]
        if not len(evidence):
            raise ValueError("Expected a nonempty expert replay")
        keys = ("action", "previous_action", "fee", "reward", "votes")
        if "half_lives" in trace:
            keys += ("half_lives", "action_q", "margins")
            evidence = evidence.reshape(len(evidence), -1)
        else:
            keys += ("mixed_q", "threshold", "weights")
        data = {key: trace[key].tolist() for key in keys}
        if "outcome_windows" in trace:
            data["outcome_windows"] = trace["outcome_windows"].tolist()
            data["market_q"] = trace["market_q"].tolist()
            data["execution_time"] = np.datetime_as_string(trace["execution_timestamp"], unit="m").tolist()
        data["evidence"] = evidence.tolist()
        data["time"] = np.datetime_as_string(trace["timestamp"], unit="m").tolist()
    payload = base64.b64encode(gzip.compress(json.dumps(data, separators=(",", ":"), allow_nan=False).encode()))
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    tmp.write_text(PAGE.replace("__PAYLOAD__", payload.decode("ascii")), encoding="utf-8")
    tmp.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.trace, args.output)
