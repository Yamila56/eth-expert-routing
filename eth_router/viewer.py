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
<p>选择一分钟，查看专家证据、加权建议与实际成交。</p>
<p id="error" role="alert"></p>
<section>
<h2>累计净损益 · USDT</h2>
<canvas id="curve" aria-label="累计净损益随时间变化，点击定位"></canvas>
<input id="cursor" type="range" min="0" max="0" value="0" aria-label="回放分钟">
<div class="controls">
<button id="prev" aria-label="上一分钟">←</button><button id="next" aria-label="下一分钟">→</button>
<label>分钟 <input id="minute" type="number" min="1" value="1" size="8"></label>
<button id="switch">下一次换仓</button><strong id="time" class="accent"></strong>
</div>
</section>
<section><h2>仓位决策</h2><div class="metrics" id="decision"></div><p id="explanation"></p></section>
<section><h2>专家证据</h2>
<table><thead><tr><th>专家</th><th>建议</th><th>适用度</th><th>近期净损益 · bp</th><th>换仓频率</th><th>权重</th></tr></thead>
<tbody id="experts"></tbody></table></section>
</main>
<script>
const el = id => document.getElementById(id);
const num = (x, digits=3) => Number(x).toFixed(digits);
const names = ["趋势 1","趋势 2","趋势 3","波动 1","波动 2","波动 3"];
const holding = n => n===1 ? "持有" : "空仓";
async function start(){
  const blob = Uint8Array.from(atob("__PAYLOAD__"), c=>c.charCodeAt(0));
  const stream = new Blob([blob]).stream().pipeThrough(new DecompressionStream("gzip"));
  const d = await new Response(stream).json(), n=d.action.length;
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
    const previous=d.previous_action[i], actual=d.action[i], q=d.mixed_q[i];
    const proposed=Number(q[1]>q[0]), edge=q[1-previous]-q[previous];
    const fields=[["上一仓位",holding(previous)],["加权建议",holding(proposed)],["最终仓位",holding(actual)],
                  ["换仓优势",num(edge,5)],["换仓门槛",num(d.threshold[i],5)],
                  ["手续费",num(d.fee[i],4)],["累计净损益",num(equity[i],4)]];
    el("decision").replaceChildren(...fields.map(([label,value])=>{
      const v=document.createElement("span");v.textContent=label+" "+value;return v;
    }));
    el("explanation").textContent=proposed!==actual ? "换仓门槛使账户维持原仓位。" :
      actual!==previous ? "加权建议通过换仓门槛，账户改变仓位。" : "加权建议维持当前仓位。";
    el("experts").replaceChildren(...names.map((name,j)=>{
      const e=d.evidence[i].slice(j*5,j*5+5), v=d.votes[i][j];
      const cells=[name,v===0?"持平":v>0?"持有":"空仓",e[4] ? num(e[3]*100,1)+"%" : "历史不足",
                   num(e[0],2),num(e[1]*100,2)+"%",num(d.weights[i][j]*100,2)+"%"];
      const tr=document.createElement("tr");
      for(const value of cells){const td=document.createElement("td");td.textContent=value;tr.append(td)}
      return tr;
    }));
    draw();
  }
  el("cursor").oninput=e=>select(+e.target.value);
  el("minute").onchange=e=>select(+e.target.value-1);
  el("prev").onclick=()=>select(i-1);el("next").onclick=()=>select(i+1);
  el("switch").onclick=()=>{const j=d.action.findIndex((a,k)=>k>i&&a!==d.previous_action[k]);if(j>=0)select(j)};
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
        if evidence.shape[1] != 30 or not len(evidence):
            raise ValueError("Expected a nonempty six-expert replay")
        data = {key: trace[key].tolist() for key in
                ("action", "previous_action", "mixed_q", "threshold", "fee", "reward", "weights", "votes", "evidence")}
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
