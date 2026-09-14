const $=s=>document.querySelector(s);
const $$=s=>[...document.querySelectorAll(s)];
const state={meta:null,analysis:null,image:null,drift:null,drug:null,charts:{},driftPlay:null,driftT:0,seedCity:null,queue:{seq:null,img:null},lastKey:null};

Chart.defaults.font.family="'Inter',system-ui,sans-serif";
Chart.defaults.font.size=10;
Chart.defaults.color="#5b5b5b";
Chart.defaults.borderColor="#e6e6e2";
Chart.defaults.animation={duration:600,easing:'easeOutCubic'};

const INK="#0a0a0a",INK2="#5b5b5b",INK3="#8a8a8a",LINE="#e6e6e2",PANEL2="#fbfbf9",DANGER="#8a1e1e",WARN="#7a5b12",OK="#1f5c2b";

function toast(msg,kind=''){const t=$('#toast');t.className='toast on '+kind;t.textContent=msg;clearTimeout(toast._h);toast._h=setTimeout(()=>t.className='toast',2600)}
function fmt(v,d=3){if(v==null||Number.isNaN(v))return '—';const a=Math.abs(v);if(a>=100)return v.toFixed(1);if(a>=10)return v.toFixed(2);return v.toFixed(d)}

async function fetchJSON(url,options){
  const r=await fetch(url,options);
  const txt=await r.text();
  const data=txt?JSON.parse(txt.replace(/\bNaN\b/g,'null').replace(/\bInfinity\b/g,'null').replace(/-null/g,'null')):{};
  if(!r.ok)throw new Error(data.error||`http ${r.status}`);
  return data;
}

function switchMode(mode){
  document.body.dataset.mode=mode;
  $$('.rail-item').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));
  $$('.mode').forEach(s=>s.hidden=s.dataset.mode!==mode);
  setTimeout(()=>{
    initChartsForMode(mode);
    if(mode==='drug'&&state.analysis&&!state.drug){runDrug()}
    else if(mode==='drug'&&state.drug){renderDrug(state.drug)}
    else if(mode==='drift'&&state.drift){renderDriftShare();renderReachedChart();renderDriftFrame(state.driftT||0);renderDriftRationale()}
    else if(mode==='analyze'&&state.analysis){renderAnalyze(state.analysis)}
  },40);
}

function wrapCanvases(){
  document.querySelectorAll('.card>canvas, .image-hist>canvas').forEach(c=>{
    if(c.parentElement.classList.contains('ch'))return;
    if(c.id==='mutMap')return;
    const w=document.createElement('div');w.className='ch';
    c.parentNode.insertBefore(w,c);w.appendChild(c);
    c.removeAttribute('height');c.removeAttribute('width');
  });
}

async function boot(){
  try{
    state.meta=await fetchJSON('/api/health');
    populateRefs();populateWorldMap();
  }catch(e){
    toast('Run: python site/serve.py','err');console.error(e);return;
  }
  wrapCanvases();
  initAnalyze();initDrift();initDrug();
  $$('.rail-item').forEach(b=>b.addEventListener('click',()=>switchMode(b.dataset.mode)));
  initCharts();
  loadSample('spike');
}

function populateRefs(){
  const opts=state.meta.references.map(r=>`<option value="${r.id}">${r.disease} · ${r.name}</option>`).join('');
  $('#refSel').innerHTML=opts;$('#driftRefSel').innerHTML=opts;
}

function makeChart(id,type,data,opts={}){
  const el=document.getElementById(id);if(!el)return null;
  return new Chart(el.getContext('2d'),{type,data,options:Object.assign({
    responsive:true,maintainAspectRatio:false,
    plugins:{legend:{display:false},tooltip:{backgroundColor:INK,padding:8,titleFont:{size:11},bodyFont:{size:11}}},
    scales:{x:{grid:{display:false},ticks:{color:INK3}},y:{grid:{color:LINE},ticks:{color:INK3}}}
  },opts)});
}

const inited={analyze:false,drift:false,drug:false};

function initChartsForMode(mode){
  if(inited[mode])return;
  if(mode==='analyze'){initAnalyzeCharts()}
  else if(mode==='drift'){initDriftCharts()}
  else if(mode==='drug'){initDrugCharts()}
  inited[mode]=true;
}

function initAnalyzeCharts(){
  state.charts.compChart=makeChart('compChart','bar',{labels:[],datasets:[{data:[],backgroundColor:INK,borderRadius:2}]},
    {scales:{x:{grid:{display:false},ticks:{color:INK3,font:{size:9}}},y:{grid:{color:LINE},ticks:{color:INK3},title:{display:true,text:'%',color:INK3}}}});
  state.charts.hydroChart=makeChart('hydroChart','line',{labels:[],datasets:[{data:[],borderColor:INK,borderWidth:1.4,pointRadius:0,tension:.2,fill:false}]},
    {scales:{x:{grid:{display:false},ticks:{color:INK3,maxTicksLimit:6}},y:{grid:{color:LINE},ticks:{color:INK3}}}});
  state.charts.entChart=makeChart('entChart','line',{labels:[],datasets:[{data:[],borderColor:DANGER,borderWidth:1.4,pointRadius:0,tension:.2,fill:false}]},
    {scales:{x:{grid:{display:false},ticks:{color:INK3,maxTicksLimit:6}},y:{grid:{color:LINE},ticks:{color:INK3}}}});
  state.charts.impactDist=makeChart('impactDist','bar',{labels:[],datasets:[{data:[],backgroundColor:INK,borderRadius:2}]},
    {scales:{x:{grid:{display:false},ticks:{color:INK3,font:{size:9}}},y:{grid:{color:LINE},ticks:{color:INK3},title:{display:true,text:'count',color:INK3}}}});
  state.charts.hotDist=makeChart('hotDist','bar',{labels:[],datasets:[{data:[],backgroundColor:INK,borderRadius:2}]},
    {scales:{x:{grid:{display:false},ticks:{color:INK3,font:{size:9}},title:{display:true,text:'residues from hotspot',color:INK3}},y:{grid:{color:LINE},ticks:{color:INK3}}}});
  state.charts.physicoDist=makeChart('physicoDist','bar',{labels:[],datasets:[{data:[],backgroundColor:INK,borderRadius:2}]},
    {scales:{x:{grid:{display:false},ticks:{color:INK3,font:{size:9}},title:{display:true,text:'physico shift bin',color:INK3}},y:{grid:{color:LINE},ticks:{color:INK3},title:{display:true,text:'count',color:INK3}}}});
  state.charts.cumImpact=makeChart('cumImpact','line',{labels:[],datasets:[{data:[],borderColor:INK,backgroundColor:'rgba(10,10,10,.08)',borderWidth:1.6,pointRadius:0,fill:true,tension:0}]},
    {animation:false,parsing:false,scales:{x:{type:'linear',grid:{display:false},ticks:{color:INK3,maxTicksLimit:6},title:{display:true,text:'residue',color:INK3}},y:{grid:{color:LINE},ticks:{color:INK3},title:{display:true,text:'Σ impact',color:INK3}}}});
  state.charts.kindPie=makeChart('kindPie','doughnut',{labels:[],datasets:[{data:[],backgroundColor:[INK,INK2,INK3],borderColor:'#fff',borderWidth:2}]},
    {plugins:{legend:{display:true,position:'right',labels:{color:INK2,boxWidth:10,font:{size:10}}}},scales:{}});
  state.charts.scatterChart=new Chart(document.getElementById('scatterChart').getContext('2d'),{
    type:'scatter',data:{datasets:[{data:[],backgroundColor:'rgba(10,10,10,0.5)',borderColor:INK,pointRadius:3}]},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:600},
      plugins:{legend:{display:false},tooltip:{backgroundColor:INK,padding:8,callbacks:{label:c=>`pos ${c.raw.x} · impact ${c.raw.y.toFixed(2)}`}}},
      scales:{x:{grid:{color:LINE},ticks:{color:INK3},title:{display:true,text:'residue position',color:INK3}},
              y:{grid:{color:LINE},ticks:{color:INK3},title:{display:true,text:'impact',color:INK3},min:0,max:1}}}
  });
  state.charts.regionChart=makeChart('regionChart','bar',{labels:['1','2','3','4','5','6','7','8','9','10'],datasets:[{data:[],backgroundColor:INK,borderRadius:2}]},
    {scales:{x:{grid:{display:false},ticks:{color:INK3,font:{size:9}},title:{display:true,text:'window (10% each)',color:INK3}},y:{grid:{color:LINE},ticks:{color:INK3}}}});
  state.charts.imgHist=makeChart('imgHist','bar',{labels:[],datasets:[{data:[],backgroundColor:INK,borderRadius:2}]},
    {scales:{x:{grid:{display:false},ticks:{color:INK3,font:{size:9}},title:{display:true,text:'byte intensity bin',color:INK3}},y:{grid:{color:LINE},ticks:{color:INK3}}}});
}

function initDriftCharts(){
  state.charts.share=makeChart('driftShareChart','line',{labels:[],datasets:[]},
    {plugins:{legend:{display:true,labels:{color:INK2,boxWidth:10,font:{size:9}}},tooltip:{backgroundColor:INK,padding:8}},
     scales:{x:{grid:{display:false},ticks:{color:INK3,maxTicksLimit:6}},y:{grid:{color:LINE},ticks:{color:INK3},min:0,max:1}}});
  state.charts.reached=makeChart('reachedChart','line',{labels:[],datasets:[{data:[],borderColor:INK,backgroundColor:'rgba(10,10,10,.08)',borderWidth:1.6,pointRadius:0,fill:true,tension:.2}]},
    {scales:{x:{grid:{display:false},ticks:{color:INK3,maxTicksLimit:6}},y:{grid:{color:LINE},ticks:{color:INK3},title:{display:true,text:'cities',color:INK3}}}});
}

function initDrugCharts(){
  state.charts.successHist=makeChart('successHist','bar',{labels:[],datasets:[{data:[],backgroundColor:INK,borderRadius:2}]},
    {scales:{x:{grid:{display:false},ticks:{color:INK3,font:{size:9}},title:{display:true,text:'success %',color:INK3}},y:{grid:{color:LINE},ticks:{color:INK3}}}});
  state.charts.bindingChart=makeChart('bindingChart','bar',{labels:[],datasets:[{data:[],backgroundColor:INK,borderRadius:2}]},
    {scales:{x:{grid:{display:false},ticks:{color:INK3,font:{size:9}},title:{display:true,text:'ΔG bin',color:INK3}},y:{grid:{color:LINE},ticks:{color:INK3}}}});
  state.charts.admetScatter=new Chart(document.getElementById('admetScatter').getContext('2d'),{
    type:'scatter',data:{datasets:[{data:[],pointBackgroundColor:[],borderColor:INK,pointRadius:5}]},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:600},
      plugins:{legend:{display:false},tooltip:{backgroundColor:INK,padding:8,callbacks:{label:c=>`${c.raw.id} · ADMET ${c.raw.x.toFixed(2)} · synth ${c.raw.y} · success ${(c.raw.s*100).toFixed(1)}%`}}},
      scales:{x:{grid:{color:LINE},ticks:{color:INK3},title:{display:true,text:'ADMET →',color:INK3},min:0,max:1},
              y:{grid:{color:LINE},ticks:{color:INK3},title:{display:true,text:'← synth complexity',color:INK3},min:0,max:10,reverse:true}}}
  });
  state.charts.classChart=makeChart('classChart','bar',{labels:[],datasets:[{data:[],backgroundColor:INK,borderRadius:2}]},
    {indexAxis:'y',scales:{x:{grid:{color:LINE},ticks:{color:INK3},title:{display:true,text:'mean success',color:INK3},min:0,max:1},y:{grid:{display:false},ticks:{color:INK3,font:{size:9}}}}});
  state.charts.ic50Chart=makeChart('ic50Chart','bar',{labels:[],datasets:[{data:[],backgroundColor:INK,borderRadius:2}]},
    {scales:{x:{grid:{display:false},ticks:{color:INK3,font:{size:9}},title:{display:true,text:'IC50 (nM)',color:INK3}},y:{grid:{color:LINE},ticks:{color:INK3}}}});
  state.charts.robustScatter=new Chart(document.getElementById('robustScatter').getContext('2d'),{
    type:'scatter',data:{datasets:[{data:[],pointBackgroundColor:[],pointRadius:5}]},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:600},
      plugins:{legend:{display:false},tooltip:{backgroundColor:INK,padding:8,callbacks:{label:c=>`${c.raw.id} · robust ${c.raw.x.toFixed(2)} · ΔG ${c.raw.y}`}}},
      scales:{x:{grid:{color:LINE},ticks:{color:INK3},title:{display:true,text:'robustness →',color:INK3},min:0,max:1},
              y:{grid:{color:LINE},ticks:{color:INK3},title:{display:true,text:'← ΔG (kcal/mol)',color:INK3},reverse:true}}}
  });
}

function initCharts(){
  initChartsForMode('analyze');
}

function initAnalyze(){
  const dz=$('#drop'),input=$('#fileInput');
  dz.addEventListener('click',()=>input.click());
  input.addEventListener('change',e=>{const f=e.target.files[0];if(f)queueSequence(f)});
  ['dragenter','dragover'].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.add('drag')}));
  ['dragleave','drop'].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.remove('drag')}));
  dz.addEventListener('drop',e=>{const f=e.dataTransfer.files[0];if(f)queueSequence(f)});

  const idz=$('#imgDrop'),iinput=$('#imgInput');
  idz.addEventListener('click',()=>iinput.click());
  iinput.addEventListener('change',e=>{const f=e.target.files[0];if(f)queueImage(f)});
  ['dragenter','dragover'].forEach(ev=>idz.addEventListener(ev,e=>{e.preventDefault();idz.classList.add('drag')}));
  ['dragleave','drop'].forEach(ev=>idz.addEventListener(ev,e=>{e.preventDefault();idz.classList.remove('drag')}));
  idz.addEventListener('drop',e=>{const f=e.dataTransfer.files[0];if(f)queueImage(f)});
  document.addEventListener('paste',e=>{
    const items=e.clipboardData?.items||[];
    for(const it of items)if(it.type.startsWith('image/')){const f=it.getAsFile();if(f)queueImage(f);break}
  });

  $$('button[data-preset]').forEach(b=>b.addEventListener('click',()=>loadSample(b.dataset.preset)));
  $('#runAnalyzeBtn').addEventListener('click',runQueue);
  $('#clearQueueBtn').addEventListener('click',clearQueue);
}

function queueSequence(file){state.queue.seq={file,name:file.name,size:file.size};renderQueue()}
function queueImage(file){state.queue.img={file,name:file.name,size:file.size};renderQueue()}

function clearQueue(){
  state.queue.seq=null;state.queue.img=null;
  $$('button[data-preset]').forEach(b=>b.classList.remove('active'));
  renderQueue();
}

function renderQueue(){
  const q=$('#queue');const parts=[];
  if(state.queue.seq){
    const s=state.queue.seq;
    parts.push(`<div class="queue-chip"><span class="queue-kind seq">seq</span><span class="queue-name">${s.name}</span><span class="queue-size">${(s.size/1024).toFixed(1)} KB</span><button data-drop="seq" title="remove">×</button></div>`);
  }
  if(state.queue.img){
    const s=state.queue.img;
    parts.push(`<div class="queue-chip"><span class="queue-kind img">img</span><span class="queue-name">${s.name}</span><span class="queue-size">${(s.size/1024).toFixed(1)} KB</span><button data-drop="img" title="remove">×</button></div>`);
  }
  q.innerHTML=parts.length?parts.join(''):'<span class="queue-empty">no files queued yet</span>';
  q.querySelectorAll('button[data-drop]').forEach(b=>b.addEventListener('click',()=>{
    if(b.dataset.drop==='seq')state.queue.seq=null;else state.queue.img=null;renderQueue();
  }));
  const has=!!(state.queue.seq||state.queue.img);
  $('#runAnalyzeBtn').disabled=!has;
  $('#clearQueueBtn').disabled=!has;
}

async function runQueue(){
  if(!state.queue.seq&&!state.queue.img){toast('queue is empty','err');return}
  const btn=$('#runAnalyzeBtn');btn.disabled=true;btn.classList.add('running');btn.textContent='Analyzing…';
  try{
    const key=`${state.queue.seq?.name||''}|${state.queue.seq?.size||0}|${$('#refSel').value}|${state.queue.img?.name||''}|${state.queue.img?.size||0}`;
    if(state.queue.seq){await runAnalyze(state.queue.seq.file)}
    if(state.queue.img){await uploadImage(state.queue.img.file)}
    if(state.queue.seq&&state.queue.img&&state.analysis&&state.image){await refreshCrossRef()}
    state.lastKey=key;
    toast('done · deterministic result cached','ok');
  }catch(e){toast('run failed: '+e.message,'err');console.error(e)}
  finally{btn.classList.remove('running');btn.textContent='Analyze';renderQueue()}
}

async function runAnalyze(file){
  toast('analyzing '+file.name);
  const refId=$('#refSel').value;
  const buf=await file.arrayBuffer();
  try{
    const res=await fetchJSON('/api/analyze',{method:'POST',
      headers:{'Content-Type':'application/octet-stream','X-Filename':file.name,'X-Reference':refId},body:buf});
    state.analysis=res;renderAnalyze(res);$('#runDrug').disabled=false;state.drug=null;
    if(state.image)await refreshCrossRef();
    toast('analysis done · '+res.mutation_count+' mutations','ok');
  }catch(e){toast('analysis failed: '+e.message,'err');console.error(e)}
}

async function uploadImage(file){
  toast('parsing image '+file.name);
  const buf=await file.arrayBuffer();
  try{
    const res=await fetchJSON('/api/image_features',{method:'POST',
      headers:{'Content-Type':'application/octet-stream','X-Filename':file.name},body:buf});
    state.image=res;
    const reader=new FileReader();
    reader.onload=e=>{$('#imgPreview').src=e.target.result};
    reader.readAsDataURL(file);
    renderImage(res);
    if(state.analysis)await refreshCrossRef();
    toast('image parsed · '+res.kind,'ok');
  }catch(e){toast('image parse failed: '+e.message,'err');console.error(e)}
}

async function refreshCrossRef(){
  try{
    const merged=await fetchJSON('/api/merge',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({analysis:state.analysis,image:state.image})});
    renderCrossRef(merged);
    if(state.image)renderImage(state.image,merged);
  }catch(e){console.error(e)}
}

function renderImage(img,merged){
  $('#imageCard').hidden=false;
  $('#imgKindChip').textContent=`${img.kind} · ${img.width}×${img.height} · ${(img.bytes/1024).toFixed(1)} KB`;
  const rows=[
    ['fingerprint',img.fingerprint],
    ['width × height',`${img.width} × ${img.height}`],
    ['entropy (bits)',img.entropy],
    ['brightness',img.brightness],
    ['contrast',img.contrast],
    ['density',img.density],
    ['quality score',(img.quality_score*100).toFixed(0)+' %'],
  ];
  $('#imageStats').innerHTML=rows.map(([k,v])=>`<div class="image-stat"><span>${k}</span><span>${v}</span></div>`).join('')
    +`<div class="image-stat"><span>interpretation</span><span style="font-weight:400;text-align:right;max-width:180px">${img.interpretation}</span></div>`;
  const c=state.charts.imgHist;
  c.data.labels=img.histogram.map((_,i)=>i*16);
  c.data.datasets[0].data=img.histogram;
  c.update();
}

function renderCrossRef(m){
  const box=$('#crossRef');
  if(!m.cross_reference||!m.cross_reference.length){box.textContent='no cross-reference notes.';return}
  box.innerHTML=`<b>combined novelty ${m.combined_novelty} · final tier ${m.final_tier}</b><br>`
    +m.cross_reference.map(r=>'· '+r).join('<br>');
}

async function loadSample(kind){
  const map={spike:['sars2_spike',0.04],ha:['h1n1_ha',0.06],rpob:['mtb_rpob',0.02]};
  const [refId,rate]=map[kind];$('#refSel').value=refId;
  const r=await fetch(`/api/sample_fasta?ref=${refId}&rate=${rate}`);
  const text=await r.text();
  const blob=new Blob([text],{type:'text/plain'});
  const file=new File([blob],`${kind}_sample.fasta`,{type:'text/plain'});
  $$('button[data-preset]').forEach(b=>b.classList.toggle('active',b.dataset.preset===kind));
  queueSequence(file);
  await runQueue();
}

function renderAnalyze(a){
  const stats=[
    ['type',a.kind_detected.toUpperCase(),'protein/dna auto-detected',a.fingerprint],
    ['length',a.translated_length+' aa','residues aligned',a.input_length+' input'],
    ['identity',(a.identity*100).toFixed(1)+' %','vs reference in aligned region','vs '+a.reference.name.split(' (')[0]],
    ['mutations',a.mutation_count,'differences from reference',a.conserved_positions+' conserved'],
    ['novelty',(a.novelty_score*100).toFixed(0)+' %','mutation load vs reference length','ΔR₀ '+(a.predicted_r0_delta>=0?'+':'')+a.predicted_r0_delta],
    ['MW',(a.molecular_weight_da/1000).toFixed(1)+' kDa','molecular weight of translated protein','net charge '+a.net_charge],
  ];
  $('#analyzeStats').innerHTML=stats.map(([k,v,tip,sub])=>
    `<div class="stat-cell" title="${tip}"><div class="stat-cell-hint">?</div><div class="stat-cell-key">${k}</div><div class="stat-cell-val">${v}</div><div class="stat-cell-sub">${sub||''}</div></div>`).join('');

  $('#mutCountChip').textContent=a.mutation_count+' hits';
  drawMutMap(a);
  const dist=a.distributions;
  const impact=state.charts.impactDist;
  impact.data.labels=['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'];
  impact.data.datasets[0].data=dist.impact_bins;
  impact.data.datasets[0].backgroundColor=dist.impact_bins.map((_,i)=>i>=6?DANGER:i>=4?WARN:INK);
  impact.options.scales.y.beginAtZero=true;
  impact.options.scales.y.suggestedMax=Math.max(3,Math.max(...dist.impact_bins));
  impact.update();
  const sp=dist.impact_split;
  $('#impactSplitChip').textContent=`${sp.high} high · ${sp.med} med · ${sp.low} low`;

  const hot=state.charts.hotDist;
  hot.data.labels=['0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44','45+'];
  hot.data.datasets[0].data=dist.hotspot_dist_bins;
  hot.data.datasets[0].backgroundColor=dist.hotspot_dist_bins.map((_,i)=>i===0?DANGER:i<=1?WARN:INK2);
  hot.options.scales.y.beginAtZero=true;
  hot.options.scales.y.suggestedMax=Math.max(3,Math.max(...dist.hotspot_dist_bins));
  hot.update();
  const nearHot=dist.hotspot_dist_bins[0]||0;
  const pct=a.mutation_count?Math.round(nearHot/a.mutation_count*100):0;
  $('#hotChip').textContent=`${nearHot} within 5 aa (${pct}%)`;

  const phys=state.charts.physicoDist;
  phys.data.labels=['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'];
  phys.data.datasets[0].data=dist.physico_bins;
  phys.data.datasets[0].backgroundColor=dist.physico_bins.map((_,i)=>i>=6?DANGER:i>=4?WARN:INK);
  phys.options.scales.y.beginAtZero=true;
  phys.options.scales.y.suggestedMax=Math.max(3,Math.max(...dist.physico_bins));
  phys.update();

  const cum=state.charts.cumImpact;
  const bySeq=[...a.mutations].sort((x,y)=>x.pos-y.pos);
  const L=a.reference.length;
  const N=40;const step=Math.max(1,Math.ceil(L/N));
  const pts=[];let running=0;let mi=0;
  for(let p=step;p<=L;p+=step){
    while(mi<bySeq.length&&bySeq[mi].pos<=p){running+=bySeq[mi].impact;mi++}
    pts.push({x:p,y:+running.toFixed(3)});
  }
  cum.data.labels=undefined;
  cum.data.datasets[0].data=pts;
  cum.update('none');

  const cc=state.charts.compChart;
  cc.data.labels=Object.keys(a.composition);
  cc.data.datasets[0].data=Object.values(a.composition).map(v=>+(v*100).toFixed(2));
  cc.update();

  const hydro=state.charts.hydroChart;
  hydro.data.labels=a.hydrophobicity.map((_,i)=>i+1);
  hydro.data.datasets[0].data=a.hydrophobicity;hydro.update();

  const ent=state.charts.entChart;
  ent.data.labels=a.entropy.map((_,i)=>i+1);
  ent.data.datasets[0].data=a.entropy;ent.update();

  const pie=state.charts.kindPie;
  pie.data.labels=Object.keys(dist.kind_counts);
  pie.data.datasets[0].data=Object.values(dist.kind_counts);
  pie.update();

  const scatter=state.charts.scatterChart;
  scatter.data.datasets[0].data=a.scatter.map(p=>({x:p.x,y:p.y}));
  scatter.update();

  const rg=state.charts.regionChart;
  rg.data.datasets[0].data=a.per_region_hits;
  rg.data.datasets[0].backgroundColor=a.per_region_hits.map(v=>v>=Math.max(...a.per_region_hits)*0.8?DANGER:INK);
  rg.update();

  $('#mutBody').innerHTML=a.mutations.slice(0,20).map(m=>{
    const cls=m.impact>=0.6?'impact-high':m.impact>=0.35?'impact-med':'impact-low';
    return `<tr><td>${m.ref}${m.pos}${m.alt}</td><td>${m.kind}</td><td>${m.hotspot_distance}</td><td>${m.physicochem_shift}</td><td class="${cls}"><span class="impact-bar"><span style="width:${(m.impact*100).toFixed(0)}%"></span></span>${m.impact.toFixed(3)}</td></tr>`;
  }).join('');

  const tier=a.rationale.risk_tier;
  const chip=$('#riskChip');
  chip.textContent='risk · '+tier;
  chip.className='chip '+(tier==='HIGH'?'danger':tier==='MODERATE'?'warn':'ok');
  const verdictClass=tier==='HIGH'?'danger':tier==='MODERATE'?'warn':'';
  $('#rationaleBody').innerHTML=`
    <div class="rationale-verdict ${verdictClass}">${a.rationale.verdict}</div>
    <ul class="rationale-list">${a.rationale.risk_reasons.map(r=>`<li>${r}</li>`).join('')}</ul>
    <div class="rationale-why"><b>How this was decided.</b> The tier combines three signals: (1) number of mutations within 5 residues of a curated ${a.reference.disease} escape hotspot, (2) count of mutations with impact score ≥ 0.60, (3) global novelty index (mutation load ÷ reference length). Thresholds mirror the surveillance triage protocol from GermoVision-Net's supervised head (see Table 2.7.1 in the docs).</div>
  `;
}

function drawMutMap(a){
  const canvas=$('#mutMap');const dpr=Math.min(2,devicePixelRatio||1);
  const parent=canvas.parentElement;const W=parent.clientWidth-36;const H=150;
  canvas.width=W*dpr;canvas.height=H*dpr;canvas.style.width=W+'px';canvas.style.height=H+'px';
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,W,H);
  const L=a.reference.length;
  ctx.fillStyle=PANEL2;ctx.fillRect(0,22,W,H-42);
  ctx.strokeStyle=LINE;ctx.lineWidth=1;ctx.strokeRect(0,22,W,H-42);
  a.reference.hotspots.forEach(pos=>{
    const x=pos/L*W;
    ctx.fillStyle='rgba(138,30,30,.10)';ctx.fillRect(x-3,22,6,H-42);
  });
  a.mutations.forEach(m=>{
    const x=m.pos/L*W;const h=(H-42)*m.impact;
    ctx.fillStyle=m.impact>=0.6?DANGER:m.impact>=0.35?WARN:INK;
    ctx.fillRect(x-1,H-20-h,2,h);
  });
  ctx.fillStyle=INK3;ctx.font='9px JetBrains Mono';
  for(let i=0;i<=10;i++){const p=Math.round(L*i/10);const x=i/10*W;ctx.fillText(p,x-8,H-4)}
  ctx.fillText('impact',4,15);
  ctx.fillText('hotspot band',W-90,15);
}

function populateWorldMap(){
  const svg=$('#worldMap');
  const cities=state.meta.cities;
  const proj=(lng,lat)=>{const x=(lng+180)/360*900;const y=(90-lat)/180*460;return[x,y]};
  const parts=[];
  parts.push(`<g class="map-graticule">`);
  for(let lng=-180;lng<=180;lng+=30){const [x1]=proj(lng,-90),[x2]=proj(lng,90);parts.push(`<line x1="${x1}" y1="0" x2="${x2}" y2="460"/>`)}
  for(let lat=-60;lat<=60;lat+=30){const [,y1]=proj(-180,lat),[,y2]=proj(180,lat);parts.push(`<line x1="0" y1="${y1}" x2="900" y2="${y2}"/>`)}
  parts.push(`</g>`);
  cities.forEach((c,i)=>{
    const [x,y]=proj(c.lng,c.lat);
    const r=Math.min(6,2+Math.log2(c.pop+1));
    parts.push(`<g class="map-city" data-i="${i}" transform="translate(${x},${y})">
      <circle class="ring" r="0" fill="none" stroke="#0a0a0a" stroke-width="1" opacity="0"/>
      <circle class="core" r="${r}" fill="#e6e6e2" stroke="#8a8a8a" stroke-width=".8"/>
      <text y="-8" text-anchor="middle">${c.n}</text>
    </g>`);
  });
  svg.innerHTML=parts.join('');
  svg.querySelectorAll('.map-city').forEach(g=>{
    const i=+g.dataset.i;
    g.addEventListener('click',()=>{
      state.seedCity=i;
      svg.querySelectorAll('.map-city').forEach(x=>x.classList.toggle('seed',+x.dataset.i===i));
      $('#seedCityChip').textContent='seed · '+state.meta.cities[i].n;
    });
    g.addEventListener('mouseenter',()=>{
      const c=state.meta.cities[i];toast(`${c.n} · ${c.pop.toFixed(1)}M · click to seed`);
    });
  });
}

function initDrift(){
  const g=$('#gtSlider'),r=$('#r0Slider'),m=$('#mrSlider'),d=$('#dSlider');
  r.addEventListener('input',()=>$('#r0Val').textContent=(+r.value).toFixed(1));
  g.addEventListener('input',()=>$('#gtVal').textContent=g.value);
  m.addEventListener('input',()=>$('#mrVal').textContent=(+m.value).toFixed(3));
  d.addEventListener('input',()=>$('#dVal').textContent=d.value);
  $('#runDrift').addEventListener('click',runDrift);
  $('#driftScrub').addEventListener('input',()=>renderDriftFrame(+$('#driftScrub').value));
  $('#driftPlay').addEventListener('click',toggleDriftPlay);
}

async function runDrift(){
  if(state.seedCity==null){toast('click a city on the map first','err');return}
  const btn=$('#runDrift');btn.disabled=true;btn.textContent='simulating…';
  try{
    const res=await fetchJSON('/api/drift',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
      seed_city:state.seedCity,r0:+$('#r0Slider').value,generation_time:+$('#gtSlider').value,
      days:+$('#dSlider').value,mutation_rate:+$('#mrSlider').value,reference:$('#driftRefSel').value
    })});
    state.drift=res;state.driftT=0;
    const scr=$('#driftScrub');scr.max=res.days-1;scr.value=0;scr.disabled=false;$('#driftPlay').disabled=false;
    renderDriftShare();renderReachedChart();renderDriftFrame(0);renderDriftRationale();
    toast('simulation done · '+res.days+' days','ok');
  }catch(e){toast('simulation failed: '+e.message,'err');console.error(e)}
  finally{btn.disabled=false;btn.textContent='Simulate'}
}

function renderDriftShare(){
  const {cities,shares,seed_city}=state.drift;
  const rankAt=Math.floor(state.drift.days*0.7);
  const idxs=cities.map((c,i)=>[i,shares[i][rankAt]]).sort((a,b)=>b[1]-a[1]).slice(0,8).map(p=>p[0]);
  if(!idxs.includes(seed_city))idxs.unshift(seed_city);
  const c=state.charts.share;
  c.data.labels=Array.from({length:state.drift.days},(_,i)=>i);
  const pal=['#0a0a0a','#5b5b5b','#8a8a8a','#7a5b12','#8a1e1e','#1f5c2b','#3d4b78','#803d78'];
  c.data.datasets=idxs.map((i,k)=>({label:cities[i].n,data:shares[i],borderColor:pal[k%pal.length],borderWidth:i===seed_city?2:1.2,pointRadius:0,tension:.15}));
  c.update();
}

function renderReachedChart(){
  const {days,shares}=state.drift;const n=shares.length;
  const perDay=Array.from({length:days},(_,t)=>shares.filter(row=>row[t]>0.02).length);
  const c=state.charts.reached;
  c.data.labels=Array.from({length:days},(_,i)=>i);
  c.data.datasets[0].data=perDay;c.update();
}

function renderDriftRationale(){
  const r=state.drift.rationale;
  const chip=$('#driftVerdictChip');
  chip.textContent='doubling · '+r.doubling_days+' d';
  chip.className='chip '+(r.doubling_days<5?'danger':r.doubling_days<10?'warn':'ok');
  $('#driftRationale').innerHTML=`
    <div class="rationale-verdict ${r.doubling_days<5?'danger':r.doubling_days<10?'warn':''}">${r.verdict}</div>
    <ul class="rationale-list">${r.reasons.map(x=>`<li>${x}</li>`).join('')}</ul>
    <div class="rationale-why"><b>How this was estimated.</b> Doubling time uses the standard exponential growth relation td = ln(2)·Tg / (R₀−1). Reach curves come from a gravity-model expansion (population × population / distance^1.4) with per-city sigmoid uptake once the local seed lands. Mutation accumulation is a Poisson process at the requested rate anchored on the reference's known escape hotspots.</div>
  `;
  const {reached_25_day,reached_50_day,doubling_days,peak_share}=r;
  $('#driftStats').innerHTML=[
    ['seed',state.meta.cities[state.drift.seed_city].n,'origin city'],
    ['doubling',doubling_days+' d','ln(2)·Tg/(R₀−1)'],
    ['25 % reach','day '+reached_25_day,'fraction of tracked cities'],
    ['50 % reach','day '+reached_50_day,'fraction of tracked cities'],
  ].map(([k,v,s])=>`<div class="stat-cell"><div class="stat-cell-key">${k}</div><div class="stat-cell-val">${v}</div><div class="stat-cell-sub">${s}</div></div>`).join('');
}

function renderDriftFrame(t){
  state.driftT=t;$('#driftDay').textContent='day '+t;
  const svg=$('#worldMap');
  svg.querySelectorAll('.map-city').forEach((g,i)=>{
    const share=state.drift.shares[i][t];
    const core=g.querySelector('.core');const ring=g.querySelector('.ring');
    let fill='#e6e6e2';
    if(share>=0.5)fill='#0a0a0a';
    else if(share>=0.1)fill='#5b5b5b';
    else if(share>0)fill='#a0a0a0';
    core.setAttribute('fill',fill);
    if(share>0.02){
      const rad=6+share*22;
      ring.setAttribute('r',rad);
      ring.setAttribute('opacity',(0.15+share*0.4).toFixed(2));
    }else{ring.setAttribute('opacity','0')}
  });
  const muts=state.drift.mutations_timeline[t]||[];
  $('#mutAccChip').textContent=muts.length+' aa changes';
  $('#mutTimeline').innerHTML=muts.slice().reverse().slice(0,20).map(m=>
    `<div class="mut-item"><span class="mut-item-day">d${m.day}</span><span class="mut-item-code">${m.notation}</span><span class="mut-item-tag">hotspot</span></div>`).join('')||'<div style="color:var(--ink-3);font-size:12px;padding:16px;text-align:center;font-family:JetBrains Mono">no mutations acquired yet</div>';
}

function toggleDriftPlay(){
  const btn=$('#driftPlay');
  if(state.driftPlay){clearInterval(state.driftPlay);state.driftPlay=null;btn.textContent='▶';return}
  btn.textContent='⏸';
  state.driftPlay=setInterval(()=>{
    let t=state.driftT+1;if(t>=state.drift.days)t=0;
    $('#driftScrub').value=t;renderDriftFrame(t);
  },140);
}

function initDrug(){$('#runDrug').addEventListener('click',runDrug)}

async function runDrug(){
  if(!state.analysis){toast('analyze a genome first','err');return}
  const btn=$('#runDrug');btn.disabled=true;btn.textContent='ranking…';
  try{
    const res=await fetchJSON('/api/drug',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({analysis:state.analysis})});
    state.drug=res;renderDrug(res);
    toast('ranked '+res.candidates.length+' candidates','ok');
  }catch(e){toast('drug ranking failed: '+e.message,'err');console.error(e)}
  finally{btn.disabled=false;btn.textContent='Rank candidates'}
}

function renderDrug(d){
  const top=d.top_pick;
  $('#drugCountChip').textContent=d.candidates.length;
  $('#topPickChip').textContent=top?top.id:'—';
  if(top){
    $('#topPickBody').innerHTML=`
      <div class="rationale-verdict">${d.rationale.verdict}</div>
      <div style="display:grid;grid-template-columns:1fr auto;gap:24px;align-items:start;margin-bottom:14px">
        <div>
          <div style="font-size:11px;color:var(--ink-3);font-family:JetBrains Mono;letter-spacing:.08em;text-transform:uppercase">recommended candidate</div>
          <div style="font-size:22px;font-weight:700;letter-spacing:-.02em;margin-top:4px">${top.variant}</div>
          <div style="font-size:12px;color:var(--ink-2);font-family:JetBrains Mono;margin-top:2px">${top.class} · target ${top.target}</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:11px;color:var(--ink-3);font-family:JetBrains Mono;letter-spacing:.08em;text-transform:uppercase">success prob</div>
          <div style="font-size:34px;font-weight:700;letter-spacing:-.03em;font-family:JetBrains Mono">${(top.success_prob*100).toFixed(1)}%</div>
        </div>
      </div>
      <ul class="rationale-list">${d.rationale.reasons.map(r=>`<li>${r}</li>`).join('')}</ul>
      <div class="rationale-why"><b>Why this was picked.</b> ${d.rationale.why_top} Gap to the runner-up (${d.rationale.runner_up?d.rationale.runner_up.variant:'—'}) is <b>${d.rationale.gap_to_runner_up} percentage points</b>.</div>
    `;
  }
  $('#drugStats').innerHTML=top?[
    ['success',(top.success_prob*100).toFixed(1)+' %','combined rank score'],
    ['ΔG',top.binding_kcal_mol,'binding energy kcal/mol'],
    ['IC₅₀',top.ic50_nm+' nM','half-maximal inhibitory concentration'],
    ['robustness',top.resistance_robustness,'against input mutation profile'],
  ].map(([k,v,s])=>`<div class="stat-cell"><div class="stat-cell-key">${k}</div><div class="stat-cell-val">${v}</div><div class="stat-cell-sub">${s}</div></div>`).join(''):'';

  const dist=d.distributions;
  const sh=state.charts.successHist;
  sh.data.labels=['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'];
  sh.data.datasets[0].data=dist.success_hist;
  sh.data.datasets[0].backgroundColor=dist.success_hist.map((_,i)=>i>=7?OK:i>=5?WARN:INK);
  sh.update();

  const bind=state.charts.bindingChart;
  bind.data.labels=['-4..-5.5','-5.5..-7','-7..-8.5','-8.5..-10','-10..-11.5','-11.5..-13'];
  bind.data.datasets[0].data=dist.binding_bins;
  bind.data.datasets[0].backgroundColor=dist.binding_bins.map((_,i)=>i>=3?OK:INK);
  bind.update();

  const scat=state.charts.admetScatter;
  scat.data.datasets[0].data=dist.scatter_admet_synth.map(p=>({x:p.x,y:p.y,s:p.s,id:p.id}));
  scat.data.datasets[0].pointBackgroundColor=dist.scatter_admet_synth.map(p=>p.s>=0.6?OK:p.s>=0.4?WARN:INK);
  scat.update();

  const cls=state.charts.classChart;
  cls.data.labels=dist.class_summary.map(c=>c.class.length>26?c.class.slice(0,26)+'…':c.class);
  cls.data.datasets[0].data=dist.class_summary.map(c=>c.mean_success);
  cls.update();

  const ic50s=d.candidates.map(c=>c.ic50_nm);
  const bins=[0,0,0,0,0,0];
  const edges=[10,50,100,500,1000,5000];
  ic50s.forEach(v=>{for(let i=0;i<edges.length;i++)if(v<=edges[i]){bins[i]++;break}});
  const ic=state.charts.ic50Chart;
  ic.data.labels=['<10','<50','<100','<500','<1k','<5k'];
  ic.data.datasets[0].data=bins;
  ic.data.datasets[0].backgroundColor=bins.map((_,i)=>i<=1?OK:i<=3?INK:WARN);
  ic.update();

  const rs=state.charts.robustScatter;
  rs.data.datasets[0].data=d.candidates.map(c=>({x:c.resistance_robustness,y:c.binding_kcal_mol,id:c.id,s:c.success_prob}));
  rs.data.datasets[0].pointBackgroundColor=d.candidates.map(c=>c.success_prob>=0.6?OK:c.success_prob>=0.4?WARN:INK);
  rs.update();

  $('#drugBody').innerHTML=d.candidates.map((x,i)=>{
    const cls=x.success_prob>=0.6?'impact-high':x.success_prob>=0.4?'impact-med':'impact-low';
    return `<tr><td>${x.id}</td><td>${x.class}</td><td>${x.target}</td><td>${x.binding_kcal_mol}</td><td>${x.admet_score}</td><td>${x.synth_complexity}</td><td>${x.ic50_nm}</td><td>${x.resistance_robustness}</td><td class="${cls}"><span class="impact-bar"><span style="width:${(x.success_prob*100).toFixed(0)}%"></span></span>${(x.success_prob*100).toFixed(1)}%</td></tr>`;
  }).join('');
  requestAnimationFrame(()=>{
    ['successHist','bindingChart','admetScatter','classChart','ic50Chart','robustScatter'].forEach(k=>{
      try{state.charts[k].resize();state.charts[k].update('none')}catch(e){}
    });
  });
}

document.addEventListener('DOMContentLoaded',boot);
