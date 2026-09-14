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
  state.charts.mlFeat=makeChart('mlFeatChart','bar',{labels:[],datasets:[{data:[],backgroundColor:INK,borderRadius:2}]},
    {indexAxis:'y',scales:{x:{grid:{color:LINE},ticks:{color:INK3},title:{display:true,text:'importance',color:INK3},min:0},y:{grid:{display:false},ticks:{color:INK3,font:{size:9}}}}});
}

function initDrugCharts(){
  state.charts.radar=new Chart(document.getElementById('radarChart').getContext('2d'),{
    type:'radar',data:{labels:['binding','potency','ADMET','robustness','ease of synth'],
      datasets:[
        {label:'class avg',data:[0,0,0,0,0],borderColor:'#c8c4b5',backgroundColor:'rgba(200,196,181,.25)',borderWidth:1.4,pointRadius:2,pointBackgroundColor:'#c8c4b5'},
        {label:'selected',data:[0,0,0,0,0],borderColor:INK,backgroundColor:'rgba(10,10,10,.14)',borderWidth:2,pointRadius:3,pointBackgroundColor:INK},
      ]},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:600},
      plugins:{legend:{display:true,labels:{color:INK2,boxWidth:10,font:{size:10}}},tooltip:{backgroundColor:INK,padding:8}},
      scales:{r:{angleLines:{color:LINE},grid:{color:LINE},ticks:{stepSize:0.25,color:INK3,font:{size:9},backdropColor:'rgba(0,0,0,0)'},pointLabels:{color:INK,font:{size:10,weight:'600'}},min:0,max:1}}}
  });
  const barLabel={anchor:'end',align:'end',color:INK,font:{size:10,weight:'700'},formatter:v=>v||''};
  state.charts.successHist=makeChart('successHist','bar',
    {labels:[],datasets:[{data:[],backgroundColor:INK,borderRadius:3,maxBarThickness:24}]},
    {layout:{padding:{top:16}},plugins:{legend:{display:false},tooltip:{backgroundColor:INK,padding:8}},
     scales:{x:{grid:{display:false},ticks:{color:INK,font:{size:10,weight:'600'},maxRotation:0}},
             y:{grid:{color:LINE},ticks:{color:INK3,font:{size:10}},beginAtZero:true}}});
  state.charts.bindingChart=makeChart('bindingChart','bar',
    {labels:[],datasets:[{data:[],backgroundColor:INK,borderRadius:3,maxBarThickness:24}]},
    {layout:{padding:{top:16}},plugins:{legend:{display:false},tooltip:{backgroundColor:INK,padding:8}},
     scales:{x:{grid:{display:false},ticks:{color:INK,font:{size:10,weight:'600'},maxRotation:0}},
             y:{grid:{color:LINE},ticks:{color:INK3,font:{size:10}},beginAtZero:true}}});
  state.charts.admetScatter=new Chart(document.getElementById('admetScatter').getContext('2d'),{
    type:'scatter',data:{datasets:[{data:[],pointBackgroundColor:[],borderColor:INK,pointRadius:6,hoverRadius:9}]},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:600},
      plugins:{legend:{display:false},tooltip:{backgroundColor:INK,padding:8,callbacks:{label:c=>`${c.raw.id} · ADMET ${c.raw.x.toFixed(2)} · synth ${c.raw.y} · success ${(c.raw.s*100).toFixed(1)}%`}}},
      scales:{x:{grid:{color:LINE},ticks:{color:INK3,font:{size:10}},title:{display:true,text:'ADMET (higher is better) →',color:INK,font:{size:10,weight:'600'}},min:0,max:1},
              y:{grid:{color:LINE},ticks:{color:INK3,font:{size:10}},title:{display:true,text:'← synth complexity (lower is better)',color:INK,font:{size:10,weight:'600'}},min:0,max:10,reverse:true}}}
  });
  state.charts.classChart=makeChart('classChart','bar',
    {labels:[],datasets:[{data:[],backgroundColor:INK,borderRadius:3,maxBarThickness:22}]},
    {indexAxis:'y',layout:{padding:{right:36}},plugins:{legend:{display:false},tooltip:{backgroundColor:INK,padding:8,callbacks:{label:c=>`mean success ${(c.raw*100).toFixed(1)}%`}}},
     scales:{x:{grid:{color:LINE},ticks:{color:INK3,font:{size:10},callback:v=>(v*100).toFixed(0)+'%'},title:{display:true,text:'mean success prob',color:INK,font:{size:10,weight:'600'}},min:0,max:1},
             y:{grid:{display:false},ticks:{color:INK,font:{size:10,weight:'600'}}}}});
  state.charts.ic50Chart=makeChart('ic50Chart','bar',
    {labels:[],datasets:[{data:[],backgroundColor:INK,borderRadius:3,maxBarThickness:26}]},
    {layout:{padding:{top:16}},plugins:{legend:{display:false},tooltip:{backgroundColor:INK,padding:8,callbacks:{label:c=>`${c.raw} candidates`}}},
     scales:{x:{grid:{display:false},ticks:{color:INK,font:{size:10,weight:'600'},maxRotation:0},title:{display:true,text:'IC₅₀ upper bound (nM)',color:INK,font:{size:10,weight:'600'}}},
             y:{grid:{color:LINE},ticks:{color:INK3,font:{size:10}},beginAtZero:true}}});
  state.charts.robustScatter=new Chart(document.getElementById('robustScatter').getContext('2d'),{
    type:'scatter',data:{datasets:[{data:[],pointBackgroundColor:[],pointRadius:6,hoverRadius:9}]},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:600},
      plugins:{legend:{display:false},tooltip:{backgroundColor:INK,padding:8,callbacks:{label:c=>`${c.raw.id} · robust ${c.raw.x.toFixed(2)} · ΔG ${c.raw.y} · success ${(c.raw.s*100).toFixed(1)}%`}}},
      scales:{x:{grid:{color:LINE},ticks:{color:INK3,font:{size:10}},title:{display:true,text:'resistance robustness →',color:INK,font:{size:10,weight:'600'}},min:0,max:1},
              y:{grid:{color:LINE},ticks:{color:INK3,font:{size:10}},title:{display:true,text:'← ΔG (kcal/mol, stronger)',color:INK,font:{size:10,weight:'600'}},reverse:true}}}
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

const CONTINENTS=[
  {name:'north america',pts:[[-115,23],[-117,32],[-120,34],[-122,37],[-124,41],[-124,45],[-124,48],[-127,50],[-132,53],[-137,58],[-146,60],[-150,58],[-155,58],[-161,60],[-166,63],[-166,66],[-162,70],[-155,71],[-149,70],[-142,70],[-135,69],[-125,69],[-115,68],[-105,68],[-95,63],[-88,66],[-82,66],[-78,65],[-75,68],[-67,63],[-62,58],[-58,54],[-55,52],[-58,48],[-63,47],[-66,45],[-68,44],[-66,43],[-70,42],[-74,40],[-76,38],[-76,35],[-79,33],[-80,29],[-82,26],[-80,25],[-84,29],[-88,30],[-91,29],[-94,29],[-97,26],[-97,23],[-100,20],[-104,20],[-107,22],[-112,23],[-115,23]]},
  {name:'south america',pts:[[-77,7],[-80,7],[-80,3],[-79,-3],[-81,-6],[-77,-10],[-76,-14],[-73,-18],[-71,-24],[-71,-30],[-72,-36],[-73,-41],[-73,-46],[-74,-50],[-71,-53],[-68,-55],[-66,-55],[-65,-52],[-64,-48],[-62,-40],[-59,-38],[-57,-35],[-56,-31],[-53,-27],[-49,-25],[-45,-23],[-42,-23],[-40,-20],[-39,-15],[-37,-11],[-35,-8],[-38,-4],[-45,-1],[-50,-1],[-52,4],[-58,7],[-62,10],[-68,11],[-72,10],[-76,8],[-77,7]]},
  {name:'africa',pts:[[-10,35],[-11,30],[-13,27],[-16,22],[-17,17],[-17,14],[-16,12],[-15,10],[-13,9],[-10,7],[-7,5],[-3,5],[0,4],[4,5],[6,4],[8,3],[9,2],[8,-1],[10,-4],[12,-5],[13,-8],[13,-13],[12,-16],[13,-18],[15,-22],[16,-26],[17,-30],[18,-33],[20,-34],[22,-34],[26,-34],[28,-33],[31,-30],[33,-27],[35,-24],[37,-21],[39,-16],[41,-13],[41,-10],[41,-6],[42,-2],[43,2],[45,5],[48,9],[51,10],[51,12],[48,12],[45,11],[43,11],[43,13],[45,17],[43,20],[40,22],[37,24],[34,28],[32,31],[27,34],[20,32],[15,33],[10,35],[5,36],[0,35],[-5,36],[-10,36],[-10,35]]},
  {name:'europe',pts:[[-9,43],[-9,38],[-7,37],[-5,36],[-1,36],[2,37],[3,41],[3,43],[6,43],[8,44],[10,44],[13,46],[13,45],[13,42],[16,41],[18,40],[18,42],[15,44],[13,45],[13,46],[16,47],[20,46],[24,44],[26,42],[27,42],[26,45],[24,46],[24,49],[24,53],[19,54],[15,54],[12,54],[8,54],[8,55],[10,58],[12,58],[13,60],[10,60],[9,63],[6,62],[5,60],[6,57],[3,53],[-1,50],[-3,50],[-2,48],[-4,48],[-4,44],[-9,43]]},
  {name:'scandinavia',pts:[[5,58],[6,60],[8,63],[10,64],[13,64],[13,67],[16,68],[18,68],[22,70],[27,70],[30,69],[28,66],[25,65],[23,64],[24,60],[27,59],[24,58],[22,60],[20,59],[17,59],[14,58],[11,58],[8,58],[5,58]]},
  {name:'uk',pts:[[-5,58],[-5,55],[-4,54],[-3,55],[-1,54],[1,52],[1,50],[-2,50],[-4,50],[-5,52],[-4,55],[-5,58]]},
  {name:'ireland',pts:[[-10,53],[-9,55],[-7,55],[-6,52],[-9,51],[-10,53]]},
  {name:'iceland',pts:[[-24,65],[-15,66],[-13,65],[-15,63],[-22,63],[-24,65]]},
  {name:'russia',pts:[[27,60],[30,60],[35,64],[40,66],[45,66],[55,68],[63,66],[70,68],[75,69],[80,72],[90,72],[105,73],[120,73],[135,72],[145,70],[155,71],[165,69],[172,68],[177,66],[178,64],[172,60],[165,58],[160,56],[154,55],[144,55],[137,52],[136,48],[133,44],[128,45],[131,42],[128,40],[125,38],[125,35],[123,32],[122,30],[122,25],[118,22],[115,20],[110,20],[108,17],[108,12],[104,10],[100,7],[100,3],[103,1],[105,-3],[112,-6],[118,-4],[130,-3],[136,-1],[141,-3],[144,-8],[146,-6],[142,0],[131,4],[123,10],[121,15],[128,25],[130,32],[125,36],[125,40],[130,44],[135,48],[133,52],[142,54],[153,60],[155,62],[145,65],[135,68],[125,68],[110,72],[95,70],[80,73],[65,72],[50,70],[42,66],[38,63],[35,60],[30,58],[27,60]]},
  {name:'india',pts:[[68,24],[72,20],[73,15],[76,10],[78,8],[80,10],[82,17],[87,22],[89,26],[92,25],[95,27],[93,28],[89,26],[82,27],[80,30],[76,32],[74,34],[70,32],[68,29],[68,24]]},
  {name:'middle east',pts:[[26,40],[28,36],[35,36],[38,35],[44,35],[48,32],[52,28],[56,24],[59,21],[54,17],[50,15],[45,12],[42,15],[43,20],[39,22],[35,22],[32,24],[30,26],[26,31],[26,40]]},
  {name:'africa horn',pts:[[41,12],[45,11],[51,11],[52,10],[49,7],[47,4],[45,3],[41,2],[41,6],[41,12]]},
  {name:'madagascar',pts:[[43,-13],[46,-15],[50,-19],[50,-25],[47,-25],[44,-22],[43,-18],[43,-13]]},
  {name:'australia',pts:[[113,-22],[114,-26],[117,-31],[122,-33],[128,-33],[132,-32],[138,-35],[141,-38],[146,-38],[150,-37],[153,-33],[153,-28],[151,-24],[146,-19],[142,-11],[136,-12],[130,-12],[125,-14],[121,-19],[115,-22],[113,-22]]},
  {name:'new zealand n',pts:[[172,-34],[176,-37],[178,-38],[175,-41],[173,-40],[172,-37],[172,-34]]},
  {name:'new zealand s',pts:[[166,-45],[171,-42],[174,-42],[174,-44],[170,-47],[166,-46],[166,-45]]},
  {name:'papua',pts:[[131,-1],[135,-3],[141,-3],[147,-6],[151,-9],[147,-11],[140,-10],[133,-6],[131,-1]]},
  {name:'sumatra',pts:[[95,5],[99,4],[102,1],[105,-2],[104,-5],[100,-2],[97,1],[95,5]]},
  {name:'borneo',pts:[[109,7],[115,6],[118,4],[119,-1],[117,-4],[113,-4],[110,-2],[109,2],[109,7]]},
  {name:'japan honshu',pts:[[133,34],[136,35],[140,37],[141,41],[139,40],[136,36],[133,34]]},
  {name:'japan hokkaido',pts:[[140,42],[143,42],[145,44],[143,45],[140,44],[140,42]]},
  {name:'japan kyushu',pts:[[130,32],[132,34],[131,32],[130,31],[130,32]]},
  {name:'philippines',pts:[[120,14],[123,13],[126,10],[125,7],[121,6],[119,10],[120,14]]},
  {name:'sri lanka',pts:[[80,9],[82,7],[82,6],[80,6],[80,9]]},
  {name:'taiwan',pts:[[120,22],[121,25],[122,25],[121,22],[120,22]]},
  {name:'cuba',pts:[[-84,22],[-80,23],[-76,22],[-74,20],[-77,20],[-82,21],[-84,22]]},
  {name:'antarctica',pts:[[-180,-72],[-160,-74],[-135,-73],[-110,-74],[-85,-73],[-60,-72],[-40,-75],[-15,-72],[10,-70],[35,-70],[60,-68],[85,-67],[110,-66],[135,-67],[160,-70],[180,-72],[180,-90],[-180,-90],[-180,-72]]},
  {name:'greenland',pts:[[-46,60],[-42,62],[-36,64],[-30,67],[-22,70],[-20,73],[-25,78],[-30,82],[-42,83],[-52,82],[-58,78],[-60,73],[-55,68],[-52,63],[-48,60],[-46,60]]},
];

function populateWorldMap(){
  const svg=$('#worldMap');
  const cities=state.meta.cities;
  const proj=(lng,lat)=>{const x=(lng+180)/360*900;const y=(90-lat)/180*460;return[x,y]};
  const parts=[];
  parts.push(`<g class="map-continents">`);
  CONTINENTS.forEach(c=>{
    const d=c.pts.map((p,i)=>{const [x,y]=proj(p[0],p[1]);return `${i?'L':'M'} ${x.toFixed(1)} ${y.toFixed(1)}`}).join(' ')+' Z';
    parts.push(`<path class="map-continent" d="${d}"/>`);
  });
  parts.push(`</g>`);
  parts.push(`<g class="map-graticule">`);
  for(let lng=-180;lng<=180;lng+=30){const [x1]=proj(lng,-90),[x2]=proj(lng,90);parts.push(`<line x1="${x1}" y1="0" x2="${x2}" y2="460"/>`)}
  for(let lat=-60;lat<=60;lat+=30){const [,y1]=proj(-180,lat),[,y2]=proj(180,lat);parts.push(`<line x1="0" y1="${y1}" x2="900" y2="${y2}"/>`)}
  parts.push(`</g>`);
  parts.push(`<g id="mapArcs"></g>`);
  parts.push(`<g id="mapCities">`);
  cities.forEach((c,i)=>{
    const [x,y]=proj(c.lng,c.lat);
    const r=Math.min(6,2+Math.log2(c.pop+1));
    parts.push(`<g class="map-city" data-i="${i}" transform="translate(${x},${y})">
      <circle class="halo" r="6" style="animation:none"/>
      <circle class="ring" r="0" fill="none" stroke="#0a0a0a" stroke-width="1" opacity="0"/>
      <circle class="core" r="${r}" fill="#c8c4b5" stroke="#5b5b5b" stroke-width=".8"/>
      <text y="-8" text-anchor="middle">${c.n}</text>
    </g>`);
  });
  parts.push(`</g>`);
  svg.innerHTML=parts.join('');
  let tip=document.querySelector('.map-card .map-tooltip');
  if(!tip){tip=document.createElement('div');tip.className='map-tooltip';document.querySelector('.map-card').appendChild(tip)}
  svg.querySelectorAll('.map-city').forEach(g=>{
    const i=+g.dataset.i;
    g.addEventListener('click',()=>{
      state.seedCity=i;
      svg.querySelectorAll('.map-city').forEach(x=>x.classList.toggle('seed',+x.dataset.i===i));
      $('#seedCityChip').textContent='seed · '+state.meta.cities[i].n;
    });
    g.addEventListener('mouseenter',e=>showCityTip(i,e));
    g.addEventListener('mousemove',e=>showCityTip(i,e));
    g.addEventListener('mouseleave',()=>{tip.classList.remove('on')});
  });
}

function showCityTip(i,e){
  const tip=document.querySelector('.map-card .map-tooltip');if(!tip)return;
  const c=state.meta.cities[i];
  const rows=[`<b>${c.n}</b>`,`<div class="tt-row">pop <b>${c.pop.toFixed(1)}M</b> · ${continentOf(c.lat,c.lng)}</div>`];
  if(state.drift){
    const arr=state.drift.arrival[i];
    const risk=state.drift.ml_risk_60d?state.drift.ml_risk_60d[i]:null;
    const share=state.drift.shares[i][state.driftT||0];
    rows.push(`<div class="tt-row">arrival <b>${isFinite(arr)?arr.toFixed(1)+' d':'—'}</b> · risk 60d <b>${risk!=null?(risk*100).toFixed(0)+'%':'—'}</b></div>`);
    rows.push(`<div class="tt-row">share today <b>${(share*100).toFixed(1)}%</b> · peak <b>${(Math.max(...state.drift.shares[i])*100).toFixed(1)}%</b></div>`);
  }else{
    rows.push(`<div class="tt-row">click to set as seed</div>`);
  }
  tip.innerHTML=rows.join('');
  const container=document.querySelector('.map-card').getBoundingClientRect();
  tip.style.left=(e.clientX-container.left+12)+'px';
  tip.style.top=(e.clientY-container.top-12)+'px';
  tip.classList.add('on');
}

function projMap(lng,lat){return[(lng+180)/360*900,(90-lat)/180*460]}

function renderMapArcs(){
  const g=document.getElementById('mapArcs');if(!g||!state.drift)return;
  const parts=[];
  const arcs=state.drift.arcs||[];
  const maxDelay=Math.max(...arcs.map(a=>a.delay),1);
  arcs.forEach(a=>{
    const risk=state.drift.ml_risk_60d?state.drift.ml_risk_60d[a.target]:0.5;
    const opacity=(0.15+risk*0.55).toFixed(2);
    const d=a.path.map((p,i)=>{const [x,y]=projMap(p[1],p[0]);return `${i?'L':'M'} ${x.toFixed(1)} ${y.toFixed(1)}`}).join(' ');
    parts.push(`<path class="map-arc map-arc-dash" d="${d}" opacity="${opacity}"/>`);
  });
  g.innerHTML=parts.join('');
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
    renderDriftShare();renderReachedChart();renderMlFeat();renderMapArcs();renderDriftFrame(0);renderDriftRationale();
    const badge=$('#mlBadge');
    if(res.ml_used){badge.textContent='ml · '+(res.ml_feature_importance?.length||0)+' features';badge.className='chip ok'}
    else{badge.textContent='ml offline (heuristic fallback)';badge.className='chip warn'}
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

function renderMlFeat(){
  const fi=state.drift.ml_feature_importance||[];
  if(!fi.length){return}
  const sorted=[...fi].sort((a,b)=>b.risk_importance-a.risk_importance).slice(0,8);
  const c=state.charts.mlFeat;
  c.data.labels=sorted.map(x=>x.feature);
  c.data.datasets[0].data=sorted.map(x=>+x.risk_importance.toFixed(3));
  c.data.datasets[0].backgroundColor=sorted.map((_,i)=>i===0?'#8a1e1e':i<3?'#7a5b12':INK);
  c.update();
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

function continentOf(lat,lng){
  if(lng>=-170&&lng<=-30&&lat>=13)return 'North America';
  if(lng>=-90&&lng<=-30&&lat>=-60&&lat<15)return 'South America';
  if(lng>=-25&&lng<=50&&lat>=-40&&lat<38)return 'Africa';
  if(lng>=-15&&lng<=45&&lat>=35)return 'Europe';
  if(lng>=45&&lat>=-15)return 'Asia';
  if(lat<-15&&lng>=110)return 'Oceania';
  return 'Other';
}

function renderCityTable(){
  if(!state.drift)return;
  const t=state.driftT||0;
  const cities=state.meta.cities;
  const rows=cities.map((c,i)=>({
    i,city:c.n,pop:c.pop,arrival:state.drift.arrival[i],
    risk:state.drift.ml_risk_60d?state.drift.ml_risk_60d[i]:null,
    share:state.drift.shares[i][t],
    peak:Math.max(...state.drift.shares[i]),
    continent:continentOf(c.lat,c.lng),
  })).sort((a,b)=>a.arrival-b.arrival);
  $('#cityTableChip').textContent=`day ${t} · ${rows.filter(r=>r.arrival<=t).length} arrived`;
  $('#cityBody').innerHTML=rows.map(r=>{
    const riskCls=r.risk==null?'':r.risk>=0.6?'impact-high':r.risk>=0.35?'impact-med':'impact-low';
    const shareCls=r.share>=0.5?'impact-high':r.share>=0.1?'impact-med':'impact-low';
    const arriv=isFinite(r.arrival)?r.arrival.toFixed(1)+' d':'—';
    return `<tr><td>${r.city}</td><td>${r.pop.toFixed(1)}</td><td>${arriv}</td><td class="${riskCls}">${r.risk==null?'—':(r.risk*100).toFixed(0)+'%'}</td><td class="${shareCls}">${(r.share*100).toFixed(1)}%</td><td>${(r.peak*100).toFixed(1)}%</td><td>${r.continent}</td></tr>`;
  }).join('');
}

function renderDriftFrame(t){
  state.driftT=t;$('#driftDay').textContent='day '+t;
  const svg=$('#worldMap');
  svg.querySelectorAll('.map-city').forEach((g,i)=>{
    const share=state.drift.shares[i][t];
    const arrivedDay=state.drift.arrival?.[i];
    const core=g.querySelector('.core');const ring=g.querySelector('.ring');const halo=g.querySelector('.halo');
    let fill='#c8c4b5';
    if(share>=0.5)fill='#0a0a0a';
    else if(share>=0.1)fill='#5b5b5b';
    else if(share>=0.02)fill='#7a5b12';
    else if(share>0)fill='#a0a0a0';
    core.setAttribute('fill',fill);
    if(share>0.02){
      const rad=6+share*22;
      ring.setAttribute('r',rad);
      ring.setAttribute('opacity',(0.15+share*0.4).toFixed(2));
    }else{ring.setAttribute('opacity','0')}
    if(halo){
      const dt=t-(arrivedDay??Infinity);
      if(dt>=0&&dt<3){halo.style.animation='halo 2.6s ease-out infinite';halo.setAttribute('stroke',share>=0.5?'#0a0a0a':'#7a5b12')}
      else{halo.style.animation='none';halo.setAttribute('opacity','0')}
    }
  });
  renderCityTable();
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

const MOL_ICONS={
  'small molecule protease inhibitor':`<g><polygon class="fill" points="32,10 52,22 52,44 32,56 12,44 12,22"/><line x1="32" y1="10" x2="32" y2="0"/><line x1="52" y1="22" x2="60" y2="18"/><line x1="52" y1="44" x2="60" y2="48"/><line x1="12" y1="44" x2="4" y2="48"/><line x1="12" y1="22" x2="4" y2="18"/></g>`,
  'monoclonal antibody':`<g><line x1="32" y1="56" x2="32" y2="30"/><line x1="32" y1="30" x2="16" y2="10"/><line x1="32" y1="30" x2="48" y2="10"/><circle cx="16" cy="10" r="4" class="fill"/><circle cx="48" cy="10" r="4" class="fill"/><rect x="28" y="46" width="8" height="10" class="fill"/></g>`,
  'peptide fusion inhibitor':`<g><circle cx="10" cy="20" r="4" class="fill"/><circle cx="22" cy="14" r="4" class="fill"/><circle cx="34" cy="20" r="4" class="fill"/><circle cx="46" cy="14" r="4" class="fill"/><circle cx="54" cy="24" r="4" class="fill"/><circle cx="46" cy="38" r="4" class="fill"/><circle cx="30" cy="46" r="4" class="fill"/><circle cx="16" cy="42" r="4" class="fill"/><polyline points="10,20 22,14 34,20 46,14 54,24 46,38 30,46 16,42"/></g>`,
  'polymerase inhibitor':`<g><polygon class="fill" points="32,8 52,22 52,42 32,56 12,42 12,22"/><rect x="24" y="24" width="16" height="16" fill="var(--panel-2)" stroke="var(--panel-2)"/><line x1="32" y1="24" x2="32" y2="40"/><line x1="24" y1="32" x2="40" y2="32"/></g>`,
  'neuraminidase inhibitor':`<g><circle cx="32" cy="32" r="16" class="fill"/><circle cx="32" cy="32" r="8" fill="var(--panel-2)" stroke="var(--panel-2)"/><line x1="16" y1="32" x2="8" y2="32"/><line x1="48" y1="32" x2="56" y2="32"/><line x1="32" y1="16" x2="32" y2="8"/><line x1="32" y1="48" x2="32" y2="56"/></g>`,
  'hemagglutinin stem antibody':`<g><line x1="32" y1="56" x2="32" y2="30"/><line x1="32" y1="30" x2="16" y2="10"/><line x1="32" y1="30" x2="48" y2="10"/><circle cx="16" cy="10" r="4" class="fill"/><circle cx="48" cy="10" r="4" class="fill"/><rect x="28" y="46" width="8" height="10" class="fill"/></g>`,
  'cap-snatching inhibitor':`<g><polygon class="fill" points="32,10 44,20 44,40 32,50 20,40 20,20"/><path d="M 20 20 L 12 14 M 44 20 L 52 14 M 20 40 L 12 46 M 44 40 L 52 46"/></g>`,
  'integrase strand-transfer inhibitor':`<g><polygon class="fill" points="10,32 22,10 46,10 58,32 46,54 22,54"/><line x1="22" y1="10" x2="46" y2="54"/><line x1="46" y1="10" x2="22" y2="54"/></g>`,
  'gp120 broadly neutralizing antibody':`<g><line x1="32" y1="56" x2="32" y2="30"/><line x1="32" y1="30" x2="16" y2="10"/><line x1="32" y1="30" x2="48" y2="10"/><circle cx="16" cy="10" r="5" class="fill"/><circle cx="48" cy="10" r="5" class="fill"/><rect x="28" y="46" width="8" height="10" class="fill"/></g>`,
  'fusion inhibitor':`<g><path d="M 12 32 C 20 12, 44 12, 52 32 S 44 52, 12 32" fill="none"/><path d="M 20 32 C 24 22, 40 22, 44 32" fill="none"/></g>`,
  'capsid inhibitor':`<g><polygon class="fill" points="32,6 54,20 54,44 32,58 10,44 10,20"/><circle cx="32" cy="32" r="6" fill="var(--panel-2)" stroke="var(--panel-2)"/></g>`,
  'rpoB rescue rifamycin analog':`<g><rect x="12" y="24" width="40" height="16" rx="8" class="fill"/><circle cx="20" cy="32" r="3" fill="var(--panel-2)"/><circle cx="32" cy="32" r="3" fill="var(--panel-2)"/><circle cx="44" cy="32" r="3" fill="var(--panel-2)"/></g>`,
  'bedaquiline analog':`<g><polygon class="fill" points="20,10 44,10 54,32 44,54 20,54 10,32"/><line x1="32" y1="10" x2="32" y2="54"/></g>`,
  'cell-wall inhibitor':`<g><rect x="10" y="12" width="12" height="12" class="fill"/><rect x="26" y="12" width="12" height="12" class="fill"/><rect x="42" y="12" width="12" height="12" class="fill"/><rect x="18" y="26" width="12" height="12" class="fill"/><rect x="34" y="26" width="12" height="12" class="fill"/><rect x="10" y="40" width="12" height="12" class="fill"/><rect x="26" y="40" width="12" height="12" class="fill"/><rect x="42" y="40" width="12" height="12" class="fill"/></g>`,
  'oxazolidinone':`<g><polygon points="32,10 46,20 40,38 24,38 18,20" class="fill"/><line x1="32" y1="38" x2="32" y2="54"/><line x1="24" y1="38" x2="20" y2="54"/><line x1="40" y1="38" x2="44" y2="54"/></g>`,
};

function setChipUnderCard(canvasId,text){
  const card=document.getElementById(canvasId)?.closest('.card');
  if(!card)return;
  let chip=card.querySelector('.card-head .chip.stat-chip');
  if(!chip){
    const head=card.querySelector('.card-head');
    if(!head)return;
    chip=document.createElement('span');chip.className='chip stat-chip';head.appendChild(chip);
  }
  chip.textContent=text;
}

function moleculeSvg(className){
  const inner=MOL_ICONS[className]||`<circle cx="32" cy="32" r="18" class="fill"/>`;
  return `<svg class="drug-molecule" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">${inner}</svg>`;
}

function successRing(pct){
  const r=22;const c=2*Math.PI*r;
  const cls=pct>=0.6?'high':pct>=0.4?'med':'low';
  const off=c*(1-pct);
  return `<div class="success-ring"><svg viewBox="0 0 56 56"><circle class="ring-bg" cx="28" cy="28" r="${r}"/><circle class="ring-fg ${cls}" cx="28" cy="28" r="${r}" stroke-dasharray="${c.toFixed(1)}" stroke-dashoffset="${off.toFixed(1)}"/><text class="ring-text" x="28" y="28">${(pct*100).toFixed(0)}%</text></svg></div>`;
}

function renderDrugCards(d){
  const cards=$('#drugCards');const top=d.candidates.slice(0,8);
  cards.innerHTML=top.map((c,i)=>{
    const rankCls=i===0?'rank1':i===1?'rank2':i===2?'rank3':'';
    return `<div class="drug-card ${i===0?'active':''}" data-idx="${i}">
      <span class="rank-pill ${rankCls}">#${i+1}</span>
      ${moleculeSvg(c.class)}
      <div class="drug-name">${c.variant}</div>
      <div class="drug-class">${c.class}</div>
      ${successRing(c.success_prob)}
    </div>`;
  }).join('');
  cards.querySelectorAll('.drug-card').forEach(el=>{
    el.addEventListener('click',()=>{
      cards.querySelectorAll('.drug-card').forEach(x=>x.classList.remove('active'));
      el.classList.add('active');
      const idx=+el.dataset.idx;
      selectDrug(d,idx);
    });
  });
  selectDrug(d,0);
}

function selectDrug(d,idx){
  const c=d.candidates[idx];
  $('#drugCardsChip').textContent=c.id;
  $('#radarChip').textContent=c.variant;
  const norm={
    binding:Math.min(1,Math.max(0,(-c.binding_kcal_mol-4)/9)),
    potency:Math.min(1,Math.max(0,1-Math.log10(Math.max(c.ic50_nm,1))/4)),
    admet:c.admet_score,
    robustness:c.resistance_robustness,
    ease:1-c.synth_complexity/10,
  };
  const sameClass=d.candidates.filter(x=>x.class===c.class);
  const avg={
    binding:sameClass.reduce((a,b)=>a+Math.min(1,Math.max(0,(-b.binding_kcal_mol-4)/9)),0)/sameClass.length,
    potency:sameClass.reduce((a,b)=>a+Math.min(1,Math.max(0,1-Math.log10(Math.max(b.ic50_nm,1))/4)),0)/sameClass.length,
    admet:sameClass.reduce((a,b)=>a+b.admet_score,0)/sameClass.length,
    robustness:sameClass.reduce((a,b)=>a+b.resistance_robustness,0)/sameClass.length,
    ease:sameClass.reduce((a,b)=>a+(1-b.synth_complexity/10),0)/sameClass.length,
  };
  const r=state.charts.radar;
  r.data.datasets[0].data=[avg.binding,avg.potency,avg.admet,avg.robustness,avg.ease];
  r.data.datasets[1].data=[norm.binding,norm.potency,norm.admet,norm.robustness,norm.ease];
  r.update();
  renderPocket(c);
}

function renderPocket(candidate){
  const svg=$('#pocketDiagram');if(!svg)return;
  const W=900,H=220;
  const a=state.analysis;
  if(!a){svg.innerHTML=`<text x="450" y="110" text-anchor="middle" class="pocket-label" fill="#8a8a8a">no sequence in memory</text>`;return}
  const L=a.reference.length;const midY=110;
  const parts=[];
  parts.push(`<line class="pocket-axis" x1="40" y1="${midY}" x2="${W-40}" y2="${midY}"/>`);
  parts.push(`<line class="pocket-rail" x1="40" y1="${midY}" x2="${W-40}" y2="${midY}"/>`);
  a.reference.hotspots.forEach(pos=>{
    const x=40+(pos/L)*(W-80);
    parts.push(`<rect class="pocket-hotspot" x="${x-8}" y="${midY-40}" width="16" height="80"/>`);
    parts.push(`<text class="pocket-label" x="${x}" y="${midY-46}" text-anchor="middle" fill="#7a5b12">${pos}</text>`);
  });
  a.mutations.forEach(m=>{
    const x=40+(m.pos/L)*(W-80);
    const h=8+m.impact*40;
    const color=m.impact>=0.6?'#8a1e1e':m.impact>=0.35?'#7a5b12':'#5b5b5b';
    parts.push(`<line class="pocket-mut" x1="${x.toFixed(1)}" y1="${(midY-h).toFixed(1)}" x2="${x.toFixed(1)}" y2="${(midY+h).toFixed(1)}" stroke="${color}" stroke-width="2.2" opacity="0.85"/>`);
  });
  const bindSeed=Math.abs(candidate.id.charCodeAt(candidate.id.length-1)+candidate.id.charCodeAt(0));
  const bindPos=Math.min(0.92,Math.max(0.08,((bindSeed*13)%97)/100));
  const bx=40+bindPos*(W-80);
  parts.push(`<path class="pocket-binder" d="M ${bx.toFixed(1)} 30 Q ${bx.toFixed(1)} 60 ${bx.toFixed(1)} ${midY-8}"/>`);
  parts.push(`<polygon class="pocket-binder-arrow" points="${bx-5},${midY-14} ${bx+5},${midY-14} ${bx},${midY-4}"/>`);
  parts.push(`<text class="pocket-label" x="${bx}" y="24" text-anchor="middle">${candidate.variant}</text>`);
  parts.push(`<text class="pocket-label" x="${bx}" y="${midY+34}" text-anchor="middle" fill="#5b5b5b">binds ~ pos ${Math.round(bindPos*L)}</text>`);
  parts.push(`<text class="pocket-label" x="46" y="${midY+8}" fill="#8a8a8a">N-term</text>`);
  parts.push(`<text class="pocket-label" x="${W-46}" y="${midY+8}" text-anchor="end" fill="#8a8a8a">C-term (${L} aa)</text>`);
  svg.innerHTML=parts.join('');
  const nearest=a.mutations.map(m=>Math.abs(m.pos-bindPos*L)).sort((a,b)=>a-b)[0]||L;
  $('#pocketChip').textContent=`nearest mutation Δ ${nearest.toFixed(0)} aa`;
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
  const total=d.candidates.length;
  const sh=state.charts.successHist;
  sh.data.labels=['5','15','25','35','45','55','65','75','85','95'];
  sh.data.datasets[0].data=dist.success_hist;
  sh.data.datasets[0].backgroundColor=dist.success_hist.map((_,i)=>i>=7?OK:i>=5?WARN:INK);
  sh.options.scales.y.suggestedMax=Math.max(3,Math.max(...dist.success_hist)+1);
  sh.update();
  const succMean=d.candidates.reduce((a,b)=>a+b.success_prob,0)/total;
  const succHi=d.candidates.filter(c=>c.success_prob>=0.6).length;
  setChipUnderCard('successHist',`mean ${(succMean*100).toFixed(0)}% · ${succHi}/${total} ≥60%`);

  const bind=state.charts.bindingChart;
  bind.data.labels=['−4','−6','−7','−8.5','−10','−11.5'];
  bind.data.datasets[0].data=dist.binding_bins;
  bind.data.datasets[0].backgroundColor=dist.binding_bins.map((_,i)=>i>=3?OK:INK);
  bind.options.scales.y.suggestedMax=Math.max(3,Math.max(...dist.binding_bins)+1);
  bind.update();
  const bindMean=d.candidates.reduce((a,b)=>a+b.binding_kcal_mol,0)/total;
  const bindStrong=d.candidates.filter(c=>c.binding_kcal_mol<=-8).length;
  setChipUnderCard('bindingChart',`mean ΔG ${bindMean.toFixed(2)} · ${bindStrong}/${total} ≤−8`);

  const scat=state.charts.admetScatter;
  scat.data.datasets[0].data=dist.scatter_admet_synth.map(p=>({x:p.x,y:p.y,s:p.s,id:p.id}));
  scat.data.datasets[0].pointBackgroundColor=dist.scatter_admet_synth.map(p=>p.s>=0.6?OK:p.s>=0.4?WARN:INK);
  scat.update();
  const admetMean=d.candidates.reduce((a,b)=>a+b.admet_score,0)/total;
  const synthMean=d.candidates.reduce((a,b)=>a+b.synth_complexity,0)/total;
  setChipUnderCard('admetScatter',`ADMET ${admetMean.toFixed(2)} · synth ${synthMean.toFixed(1)}/10`);

  const cls=state.charts.classChart;
  cls.data.labels=dist.class_summary.map(c=>c.class.length>26?c.class.slice(0,26)+'…':c.class);
  cls.data.datasets[0].data=dist.class_summary.map(c=>c.mean_success);
  cls.data.datasets[0].backgroundColor=dist.class_summary.map(c=>c.mean_success>=0.6?OK:c.mean_success>=0.4?WARN:INK);
  cls.options.layout={padding:{right:60}};
  cls.update();
  const topClass=dist.class_summary[0];
  setChipUnderCard('classChart',topClass?`leader · ${topClass.class.split(' ').slice(0,2).join(' ')} @ ${(topClass.mean_success*100).toFixed(0)}%`:'—');

  const ic50s=d.candidates.map(c=>c.ic50_nm);
  const bins=[0,0,0,0,0,0];
  const edges=[10,50,100,500,1000,5000];
  ic50s.forEach(v=>{for(let i=0;i<edges.length;i++)if(v<=edges[i]){bins[i]++;break}});
  const ic=state.charts.ic50Chart;
  ic.data.labels=['≤10','≤50','≤100','≤500','≤1k','≤5k'];
  ic.data.datasets[0].data=bins;
  ic.data.datasets[0].backgroundColor=bins.map((_,i)=>i<=1?OK:i<=3?INK:WARN);
  ic.options.scales.y.suggestedMax=Math.max(3,Math.max(...bins)+1);
  ic.update();
  const ic50Median=[...ic50s].sort((a,b)=>a-b)[Math.floor(ic50s.length/2)];
  const sub100=ic50s.filter(v=>v<=100).length;
  setChipUnderCard('ic50Chart',`median ${ic50Median} nM · ${sub100}/${total} sub-100 nM`);

  const rs=state.charts.robustScatter;
  rs.data.datasets[0].data=d.candidates.map(c=>({x:c.resistance_robustness,y:c.binding_kcal_mol,id:c.id,s:c.success_prob}));
  rs.data.datasets[0].pointBackgroundColor=d.candidates.map(c=>c.success_prob>=0.6?OK:c.success_prob>=0.4?WARN:INK);
  rs.update();
  const robMean=d.candidates.reduce((a,b)=>a+b.resistance_robustness,0)/total;
  const bestBoth=d.candidates.filter(c=>c.resistance_robustness>=0.7&&c.binding_kcal_mol<=-8).length;
  setChipUnderCard('robustScatter',`mean robust ${robMean.toFixed(2)} · ${bestBoth}/${total} top-left`);

  $('#drugBody').innerHTML=d.candidates.map((x,i)=>{
    const cls=x.success_prob>=0.6?'impact-high':x.success_prob>=0.4?'impact-med':'impact-low';
    return `<tr><td>${x.id}</td><td>${x.class}</td><td>${x.target}</td><td>${x.binding_kcal_mol}</td><td>${x.admet_score}</td><td>${x.synth_complexity}</td><td>${x.ic50_nm}</td><td>${x.resistance_robustness}</td><td class="${cls}"><span class="impact-bar"><span style="width:${(x.success_prob*100).toFixed(0)}%"></span></span>${(x.success_prob*100).toFixed(1)}%</td></tr>`;
  }).join('');
  renderDrugCards(d);
  requestAnimationFrame(()=>{
    ['successHist','bindingChart','admetScatter','classChart','ic50Chart','robustScatter','radar'].forEach(k=>{
      try{state.charts[k].resize();state.charts[k].update('none')}catch(e){}
    });
  });
}

document.addEventListener('DOMContentLoaded',boot);
