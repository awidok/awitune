var sel=null,tab='agent',viewingFile=false;
function gsu(){try{return new URL(window.location.href).searchParams.get('exp')}catch(e){return null}}
function ssu(n){try{var u=new URL(window.location.href);if(n)u.searchParams.set('exp',n);else u.searchParams.delete('exp');window.history.replaceState({},'',u.toString())}catch(e){}}
function fe(s){if(!s)return'\u2026';var sec=Math.floor((new Date()-new Date(s))/1000);if(sec<0)return'0s';var h=Math.floor(sec/3600),m=Math.floor((sec%3600)/60),ss=sec%60;if(h>0)return h+'h'+m+'m';if(m>0)return m+'m'+ss+'s';return ss+'s'}
function fm(min){if(!min&&min!==0)return'\u2014';if(min<1)return'<1m';if(min>=60)return Math.floor(min/60)+'h'+Math.round(min%60)+'m';return Math.round(min)+'m'}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}

document.getElementById('ltb').onclick=function(e){
    if(!e.target.classList.contains('t'))return;
    document.querySelectorAll('.t').forEach(function(t){t.classList.remove('a')});
    e.target.classList.add('a');tab=e.target.dataset.t;loadTab();
};

async function rf(){
try{
    var r=await fetch('/api/state'),d=await r.json();
    document.getElementById('bv').textContent=d.best_score>0?d.best_score.toFixed(6):'\u2014';
    document.getElementById('bl').textContent=d.metric?'Best '+d.metric:'Best Score';

    // Worker
    var wsE=document.getElementById('ws'),wbE=document.getElementById('wb');
    if(d.worker_running){wsE.innerHTML='<span class="dot g"></span> Worker';wbE.textContent='\u23F9 Worker';wbE.className='bt r'}
    else{wsE.innerHTML='<span class="dot r"></span> Worker';wbE.textContent='\u25B6 Worker';wbE.className='bt g'}

    // Proxy
    var psE=document.getElementById('ps'),pbE=document.getElementById('pb');
    if(d.proxy_running){psE.innerHTML='<span class="dot g"></span> Proxy';pbE.textContent='\u23F9 Proxy';pbE.className='bt r'}
    else{psE.innerHTML='<span class="dot r"></span> Proxy';pbE.textContent='\u25B6 Proxy';pbE.className='bt g'}

    // Ideas
    var afE=document.getElementById('afs');
    afE.innerHTML=d.worker_running?'<span class="dot g"></span> Ideas '+(d.ideas_used||0):'<span class="dot r"></span> Ideas';

    // GPUs
    var totalS=(d.available_gpus?d.available_gpus.length:0)*(d.slots_per_gpu||1);
    var usedS=Object.values(d.used_slots||{}).reduce(function(a,b){return a+(b||0)},0);
    document.getElementById('gpuc').textContent=Math.max(0,totalS-usedS)+'/'+totalS+' free';
    document.getElementById('gbar').innerHTML=(d.available_gpus||[]).map(function(g){
        var u=(d.used_slots&&d.used_slots[String(g)])||0;
        return '<div class="gc '+(u>0?'bu':'fr')+'">GPU '+g+'<br>'+u+'/'+(d.slots_per_gpu||1)+'</div>';
    }).join('');

    // Queue
    if(d.queue.length>0){
        document.getElementById('qc').style.display='';
        document.getElementById('qnn').textContent='('+d.queue.length+')';
        document.getElementById('ql').innerHTML=d.queue.map(function(q,i){
            var n=q.name||q.idea_name||q.id||'task';
            return '<div class="qi"><span class="qn" title="'+esc(n)+'">'+esc(n)+'</span><span class="qp">'+esc(q.prompt||'')+'</span><button class="bt sm" onclick="rq('+i+')">✕</button></div>';
        }).join('');
    } else {
        document.getElementById('qc').style.display='none';
    }

    // Experiments
    var exps=d.history||[],oM=d.oof_progress||{};
    document.getElementById('ec').textContent='('+exps.length+')';
    document.getElementById('el').innerHTML=exps.map(function(e){
        var ac=sel===e.name?'a':'';
        var iA=e.task_type==='analysis',iS=e.task_type==='stacking',iO=e.task_type==='oof_fold';
        var iSm=e.name&&e.name.indexOf('smart_')===0;
        var sc='fl',sv=iA?'\u2014':'N/A',cv2=iA?'\u2014':'N/A';
        var gpu=e.gpu_id!=null?e.gpu_id:(e.gpu!=null?e.gpu:'?');

        if(e.status==='running'){sc='rn';sv='\u23F3'}
        else if(e.status==='queued'){sc='';sv='\u23F8'}
        else if(e.test_score&&e.test_score>0){sv=parseFloat(e.test_score).toFixed(6);sc=e.improved?'ok':''}
        if(e.cv_score!=null) cv2=parseFloat(e.cv_score).toFixed(6);

        var ts2;
        if(e.status==='running') ts2='<span style="color:#d29922">\u23F1'+fe(e.started_at)+'</span>';
        else if(e.elapsed_min) ts2=fm(e.elapsed_min);
        else ts2=e.status||'\u2026';

        var bg='';
        if(iA) bg+='<span class="badge ba">analysis</span> ';
        else if(iS) bg+='<span class="badge bs2">stack</span> ';
        else if(iO) bg+='<span class="badge bo">oof</span> ';
        if(iSm) bg+='<span class="badge bsm">smart</span> ';

        var sb=e.status==='running'?'<button class="bt r sm" onclick="event.stopPropagation();ke(\''+e.name+'\')" title="Stop">\u23F9</button>':'';
        var fb=e.status==='completed'&&e.test_score&&!iA?'<button class="bt sm" onclick="event.stopPropagation();fk(\''+e.name+'\')" title="Fork">\uD83C\uDF74</button>':'';
        var rb=e.status!=='running'&&e.status!=='queued'?'<button class="bt sm" onclick="event.stopPropagation();re(\''+e.name+'\')" title="Restart">\uD83D\uDD04</button>':'';
        var cvb=!iA&&!iO?'<button class="bt sm" onclick="event.stopPropagation();cv(\''+e.name+'\')" title="OOF/CV">\uD83E\uDDEE</button>':'';
        var db2='<button class="bt sm" onclick="event.stopPropagation();de(\''+e.name+'\')" title="Delete">\uD83D\uDDD1</button>';

        var pt=e.parent_experiment?'<span style="color:#58a6ff;font-size:8px">\u2190 '+e.parent_experiment.slice(0,18)+'</span>':'';
        var cc=iA?'':' \u00B7 cv:'+cv2;
        var oo=oM[e.name];
        var oc=oo?' \u00B7 oof:'+oo.completed+'/'+oo.total:'';

        return '<div class="ei '+ac+'" onclick="se(\''+e.name+'\')"><div class="left"><div class="n">'+bg+'<span title="'+esc(e.name)+'">'+esc(e.name)+'</span></div><div class="m">gpu'+gpu+' \u00B7 '+ts2+cc+oc+' '+pt+'</div></div><div class="act"><span class="s '+sc+'">'+sv+'</span>'+sb+fb+rb+cvb+db2+'</div></div>';
    }).join('');

    if(sel&&!exps.some(function(e){return e.name===sel})){sel=null;ssu(null)}
    if(!sel&&exps.length>0){var rn=exps.find(function(e){return e.status==='running'});se(rn?rn.name:exps[exps.length-1].name)}

    // Experiment header + metrics for selected
    if(sel){
        var s=exps.find(function(e){return e.name===sel});
        if(s){
            // Experiment name header
            var eh=document.getElementById('expHeader');
            eh.style.display='';
            document.getElementById('expName').textContent=s.name;
            var badges='';
            if(s.task_type==='analysis') badges+='<span class="badge ba">analysis</span>';
            else if(s.task_type==='stacking') badges+='<span class="badge bs2">stacking</span>';
            else if(s.task_type==='oof_fold') badges+='<span class="badge bo">oof</span>';
            if(s.name&&s.name.indexOf('smart_')===0) badges+='<span class="badge bsm">smart</span>';
            document.getElementById('expBadges').innerHTML=badges;
            var stColor=s.status==='running'?'#d29922':(s.status==='completed'?'#3fb950':(s.status==='queued'?'#8b949e':'#f85149'));
            document.getElementById('expStatus').innerHTML='<span style="color:'+stColor+'">\u25CF '+s.status+'</span>';

            // Expandable prompt (preserve expand/collapse state across refreshes)
            var pw=document.getElementById('expPromptWrap');
            if(s.prompt){
                pw.style.display='';
                var short=s.prompt.length>150?s.prompt.slice(0,150)+'\u2026 \u25BC click to expand':s.prompt;
                document.getElementById('expPromptShort').innerHTML='\uD83D\uDCDD '+esc(short);
                document.getElementById('expPromptFull').textContent=s.prompt;
                // Don't reset display state — preserve user's expand/collapse choice
            } else {
                pw.style.display='none';
            }

            // OOF card
            var oo=oM[s.name],oC='';
            if(oo){
                var fl=(oo.folds||[]).map(function(f){
                    var st=f.status||'queued';
                    var co=st==='completed'?'#3fb950':(st==='running'?'#d29922':(st==='failed'?'#f85149':'#8b949e'));
                    var ct=(f.cv_score!=null)?Number(f.cv_score).toFixed(5):'\u2014';
                    var kb=st==='running'?'<button class="bt r sm" onclick="event.stopPropagation();ke(\''+f.name+'\')">\u23F9</button>':'';
                    var rb2=st!=='running'&&st!=='queued'?'<button class="bt sm" onclick="event.stopPropagation();re(\''+f.name+'\')">\uD83D\uDD04</button>':'';
                    return '<span style="display:inline-flex;gap:3px;align-items:center;background:#0d1117;border:1px solid #30363d;border-radius:3px;padding:1px 5px;margin:1px;cursor:pointer;font-size:9px" onclick="ofl(\''+f.name+'\')"><span style="color:'+co+'">\u25CF</span>'+(f.name||'').slice(0,25)+' <span style="color:#8b949e">cv:'+ct+'</span>'+kb+rb2+'</span>';
                }).join('');
                oC='<div class="mc" style="min-width:240px;text-align:left"><div class="l">OOF</div><div style="font-size:10px;color:#c9d1d9;margin-top:1px">'+oo.running+'\u25B6 '+oo.queued+'\u23F8 '+oo.completed+'\u2713 '+oo.failed+'\u2717</div><div style="margin-top:3px;max-height:60px;overflow:auto">'+(fl||'\u2014')+'</div></div>';
            }

            var tv,tc2='#d29922';
            if(s.status==='running') tv=fe(s.started_at);
            else if(s.elapsed_min){tv=fm(s.elapsed_min);tc2='#8b949e'}
            else{tv='\u2014';tc2='#8b949e'}

            var tS=s.test_score&&s.test_score!=='N/A'?parseFloat(s.test_score).toFixed(6):'\u2014';
            var vS=s.val_score&&s.val_score!=='N/A'?parseFloat(s.val_score).toFixed(6):'\u2014';
            var cS=s.cv_score!=null?parseFloat(s.cv_score).toFixed(6):'\u2014';
            var eC=s.exit_code===0?'#3fb950':(s.status==='running'?'#d29922':'#f85149');
            var eV=s.status==='running'?'\u23F3':(s.exit_code!=null?s.exit_code:'\u2014');

            document.getElementById('mb').innerHTML=
                '<div class="mc"><div class="v" style="color:'+(tS!=='\u2014'?'#3fb950':'#8b949e')+'">'+tS+'</div><div class="l">Test</div></div>'+
                '<div class="mc"><div class="v" style="color:#58a6ff">'+vS+'</div><div class="l">Val</div></div>'+
                '<div class="mc"><div class="v" style="color:#a371f7">'+cS+'</div><div class="l">CV</div></div>'+
                '<div class="mc"><div class="v" style="color:'+tc2+'">'+tv+'</div><div class="l">'+(s.status==='running'?'Running':'Time')+'</div></div>'+
                '<div class="mc"><div class="v" style="color:'+eC+'">'+eV+'</div><div class="l">Exit</div></div>'+
                (s.parent_experiment?'<div class="mc"><div class="v" style="font-size:11px;color:#58a6ff;cursor:pointer" onclick="se(\''+s.parent_experiment+'\')">\u2190 '+s.parent_experiment.slice(0,22)+'</div><div class="l">Parent</div></div>':'')+
                oC;
        }
    } else {
        document.getElementById('expHeader').style.display='none';
    }
}catch(e){console.error(e)}
if(!viewingFile&&(tab==='agent'||tab==='events'||tab==='tasks'||tab==='reports')) loadTab();
}

function se(n){
    if(sel!==n){
        // Reset prompt expand state when switching experiments
        var pf=document.getElementById('expPromptFull'),ps=document.getElementById('expPromptShort');
        if(pf)pf.style.display='none';
        if(ps)ps.style.display='';
    }
    sel=n;ssu(n);viewingFile=false;rf();loadTab();
}
function isNB(el){return el.scrollHeight-el.scrollTop-el.clientHeight<150}

async function loadTab(){
    if(!sel)return;viewingFile=false;
    var lc=document.getElementById('lc'),wb=isNB(lc);
    try{
        if(tab==='agent'){
            var r=await fetch('/api/log/'+sel+'/agent');lc.textContent=await r.text();
            if(wb)lc.scrollTop=lc.scrollHeight;
        }
        else if(tab==='events'){
            var r=await fetch('/api/events/'+sel),d=await r.json();
            if(!d.events||d.events.length===0){
                lc.innerHTML='<div style="padding:10px;color:#8b949e">No events yet</div>';
            } else {
                var cols={text:'#c9d1d9',tool_use:'#d29922',tool_result:'#58a6ff',result:'#3fb950',error:'#f85149',raw:'#8b949e'};
                var TRUNC=300;
                lc.innerHTML='<div style="padding:4px 0;color:#8b949e;font-size:10px">'+d.raw_lines+' lines, '+d.events.length+' events</div>'+
                d.events.map(function(ev){
                    var c=cols[ev.type]||'#8b949e',ct='',fullCt='';
                    if(ev.type==='text'){ct=esc(ev.text);fullCt=ct;ct='<span style="color:#8b949e">T'+ev.turn+':</span> '+ct}
                    else if(ev.type==='tool_use'){ct='<span style="color:#8b949e">T'+ev.turn+':</span> \uD83D\uDD27 <b>'+esc(ev.tool)+'</b> '+esc(ev.input);fullCt=esc(ev.input)}
                    else if(ev.type==='tool_result'){ct='\uD83D\uDCCB '+esc(ev.tool)+': '+esc(ev.output);fullCt=esc(ev.output)}
                    else if(ev.type==='result'){ct='\u2705 '+esc(ev.text);fullCt=esc(ev.text)}
                    else{ct=esc(ev.raw||ev.text||JSON.stringify(ev));fullCt=ct}
                    // Make long events expandable
                    if(fullCt.length>TRUNC){
                        var shortCt=ct.slice(0,TRUNC)+'\u2026';
                        return '<div style="padding:2px 0;border-bottom:1px solid #21262d;color:'+c+';font-size:11px;cursor:pointer" onclick="toggleEvt(this)">['+ev.type+'] <span class="evt-short">'+shortCt+' <span style="color:#58a6ff;font-size:9px">\u25BC expand</span></span><span class="evt-full" style="display:none;white-space:pre-wrap">'+ct+'</span></div>';
                    }
                    return '<div style="padding:2px 0;border-bottom:1px solid #21262d;color:'+c+';font-size:11px">['+ev.type+'] '+ct+'</div>';
                }).join('');
                if(wb)lc.scrollTop=lc.scrollHeight;
            }
        }
        else if(tab==='tasks'){
            var r=await fetch('/api/tasks/'+sel),d=await r.json();
            if(!d.tasks||d.tasks.length===0){
                lc.innerHTML='<div style="padding:10px;color:#8b949e">No tasks</div>';
            } else {
                var tf2=(d.task_files||[]).slice().sort(function(a,b){return(b.timestamp||0)-(a.timestamp||0)});
                lc.innerHTML=tf2.map(function(tf){
                    var sz=tf.size?(parseInt(tf.size)>1024?(parseInt(tf.size)/1024).toFixed(1)+'KB':tf.size+'B'):'?';
                    var cl=d.container?'onclick="vtask(\''+sel+'\',\''+tf.id+'\')" style="cursor:pointer"':'';
                    return '<div class="fi" '+cl+'><span>\uD83D\uDCC4 '+tf.id+'</span><span class="sz">'+(tf.modified||'?')+' \u00B7 '+sz+'</span></div>';
                }).join('')||d.tasks.map(function(t){
                    var cl=d.container?'onclick="vtask(\''+sel+'\',\''+t.id+'\')" style="cursor:pointer"':'';
                    return '<div class="fi" '+cl+'><span>\u23F3 '+t.id+'</span><span class="sz">Turn '+t.turn+'</span></div>';
                }).join('');
            }
        }
        else if(tab==='workspace'){
            var r=await fetch('/api/files/'+sel+'/workspace'),d=await r.json();
            lc.innerHTML=d.files.map(function(f){
                return '<div class="fi" onclick="vf(\''+sel+'\',\'workspace/'+f.path+'\')"><span>'+esc(f.name)+'</span><span class="sz">'+f.size+'</span></div>';
            }).join('')||'<div style="padding:10px;color:#8b949e">No files</div>';
        }
        else if(tab==='output'){
            var r=await fetch('/api/files/'+sel+'/output'),d=await r.json();
            lc.innerHTML=d.files.map(function(f){
                var dl=f.name==='submission.parquet'?'<button class="bt g" onclick="event.stopPropagation();window.location=\'/api/download/'+sel+'/submission\'" style="padding:1px 8px;font-size:10px;margin-right:6px">\u2B07 Download</button>':'';
                return '<div class="fi" onclick="vf(\''+sel+'\',\'output/'+f.path+'\')"><span>'+esc(f.name)+'</span><span>'+dl+'<span class="sz">'+f.size+'</span></span></div>';
            }).join('')||'<div style="padding:10px;color:#8b949e">No files</div>';
        }
        else if(tab==='reports'){
            var r=await fetch('/api/analyst_reports'),d=await r.json();
            if(!d.reports||d.reports.length===0){
                lc.innerHTML='<div style="padding:10px;color:#8b949e">No analyst reports yet.</div>';
            } else {
                lc.innerHTML=d.reports.map(function(rp){
                    return '<div class="fi" onclick="viewReport(\''+rp.name+'\')"><span>\uD83D\uDCCA '+esc(rp.name)+'</span><span class="sz">'+(rp.size/1024).toFixed(1)+'KB \u00B7 '+rp.modified.slice(0,16)+'</span></div>';
                }).join('');
            }
        }
    }catch(e){lc.textContent='Error: '+e.message}
}

async function vf(exp,path){
    viewingFile=true;var lc=document.getElementById('lc');
    try{var r=await fetch('/api/file/'+exp+'?path='+encodeURIComponent(path));lc.textContent=await r.text()}
    catch(e){lc.textContent='Error: '+e.message}
}

async function vtask(exp,taskId){
    viewingFile=true;var lc=document.getElementById('lc');
    try{
        var r=await fetch('/api/task_output/'+exp+'/'+taskId),text=await r.text();
        lc.innerHTML='<div style="padding:4px 8px;background:#1c2128;border-bottom:1px solid #30363d;font-size:11px;color:#58a6ff;display:flex;justify-content:space-between"><span>Task: '+taskId+'</span><span><button class="bt" onclick="vtask(\''+exp+'\',\''+taskId+'\')" style="padding:1px 8px;font-size:10px">\u21BB Refresh</button> <button class="bt" onclick="viewingFile=false;loadTab()" style="padding:1px 8px;font-size:10px">\u2190 Back</button></span></div><pre style="padding:8px;margin:0;white-space:pre-wrap;word-break:break-all;font-size:11px">'+esc(text)+'</pre>';
        lc.scrollTop=lc.scrollHeight;
    }catch(e){lc.textContent='Error: '+e.message}
}

function onTTC(){
    var tt=document.getElementById('tti').value,bi=document.getElementById('bi'),pi=document.getElementById('pi');
    if(tt==='analysis'){bi.style.display='none';pi.placeholder='What to analyze?'}
    else{bi.style.display='';pi.placeholder="Describe what to do \u2014 e.g. 'Try TabNet' or 'Stack exp_A and exp_B'"}
}

function togglePrompt(){
    var sh=document.getElementById('expPromptShort'),fl=document.getElementById('expPromptFull');
    if(fl.style.display==='none'){fl.style.display='';sh.style.display='none'}
    else{fl.style.display='none';sh.style.display=''}
}

function toggleEvt(el){
    var full=el.querySelector('.evt-full'),short=el.querySelector('.evt-short');
    if(!full||!short)return;
    if(full.style.display==='none'){full.style.display='';short.style.display='none'}
    else{full.style.display='none';short.style.display=''}
}

async function viewReport(name){
    viewingFile=true;var lc=document.getElementById('lc');
    try{var r=await fetch('/api/analyst_reports/'+name);lc.textContent=await r.text()}
    catch(e){lc.textContent='Error: '+e.message}
}

async function la(){
    var prompt=document.getElementById('pi').value,name=document.getElementById('ni').value;
    var base_experiment=document.getElementById('bi').value,task_type=document.getElementById('tti').value;
    await fetch('/api/launch',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({prompt:prompt,name:name,base_experiment:base_experiment,task_type:task_type})});
    document.getElementById('pi').value='';document.getElementById('ni').value='';
    document.getElementById('bi').value='';document.getElementById('tti').value='experiment';onTTC();rf();
}

async function smartLaunch(){
    var instruction=document.getElementById('pi').value.trim();
    if(!instruction){alert('Enter a prompt first');return}
    var btn=document.getElementById('slb'),orig=btn.textContent;
    btn.textContent='\uD83E\uDDE0 Thinking\u2026';btn.disabled=true;
    try{
        var r=await fetch('/api/smart_launch',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({instruction:instruction})});
        var d=await r.json();
        if(!r.ok){alert('Smart launch failed: '+(d.message||'unknown error'));return}
        document.getElementById('pi').value='';document.getElementById('ni').value='';
        document.getElementById('bi').value='';document.getElementById('tti').value='experiment';onTTC();
        if(d.id)se(d.id);rf();
    }catch(e){alert('Smart launch error: '+e.message)}
    finally{btn.textContent=orig;btn.disabled=false}
}

function fk(expName){
    var bi=document.getElementById('bi');bi.value=expName;
    if(!bi.querySelector('option[value="'+expName+'"]')){
        var opt=document.createElement('option');opt.value=expName;opt.textContent='\uD83C\uDF74 '+expName;
        bi.appendChild(opt);bi.value=expName;
    }
    document.getElementById('pi').focus();
    document.getElementById('pi').placeholder='Modify "'+expName+'"';
}

async function loadBases(){
    try{
        var r=await fetch('/api/experiments/bases'),d=await r.json(),bi=document.getElementById('bi'),cur=bi.value;
        var opts=['<option value="">\uD83D\uDCE6 default</option>'];
        for(var i=0;i<d.bases.length;i++){var b=d.bases[i];
            var score=b.test_score?parseFloat(b.test_score).toFixed(4):'?';
            var star=b.improved?'\u2605 ':'';
            opts.push('<option value="'+b.name+'">'+star+score+' \u2014 '+b.name+'</option>');
        }
        bi.innerHTML=opts.join('');if(cur)bi.value=cur;
    }catch(e){}
}

async function rq(idx){
    await fetch('/api/queue/remove',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index:idx})});rf();
}
async function tw(){
    var r=await fetch('/api/state'),d=await r.json();
    if(d.worker_running)await fetch('/api/worker/stop',{method:'POST'});
    else await fetch('/api/worker/start',{method:'POST'});rf();
}
async function tp(){
    var r=await fetch('/api/state'),d=await r.json();
    if(d.proxy_running)await fetch('/api/proxy/stop',{method:'POST'});
    else await fetch('/api/proxy/start',{method:'POST'});rf();
}
async function ke(name){if(!confirm('Stop '+name+'?'))return;await fetch('/api/kill/'+name,{method:'POST'});rf()}
async function re(name){if(!confirm('Restart '+name+'?'))return;var r=await fetch('/api/restart/'+name,{method:'POST'}),d=await r.json();if(d.id)se(d.id);rf()}
async function de(name){if(!confirm('Delete '+name+'?'))return;await fetch('/api/delete/'+name,{method:'POST'});if(sel===name){sel=null;ssu(null)}rf()}
async function cv(name){
    if(!confirm('Queue OOF/CV for '+name+'?'))return;
    try{var r=await fetch('/api/oof/'+name,{method:'POST'}),d=await r.json();
    if(r.status===409)alert(d.message||'OOF already running');
    else if(!r.ok)alert(d.message||'Failed');
    else alert('Queued '+(d.count||0)+' OOF folds')}
    catch(e){alert('Failed: '+e.message)}rf();
}
async function ofl(name){
    viewingFile=true;var lc=document.getElementById('lc');
    try{var r=await fetch('/api/log/'+name+'/agent'),txt=await r.text();
    lc.innerHTML='<div style="padding:4px 8px;background:#1c2128;border-bottom:1px solid #30363d;font-size:11px;color:#58a6ff;display:flex;justify-content:space-between"><span>OOF: '+name+'</span><span><button class="bt" onclick="ofl(\''+name+'\')" style="padding:1px 8px;font-size:10px">\u21BB</button> <button class="bt" onclick="viewingFile=false;loadTab()" style="padding:1px 8px;font-size:10px">\u2190 Back</button></span></div><pre style="padding:8px;margin:0;white-space:pre-wrap;word-break:break-all;font-size:11px">'+esc(txt)+'</pre>';
    lc.scrollTop=lc.scrollHeight}catch(e){lc.textContent='Error: '+e.message}
}
function tc(){document.body.classList.toggle('compact');localStorage.setItem('dash_compact',document.body.classList.contains('compact')?'1':'0')}
function tfl(){document.body.classList.toggle('focus-log');localStorage.setItem('dash_focus_log',document.body.classList.contains('focus-log')?'1':'0')}
async function showProxyLog(){
    viewingFile=true;var lc=document.getElementById('lc');
    document.querySelectorAll('.t').forEach(function(t){t.classList.remove('a')});
    try{var r=await fetch('/api/log/proxy');lc.textContent=await r.text();lc.scrollTop=lc.scrollHeight}
    catch(e){lc.textContent='Error: '+e.message}
}
async function showOrchestratorLog(){
    viewingFile=true;var lc=document.getElementById('lc');
    document.querySelectorAll('.t').forEach(function(t){t.classList.remove('a')});
    try{var r=await fetch('/api/log/orchestrator');lc.textContent=await r.text();lc.scrollTop=lc.scrollHeight}
    catch(e){lc.textContent='Error: '+e.message}
}

if(localStorage.getItem('dash_compact')==='1')document.body.classList.add('compact');
if(localStorage.getItem('dash_focus_log')==='1')document.body.classList.add('focus-log');
sel=gsu();rf();loadBases();setInterval(rf,5000);setInterval(loadBases,15000);
