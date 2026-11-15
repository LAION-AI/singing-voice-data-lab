'use strict';
const $=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let manifest,clip,selection=0;const clips=new Map(),blobs=new Map();
function status(text,error=false){$('status').textContent=text;$('status').classList.toggle('error',error)}
async function member(path){
 const info=clip.members[path];if(!info)throw new Error('Missing published file: '+path);
 if(blobs.has(path))return blobs.get(path);
 const end=info.start+info.size-1;
 const response=await fetch(manifest.archive_url+'?download=true&member='+encodeURIComponent(path),{headers:{Range:`bytes=${info.start}-${end}`}});
 if(response.status!==206){if(response.body)await response.body.cancel();throw new Error('The data server did not honour the byte-range request. Please retry.');}
 const buffer=await response.arrayBuffer();if(buffer.byteLength!==info.size)throw new Error('Incomplete audio range. Please retry.');
 const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),b=>b.toString(16).padStart(2,'0')).join('');
 if(digest!==info.sha256)throw new Error('File checksum mismatch. Please retry.');
 const type=path.endsWith('.mp3')?'audio/mpeg':path.endsWith('.flac')?'audio/flac':'application/octet-stream';
 const result={url:URL.createObjectURL(new Blob([buffer],{type})),size:buffer.byteLength};blobs.set(path,result);
 if(blobs.size>28){for(const [key,value] of blobs){if(![...document.querySelectorAll('audio')].some(a=>a.src===value.url)){URL.revokeObjectURL(value.url);blobs.delete(key);break;}}}
 return result;
}
function stop(){document.querySelectorAll('audio').forEach(a=>a.pause())}
async function audioCard(path,label,preload,ticket){
 const box=document.createElement('article');box.className='player';const heading=document.createElement('h3');heading.textContent=label;
 const button=document.createElement('button');button.textContent='Load / play audio';const audio=document.createElement('audio');audio.controls=true;audio.preload='none';
 audio.addEventListener('play',()=>document.querySelectorAll('audio').forEach(other=>{if(other!==audio)other.pause()}));
 async function load(play){
  button.disabled=true;button.textContent='Loading…';
  try{const data=await member(path);if(ticket!==selection)return;audio.src=data.url;audio.dataset.member=path;button.textContent='Play';button.disabled=false;
   if(play){try{await audio.play()}catch{status('Audio ready. Press Play in the player.')}}
  }catch(error){button.disabled=false;button.textContent='Retry audio';status(error.message,true)}
 }
 button.onclick=()=>audio.src?audio.play().catch(e=>status(e.message,true)):load(true);
 box.append(heading,button,audio);$('players').append(box);if(preload)load(false);
}
function draw(){
 const method=$('method').value,notes=clip.methods[method].notes,duration=clip.metadata.duration_seconds;
 let pitches=notes.map(n=>n.pitch).concat(clip.pitch.values.filter(x=>x!==null));const lo=Math.floor(Math.min(...pitches)-2),hi=Math.ceil(Math.max(...pitches)+2);
 const X=t=>55+Math.max(0,Math.min(duration,t))/duration*1015,Y=p=>280-(p-lo)/(hi-lo||1)*245;
 let svg='<svg viewBox="0 0 1100 325" role="img" aria-label="Predicted notes and continuous automatic pitch">';
 for(let i=0;i<=5;i++){const p=lo+(hi-lo)*i/5,y=Y(p);svg+=`<path d="M55 ${y}H1070" stroke="#eef0f5"/><text x="44" y="${y+4}" text-anchor="end" font-size="11" fill="#6a778b">${Math.round(p)}</text>`;}
 for(let i=0;i<=5;i++){const t=duration*i/5;svg+=`<text x="${X(t)}" y="302" text-anchor="middle" font-size="11" fill="#6a778b">${t.toFixed(0)}s</text>`;}
 let path='',active=false;clip.pitch.values.forEach((p,i)=>{if(p===null){active=false;return;}path+=(active?'L':'M')+X(clip.pitch.times[i]).toFixed(2)+' '+Y(p).toFixed(2)+' ';active=true});svg+=`<path d="${path}" fill="none" stroke="#d89529" stroke-width="1"/>`;
 for(const n of notes){const e=n.ensemble||{},x1=X(n.onset),x2=X(n.offset);if(e.changed)svg+=`<path d="M${x1} ${Y(e.original_rosvot_pitch)}H${x2}" stroke="#89909b" stroke-width="2" stroke-dasharray="4 3"/>`;
  svg+=`<path d="M${x1} ${Y(n.pitch)}H${x2}" stroke="${e.added?'#208264':e.changed?'#c64363':'#6765d7'}" stroke-width="5"><title>${esc(n.pitch_name||n.pitch)} · ${n.onset.toFixed(3)}–${n.offset.toFixed(3)}s</title></path>`;}
 svg+='<text x="565" y="323" text-anchor="middle" font-size="11" fill="#6a778b">Seconds from start of full recording · vertical axis: MIDI pitch</text></svg>';$('plot').innerHTML=svg;
 $('notes').innerHTML=notes.map(n=>`<tr><td>${n.pitch}</td><td>${esc(n.pitch_name||'')}</td><td>${n.onset.toFixed(4)}</td><td>${n.offset.toFixed(4)}</td><td>${(n.offset-n.onset).toFixed(4)}</td></tr>`).join('');$('tokens').textContent=clip.methods[method].tokens;
 const changed=notes.filter(n=>n.ensemble?.changed).length,added=notes.filter(n=>n.ensemble?.added).length;
 $('decision').textContent=`${notes.length} notes · ${changed} changed-pitch notes/segments · ${added} added gap notes. Compare the piano and vocal overlay to assess fragmentation.`;
}
async function update(){
 const ticket=++selection;stop();$('players').replaceChildren();status('Loading recording…');
 try{const sid=$('recording').value;if(!clips.has(sid)){const response=await fetch('clips/'+sid+'.json');if(!response.ok)throw new Error('Clip metadata unavailable');clips.set(sid,await response.json())}if(ticket!==selection)return;clip=clips.get(sid);
  const m=clip.metadata;$('meta').textContent=`${sid} · ${m.genre||'singing'} · ${m.language||'unknown'} · ${m.duration_seconds.toFixed(1)} s · source clip ${m.source_id}. Notes and tokens cover the full recording.`;
  const full=$('full').checked,p=$('method').value,suffix=full?'.mp3':'_preview.mp3';
  audioCard(sid+'/'+(full?'original.flac':'original_preview.mp3'),'Original LeVo2 recording',false,ticket);
  audioCard(sid+'/'+(full?'lead.mp3':'lead_preview.mp3'),'Extracted lead vocal · matched level',true,ticket);
  for(const [kind,label] of [['epiano','Melody as electric piano'],['choir','Melody as synthetic choir'],['vocal_epiano','Lead vocal + piano'],['vocal_choir','Lead vocal + choir']])audioCard(sid+'/'+p+'/'+kind+suffix,label,kind==='epiano',ticket);
  draw();status('Ready. Lead and piano are prepared first; other players load on demand.');
 }catch(error){status(error.message,true)}
}
document.querySelectorAll('[data-download]').forEach(button=>button.addEventListener('click',async()=>{button.disabled=true;try{const name=button.dataset.download,sid=$('recording').value,path=name==='f0.npz'?sid+'/'+name:sid+'/'+$('method').value+'/'+name,data=await member(path);const link=document.createElement('a');link.href=data.url;link.download=sid+'.'+$('method').value+'.'+name;link.click()}catch(error){status(error.message,true)}finally{button.disabled=false}}));
(async()=>{try{const response=await fetch('manifest.json');manifest=await response.json();for(const m of manifest.recordings){const option=new Option(m.sample_id+' · '+(m.genre||'singing'),m.sample_id);$('recording').add(option)}for(const [key,label] of Object.entries(manifest.methods))$('method').add(new Option(label,key));$('method').value=manifest.default;for(const id of ['recording','method','full'])$(id).addEventListener('change',update);await update()}catch(error){status(error.message,true)}})();
