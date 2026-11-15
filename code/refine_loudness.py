"""Measured two-pass correction of first-pass loudness outliers; CPU only."""
import concurrent.futures,json,subprocess,time,math
from pathlib import Path
from normalize_listening import R,B,F

def fix(sid,key,inputs,pre='',reuse=None):
 d=B/sid;meta=json.loads((d/'processing_metadata.json').read_text());o=d/'listening_v2';log=o/(key+'.json');old=json.loads(log.read_text())
 if reuse:stats=reuse
 else:
  args=[F,'-nostdin','-v','info','-threads','1','-filter_complex_threads','1','-y']
  for p in inputs:args+=['-i',str(p)]
  args+=['-filter_complex',(pre or '[0:a]')+'loudnorm=I=-18:TP=-1.5:LRA=50:print_format=json','-f','null','-']
  x=subprocess.run(args,capture_output=True,check=True).stderr.decode();stats=json.loads(x[x.rfind('{'):x.rfind('}')+1])
 if not math.isfinite(float(stats['input_i'])):return False
 args=[F,'-nostdin','-v','info','-threads','1','-filter_complex_threads','1','-y']
 for p in inputs:args+=['-i',str(p)]
 settings='loudnorm=I=-18:TP=-1.5:LRA=50:linear=true:print_format=json:'+':'.join(f'{dest}={stats[src]}' for dest,src in [('measured_I','input_i'),('measured_TP','input_tp'),('measured_LRA','input_lra'),('measured_thresh','input_thresh'),('offset','target_offset')])
 graph=(pre or '[0:a]')+settings+',aresample=44100,asplit=2[full][clip];[clip]atrim=start='+str(meta['preview_start_seconds'])+':duration='+str(meta['preview_duration_seconds'])+',asetpts=PTS-STARTPTS[preview]'
 full=o/(key+'.mp3');prev=o/(key+'_preview.mp3');tf=o/(key+'.new.mp3');tp=o/(key+'_preview.new.mp3')
 args+=['-filter_complex',graph,'-map','[full]','-c:a','libmp3lame','-b:a','160k','-ar','44100',str(tf),'-map','[preview]','-c:a','libmp3lame','-b:a','160k','-ar','44100',str(tp)]
 x=subprocess.run(args,capture_output=True,check=True).stderr.decode();new=json.loads(x[x.rfind('{'):x.rfind('}')+1]);tf.replace(full);tp.replace(prev)
 old.update(first_pass_measurement=old.get('first_pass_measurement',old['measurement']),measurement=new,normalization_version=2,method='Measured two-pass EBU R128 -18 LUFS / -1.5 dBTP; linear when feasible, otherwise peak-limited dynamic correction; listening only',second_pass_filter=settings)
 log.write_text(json.dumps(old,indent=2));return True

def track(sid):
 d=B/sid/'listening_v2';deadline=time.time()+3600
 if (d/'CALIBRATED.json').exists():return
 while not (d/'procedural_vocal_choir.json').exists():
  if time.time()>deadline:raise RuntimeError(sid+' incomplete')
  time.sleep(3)
 changed=set()
 for key in ['lead']+[p+'_'+t for p in ['game','rosvot','stars','procedural'] for t in ['epiano','choir']]:
  old=json.loads((d/(key+'.json')).read_text())
  if abs(float(old['measurement']['output_i'])+18)>.4 and old.get('normalization_version')!=2:
   if fix(sid,key,[B/x for x in old['inputs']],reuse=old['measurement']):changed.add(key)
 for p in ['game','rosvot','stars','procedural']:
  for t in ['epiano','choir']:
   key=p+'_vocal_'+t;old=json.loads((d/(key+'.json')).read_text())
   if old.get('normalization_version')!=2 and (abs(float(old['measurement']['output_i'])+18)>.4 or 'lead' in changed or p+'_'+t in changed):fix(sid,key,[d/'lead.mp3',d/(p+'_'+t+'.mp3')],'[0:a][1:a]amix=inputs=2:duration=longest:normalize=1,')
 (d/'CALIBRATED.json').write_text(json.dumps(dict(method='Measured normalization; two-pass correction of >0.4 LU first-pass deviations',complete=True)))
 print('CALIBRATED',sid,flush=True)
if __name__=='__main__':
 ids=[x['sample_id'] for x in json.loads((B/'manifest.json').read_text())]
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(track,ids))
 print('DONE',len(ids),flush=True)
