"""Final scalar correction for residual LUFS deviations; preserve track timing."""
import concurrent.futures,json,time,subprocess,math
from pathlib import Path
from normalize_listening import B,R,F

def track(sid):
 d=B/sid/'listening_v2';deadline=time.time()+3600
 while not (d/'CALIBRATED.json').exists():
  if time.time()>deadline:raise RuntimeError(sid)
  time.sleep(3)
 for p in sorted(d.glob('*.json')):
  if p.name in ['CALIBRATED.json','READY.json']:continue
  x=json.loads(p.read_text())
  if 'final_scalar_db' in x:continue
  measured=x['measurement'];i=float(measured['output_i']);tp=float(measured['output_tp'])
  gain=min(-18-i,-1.0-tp) if math.isfinite(i) and math.isfinite(tp) else 0.
  if abs(gain)>.25:
   for suffix in ['.mp3','_preview.mp3']:
    old=d/(p.stem+suffix);tmp=d/(p.stem+suffix.replace('.mp3','.final.mp3'))
    subprocess.run([F,'-nostdin','-v','error','-threads','1','-y','-i',str(old),'-af',f'volume={gain}dB','-c:a','libmp3lame','-b:a','160k',str(tmp)],check=True)
    tmp.replace(old)
  else:gain=0.
  x.update(final_scalar_db=gain,final_expected_lufs=i+gain,final_expected_true_peak_dbtp=tp+gain,final_scalar_note='Static gain applied to full and preview MP3; peak headroom caps amplification. Expected LUFS based on measured renderer output; MP3 encoding can introduce small changes.')
  p.write_text(json.dumps(x,indent=2))
 (d/'READY.json').write_text(json.dumps(dict(complete=True,loudness_target_lufs=-18,finalized_unix=time.time())))
 print('READY',sid,flush=True)
if __name__=='__main__':
 ids=[x['sample_id'] for x in json.loads((B/'manifest.json').read_text())]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(track,ids))
 rows=[json.loads(p.read_text()) for sid in ids for p in (B/sid/'listening_v2').glob('*.json') if p.name not in ['CALIBRATED.json','READY.json']]
 (R/'evidence/v2/loudness_final.json').write_text(json.dumps(rows,indent=2));print('DONE',len(rows),flush=True)
