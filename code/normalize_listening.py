"""Listening-only EBU R128 derivatives. Preserve all scientific audio and note timing."""
import concurrent.futures,json,os,subprocess,time,shutil
from pathlib import Path
R=Path(os.environ.get('SINGING_BENCH_ROOT',Path(__file__).resolve().parents[1])); B=R/'benchmark'; F=os.environ.get('FFMPEG_BINARY') or os.environ.get('FFMPEG') or shutil.which('ffmpeg') or 'ffmpeg'

def convert(sid,key,inputs,pre=''):
 d=B/sid; meta=json.loads((d/'processing_metadata.json').read_text());out=d/'listening_v2';out.mkdir(exist_ok=True)
 full=out/(key+'.mp3');preview=out/(key+'_preview.mp3');log=out/(key+'.json')
 if full.exists() and preview.exists() and log.exists():return json.loads(log.read_text())
 args=[F,'-nostdin','-v','info','-threads','1','-filter_complex_threads','1','-y']
 for p in inputs:args+=['-i',str(p)]
 norm='loudnorm=I=-18:TP=-1.5:LRA=50:linear=false:print_format=json'
 graph=(pre or '[0:a]')+norm+',aresample=44100,asplit=2[full][clip];[clip]atrim=start='+str(meta['preview_start_seconds'])+':duration='+str(meta['preview_duration_seconds'])+',asetpts=PTS-STARTPTS[preview]'
 args+=['-filter_complex',graph,'-map','[full]','-c:a','libmp3lame','-b:a','160k','-ar','44100',str(full),'-map','[preview]','-c:a','libmp3lame','-b:a','160k','-ar','44100',str(preview)]
 t=time.perf_counter();p=subprocess.run(args,capture_output=True)
 if p.returncode:raise RuntimeError(p.stderr.decode()[-1000:])
 s=p.stderr.decode();stats=json.loads(s[s.rfind('{'):s.rfind('}')+1])
 record=dict(sample_id=sid,kind=key,method='EBU R128 / ffmpeg loudnorm dynamic, I=-18 LUFS, TP=-1.5 dBTP, LRA=50; playback derivative only',inputs=[str(x.relative_to(B)) for x in inputs],measurement=stats,wall_seconds=time.perf_counter()-t,preview_start=meta['preview_start_seconds'],pitch_timing_unchanged=True)
 log.write_text(json.dumps(record,indent=2));return record

def track(sid):
 rows=[convert(sid,'lead',[B/sid/'lead_vocal.flac'])]
 for pipe in ['game','rosvot','stars','procedural']:
  for timbre in ['epiano','choir']:
   key=f'{pipe}_{timbre}';rows.append(convert(sid,key,[B/sid/'renders'/(key+'.flac')]))
   rows.append(convert(sid,f'{pipe}_vocal_{timbre}',[B/sid/'listening_v2/lead.mp3',B/sid/'listening_v2'/(key+'.mp3')],'[0:a][1:a]amix=inputs=2:duration=longest:normalize=1,'))
 print('NORMALIZED',sid,flush=True);return rows
if __name__=='__main__':
 ids=[x['sample_id'] for x in json.loads((B/'manifest.json').read_text())]
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:rows=sum(list(ex.map(track,ids)),[])
 (R/'evidence/v2/loudness.json').write_text(json.dumps(rows,indent=2));print('DONE',len(rows),flush=True)
