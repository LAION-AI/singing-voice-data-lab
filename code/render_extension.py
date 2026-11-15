"""Render new note pipelines and normalize the exact existing excerpts; CPU only."""
import argparse,concurrent.futures,json,subprocess,time
from common import ROOT,BENCH,read,samples,metric
from normalize_listening import convert
from refine_loudness import fix
from finalize_listening import track as final_scalar

def track(sid,pipelines):
 d=BENCH/sid
 for pipe in pipelines:
  if not (d/'note_events'/f'{pipe}.json').exists():continue
  start=time.perf_counter()
  for timbre in ['epiano','choir']:
   key=f'{pipe}_{timbre}';record=convert(sid,key,[d/'renders'/(key+'.flac')])
   if abs(float(record['measurement']['output_i'])+18)>.4 and record.get('normalization_version')!=2:fix(sid,key,[d/'renders'/(key+'.flac')],reuse=record['measurement'])
   overlay=f'{pipe}_vocal_{timbre}'
   record=convert(sid,overlay,[d/'listening_v2/lead.mp3',d/'listening_v2'/(key+'.mp3')],'[0:a][1:a]amix=inputs=2:duration=longest:normalize=1,')
   if abs(float(record['measurement']['output_i'])+18)>.4 and record.get('normalization_version')!=2:fix(sid,overlay,[d/'listening_v2/lead.mp3',d/'listening_v2'/(key+'.mp3')],'[0:a][1:a]amix=inputs=2:duration=longest:normalize=1,',reuse=record['measurement'])
  final_scalar(sid)
  metric('normalize_'+pipe,sid,time.perf_counter()-start,read(d/'processing_metadata.json')['duration_seconds'],hardware_scope='CPU only')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ids',nargs='*');p.add_argument('--pipelines',nargs='+',default=['muscriptor_large','muscriptor_medium']);a=p.parse_args();ids=[x['sample_id'] for x in samples(a.ids)]
 cmd=['bash',str(ROOT/'code/env.sh'),str(ROOT/'code/render.py'),'--pipelines',*a.pipelines]
 if a.ids:cmd+=['--ids',*ids]
 subprocess.run(cmd,check=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(lambda sid:track(sid,a.pipelines),ids))
