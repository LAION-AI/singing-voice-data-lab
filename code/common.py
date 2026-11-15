import contextlib, datetime, hashlib, json, os, shutil, subprocess, time, traceback
from pathlib import Path
import numpy as np

ROOT=Path(os.environ.get('SINGING_BENCH_ROOT',Path(__file__).resolve().parents[1]))
BENCH=ROOT/'benchmark'
FFMPEG=os.environ.get('FFMPEG_BINARY') or shutil.which('ffmpeg') or 'ffmpeg'

def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
    tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def samples(ids=None):
    all=read(BENCH/'manifest.json')
    if ids and (ROOT/'work/reserve_samples.json').exists():
        known={x['sample_id'] for x in all}
        all += [x for x in read(ROOT/'work/reserve_samples.json') if x['sample_id'] not in known]
    return [x for x in all if not ids or x['sample_id'] in ids]
def decode(path,sr=44100,channels=1):
    p=subprocess.run([FFMPEG,'-v','error','-i',str(path),'-f','f32le','-acodec','pcm_f32le','-ar',str(sr),'-ac',str(channels),'-'],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    return np.frombuffer(p.stdout,dtype='<f4').reshape(-1,channels).copy()
def encode_mp3(y,sr,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if y.ndim==1:y=y[:,None]
    subprocess.run([FFMPEG,'-v','error','-y','-f','f32le','-ar',str(sr),'-ac',str(y.shape[1]),'-i','-','-codec:a','libmp3lame','-b:a','160k',str(path)],input=np.asarray(y,dtype='<f4').tobytes(),check=True,stderr=subprocess.PIPE)
def sync():
    import torch
    if torch.cuda.is_available():torch.cuda.synchronize()
def hardware():
    import torch,platform
    return dict(platform=platform.system(),architecture=platform.machine(),python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,vram_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,job_id=os.environ.get('SLURM_JOB_ID'))
def metric(stage,sample_id,seconds,audio_seconds=0,**extra):
    row=dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,sample_id=sample_id,wall_seconds=float(seconds),audio_seconds=float(audio_seconds),**extra)
    path=ROOT/'logs'/f'timing-{stage}-{os.getpid()}.jsonl'
    with path.open('a') as f:f.write(json.dumps(row,default=str)+'\n')
    print(json.dumps(row,default=str),flush=True)
def failure(stage,sample_id,e):
    write(BENCH/sample_id/'status'/f'{stage}.json',dict(status='failed',error_type=type(e).__name__,error=str(e)[:1500],traceback=traceback.format_exc(),time=time.time()))
    print('FAILED',stage,sample_id,type(e).__name__,str(e)[:300],flush=True)
def complete(stage,sample_id,**extra):write(BENCH/sample_id/'status'/f'{stage}.json',dict(status='complete',time=time.time(),**extra))

