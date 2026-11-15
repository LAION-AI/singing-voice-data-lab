"""Portable entry point for the reproducible clone-based singing annotation lab."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
DEFAULT='rosvot_procedural_segmented'

def environment(work):
    env=dict(os.environ,SINGING_BENCH_ROOT=str(work),SINGING_PORTABLE='1',
        FFMPEG_BINARY=os.environ.get('FFMPEG_BINARY') or shutil.which('ffmpeg') or 'ffmpeg')
    env['PYTHONPATH']=str(HERE/'code')+os.pathsep+env.get('PYTHONPATH','')
    return env

def prepare(work):
    work.mkdir(parents=True,exist_ok=True)
    for folder in ['benchmark','logs','work','evidence','evidence/v2','evidence/v3','evidence/v4_ensemble']:(work/folder).mkdir(exist_ok=True,parents=True)
    for name,source in [('models',HERE/'models'),('upstream',HERE/'third_party'),('code',HERE/'code')]:
        link=work/name
        if not link.exists():link.symlink_to(source,target_is_directory=True)
    (work/'evidence/soundfont.json').write_text(json.dumps(dict(name='GeneralUser GS v2.0.3',source='https://github.com/mrbumpy409/GeneralUser-GS/tree/684543d5e5efaef08d02be50dcda8d552478fa60',sha256='9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe'))+'\n')
    (work/'evidence/parakeet_revision.json').write_text(json.dumps(dict(revision='2bf128600aac4b16946f7ed8372e56117fe5e23b'))+'\n')
    model_lock=HERE/'models/muscriptor_lock.json'
    if model_lock.exists():shutil.copy2(model_lock,work/'evidence/v3/muscriptor_lock.json')

def stage(work,name,args):
    subprocess.run([sys.executable,str(HERE/'code'/name),*args],env=environment(work),check=True)

def ingest(work,inputs,already_lead=False):
    from hashlib import sha256
    import soundfile as sf
    prepare(work);manifest=[]
    for index,source in enumerate(inputs,1):
        sid=f'song_{index:05d}';d=work/'benchmark'/sid;d.mkdir(exist_ok=True)
        for folder in ['status','preview','note_events','midi','flattened_midi','renders','asr','pitch','listening_v2']:(d/folder).mkdir(exist_ok=True)
        target=d/('source'+source.suffix.lower())
        if not target.exists():shutil.copyfile(source,target)
        h=sha256()
        with target.open('rb') as f:
            for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
        item=dict(sample_id=sid,group='user',dataset='user-provided',source_id=source.stem,
            source_audio=target.name,source_sha256=h.hexdigest(),language='unknown',original_generation_prompt=None)
        if already_lead:
            lead=d/'lead_vocal.flac'
            subprocess.run([environment(work)['FFMPEG_BINARY'],'-v','error','-y','-i',str(target),'-ar','44100','-ac','2','-c:a','flac',str(lead)],check=True)
            duration=sf.info(lead).duration;item.update(duration_seconds=duration,preview_start_seconds=0,preview_duration_seconds=min(35,duration))
            (d/'status/stems.json').write_text(json.dumps(dict(status='complete',source='user-declared isolated lead; no separation performed'))+'\n')
        (d/'processing_metadata.json').write_text(json.dumps(item,indent=2)+'\n');manifest.append(item)
    (work/'benchmark/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run');p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--pipeline',default=DEFAULT,choices=['rosvot','procedural','rosvot_procedural_ensemble','rosvot_procedural_gapfill','rosvot_procedural_vote_gapfill',DEFAULT]);p.add_argument('--already-lead',action='store_true');p.add_argument('--render',action='store_true');p.add_argument('--device',default='cuda');p.add_argument('--separator-batch-size',type=int,default=1)
    p=sub.add_parser('extract');p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--separator-batch-size',type=int,default=1)
    p=sub.add_parser('download-models');p.add_argument('--extras',nargs='*',default=[])
    p=sub.add_parser('annotate-existing');p.add_argument('--workspace',type=Path,required=True);p.add_argument('--pipeline',default=DEFAULT)
    p=sub.add_parser('stage');p.add_argument('worker',choices=['worker_stems.py','worker_f0.py','worker_ensemble.py','worker_render.py','worker_asr.py','worker_caption.py','worker_notes.py','worker_muscriptor.py','worker_whisper_caption.py']);p.add_argument('--workspace',type=Path,required=True);p.add_argument('arguments',nargs=argparse.REMAINDER)
    p=sub.add_parser('flatten');p.add_argument('notes',type=Path);p.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.command=='download-models':
        subprocess.run([sys.executable,str(HERE/'tools/download_models.py'),'--extras',*a.extras],check=True);return
    if a.command=='flatten':
        sys.path.insert(0,str(HERE/'code'));from midi_tokens import encode,roundtrip
        notes=json.loads(a.notes.read_text());roundtrip(notes['notes'],notes['track_duration_seconds']);a.output.write_text(encode(notes['notes'],notes['track_duration_seconds']));return
    work=(a.output if a.command in ['run','extract'] else a.workspace).resolve();prepare(work)
    if a.command=='annotate-existing':stage(work,'worker_ensemble.py',['--pipeline',a.pipeline]);return
    if a.command=='stage':stage(work,a.worker,a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments);return
    extensions={'.wav','.flac','.mp3','.m4a','.ogg','.opus','.aac'}
    inputs=sorted(p for p in a.input.iterdir() if p.suffix.lower() in extensions) if a.input.is_dir() else [a.input]
    if not inputs or not all(p.is_file() for p in inputs):raise ValueError('No input audio files found')
    if (work/'benchmark/manifest.json').exists():raise RuntimeError('Output already contains a manifest. Resume with the stage/annotate-existing commands instead of replacing inputs.')
    ingest(work,inputs,getattr(a,'already_lead',False))
    if not getattr(a,'already_lead',False):stage(work,'worker_stems.py',['--device',a.device,'--batch-size',str(a.separator_batch_size)])
    if a.command=='extract':return
    if not a.device.startswith('cuda'):raise RuntimeError('RMVPE and ROSVOT workers currently require CUDA; lead extraction alone also supports CPU.')
    stage(work,'worker_f0.py',[]);stage(work,'worker_notes.py',['--models','rosvot'])
    stage(work,'worker_ensemble.py',['--pipeline',a.pipeline])
    if a.render:stage(work,'worker_render.py',['--pipeline',a.pipeline])

if __name__=='__main__':main()
