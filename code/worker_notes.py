"""Run independent upstream models; preserve raw outputs and globalize chunk time."""
import argparse,os,re,shutil,subprocess,sys,time,traceback,hashlib
from pathlib import Path
import numpy as np,soundfile as sf,torch
from common import *
from worker_f0 import export
from midi_tokens import load_midi

def command(cmd,cwd,log):
    with Path(log).open('w') as f:
        env=dict(os.environ, PYTHONPATH=str(cwd)+os.pathsep+os.environ.get('PYTHONPATH',''))
        p=subprocess.run(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,env=env)
    if p.returncode:raise RuntimeError(f'Upstream process failed ({p.returncode}); see {Path(log).name}')

def prepare(items):
    cohort=hashlib.sha256(','.join(x['sample_id'] for x in items).encode()).hexdigest()[:12]
    chunkdir=ROOT/'work/note_chunks'/cohort;chunkdir.mkdir(exist_ok=True,parents=True);chunks=[];stars=[]
    import cmudict
    lexicon=cmudict.dict()
    allowed=set(read(ROOT/'upstream/STARS/chinese_and_english_phone_set.json'))
    for item in items:
        sid=item['sample_id'];d=BENCH/sid;audio=decode(d/'lead_vocal.flac',24000,1)[:,0]
        asr=read(d/'asr/parakeet.json') if (d/'asr/parakeet.json').exists() else dict(words=[])
        for a in range(0,len(audio),30*24000):
            b=min(len(audio),a+30*24000);name=f'{sid}__{a//720000:03d}';wav=chunkdir/(name+'.wav')
            if not wav.exists():sf.write(wav,audio[a:b],24000,subtype='PCM_16')
            words=[w for w in asr.get('words',[]) if a/24000<=(w['onset']+w['offset'])/2<b/24000]
            text=' '.join(w['word'].strip() for w in words)
            (chunkdir/(name+'.txt')).write_text(text)
            chunk=dict(item_name=name,wav_fn=str(wav),sample_id=sid,offset_seconds=a/24000,duration_seconds=(b-a)/24000,text=text)
            chunks.append(chunk)
            ww=[];ph=[];ph2=[];oov=[]
            for word in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",text):
                phones=lexicon.get(word.lower(),[[]])[0]
                phones=[p if p in allowed else re.sub(r'\d$','',p) for p in phones]
                if not phones or any(p not in allowed for p in phones):
                    oov.append(word);continue
                ww.append(word);ph+=phones;ph2 += [len(ww)-1]*len(phones)
            chunk['stars_omitted_oov_words']=oov
            if ph:
                stars.append(dict(item_name=name,wav_fn=str(wav),word=ww,ph=ph,ph2words=ph2))
    write(chunkdir/'chunks.json',chunks);write(chunkdir/'stars.json',stars)
    write(chunkdir/'rosvot.json',[dict(item_name=x['item_name'],wav_fn=x['wav_fn']) for x in chunks])
    return chunkdir,chunks

def collect(model,items,chunks,rawdir,elapsed):
    import textgrid
    total=sum(read(BENCH/x['sample_id']/'processing_metadata.json')['duration_seconds'] for x in items)
    for item in items:
        sid=item['sample_id'];d=BENCH/sid;allnotes=[];missing=[];rawfiles=[]
        own=[c for c in chunks if c['sample_id']==sid]
        for c in own:
            name=c['item_name']
            files=list(rawdir.rglob(name+'*.mid'))
            if not files:
                # STARS skips transcriptless chunks; this is a real missing region.
                missing.append(dict(chunk=name,start=c['offset_seconds'],duration=c['duration_seconds'],reason='No MIDI emitted by upstream model'));continue
            path=files[0];notes=load_midi(path)
            for n in notes:
                n['onset']+=c['offset_seconds'];n['offset']+=c['offset_seconds'];n['chunk']=name
            allnotes+=notes
            dest=d/'raw'/model;dest.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dest/path.name);rawfiles.append(str((dest/path.name).relative_to(d)))
        # Retain every matching raw output, including GAME CSV/TXT and upstream diagnostics.
        for raw in rawdir.rglob('*'):
            if raw.is_file() and any(c['item_name'] in raw.name for c in own):
                dest=d/'raw'/model/raw.relative_to(rawdir);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(raw,dest)
        meta=read(d/'processing_metadata.json');duration=meta['duration_seconds']
        # No re-tuning, time warping, or merging that hides note boundary errors.
        if allnotes:
            export(d,model,sorted(allnotes,key=lambda n:(n['onset'],n['pitch'])),duration,dict(upstream=model,chunk_seconds=30,raw_files=rawfiles,missing_chunks=missing,conditioning_omitted_oov_words=[dict(chunk=c['item_name'],words=c.get('stars_omitted_oov_words',[])) for c in own] if model=='stars' else [],notes_unmodified_except_global_chunk_offset=True))
            if missing:complete(model,sid,availability='partial',missing_chunks=missing,note_count=len(allnotes))
        else:failure(model,sid,RuntimeError('No notes emitted; inspect preserved upstream log'))
        metric(model,sid,elapsed*duration/total,duration,timing_scope='share of measured bulk wall time proportional to source duration; use bulk row for actual wall time',batch_size=4,precision='float32')
        if model=='stars':
            phones=[];words=[];techniques=[]
            for c in own:
                name=c['item_name'];tgfiles=list(rawdir.rglob(name+'*.TextGrid'))
                for path in tgfiles:
                    dest=d/'raw/stars';dest.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest/path.name)
                    tg=textgrid.TextGrid.fromFile(str(path))
                    for tier in tg:
                        vals=[dict(label=i.mark,onset=i.minTime+c['offset_seconds'],offset=i.maxTime+c['offset_seconds']) for i in tier]
                        if tier.name=='phones':phones+=vals
                        elif tier.name=='words':words+=vals
                        else:techniques.append(dict(chunk=name,tier=tier.name,intervals=vals))
            if phones:write(d/'alignment/stars_phonemes.json',dict(phones=phones,words=words,provenance='STARS joint singing phoneme alignment conditioned on CMU phones of the Parakeet ASR hypothesis',techniques=techniques))

def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='*');p.add_argument('--models',nargs='*',default=['game','rosvot','tifa','stars']);a=p.parse_args()
    items=samples(a.ids)
    a.models=[model for model in a.models if not all((BENCH/x['sample_id']/'status'/f'{model}.json').exists() and read(BENCH/x['sample_id']/'status'/f'{model}.json').get('status')=='complete' for x in items)]
    if not a.models:
        print('SKIP: all requested note/alignment outputs already complete',flush=True);return
    all_items=items
    for model in a.models:
        items=[x for x in all_items if not ((BENCH/x['sample_id']/'status'/f'{model}.json').exists() and read(BENCH/x['sample_id']/'status'/f'{model}.json').get('status') in ['complete','unavailable'])]
        if not items:continue
        chunkdir,chunks=prepare(items)
        duration=sum(read(BENCH/x['sample_id']/'processing_metadata.json')['duration_seconds'] for x in items)
        rawdir=ROOT/'work'/('raw_'+model+('_'+'-'.join(a.ids) if a.ids else '_full'));rawdir.mkdir(exist_ok=True)
        log=ROOT/'logs'/f'{os.environ.get("SLURM_JOB_ID","local")}-{model}-upstream-{len(items)}.log'
        upstream=ROOT/'upstream'/dict(game='GAME',rosvot='ROSVOT',stars='STARS',tifa='TIFA')[model]
        cmd=[sys.executable,'-B']
        try:
            if model=='game':
                cmd += ['infer.py','extract',str(chunkdir),'-m',str(ROOT/'models/game/GAME-1.0-large/model.pt'),'--glob','*.wav','--output-dir',str(rawdir),'--output-formats','mid,csv,txt','--pitch-format','number','--batch-size','4','--num-workers','0']
            elif model=='rosvot':
                # Official relative checkpoint paths are made explicit, with no parameter changes.
                link=upstream/'checkpoints'
                if not link.exists():link.symlink_to(ROOT/'models/rosvot/checkpoints',target_is_directory=True)
                cmd += ['inference/rosvot.py','-o',str(rawdir),'--metadata',str(chunkdir/'rosvot.json'),'--bsz','4','--max_tokens','20000','--max_frames','10000','--ds_workers','0','--sync_saving']
            elif model=='stars':
                link=upstream/'checkpoints'
                if not link.exists():link.symlink_to(ROOT/'models/stars',target_is_directory=True)
                cmd += ['inference/stars.py','--ckpt',str(ROOT/'models/stars/stars_chinese_english_bilingual/model_ckpt_steps_300000.ckpt'),'--config','configs/stars_bilingual.yaml','--phset','chinese_and_english_phone_set.json','-o',str(rawdir),'--metadata',str(chunkdir/'stars.json'),'--bsz','4','--max_tokens','20000','--ds_workers','0']
            else:
                cmd += ['infer.py',str(chunkdir),'-m',str(ROOT/'models/tifa/TIFA-1.0-ST/model.pt'),'-l','en','--fill','SP','--stat','-o',str(rawdir),'--batch-size','4','--num-workers','0']
            sync();t=time.perf_counter();command(cmd,upstream,log);sync();elapsed=time.perf_counter()-t
            metric(model+'_bulk',None,elapsed,duration,tracks=len(items),sample_ids=[x['sample_id'] for x in items],chunks=len(chunks),hardware=hardware(),command=cmd,batch_size=4)
            if model!='tifa':collect(model,items,chunks,rawdir,elapsed)
            else:
                import textgrid
                for item in items:
                    sid=item['sample_id'];d=BENCH/sid;phones=[];words=[];diagnostics=[]
                    for c in [x for x in chunks if x['sample_id']==sid]:
                        for path in rawdir.rglob(c['item_name']+'.TextGrid'):
                            dest=d/'raw/tifa';dest.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest/path.name)
                            tg=textgrid.TextGrid.fromFile(str(path))
                            for tier in tg:
                                vals=[dict(label=i.mark,onset=i.minTime+c['offset_seconds'],offset=i.maxTime+c['offset_seconds']) for i in tier]
                                if tier.name=='phones':phones+=vals
                                elif tier.name=='texts':words+=vals
                    if phones:
                        write(d/'alignment/phonemes.json',dict(status='complete',phones=phones,words=words,provenance='TIFA-1.0-ST forced singing alignment of Parakeet hypothesis; not native ASR timestamps',language='en',chunk_seconds=30));complete('tifa',sid,phonemes=len(phones))
                    else:failure('tifa',sid,RuntimeError('TIFA emitted no phoneme alignments'))
        except Exception as e:
            traceback.print_exc()
            for x in items:failure(model,x['sample_id'],e)

if __name__=='__main__':
    from filelock import FileLock
    own_ids=sys.argv[sys.argv.index('--ids')+1:] if '--ids' in sys.argv else []
    lockname='notes-'+hashlib.sha256(','.join(own_ids).encode()).hexdigest()[:12]+'.lock' if own_ids else 'notes.lock'
    # Optional bounded hold for auxiliary inference in this same allocation.
    # The main controller reaches this only after its primary ASR/caption workers finish.
    if not own_ids:
        deadline=time.monotonic()+1800
        while (ROOT/'work/HOLD_ALLOCATION').exists():
            if time.monotonic()>deadline:raise RuntimeError('Bounded auxiliary hold expired')
            time.sleep(5)
    with FileLock(str(ROOT/'work'/lockname)):main()
