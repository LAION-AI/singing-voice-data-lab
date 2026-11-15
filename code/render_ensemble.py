"""Render only the new ensemble; leave all existing listening files untouched."""
import concurrent.futures
import argparse
import json
import math
import subprocess
import time
import numpy as np
import soundfile as sf
from common import ROOT, BENCH, read, write, samples, decode
from ensemble_notes import PIPELINE, EVIDENCE
from render import synthesize
from normalize_listening import convert, F
from refine_loudness import fix


def finish_scalar(sid,key):
    directory=BENCH/sid/'listening_v2';path=directory/(key+'.json');data=read(path)
    if 'final_scalar_db' in data:return
    measured=data['measurement'];i=float(measured['output_i']);peak=float(measured['output_tp'])
    gain=min(-18-i,-1-peak) if math.isfinite(i) and math.isfinite(peak) else 0.
    if abs(gain)>.25:
        for suffix in ['.mp3','_preview.mp3']:
            old=directory/(key+suffix);tmp=old.with_suffix('.final.mp3')
            subprocess.run([F,'-nostdin','-v','error','-threads','1','-y','-i',str(old),'-af',f'volume={gain}dB','-c:a','libmp3lame','-b:a','160k',str(tmp)],check=True)
            tmp.replace(old)
    else:gain=0.
    data.update(final_scalar_db=gain,final_expected_lufs=i+gain,
                final_expected_true_peak_dbtp=peak+gain,
                final_scalar_note='Same static gain on full and preview; timing unchanged. MP3 encoding can slightly change levels.')
    write(path,data)


def normalize(sid,key,inputs,pre=''):
    record=convert(sid,key,inputs,pre)
    if abs(float(record['measurement']['output_i'])+18)>.4 and record.get('normalization_version')!=2:
        fix(sid,key,inputs,pre,reuse=record['measurement'])
    finish_scalar(sid,key)


def track(sid,pipeline=PIPELINE,retain_raw=True):
    started=time.perf_counter();directory=BENCH/sid;meta=read(directory/'processing_metadata.json')
    duration=meta['duration_seconds'];sr=44100
    notes=read(directory/'note_events'/f'{pipeline}.json')['notes']
    outputs=directory/'renders';outputs.mkdir(exist_ok=True)
    vocal=None
    for timbre,program in [('epiano',4),('choir',52)]:
        key=f'{pipeline}_{timbre}';raw=outputs/(key+'.flac');raw_mix=outputs/(f'{pipeline}_vocal_{timbre}.flac')
        listening=directory/'listening_v2'
        complete=all((listening/(k+suffix)).exists() for k in [key,f'{pipeline}_vocal_{timbre}'] for suffix in ['.mp3','_preview.mp3','.json'])
        if complete and not retain_raw:continue
        if not raw.exists() or (retain_raw and not raw_mix.exists()):
            y=synthesize(notes,duration,program,sr)
            sf.write(raw,y,sr,subtype='PCM_16')
            if retain_raw:
                if vocal is None:vocal=decode(directory/'lead_vocal.flac',sr,2)
                overlay=.72*vocal[:len(y)]+.38*y[:len(vocal)]
                peak=float(np.max(np.abs(overlay)))
                if peak>.98:overlay*=.98/peak
                sf.write(raw_mix,overlay,sr,subtype='PCM_16')
        normalize(sid,key,[raw])
        normalize(sid,f'{pipeline}_vocal_{timbre}',[
            directory/'listening_v2/lead.mp3',directory/'listening_v2'/(key+'.mp3')],
            '[0:a][1:a]amix=inputs=2:duration=longest:normalize=1,')
        if not retain_raw:raw.unlink(missing_ok=True)
    elapsed=time.perf_counter()-started
    write(outputs/f'{pipeline}_config.json',dict(
        renderer='tinysoundfont 0.3.7 / TinySoundFont',sample_rate=sr,
        soundfont=read(ROOT/'evidence/soundfont.json'),programs=dict(epiano=4,choir=52),
        gain_db=-9,velocity=90,algorithmic_latency_seconds=0,
        original_vocal_timing_unchanged=True,raw_overlay_gains=dict(vocal=.72,instrument=.38),
        listening='same EBU R128 -18 LUFS normalization as existing pipelines; equal normalized inputs for overlays',
        preview_start_seconds=meta['preview_start_seconds'],cpu_only=True,raw_flac_retained=retain_raw))
    row=dict(sample=sid,pipeline=pipeline,wall_seconds=elapsed,audio_seconds=duration)
    write(directory/'status'/f'render_{pipeline}.json',dict(status='complete',**row))
    print('ENSEMBLE_RENDERED',sid,pipeline,round(elapsed,2),flush=True)
    return row


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pipelines',nargs='+',default=[PIPELINE]);parser.add_argument('--compact',action='store_true');parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    assert 1<=args.workers<=4
    started=time.perf_counter()
    tasks=[(x['sample_id'],pipeline) for pipeline in args.pipelines for x in samples()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows=list(executor.map(lambda task:track(*task,retain_raw=not args.compact),tasks))
    record='render_timings.json' if args.pipelines==[PIPELINE] else 'render_combinations_timings.json'
    write(EVIDENCE/record,dict(cpu_only=True,workers=args.workers,
        elapsed_wall_seconds=time.perf_counter()-started,tracks=rows))
    print('ALL_ENSEMBLE_RENDERS_COMPLETE',len(rows),flush=True)
