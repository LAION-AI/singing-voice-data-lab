"""Offline sample-accurate SoundFont rendering; never shifts the original vocal."""
import argparse,time
import numpy as np,soundfile as sf
from common import *

def synthesize(notes,duration,program,sr=44100):
    from tinysoundfont import Synth
    synth=Synth(gain=-9,samplerate=sr)
    font=synth.sfload(str(ROOT/'models/generaluser.sf2'))
    synth.program_select(0,font,0,program)
    ev=[]
    for n in notes:
        ev.extend([(round(n['onset']*sr),1,n['pitch']),(round(n['offset']*sr),0,n['pitch'])])
    nframes=round(duration*sr);out=np.zeros((nframes,2),dtype=np.float32);pos=0
    for frame,on,pitch in sorted(ev)+[(nframes,2,0)]:
        frame=min(nframes,max(pos,frame))
        if frame>pos:out[pos:frame]=np.frombuffer(synth.generate_simple(frame-pos),dtype=np.float32).reshape(-1,2)
        pos=frame
        if on==1:synth.noteon(0,int(pitch),90)
        elif on==0:synth.noteoff(0,int(pitch))
    peak=float(np.max(np.abs(out)))
    if peak>.92:out*=.92/peak
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='*');p.add_argument('--pipelines',nargs='+',default=['game','rosvot','stars','procedural']);a=p.parse_args()
    for item in samples(a.ids):
        sid=item['sample_id'];d=BENCH/sid
        if not (d/'lead_vocal.flac').exists():continue
        meta=read(d/'processing_metadata.json');dur=meta['duration_seconds'];start=meta['preview_start_seconds'];length=meta['preview_duration_seconds'];sr=44100
        vocal=decode(d/'lead_vocal.flac',sr,2)
        for model in a.pipelines:
            path=d/'note_events'/f'{model}.json'
            if not path.exists():continue
            expected=[d/'renders'/f'{model}_{kind}.flac' for kind in ['epiano','choir','vocal_epiano','vocal_choir']]+[d/'preview'/f'{model}_{kind}.mp3' for kind in ['epiano','choir','vocal_epiano','vocal_choir']]
            if all(p.exists() for p in expected) and (d/'status'/f'render_{model}.json').exists():continue
            notes=read(path)['notes'];t=time.perf_counter()
            for timbre,program in [('epiano',4),('choir',52)]:
                out=d/'renders'/f'{model}_{timbre}.flac'
                if all(p.exists() for p in [out,d/'renders'/f'{model}_vocal_{timbre}.flac',d/'preview'/f'{model}_{timbre}.mp3',d/'preview'/f'{model}_vocal_{timbre}.mp3']):continue
                y=synthesize(notes,dur,program,sr)
                out.parent.mkdir(exist_ok=True);sf.write(out,y,sr,subtype='PCM_16')
                # One shared scalar only: no pitch correction or vocal time warping.
                overlay=.72*vocal[:len(y)]+.38*y[:len(vocal)]
                peak=float(np.max(np.abs(overlay)))
                if peak>.98:overlay*=.98/peak
                sf.write(d/'renders'/f'{model}_vocal_{timbre}.flac',overlay,sr,subtype='PCM_16')
                lo=round(start*sr);hi=round((start+length)*sr)
                encode_mp3(y[lo:hi],sr,d/'preview'/f'{model}_{timbre}.mp3')
                encode_mp3(overlay[lo:hi],sr,d/'preview'/f'{model}_vocal_{timbre}.mp3')
            write(d/'renders'/f'{model}_config.json',dict(renderer='tinysoundfont 0.3.7 / TinySoundFont',soundfont=read(ROOT/'evidence/soundfont.json'),sample_rate=sr,gain_db=-9,velocity=90,algorithmic_latency_seconds=0,latency_method='Offline event-to-sample scheduling without a real-time output buffer. SoundFont natural attack is retained, not treated as latency.',overlay_gains=dict(vocal=.72,instrument=.38),peak_limiter='shared scalar if absolute peak exceeds 0.98',vocal_timing_unmodified=True,preview_start_seconds=start))
            complete('render_'+model,sid);metric('render_'+model,sid,time.perf_counter()-t,dur,hardware_scope='CPU only')

if __name__=='__main__':main()

