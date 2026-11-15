import argparse,sys,time
import numpy as np,soundfile as sf,torch
from scipy.ndimage import median_filter
from common import *
from midi_tokens import name,roundtrip,save_midi,encode

def segment(f0,hop=.01):
    voiced=f0>0;pitch=np.zeros_like(f0);pitch[voiced]=69+12*np.log2(f0[voiced]/440)
    # Median smoothing plus sustained-change hysteresis suppresses vibrato splitting.
    smooth=median_filter(pitch,size=9,mode='nearest');out=[];i=0
    while i<len(f0):
        if not voiced[i]:i+=1;continue
        start=i;center=float(np.median(pitch[i:min(i+10,len(f0))][voiced[i:min(i+10,len(f0))]]));i+=1
        while i<len(f0):
            if not voiced[i] and not voiced[i:min(i+4,len(f0))].any():break
            if voiced[i] and abs(smooth[i]-center)>.8:
                window=smooth[i:min(i+7,len(f0))]
                if len(window)>=5 and np.all(np.abs(window-center)>.65):break
            if voiced[i]:center=.98*center+.02*smooth[i]
            i+=1
        end=i
        vals=f0[start:end];vals=vals[vals>0]
        if (end-start)*hop>=.09 and len(vals):
            p=int(np.clip(np.floor(69+12*np.log2(np.median(vals)/440)+.5),0,127))
            out.append(dict(pitch=p,onset=start*hop,offset=end*hop,confidence=None,median_f0_hz=float(np.median(vals)),pitch_deviation_cents=float(np.median(1200*np.log2(vals/(440*2**((p-69)/12)))))))
        i=max(i,start+1)
    return out

def export(d,model,notes,duration,provenance):
    for n in notes:n.update(pitch_name=name(n['pitch']),duration=n['offset']-n['onset'])
    report=roundtrip(notes,duration)
    write(d/'note_events'/f'{model}.json',dict(schema='singing-note-events-1.0',pipeline=model,track_duration_seconds=duration,notes=notes,provenance=provenance,roundtrip=report))
    save_midi(notes,d/'midi'/f'{model}.mid')
    f=d/'flattened_midi'/f'{model}.txt';f.parent.mkdir(parents=True,exist_ok=True);f.write_text(encode(notes,duration))
    complete(model,d.name,note_count=len(notes),roundtrip=report)

def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='*');args=p.parse_args()
    sys.path.insert(0,str(ROOT/'upstream/STARS'))
    from modules.pe.rmvpe import RMVPE
    t=time.perf_counter();model=RMVPE(str(ROOT/'models/stars/rmvpe/model.pt'),device='cuda');sync()
    metric('rmvpe_load',None,time.perf_counter()-t,hardware=hardware())
    for item in samples(args.ids):
        sid=item['sample_id'];d=BENCH/sid
        if (d/'status/procedural.json').exists() and read(d/'status/procedural.json')['status']=='complete':continue
        try:
            t=time.perf_counter();audio=decode(d/'lead_vocal.flac',16000,1)[:,0];f0s=[]
            # Independent 30s windows with 0.5s overlap; retain each core exactly once.
            core=30*16000;margin=8000
            for a in range(0,len(audio),core):
                lo=max(0,a-margin);hi=min(len(audio),a+core+margin)
                f=np.asarray(model.infer_from_audio(audio[lo:hi],sample_rate=16000,thred=.03))
                start=round((a-lo)/160);end=start+int(np.ceil(min(core,len(audio)-a)/160))
                f0s.append(f[start:end])
            sync();f0=np.concatenate(f0s);times=np.arange(len(f0))*.01
            assert np.isfinite(f0).all()
            (d/'pitch').mkdir(exist_ok=True);np.savez_compressed(d/'pitch/f0.npz',time_seconds=times,f0_hz=f0,voiced=f0>0)
            np.save(d/'pitch/f0.npy',f0)
            write(d/'pitch/config.json',dict(model='RMVPE',checkpoint_revision='verstar/STARS@744a7ad02e1d788452293cd903ea6a933f7862c4',hop_seconds=.01,threshold=.03,confidence='voiced boolean; calibrated frame probabilities are not exported by this API',chunk_seconds=30,context_margin_seconds=.5))
            export(d,'procedural',segment(f0),len(audio)/16000,dict(model='RMVPE + median/hysteresis v1',minimum_note_seconds=.09,median_window_seconds=.09,sustained_pitch_change_semitones=.8))
            metric('rmvpe_procedural',sid,time.perf_counter()-t,len(audio)/16000,batch_size=1,precision='float32')
        except Exception as e:failure('procedural',sid,e)

if __name__=='__main__':main()
