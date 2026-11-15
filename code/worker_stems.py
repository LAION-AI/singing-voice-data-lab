"""Exact Mega53 lead-head extraction using bundled, tensor-identical weights."""
import argparse,contextlib,json,sys,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from common import ROOT,BENCH,read,write,sha,samples,decode,encode_mp3,complete,metric

def load_model(device):
    from safetensors.torch import load_file
    sys.path.insert(0,str(ROOT/'upstream/MEGA53'))
    from models.bs_roformer.bs_roformer import BSRoformer
    directory=ROOT/'models/mega53_lead_v1';manifest=read(directory/'manifest.json')
    state={}
    for shard in manifest['shards']:
        path=directory/shard['file']
        if sha(path)!=shard['sha256']:raise RuntimeError('Bundled weight checksum mismatch: '+shard['file'])
        state.update(load_file(str(path),device='cpu'))
    model=BSRoformer(**read(directory/'config.json'));model.load_state_dict(state,strict=True)
    return model.to(device).eval(),manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='*');p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=1);a=p.parse_args()
    if a.batch_size<1:raise ValueError('batch-size must be positive')
    device=torch.device(a.device);model,manifest=load_model(device)
    sr=44100;chunk=20*sr;hop=chunk//2;window=np.maximum(np.hanning(chunk).astype(np.float32),1e-5)
    for item in samples(a.ids):
        sid=item['sample_id'];d=BENCH/sid;status=d/'status/stems.json'
        if status.exists() and read(status).get('status')=='complete':continue
        start=time.perf_counter();audio=decode(d/item['source_audio'],sr,2);n=len(audio)
        output=np.zeros_like(audio);weight=np.zeros(n,np.float32)
        positions=[]
        for at in range(0,n,hop):
            positions.append(at)
            if at+chunk>=n:break
        for begin in range(0,len(positions),a.batch_size):
            offsets=positions[begin:begin+a.batch_size];inputs=np.zeros((len(offsets),2,chunk),np.float32)
            for i,at in enumerate(offsets):inputs[i,:,:min(chunk,n-at)]=audio[at:at+chunk].T
            xt=torch.from_numpy(inputs).to(device)
            amp=torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' else contextlib.nullcontext()
            with torch.inference_mode(),amp:pred=model(xt)[:,0].float().cpu().numpy()
            for i,at in enumerate(offsets):
                length=min(chunk,n-at);output[at:at+length]+=pred[i,:,:length].T*window[:length,None];weight[at:at+length]+=window[:length]
        output/=np.maximum(weight[:,None],1e-5);assert np.isfinite(output).all()
        sf.write(d/'lead_vocal.flac',output,sr,subtype='PCM_24');duration=n/sr
        preview=max(range(0,max(1,int(duration)-30),5),key=lambda s:float(np.mean(output[s*sr:min(n,(s+30)*sr)]**2))) if duration>45 else 0
        length=min(35,duration-preview);(d/'preview').mkdir(exist_ok=True)
        encode_mp3(audio[round(preview*sr):round((preview+length)*sr)],sr,d/'preview/original.mp3')
        encode_mp3(output[round(preview*sr):round((preview+length)*sr)],sr,d/'preview/lead.mp3')
        item.update(duration_seconds=duration,preview_start_seconds=preview,preview_duration_seconds=length,
            lead_sample_rate=sr,lead_sha256=sha(d/'lead_vocal.flac'),extraction=dict(
                model='MVSEP Mega53 v1: unchanged backbone + lead-vocal head',original_head_index=27,
                checkpoint_sha256=manifest['source_checkpoint_sha256'],pruned_tensor_exact=True,
                sample_rate=sr,channels=2,chunk_seconds=20,overlap=2,batch_size=a.batch_size,
                precision='float16 autocast' if device.type=='cuda' else 'float32',
                postprocessing='weighted Hann overlap-add; no pitch or time correction'),
            quality_flags=['Separation is not certified monophonic; backing voices and artifacts may remain.'])
        write(d/'processing_metadata.json',item);complete('stems',sid)
        metric('mega53_lead',sid,time.perf_counter()-start,duration,batch_size=a.batch_size)

if __name__=='__main__':main()
