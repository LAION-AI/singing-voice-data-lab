"""Resident, batched ROSVOT/RWBD + frozen Procedural note annotation.

No per-recording files are required. Audio is already an isolated solo voice;
this module does not run the separator or alter the source audio/tokens.
ROSVOT/RMVPE/RWBD implementations and weights retain their upstream terms.
"""
import importlib.util,math,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from worker_f0 import segment
from search_ensemble import context_arrays,candidate_notes
from worker_ensemble import RULES,DEFAULT
from midi_tokens import roundtrip,encode,midi_bytes,name


class Annotator:
    def __init__(self,upstream,models,batch_size=8):
        import torch
        self.torch=torch;self.device=torch.device('cuda:0');self.batch_size=batch_size
        upstream,models=Path(upstream),Path(models)
        sys.path.insert(0,str(upstream))
        spec=importlib.util.spec_from_file_location('packed_rosvot_upstream',upstream/'inference/rosvot.py')
        ros=importlib.util.module_from_spec(spec);spec.loader.exec_module(ros);self.ros=ros
        obj=ros.RosvotInfer.__new__(ros.RosvotInfer)
        obj.args=SimpleNamespace(ckpt=str(models/'rosvot/checkpoints/rosvot/model.pt'),wbd_ckpt=str(models/'rosvot/checkpoints/rwbd/model.pt'),wbd_config='',max_frames=10000,wbd_thr=.5)
        obj.apply_rwbd=True
        obj.hparams=ros.set_hparams(config=str(models/'rosvot/checkpoints/rosvot/config.yaml'),
            print_hparams=False,hparams_str='max_frames=10000,note_bd_threshold=0.85')
        obj.hparams['pe_ckpt']=str(models/'rosvot/checkpoints/rmvpe/model.pt')
        self.hp=obj.hparams;self.model,self.wbd=obj.build_model(self.device,verbose=False)
        self.mel=ros.MelNet(self.hp);self.mel.to(self.device)
        self.pe=ros.RMVPE(str(models/'stars/rmvpe/model.pt'),device=self.device)
        self.model.eval();self.wbd.eval();self.chunk_seconds=30
        torch.set_num_threads(4)

    def rosvot(self,waves):
        """Upstream forward and predicted word boundaries, with in-memory inputs."""
        torch,ros,hp=self.torch,self.ros,self.hp
        items=[dict(item_name=f'chunk-{i}',wav_fn=np.asarray(w,dtype=np.float32)) for i,w in enumerate(waves)]
        dataset=ros.RosvotInferDataset(items,hp,0);result=[None]*len(items)
        order=sorted(range(len(items)),key=lambda i:len(waves[i]))
        with torch.inference_mode():
            for start in range(0,len(order),self.batch_size):
                ids=order[start:start+self.batch_size];batch=dataset.collater([dataset[i] for i in ids]);batch=ros.move_to_cuda(batch,0)
                f0s,uvs=self.pe.get_pitch_batch(batch['wav'],sample_rate=hp['audio_sample_rate'],hop_size=hp['hop_size'],lengths=batch['real_lens'],fmax=hp['f0_max'],fmin=hp['f0_min'])
                ff,uu,pp=[],[],[]
                for i,(f0,uv) in enumerate(zip(f0s,uvs)):
                    size=batch['lens'][i];f0,uv=ros.norm_interp_f0(f0[:size]);f0=ros.pad_or_cut_xd(torch.FloatTensor(f0),size,0);uv=ros.pad_or_cut_xd(torch.FloatTensor(uv),size,0)
                    ff.append(f0);uu.append(uv);pp.append(ros.f0_to_coarse(ros.denorm_f0(f0,uv)))
                pitch=ros.collate_1d_or_2d(pp).to(self.device);uv=ros.collate_1d_or_2d(uu).long().to(self.device)
                mel=self.mel(batch['wav']);mel=ros.pad_or_cut_xd(mel,max(batch['lens']),1)
                mel=(mel.transpose(1,2)*batch['mel_nonpadding'].unsqueeze(1)).transpose(1,2);nonpadding=mel.abs().sum(-1)>0
                word=self.wbd(mel=mel[:,:,:hp.get('wbd_use_mel_bins',80)],pitch=pitch,uv=uv,non_padding=nonpadding,train=False)['word_bd_pred']
                out=self.model(mel=mel[:,:,:hp.get('use_mel_bins',80)],word_bd=word,pitch=pitch,uv=uv,non_padding=nonpadding,train=False)
                lengths=out['note_lengths'].cpu().numpy()
                for j,idx in enumerate(ids):
                    bd=out['note_bd_pred'][j].cpu().numpy()[:batch['real_lens'][j]]
                    pitches=out['note_pred'][j].cpu().numpy()[:lengths[j]]
                    intervals=ros.boundary2Interval(bd)*hp['hop_size']/hp['audio_sample_rate']
                    notes=[]
                    for p,(a,b) in zip(pitches,intervals):
                        # Upstream MIDI export ignores rest pitches, retaining pitch > 0.
                        if int(p)>0 and float(b)>float(a):notes.append(dict(pitch=int(p),onset=float(a),offset=float(b),confidence=None))
                    result[idx]=notes
        return result

    def annotate(self,audios):
        """Input: [{'wave24':mono array,'wave16':mono array,'duration':seconds}]."""
        torch=self.torch;chunks=[];owners=[]
        for i,item in enumerate(audios):
            for a in range(0,len(item['wave24']),720000):
                chunks.append(item['wave24'][a:a+720000]);owners.append((i,a/24000))
        raw=self.rosvot(chunks);rosnotes=[[] for _ in audios]
        for notes,(i,offset),wave in zip(raw,owners,chunks):
            duration=len(wave)/24000
            for n in notes:
                a=max(0.,n['onset']);b=min(duration,n['offset'])
                if b>a:rosnotes[i].append(dict(n,onset=a+offset,offset=b+offset))
        # Preserve the original 30-s cores, 0.5-s context and threshold .03.
        # Single-window RMVPE calls retain the baseline's endpoint padding behavior.
        results=[]
        with torch.inference_mode():
            for item,notes in zip(audios,rosnotes):
                wave=item['wave16'];parts=[];core=480000;margin=8000
                for a in range(0,len(wave),core):
                    lo=max(0,a-margin);hi=min(len(wave),a+core+margin)
                    f=np.asarray(self.pe.infer_from_audio(wave[lo:hi],sample_rate=16000,thred=.03))
                    begin=round((a-lo)/160);end=begin+int(np.ceil(min(core,len(wave)-a)/160));parts.append(f[begin:end])
                f0=np.concatenate(parts);times=np.arange(len(f0))*.01
                assert np.isfinite(f0).all() and np.all(f0>=0)
                proc=segment(f0);duration=item['duration'];ros=dict(notes=notes,track_duration_seconds=duration)
                context=context_arrays(ros,dict(notes=proc),dict(time_seconds=times,f0_hz=f0))
                merged=candidate_notes(context,RULES[DEFAULT]);proof=roundtrip(merged,duration)
                for n in merged:n.update(pitch_name=name(n['pitch']),duration=n['offset']-n['onset'])
                notes_dict=dict(schema='singing-note-events-1.0',pipeline=DEFAULT,track_duration_seconds=duration,notes=merged,
                    roundtrip=proof,provenance=dict(rule=RULES[DEFAULT],rosvot_note_threshold=.85,rwbd_threshold=.5,
                    procedural_threshold=.03,rosvot_chunk_seconds=30,procedural_context_seconds=.5,
                    rosvot_batch_size=self.batch_size,source_assumed_isolated=True,separator_used=False,
                    endpoints_clipped_to_available_chunk_audio=True,human_verified=False,
                    evaluation_caveat='Automatic note estimates; shared RMVPE is not independent ground truth.'))
                results.append(dict(annotation=notes_dict,rosvot=ros,procedural=dict(notes=proc,track_duration_seconds=duration),
                    flattened_midi=encode(merged,duration),midi=midi_bytes(merged),time_seconds=times,f0_hz=f0))
        return results
