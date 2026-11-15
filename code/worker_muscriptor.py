"""Full-track MuScriptor voice-only transcription. Run only on allocated GPU.

Raw streaming events and upstream MIDI are retained. No beat tracking, latency
correction, quantization, fill-gaps, pitch correction, or monophonic rewrite.
"""
import argparse, dataclasses, json, os, sys, time
from pathlib import Path
from common import ROOT, BENCH, read, write, sha, samples, decode, sync, metric, hardware, failure
from midi_tokens import load_midi
from worker_f0 import export


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['large','medium'],required=True);p.add_argument('--ids',nargs='*');a=p.parse_args()
    import torch
    if not torch.cuda.is_available():raise RuntimeError('MuScriptor inference requires a CUDA GPU')
    sys.path.insert(0,str(ROOT/'upstream/MuScriptor'))
    from muscriptor.transcription_model import TranscriptionModel
    from muscriptor.events import NoteStartEvent, NoteEndEvent, ProgressEvent
    from muscriptor.utils.beats import BeatGrid
    pipe='muscriptor_'+a.model;folder=ROOT/'models'/pipe;weights=folder/'model.safetensors'
    if not weights.exists():raise RuntimeError('Checkpoint absent. Grant HF gate access and run download_muscriptor.py first; no alternative model is substituted.')
    lock=read(ROOT/'evidence/v3/muscriptor_lock.json');info=lock['models'][a.model]
    t=time.perf_counter();model=TranscriptionModel.load_model(weights,device='cuda',dtype='float32');sync()
    metric(pipe+'_load',None,time.perf_counter()-t,hardware=hardware(),model_revision=info['revision'])
    settings=dict(instruments=['voice'],use_sampling=False,temperature=1.,cfg_coef=1.,batch_size=1,beam_size=1,prelude_forcing=True,no_eos_is_ok=True)
    failures=0
    for item in samples(a.ids):
        sid=item['sample_id'];d=BENCH/sid;dest=d/'status'/f'{pipe}.json'
        if dest.exists() and read(dest).get('status')=='complete':continue
        try:
            torch.cuda.reset_peak_memory_stats();t=time.perf_counter();audio=decode(d/'lead_vocal.flac',16000,1)[:,0];duration=len(audio)/16000
            raw=d/'raw'/pipe;raw.mkdir(parents=True,exist_ok=True)
            events=list(model.transcribe((torch.from_numpy(audio.copy())[None],16000),**settings));sync();elapsed=time.perf_counter()-t
            write(raw/'streaming_events.json',[dict(type=type(e).__name__,**dataclasses.asdict(e)) for e in events])
            midi=model.events_to_midi_bytes(iter(events),beat_grid=BeatGrid(bpm=120.,beats_per_bar=None,first_downbeat=0.,onset_delay=0.),export=None);(raw/'native.mid').write_bytes(midi)
            notes=load_midi(raw/'native.mid');progress=[e for e in events if isinstance(e,ProgressEvent)]
            if not progress or progress[-1].completed != progress[-1].total:raise RuntimeError('Incomplete native chunk stream')
            # Preserve upstream voice-only note events, including genuine overlap errors.
            overlap=sum(n['onset']<prior['offset'] for prior,n in zip(notes,notes[1:]))
            provenance=dict(model=info['repo'],revision=info['revision'],checkpoint_sha256=info['sha256'],implementation=lock['implementation'],settings=settings,precision='float32 weights; upstream CUDA autocast float16',input_audio_sha256=sha(d/'lead_vocal.flac'),input_sample_rate=16000,input_duration_seconds=duration,raw_streaming_events=f'raw/{pipe}/streaming_events.json',native_midi=f'raw/{pipe}/native.mid',native_export_cleanup='Upstream validate_notes(fix=True) and same-pitch overlap trim; no additional correction',canonical_conversion='Absolute seconds from native MIDI; fixed-velocity benchmark serialization',monophonic_request='voice instrument hard mask; different-pitch overlap is retained and flagged, not artificially repaired',adjacent_note_overlaps=overlap,chunks_completed=progress[-1].completed,time_adjustment='none; onset_delay explicitly fixed to 0 in a constant 120-BPM serialization grid; natural upstream timing lag is retained',created_unix=time.time())
            export(d,pipe,notes,duration,provenance)
            metric(pipe,sid,elapsed,duration,batch_size=1,chunk_seconds=5,precision=provenance['precision'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),note_count=len(notes),hardware=hardware())
        except Exception as e:
            failure(pipe,sid,e);failures+=1
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
