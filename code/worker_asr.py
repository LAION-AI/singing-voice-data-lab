import argparse,time,re,sys,traceback
import numpy as np,soundfile as sf,torch
from common import *

def extract_words(raw):
    words=[]
    if isinstance(raw,dict):
        if isinstance(raw.get('words'),list):
            for w in raw['words']:
                if not isinstance(w,dict):continue
                ts=w.get('timestamp',w.get('timestamps'))
                a=w.get('start',w.get('start_time',ts[0] if ts else None));b=w.get('end',w.get('end_time',ts[1] if ts else None))
                text=w.get('word',w.get('text',''))
                if a is not None and b is not None:words.append(dict(word=text,onset=float(a),offset=float(b),confidence=w.get('confidence')))
            if words:return words
        if 'chunks' in raw:
            for w in raw['chunks']:
                ts=w.get('timestamp',w.get('timestamps'));text=w.get('text',w.get('word',''))
                if ts and ts[0] is not None and ts[1] is not None:words.append(dict(word=text,onset=float(ts[0]),offset=float(ts[1]),confidence=w.get('confidence')))
            if words:return words
        if isinstance(raw.get('segments'),list):
            for s in raw['segments']:words+=extract_words(s)
    return words

def main():
    p=argparse.ArgumentParser();p.add_argument('model',choices=['parakeet','nemotron']);p.add_argument('--ids',nargs='*');a=p.parse_args()
    sync();t=time.perf_counter()
    if a.model=='parakeet':
        import moondream
        path=ROOT/'models/parakeet'
        model=moondream.photon('moondream/parakeet-redux',device='cuda',model_path=str(path),dtype=torch.bfloat16)
        modelid='moondream/parakeet-redux';revision=read(ROOT/'evidence/parakeet_revision.json')['revision']
    else:
        from transformers import AutoModelForMultimodalLM,AutoProcessor
        path=ROOT/'models/nemotron';processor=AutoProcessor.from_pretrained(path)
        model=AutoModelForMultimodalLM.from_pretrained(path,torch_dtype=torch.bfloat16,device_map='cuda').eval()
        modelid='nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B';revision='c17667641642c4265ff2b7037dd101b7d4567634'
    sync();metric(a.model+'_load',None,time.perf_counter()-t,hardware=hardware(),model=modelid,revision=revision)
    for item in samples(a.ids):
        sid=item['sample_id'];d=BENCH/sid;out=d/'asr'/f'{a.model}.json'
        if out.exists() and read(out).get('status')=='complete':continue
        try:
            sync();t=time.perf_counter();audio=decode(d/'lead_vocal.flac',16000,1)[:,0];words=[];raws=[];texts=[]
            if a.model=='parakeet':
                wav=ROOT/'work'/f'{sid}-asr.wav';sf.write(wav,audio,16000,subtype='PCM_16')
                raw=model.transcribe(audio=str(wav),timestamps='word');raws=[raw];words=extract_words(raw)
                text=raw.get('text',raw.get('transcription',''))
                if not text and raw.get('segments'):text=' '.join(s.get('text','') for s in raw['segments'])
                if not text:text=' '.join(w['word'] for w in words)
            else:
                # 30s disjoint cores, 1s context: retain words by midpoint in core.
                for start in range(0,len(audio),30*16000):
                    lo=max(0,start-16000);hi=min(len(audio),start+31*16000)
                    segment=audio[lo:hi];offset=lo/16000
                    inputs=processor.apply_transcription_request(segment,prompt=None).to(model.device)
                    with torch.inference_mode():outputs=model.generate(**inputs,max_new_tokens=1024,do_sample=False,return_word_timestamp_logits=True)
                    text0=processor.decode(outputs.sequences[:,inputs.input_ids.shape[1]:],skip_special_tokens=False)
                    try:
                        aligned=processor.align_word_timestamps(text0,outputs.word_timestamp_logits,inputs.input_features_mask,outputs.speaker_logits)[0]
                    except ValueError as exc:
                        if 'empty once its special tokens are removed' not in str(exc):raise
                        # A legitimately empty chunk has no alignable words; it must
                        # not discard all the other chunks of a complete song.
                        aligned=dict(chunks=[],diagnostic='Native hypothesis is empty after special-token removal; no timestamps fabricated')
                    raws.append(dict(chunk_offset_seconds=offset,core_start_seconds=start/16000,text=text0,alignment=aligned))
                    ww=extract_words(aligned)
                    for w in ww:
                        w['onset']+=offset;w['offset']+=offset
                        mid=(w['onset']+w['offset'])/2
                        if start/16000<=mid<min(len(audio)/16000,(start/16000)+30):words.append(w)
                text=' '.join(w['word'].strip() for w in words)
                if not text:text=' '.join(str(x['text']) for x in raws)
            sync();elapsed=time.perf_counter()-t
            write(out,dict(status='complete',text=text,words=words,language=read(d/'processing_metadata.json').get('language','unknown'),language_provenance='input metadata; not an independent language detector',model=modelid,revision=revision,alignment_provenance='native word timestamps (Parakeet Photon)' if a.model=='parakeet' else 'native Nemotron auxiliary CTC word alignment, chunk offsets added once',raw_outputs=raws,input_audio='lead_vocal.flac',duration_seconds=len(audio)/16000,word_count=len(words),seconds=elapsed,config=dict(precision='bfloat16',timestamps='word',max_new_tokens=1024 if a.model=='nemotron' else None),diagnostics=dict(empty_transcription=not bool(text.strip()),empty_word_timestamps=not bool(words),invalid_word_spans=sum(w['offset']<w['onset'] or w['onset']<0 for w in words))))
            txt=d/'asr'/f'{a.model}.txt';txt.write_text(text+'\n')
            if a.model=='parakeet':
                canonical=dict(status='complete',text=text,words=words,selected_hypothesis='parakeet',selection_rule='Predeclared Parakeet hypothesis, no outcome-based cherry picking',uncertainty='Automatic singing ASR, not manually verified; compare both full hypotheses and intended lyrics',reference_lyrics_verification='intended lyrics are not ground truth')
                write(d/'asr/canonical_transcript.json',canonical);(d/'asr/canonical_transcript.txt').write_text(text+'\n');write(d/'alignment/words.json',dict(words=words,provenance='Native Parakeet Redux word timestamps, absolute seconds from start of complete separated track'))
            complete(a.model,sid);metric(a.model,sid,elapsed,len(audio)/16000,batch_size=1,precision='bfloat16',words=len(words))
        except Exception as e:failure(a.model,sid,e);traceback.print_exc()

if __name__=='__main__':main()
