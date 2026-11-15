import argparse,time,sys,datetime,re
import soundfile as sf,numpy as np,torch
from common import *
from caption_prompts import BLIND,AWARE

def main():
    p=argparse.ArgumentParser();p.add_argument('model',choices=['lfm3','moss']);p.add_argument('--ids',nargs='*');a=p.parse_args()
    targets=samples(a.ids)
    if all(all((BENCH/x['sample_id']/'captions'/f'{a.model}_{condition}.json').exists() and read(BENCH/x['sample_id']/'captions'/f'{a.model}_{condition}.json').get('status') in ['complete','unavailable'] for condition in ['blind','prompt_aware']) for x in targets):
        print('SKIP: all requested captions already complete',flush=True);return
    sync();t=time.perf_counter()
    if a.model=='lfm3':
        from transformers import AutoModelForMultimodalLM,AutoProcessor
        model=AutoModelForMultimodalLM.from_pretrained(ROOT/'models/lfm3',trust_remote_code=True,torch_dtype=torch.bfloat16,device_map='cuda').eval()
        processor=AutoProcessor.from_pretrained(ROOT/'models/lfm3',trust_remote_code=True)
        model_id='LiquidAI/LFM3-Audio-2B';revision='2d6c1f99c129b3ef07a8d168bc0ff08e2bdb93f1'
    else:
        sys.path.insert(0,str(ROOT/'upstream/MOSS-Audio'))
        from src.modeling_moss_audio import MossAudioModel
        from src.processing_moss_audio import MossAudioProcessor
        model=MossAudioModel.from_pretrained(ROOT/'models/moss_audio',trust_remote_code=True,dtype=torch.bfloat16,device_map='cuda').eval()
        processor=MossAudioProcessor.from_pretrained(ROOT/'models/moss_audio',trust_remote_code=True,enable_time_marker=True)
        model_id='OpenMOSS-Team/MOSS-Audio-8B-Instruct';revision='d4dc5a6d8cd79b43dcd82884c75e303b1ecd016d'
    sync();metric(a.model+'_load',None,time.perf_counter()-t,hardware=hardware(),model=model_id,revision=revision)
    for item in samples(a.ids):
        sid=item['sample_id'];d=BENCH/sid;meta=read(d/'processing_metadata.json')
        audio=decode(d/'lead_vocal.flac',16000,1)[:,0]
        # The same complete lead-vocal waveform is used in all four conditions.
        wav=ROOT/'work'/f'{sid}-caption.wav'
        if not wav.exists():sf.write(wav,audio,16000,subtype='FLOAT')
        audio_hash=sha(wav)
        for condition in ['blind','prompt_aware']:
            stage=f'{a.model}_{condition}';out=d/'captions'/f'{stage}.json'
            if out.exists() and read(out).get('status')=='complete':continue
            original=item.get('original_generation_prompt')
            if condition=='prompt_aware' and original is None:
                write(out,dict(status='unavailable',reason='No original generation prompt exists',model=model_id,revision=revision));continue
            prompt=BLIND if condition=='blind' else AWARE.format(original=original)
            try:
                sync();tt=time.perf_counter()
                if a.model=='lfm3':
                    conversation=[{'role':'user','content':[{'type':'audio','audio':str(wav)},{'type':'text','text':prompt}]}]
                    inputs=processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt').to('cuda')
                    with torch.inference_mode():
                        outputs=model.generate(**inputs,max_new_tokens=256,do_sample=False,suppress_mixed=True,suppress_audio_out=True,return_dict_in_generate=True)
                    ids=outputs.sequences[0,inputs['input_ids'].shape[1]:]
                    text=processor.decode(ids,skip_special_tokens=True)
                else:
                    inputs=processor(text=prompt,audios=[audio],return_tensors='pt').to(model.device)
                    if inputs.get('audio_data') is not None:inputs['audio_data']=inputs['audio_data'].to(model.dtype)
                    inputs['audio_input_mask']=inputs['input_ids']==processor.audio_token_id
                    with torch.inference_mode():outputs=model.generate(**inputs,max_new_tokens=256,do_sample=False,num_beams=1,use_cache=True)
                    ids=outputs[0,inputs['input_ids'].shape[1]:];text=processor.decode(ids,skip_special_tokens=True)
                sync();elapsed=time.perf_counter()-tt
                count=len(re.findall(r"\b[\w'-]+\b",text))
                write(out,dict(status='complete',model=model_id,revision=revision,condition=condition,prompt=prompt,
                               original_generation_prompt=original if condition=='prompt_aware' else None,
                               caption=text,word_count=count,within_requested_word_range=50<=count<=80,
                               audio_sha256=audio_hash,input_audio='complete lead_vocal.flac decoded mono 16000 Hz float32',input_duration_seconds=len(audio)/16000,
                               generation=dict(do_sample=False,max_new_tokens=256,precision='bfloat16',output_tokens=int(len(ids)),hit_token_limit=len(ids)>=256),
                               timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=elapsed))
                complete(stage,sid);metric(stage,sid,elapsed,len(audio)/16000,batch_size=1,precision='bfloat16',output_tokens=int(len(ids)),input_tokens=int(inputs['input_ids'].shape[1]))
            except Exception as e:failure(stage,sid,e)

if __name__=='__main__':
    from filelock import FileLock
    import sys
    model=sys.argv[1]
    cohort='-'+'-'.join(sys.argv[sys.argv.index('--ids')+1:]) if '--ids' in sys.argv else ''
    with FileLock(str(ROOT/'work'/f'caption-{model}{cohort}.lock')):main()

