"""Write note sidecars and self-contained audio/token data directly into TARs."""
import gzip,hashlib,io,json,math,tarfile,time
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

def dumps(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def put(tar,name,payload):
    info=tarfile.TarInfo(name);info.size=len(payload);info.mtime=0;info.mode=0o644;tar.addfile(info,io.BytesIO(payload))
def audio_input(payload):
    wave,sr=sf.read(io.BytesIO(payload),dtype='float32',always_2d=True)
    assert np.isfinite(wave).all() and len(wave)>0
    mono=wave.mean(axis=1)
    def resample(rate):
        if sr==rate:return mono.copy()
        div=math.gcd(sr,rate);return np.asarray(resample_poly(mono,rate//div,sr//div),dtype=np.float32)
    return dict(wave24=resample(24000),wave16=resample(16000),duration=len(wave)/sr)

def process_task(rows,annotator,notes_path,data_path,*,metadata_filter=lambda x:x,batch_size=16,progress=None):
    """All rows share one source TAR; each output becomes visible only when closed."""
    notes_path,data_path=Path(notes_path),Path(data_path)
    notes_path.parent.mkdir(parents=True,exist_ok=True);data_path.parent.mkdir(parents=True,exist_ok=True)
    assert not notes_path.exists() and not data_path.exists()
    start=time.monotonic();done=0;seconds=0.;note_count=0;missing_scripts=0
    source_path=Path(rows[0]['audio_tar'])
    # A sequential read keeps TAR header seeks off shared storage.
    with tarfile.open(fileobj=io.BytesIO(source_path.read_bytes())) as source:
        members={m.name:m for m in source if m.isfile()}
        token_source=None
        if rows[0].get('token_tar'):
            token_source=tarfile.open(fileobj=io.BytesIO(Path(rows[0]['token_tar']).read_bytes()))
            token_members={m.name:m for m in token_source if m.isfile()}
        else:token_source=source;token_members=members
        try:
            with tarfile.open(str(notes_path)+'.partial','w') as notes_tar,tarfile.open(str(data_path)+'.partial','w') as data_tar:
                for begin in range(0,len(rows),batch_size):
                    batch=rows[begin:begin+batch_size]
                    raw_audio=[source.extractfile(members[r['audio_member']]).read() for r in batch]
                    waves=[audio_input(raw) for raw in raw_audio]
                    results=annotator.annotate(waves)
                    for r,raw,wave,result in zip(batch,raw_audio,waves,results):
                        uid=r['uid'];source_meta=json.load(source.extractfile(members[r['metadata_member']]))
                        if source_meta.get('schema')=='synthetic-singing-with-notes-1.0':source_meta=source_meta['original_metadata']
                        expected=r['source_metadata_index'].get('audio_sha256',r['source_metadata_index'].get('source_audio_sha256'))
                        digest=hashlib.sha256(raw).hexdigest()
                        if expected:assert digest==expected, 'Source audio checksum mismatch: '+uid
                        original=r.get('instruction');flat=result['flattened_midi'];general=script=None;with_midi=None
                        if original:
                            general,sep,script=original.partition('SCRIPT:')
                            assert sep and general.lstrip().startswith('GENERAL:')
                            with_midi=general.rstrip()+'\nMIDI Annotation:\n'+flat.rstrip()+'\nSCRIPT:'+script
                        else:missing_scripts+=1
                        source_id=r['source_metadata_index']['id']
                        record=dict(schema='synthetic-singing-with-notes-1.0',id=uid,source_id=source_id,
                            source_repository=r['source_repository'],source_revision=r['source_revision'],source_shard=r['source_shard'],
                            source_audio_member=r['audio_member'],source_audio_sha256=digest,modality=r['modality'],language=r['language'],
                            speaker_alias=r['alias'],duration_seconds=wave['duration'],original_metadata=source_meta,
                            original_index=r['source_metadata_index'],plain_text=r['plain'],prompt_original=original,
                            general=general.rstrip() if general else None,midi_annotation=flat,script=script,
                            prompt_with_midi=with_midi,training_selected_for_experimental_r128=r['training_member'],
                            script_available=bool(original),human_quality_verified=False,quality_filtering_complete=False,
                            annotation_pipeline=result['annotation']['pipeline'],note_count=len(result['annotation']['notes']),
                            voiced_frame_fraction=float((result['f0_hz']>0).mean()),
                            acoustic_annotation_locator=dict(repository=r['source_repository'],revision=r['source_revision'],
                                shard=r['source_metadata_index'].get('annotation_shard',r['source_shard']),
                                member=r.get('acoustic_member',r['source_metadata_index'].get('annotation_member'))))
                        record=metadata_filter(record)
                        pitch=io.BytesIO();np.savez_compressed(pitch,time_seconds=result['time_seconds'],f0_hz=result['f0_hz'],voiced=result['f0_hz']>0)
                        content=[(uid+'.json',dumps(record)),(uid+'.notes.json',dumps(metadata_filter(result['annotation']))),
                            (uid+'.rosvot.json',dumps(result['rosvot'])),(uid+'.procedural.json',dumps(result['procedural'])),
                            (uid+'.mid',result['midi']),(uid+'.midi.txt',flat.encode()),(uid+'.pitch.npz',pitch.getvalue())]
                        if with_midi:content.append((uid+'.prompt.txt',metadata_filter(with_midi).encode()))
                        for name,payload in content:
                            put(notes_tar,name,payload);put(data_tar,name,payload)
                        put(data_tar,uid+Path(r['audio_member']).suffix,raw)
                        frames=[]
                        for key,suffix,books in [('q12_member','.codes.npy',12),('q32_member','.codes32.npy',32)]:
                            payload=token_source.extractfile(token_members[r[key]]).read();array=np.load(io.BytesIO(payload),allow_pickle=False)
                            assert array.ndim==2 and array.shape[1]==books and array.dtype.kind in 'iu' and array.min()>=0 and array.max()<=1023
                            frames.append(array.shape[0]);put(data_tar,uid+suffix,payload)
                        assert frames[0]==frames[1]
                        if r.get('acoustic_member') and r['acoustic_member'] in members:
                            ann=json.loads(gzip.decompress(source.extractfile(members[r['acoustic_member']]).read()))
                            put(data_tar,uid+'.acoustic.json.gz',gzip.compress(dumps(metadata_filter(ann)),compresslevel=3,mtime=0))
                        done+=1;seconds+=wave['duration'];note_count+=len(result['annotation']['notes'])
                    if progress:progress(dict(completed=done,total=len(rows),seconds=seconds,elapsed=time.monotonic()-start))
        finally:
            if token_source is not source:token_source.close()
    Path(str(notes_path)+'.partial').replace(notes_path);Path(str(data_path)+'.partial').replace(data_path)
    return dict(samples=done,audio_seconds=seconds,notes=note_count,missing_scripts=missing_scripts,
                wall_seconds=time.monotonic()-start,resampling='soundfile decode; mono mean; scipy resample_poly to 24k/16k',
                audio_bytes_unchanged=True,token_bytes_unchanged=True)
