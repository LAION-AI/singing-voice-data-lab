"""Keep Mega53's unchanged shared tensors and lead-vocal head, in small shards."""
import argparse,hashlib,json,re,sys
from pathlib import Path
import torch
import yaml
from safetensors.torch import save_file,load_file

SOURCE_SHA256='c62820893bbf86d4e734f966bd142d9157cfc8bb8e79e9d8f9ea553f3ff3519f'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert sha(a.checkpoint)==SOURCE_SHA256,'Input is not the pinned official checkpoint'
    cfg=yaml.full_load(a.config.read_text());instruments=cfg['training']['instruments'];head=instruments.index('lead-vocal')
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    source=source.get('state_dict',source);selected={};mapping={}
    for key,tensor in source.items():
        match=re.match(r'mask_estimators\.(\d+)\.',key)
        if match and int(match.group(1))!=head:continue
        target=key.replace(f'mask_estimators.{head}.','mask_estimators.0.',1) if match else key
        selected[target]=tensor.detach().contiguous();mapping[target]=key
    model_cfg=cfg['model'];model_cfg['num_stems']=1
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'config.json').write_text(json.dumps(model_cfg,indent=2)+'\n')
    groups=[];group={};size=0
    for key,tensor in sorted(selected.items()):
        n=tensor.numel()*tensor.element_size()
        if group and size+n>32*1024*1024:groups.append(group);group={};size=0
        group[key]=tensor;size+=n
    if group:groups.append(group)
    manifest=dict(architecture='MVSEP Mega53 BS-RoFormer v1, shared backbone + original lead-vocal head only',
        original_head_index=head,original_stems=53,retained_stems=['lead-vocal'],quantized=False,
        source_url='https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21',
        source_checkpoint_sha256=SOURCE_SHA256,
        upstream_code_commit='f0b527736e7da24eae5b561a2bb1efd0569ec757',
        license='MIT',license_confirmation='https://github.com/ZFTurbo/Music-Source-Separation-Training/issues/245#issuecomment-5834608838',
        weight_map={},shards=[],tensor_values_equal_to_source=True,parameters=sum(x.numel() for x in selected.values()))
    for i,group in enumerate(groups,1):
        name=f'model-{i:05d}-of-{len(groups):05d}.safetensors';path=a.output/name
        save_file(group,str(path),metadata={'format':'pt','origin':'MVSEP Mega53 v1','stem':'lead-vocal','precision':'unchanged'})
        restored=load_file(str(path),device='cpu')
        for key,value in restored.items():
            assert torch.equal(value,source[mapping[key]]) and value.dtype==source[mapping[key]].dtype
            manifest['weight_map'][key]=name
        manifest['shards'].append(dict(file=name,bytes=path.stat().st_size,sha256=sha(path)))
    # Strictly validate the pruned model's complete state; no neural forward.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/MEGA53'))
    from models.bs_roformer.bs_roformer import BSRoformer
    model=BSRoformer(**model_cfg);model.load_state_dict(selected,strict=True)
    manifest['strict_model_load_verified']=True
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'parameters':manifest['parameters'],'bytes':sum(x['bytes'] for x in manifest['shards']),'shards':len(groups),'tensor_exact':True,'strict_load':True}),flush=True)

if __name__=='__main__':main()
