#!/usr/bin/env python3
"""Run the pure-C GGUF image consumer and record executable/input provenance."""
import argparse,json,os,subprocess,time
from pathlib import Path
from check_parity import sha256_file
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['runner','library','module','backbone','branch','mhr','image-input','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--backend',choices=['CPU','Vulkan'],required=True);p.add_argument('--device',type=int,default=0)
    p.add_argument('--description',default='-');p.add_argument('--threads',type=int,default=6);a=p.parse_args()
    paths={k:getattr(a,k) for k in ['runner','library','module','backbone','branch','mhr','image_input']}
    identities={k:sha256_file(v) for k,v in paths.items()}
    expected=dict(backbone='9228c12b5b34cdb3627fce731a3e3d5890c54dc78ca7eb357d66056893f5bf1f',branch='eebd51ac66bab764b52c7c51a47671bfa6152980a5bc7da6ea7fe20f6ae61431',mhr='d52ab772628fb6d851550381b428185a32da6398793f0ff70299bad1999ba8f8')
    if any(identities[k]!=v for k,v in expected.items()):raise ValueError('unexpected GGUF identity')
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise FileExistsError('output must be empty')
    env=os.environ.copy()
    env['LD_LIBRARY_PATH']=str(a.library.resolve().parent)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    if a.backend=='Vulkan':env.update(GGML_VK_DISABLE_F16='1',GGML_VK_DISABLE_COOPMAT='1',GGML_VK_DISABLE_COOPMAT2='1')
    argv=[str(a.runner.resolve()),str(a.module.resolve()),a.backend,str(a.device),a.description,*[str(paths[k].resolve()) for k in ['backbone','branch','mhr','image_input']],str((a.output/'result.bin').resolve()),str(a.threads)]
    started=time.monotonic();subprocess.run(argv,env=env,check=True);elapsed=time.monotonic()-started
    if any(sha256_file(v)!=identities[k] for k,v in paths.items()):raise ValueError('input or binary changed during run')
    report=dict(scope='pure C shared-library inference, copied image, model freed before result read; not optimized timing',argv=argv,sha256=identities,
                result_sha256=sha256_file(a.output/'result.bin'),script_sha256=sha256_file(Path(__file__)),elapsed_seconds=elapsed,returncode=0)
    with (a.output/'run.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
