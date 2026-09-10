"""Streaming real-module/real-SDPA observations for the full trained backbone."""
import json
from pathlib import Path

import torch
from safetensors.numpy import save_file
from inspect_math_sdpa import ObserveMathSDPA
from capture_dino_backbone import digest


class TrainedDinoObserver:
    def __init__(self, model, directory):
        self.directory = Path(directory)
        self.directory.mkdir()
        self.records, self.hooks, self.restore = [], [], []
        self.expected_blocks=len(model.blocks)
        for index, block in enumerate(model.blocks):
            self.attach(block, index)

    def attach(self, block, index):
        taps = {}
        def capture(name, value):
            taps[name] = value.detach().float().cpu().numpy().copy()
        def begin(_module, args, kwargs):
            if taps:
                raise ValueError('block observer has unconsumed data')
            rope = kwargs.get('rope', args[1] if len(args)>1 else None)
            if rope is None:
                raise ValueError('missing original RoPE input')
            prefix = args[0].shape[1]-rope[0].shape[0]
            capture('00.rope_sin', torch.nn.functional.pad(rope[0], (0,0,prefix,0), value=0))
            capture('01.rope_cos', torch.nn.functional.pad(rope[1], (0,0,prefix,0), value=1))
        self.hooks.append(block.register_forward_pre_hook(begin, with_kwargs=True))
        for attr, name in [('norm1','02.norm1'), ('attn.qkv','03.qkv'), ('attn.proj','12.attn_proj'),
                           ('ls1','13.ls1'), ('norm2','15.norm2'), ('mlp.w1','16.w1'),
                           ('mlp.w2','17.w2'), ('mlp.w3','19.w3'), ('ls2','20.ls2')]:
            self.hooks.append(block.get_submodule(attr).register_forward_hook(
                lambda _m, _a, v, name=name: capture(name, v)))
        for attr, name in [('attn.proj','11.attention'), ('norm2','14.residual1'), ('mlp.w3','18.hidden')]:
            self.hooks.append(block.get_submodule(attr).register_forward_pre_hook(
                lambda _m, a, name=name: capture(name, a[0])))
        apply_rope = block.attn.apply_rope
        def observed_rope(q, k, rope):
            capture('04.q', q); capture('05.k', k)
            result = apply_rope(q, k, rope)
            capture('07.q_rope', result[0]); capture('08.k_rope', result[1])
            return result
        compute = block.attn.compute_attention
        def observed_attention(*args, **kwargs):
            observer = ObserveMathSDPA()
            with observer:
                result = compute(*args, **kwargs)
            if set(observer.taps) != {'logits', 'probabilities'}:
                raise ValueError('missing original math-SDPA softmax observation')
            capture('09.logits', observer.taps['logits'])
            capture('10.probs', observer.taps['probabilities'])
            return result
        block.attn.apply_rope = observed_rope
        block.attn.compute_attention = observed_attention
        self.restore.append((block.attn, apply_rope, compute))
        def finish(_module, _args, output):
            capture('21.output', output)
            b,n,total=taps['03.qkv'].shape
            taps['06.v'] = taps['03.qkv'].reshape(b,n,3,block.attn.num_heads,total//(3*block.attn.num_heads))[:,:,2].transpose(0,2,1,3).copy()
            if len(taps) != 22 or index != len(self.records):
                raise ValueError('block observation coverage/order mismatch')
            name = f'block.{index}.safetensors'
            save_file(taps, self.directory/name)
            self.records.append(dict(index=index, file=name, sha256=digest(self.directory/name),
                                     order=sorted(taps), shapes={key:list(value.shape) for key,value in taps.items()}))
            taps.clear()
            print(f'Streamed original block {index} operations', flush=True)
        self.hooks.append(block.register_forward_hook(finish))

    def close(self):
        for hook in self.hooks:
            hook.remove()
        for module, apply_rope, compute in self.restore:
            module.apply_rope, module.compute_attention = apply_rope, compute
        if len(self.records) != self.expected_blocks:
            raise ValueError('incomplete backbone operation capture')
        manifest = dict(schema_version=1, scope=f'all {self.expected_blocks} blocks in one own-intermediate original backbone forward',
                        logits_probabilities='actual math SDPA dispatch, not auxiliary formulas', blocks=self.records,
                        observer_sha256=digest(Path(__file__)), sdpa_observer_sha256=digest(Path(__file__).with_name('inspect_math_sdpa.py')))
        (self.directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
