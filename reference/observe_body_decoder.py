"""Observe actual trained decoder operations without replacing its arithmetic."""
from inspect_math_sdpa import ObserveMathSDPA

class BodyDecoderObserver:
    def __init__(self,decoder,keep):
        self.hooks=[];self.restore=[]
        for index,net in enumerate(decoder.layers):
            def capture(name,value,index=index):keep(f'layer.{index}.decoder.'+name,value)
            def post(module,name,capture=capture):self.hooks.append(module.register_forward_hook(lambda _m,_a,v:capture(name,v)))
            def pre(module,name,capture=capture):self.hooks.append(module.register_forward_pre_hook(lambda _m,a:capture(name,a[0])))
            post(net.ln_pe_1,'00.token_pe');post(net.ln_pe_2,'01.context_pe')
            post(net.ln1,'02.ln1');pre(net.ln2_1,'20.self_residual');post(net.ln2_1,'21.ln2_1');post(net.ln2_2,'22.ln2_2')
            pre(net.ln3,'40.cross_residual');post(net.ln3,'41.ln3')
            post(net.ffn.layers[0][0],'42.ffn_linear1');post(net.ffn.layers[0][1],'43.ffn_gelu')
            post(net.ffn.layers[1],'44.ffn_linear2');post(net.ffn,'45.ffn_residual')
            self.hooks.append(net.register_forward_hook(lambda _m,_a,v,capture=capture:(capture('90.tokens',v[0]),capture('91.context',v[1])) and None))
            for module,stage in [(net.self_attn,'10.self'),(net.cross_attn,'30.cross')]:
                def inputs(_m,_a,kw,capture=capture,stage=stage):
                    for i,key in enumerate(['q','k','v']):capture(f'{stage}.0{i}.{key}_input',kw[key])
                self.hooks.append(module.register_forward_pre_hook(inputs,with_kwargs=True))
                for i,key in enumerate(['q','k','v']):
                    self.hooks.append(getattr(module,key+'_proj').register_forward_hook(
                        lambda _m,_a,v,key=key,i=i,module=module,stage=stage,capture=capture:capture(f'{stage}.0{i+3}.{key}',module._separate_heads(v))))
                pre(module.proj,stage+'.08.attended');post(module,stage+'.09.output')
                original=module.forward;self.restore.append((module,original))
                def forward(*args,original=original,capture=capture,stage=stage,**kwargs):
                    observer=ObserveMathSDPA()
                    with observer:value=original(*args,**kwargs)
                    if set(observer.taps)!={'logits','probabilities'}:raise ValueError('missing actual decoder SDPA taps')
                    capture(stage+'.06.logits',observer.taps['logits']);capture(stage+'.07.probs',observer.taps['probabilities'])
                    return value
                module.forward=forward

    def close(self):
        for hook in self.hooks:hook.remove()
        for module,original in self.restore:module.forward=original
