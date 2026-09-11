"""Extra real-browser pipeline checks; called by profile_live_demo --qa-pipeline.
No persisted jobs, fake inference, or production server reconfiguration.
"""
def verify(c, box):
    result={}
    # The actual worker and fallback must preserve the same JPEG pixel content
    # for a frozen deterministic opaque image, including resized input.
    result['encoding']=c.evaluate("""(async()=>{
      const {createFrameEncoder}=await import('./frame-pipeline.js');
      const canvas=document.createElement('canvas');canvas.width=320;canvas.height=240;
      const ctx=canvas.getContext('2d'),pixels=ctx.createImageData(320,240);
      for(let i=0;i<pixels.data.length;i+=4){pixels.data[i]=(i/4)%256;pixels.data[i+1]=(i/1280)%256;pixels.data[i+2]=127;pixels.data[i+3]=255}
      ctx.putImageData(pixels,0,0);
      const worker=createFrameEncoder(),fallback=createFrameEncoder({forceFallback:true});
      try{
        const checks=[];
        for(const [w,h] of [[320,240],[160,120]]){
          const a=await worker.encode(canvas,w,h),b=await fallback.encode(canvas,w,h);
          async function read(blob){const im=await createImageBitmap(blob),c=new OffscreenCanvas(w,h),x=c.getContext('2d');
            if(im.width!==w||im.height!==h)throw Error('JPEG dimensions changed');x.drawImage(im,0,0);im.close();return x.getImageData(0,0,w,h).data}
          const aa=await read(a.blob),bb=await read(b.blob);let max=0;
          for(let i=0;i<aa.length;i++)max=Math.max(max,Math.abs(aa[i]-bb[i]));
          if(max!==0)throw Error('Worker/fallback JPEG pixels differ: '+max);
          checks.push({width:w,height:h,max_pixel_difference:max});
          if(w===320&&typeof VideoFrame!=='undefined'){
            const videoFrame=new VideoFrame(canvas,{timestamp:0});
            try{
              const copied=await worker.encode(videoFrame,w,h),cc=await read(copied.blob);
              let difference=0;for(let i=0;i<cc.length;i++)difference=Math.max(difference,Math.abs(cc[i]-bb[i]));
              if(difference!==0||copied.timing.capture_path!=='video-frame-copy')throw Error('Direct frame copy differs');
              checks.push({path:copied.timing.capture_path,max_pixel_difference:difference});
            }finally{videoFrame.close()}
          }
        }
        const cancelled=worker.encode(canvas,320,240);worker.close();
        try{await cancelled;throw Error('Cancelled encoding succeeded')}catch(e){if(e.name!=='AbortError')throw e}
        return {checks,cancelled:true};
      }finally{worker.close();fallback.close()}
    })()""")

    def start(hz):
        c.evaluate("document.querySelector('#camera-open').click()")
        c.wait("!document.querySelector('#track-start').disabled",30)
        if box:
            import json
            c.evaluate('(()=>{const b='+json.dumps(box)+';["x0","y0","x1","y1"].forEach((id,i)=>document.getElementById(id).value=b[i])})()')
        c.evaluate(f"document.querySelector('#track-hz').value={hz};document.querySelector('#track-start').click()")

    def stop():
        c.evaluate("document.querySelector('#track-stop').click()")
        c.wait('!window.sam3dQA.tracking.running',15)
        assert c.evaluate('!window.sam3dQA.tracking.encoding && window.sam3dQA.tracking.prepared===0')

    # Cancel immediately, then reuse the same UI with a fresh worker/session.
    start(10);stop();result['immediate_cancel']=True
    start(3);c.wait('window.sam3dQA.tracking.frames>=5',60)
    d=c.evaluate('window.sam3dQA.tracking');stop()
    assert all(t['interval_ms']>=1000/3-1 for t in d['timings'][1:])
    result['restart_low_cap']={'frames':d['frames'],'encoder':d['encoder']}

    # Feature-detection fallback, not a mock encoder or mocked model response.
    c.evaluate('window.savedWorker=window.Worker;window.Worker=undefined')
    try:
        start(10);c.wait('window.sam3dQA.tracking.frames>=4',60)
        d=c.evaluate('window.sam3dQA.tracking');stop()
        assert d['encoder']=='canvas-fallback' and not d['errors']
        result['fallback']={'frames':d['frames'],'encoder':d['encoder']}
    finally:c.evaluate('window.Worker=window.savedWorker;delete window.savedWorker')

    # Browser variants may expose the processor only in workers or not at all.
    c.evaluate('window.savedProcessor=window.MediaStreamTrackProcessor;window.MediaStreamTrackProcessor=undefined')
    try:
        start(10);c.wait('window.sam3dQA.tracking.frames>=4',60)
        d=c.evaluate('window.sam3dQA.tracking');stop()
        assert d['cameraSource']=='video-element' and not d['errors']
        result['processor_fallback']={'frames':d['frames'],'source':d['cameraSource']}
    finally:c.evaluate('window.MediaStreamTrackProcessor=window.savedProcessor;delete window.savedProcessor')

    # Observe real responses without replacing them; compare the settled display
    # arrays with the actual complete frame payload, not merely finite numbers.
    c.evaluate("""(async()=>{
      const {decodeFrame}=await import('./tracking-math.js');let faces=null,count=0;
      window.savedFetch=window.fetch;window.qaReply=null;
      window.fetch=async(...args)=>{
        const r=await window.savedFetch(...args);
        if(r.ok&&String(args[0]).includes('/frame?')){
          const f=decodeFrame(await r.clone().arrayBuffer(),faces);faces=f.body.faces;
          window.qaReply={number:++count,body:f.body};
        }
        return r;
      };
    })()""")
    try:
        start(10);c.wait('window.sam3dQA.tracking.frames>=5',60)
        result['settled_display_exact']=c.wait("""(()=>{
          if(window.sam3dQA.tracking.settledFrame!==window.qaReply?.number)return false;
          const a=window.sam3dQA.state.result,b=window.qaReply.body;
          for(const k of ['vertices','joints','camera_translation'])
            if(!a.tensors[k].every((v,i)=>Object.is(v,b.tensors[k][i])))throw Error('Settled display differs from native response: '+k);
          return true;
        })()""",30)
        stop()
    finally:c.evaluate('window.fetch=window.savedFetch;delete window.savedFetch;delete window.qaReply')

    # Delay real outgoing requests to exercise replacement under backpressure.
    c.evaluate("""window.savedFetch=window.fetch;window.fetch=async(...args)=>{
      if(String(args[0]).includes('/frame?'))await new Promise(r=>setTimeout(r,450));
      return window.savedFetch(...args);
    }""")
    try:
        start(10);c.wait('window.sam3dQA.tracking.frames>=5',60)
        d=c.evaluate('window.sam3dQA.tracking');stop()
        assert d['maxInFlight']<=2 and d['maxPrepared']<=1 and d['dropped']>0
        assert max(t['queue_ms'] for t in d['timings'][1:])<250
        assert not d['errors']
        result['slow_transport']={k:d[k] for k in ['frames','dropped','maxPrepared','maxInFlight']}
    finally:c.evaluate('window.fetch=window.savedFetch;delete window.savedFetch')
    assert c.evaluate('window.sam3dQA.state.errors.length===0 && window.sam3dQA.tracking.errors.length===0')
    return result
