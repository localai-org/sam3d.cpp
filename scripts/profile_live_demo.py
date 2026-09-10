#!/usr/bin/env python3
"""Profile real webcam-path requests, including browser JPEG time and server stages.
Uses a virtual camera, actual native inference, and no persistent generations.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
import urllib.request
from devtools import CDP

def summary(values):
    if not values or any(not math.isfinite(v) or v<0 for v in values):
        raise ValueError('missing/nonfinite timing samples')
    return dict(count=len(values),mean=statistics.mean(values),median=statistics.median(values),
                minimum=min(values),maximum=max(values),p95=sorted(values)[math.ceil(.95*len(values))-1])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',required=True);p.add_argument('--camera',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--chrome',default='chromium')
    p.add_argument('--hz',type=int,default=10);p.add_argument('--frames',type=int,default=40)
    p.add_argument('--warmup',type=int,default=5);p.add_argument('--box',type=float,nargs=4)
    p.add_argument('--hardware-webgl',action='store_true')
    p.add_argument('--qa-pipeline',action='store_true',help='also test JPEG pixel equality, cancellation, fallback and slow transport')
    a=p.parse_args()
    if not 1<=a.hz<=30 or not 5<=a.frames<=180 or not 1<=a.warmup<=20:p.error('invalid profile bounds')
    a.output.mkdir(parents=True,exist_ok=False)
    report={'passed':False,'url':a.url,'cap_hz':a.hz,'warmup':a.warmup,'scope':'actual live browser/native pipeline; virtual camera; overlapping stage timings must not be summed'}
    # A freshly restarted server hashes model files before it starts listening.
    deadline=time.monotonic()+30
    while True:
        try:
            with urllib.request.urlopen(a.url.rstrip('/')+'/api/config',timeout=5) as response:json.load(response)
            break
        except Exception:
            if time.monotonic()>=deadline:raise
            time.sleep(.2)
    with tempfile.TemporaryDirectory(prefix='sam3d-live-profile-',ignore_cleanup_errors=True) as profile,(a.output/'chrome.log').open('w') as log:
        args=[a.chrome,'--headless=new','--no-sandbox','--disable-dev-shm-usage','--remote-debugging-port=0',f'--user-data-dir={profile}','--window-size=1500,1000','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream',f'--use-file-for-fake-video-capture={a.camera.resolve()}']
        if not a.hardware_webgl:args+=['--use-angle=swiftshader','--enable-unsafe-swiftshader']
        if a.url.startswith('http:'):args+=[f'--unsafely-treat-insecure-origin-as-secure={a.url}']
        browser=subprocess.Popen(args+['about:blank'],stdout=log,stderr=log)
        c=None
        try:
            portfile=Path(profile)/'DevToolsActivePort';end=time.monotonic()+20
            while not portfile.exists():
                if browser.poll() is not None or time.monotonic()>end:raise RuntimeError('Chrome startup failed')
                time.sleep(.1)
            port=int(portfile.read_text().splitlines()[0])
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list') as r:targets=json.load(r)
            c=CDP(next(t['webSocketDebuggerUrl'] for t in targets if t['type']=='page'))
            for method in ['Page.enable','Runtime.enable','Network.enable']:c.call(method)
            c.call('Page.navigate',dict(url=a.url));c.wait('window.sam3dQA?.state.ready')
            report['secure_context']=c.evaluate('isSecureContext');assert report['secure_context']
            c.evaluate("document.querySelector('#input-mode').value='live';document.querySelector('#input-mode').dispatchEvent(new Event('change'))")
            c.wait("window.sam3dQA.tracking.mode==='live'")
            c.evaluate("document.querySelector('#camera-open').click()")
            c.wait("!document.querySelector('#track-start').disabled",30)
            if a.box:
                c.evaluate('(()=>{const b='+json.dumps(a.box)+';["x0","y0","x1","y1"].forEach((id,i)=>document.getElementById(id).value=b[i])})()')
            report['dimensions']=c.evaluate('({width:document.querySelector("#photo").width,height:document.querySelector("#photo").height})')
            c.evaluate(f"document.querySelector('#track-hz').value={a.hz};document.querySelector('#follow-box').checked=false;document.querySelector('#track-start').click()")
            c.wait('window.sam3dQA.tracking.frames>='+str(a.frames+a.warmup),120)
            # Wait for the final measured result's short blend/render, not just
            # the HTTP response. No delay is introduced into the application.
            c.wait('window.sam3dQA.tracking.timings['+str(a.frames+a.warmup-1)+'].settled_render_ms!==undefined',10)
            diagnostic=c.evaluate('window.sam3dQA.tracking');report['raw']=diagnostic
            report['browser_errors']=c.evaluate('window.sam3dQA.state.errors');assert not report['browser_errors'] and not diagnostic['errors']
            report['finite_geometry']=c.evaluate("['vertices','joints','camera_translation'].every(k=>window.sam3dQA.state.result.tensors[k].every(x=>Number.isFinite(x)&&Math.abs(x)<=100))")
            assert report['finite_geometry']
            frames=diagnostic['timings'][a.warmup:a.warmup+a.frames]
            report['client_ms']={key:summary([f[key] for f in frames]) for key in ['capture_ms','jpeg_ms','headers_ms','download_ms','unpack_ms','request_ms','total_ms','interval_ms']}
            for key in ['queue_ms','source_age_ms','worker_encode_ms','worker_draw_ms','worker_blob_ms','first_render_ms','response_to_render_ms','settled_render_ms']:
                if all(key in f for f in frames):report['client_ms'][key]=summary([f[key] for f in frames])
            if 'maxPrepared' in diagnostic:
                assert diagnostic['maxPrepared']<=1 and diagnostic['maxInFlight']==1
                assert all(b['capture_time']-a['capture_time']>=1/report['cap_hz']-.001 for a,b in zip(frames,frames[1:]))
                report['pipeline']={k:diagnostic[k] for k in ['encoder','maxPrepared','dropped','maxInFlight']}
                report['pipeline']['frame_sources']=sorted(set(f.get('frame_source','unrecorded') for f in frames))
            if 'blend_ms' in diagnostic:
                report['presentation']={'blend_ms':diagnostic['blend_ms'],'scope':'capture to Three.js render submission; not camera exposure or GPU/compositor/physical scanout'}
                assert all(f['total_ms']<=f['first_render_ms']<=f['settled_render_ms'] for f in frames)
            keys=set.intersection(*(set(f['server']) for f in frames))
            report['server_ms']={key:summary([f['server'][key] for f in frames]) for key in sorted(keys)}
            assert 'infer' in keys and 'server' in keys,'native stage instrumentation not active'
            report['warm_hz']=1000/statistics.mean(f['interval_ms'] for f in frames)
            report['upload_bytes']=summary([f['upload_bytes'] for f in frames])
            report['render_update_ms']=summary(diagnostic['renderTimings'])
            report['webgl']=c.evaluate("(()=>{const g=document.querySelector('#viewer canvas').getContext('webgl2'),e=g.getExtension('WEBGL_debug_renderer_info');return e?g.getParameter(e.UNMASKED_RENDERER_WEBGL):g.getParameter(g.RENDERER)})()")
            c.screenshot(a.output/'live.png')
            c.evaluate("document.querySelector('#track-stop').click()");c.wait('!window.sam3dQA.tracking.running')
            # Same frozen pixels, quality and browser; isolate API scheduling from
            # encoder work. This is diagnostic only, not an alternate live path.
            jpeg=c.evaluate("""(async()=>{
              const source=document.querySelector('#photo');
              const canvas=document.createElement('canvas');
              canvas.width=source.width;canvas.height=source.height;
              canvas.getContext('2d').drawImage(source,0,0);
              const off=new OffscreenCanvas(canvas.width,canvas.height);
              off.getContext('2d').drawImage(canvas,0,0);
              const result={to_blob:[],to_data_url:[],offscreen_blob:[]};
              for(let i=0;i<15;i++){
                let t=performance.now();
                const b=await new Promise(r=>canvas.toBlob(r,'image/jpeg',.94));
                if(!b?.size)throw Error('Empty JPEG');
                result.to_blob.push(performance.now()-t);
                t=performance.now();const u=canvas.toDataURL('image/jpeg',.94);
                if(!u.startsWith('data:image/jpeg;'))throw Error('Invalid JPEG');
                result.to_data_url.push(performance.now()-t);
                t=performance.now();const o=await off.convertToBlob({type:'image/jpeg',quality:.94});
                if(!o.size)throw Error('Empty offscreen JPEG');
                result.offscreen_blob.push(performance.now()-t);
              }
              return result;
            })()""")
            report['frozen_jpeg_ms']={key:summary(value[3:]) for key,value in jpeg.items()}
            if a.qa_pipeline:
                from qa_live_pipeline import verify
                report['pipeline_qa']=verify(c,a.box)
            report['passed']=True
        finally:
            if c:
                try:c.evaluate("document.querySelector('#track-stop').click()")
                except Exception:pass
            (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            browser.terminate()
            try:browser.wait(timeout=10)
            except subprocess.TimeoutExpired:browser.kill();browser.wait()
    print(json.dumps({k:v for k,v in report.items() if k!='raw'},indent=2))

if __name__=='__main__':main()
