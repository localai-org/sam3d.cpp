#!/usr/bin/env python3
"""Real image upload -> native body inference -> browser render/export/history QA.
Requires a running demo with a prepared original reference. No mocked inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request
from devtools import CDP

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8097')
    p.add_argument('--chrome',default='chromium')
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse',help='inspect an already-complete real job instead of another inference')
    p.add_argument('--warm-repeat',action='store_true',help='generate again through the UI to test resident-session reuse')
    p.add_argument('--expect-precision',choices=['f32','bf16'],help='verify actual server/job precision, including the UI label')
    p.add_argument('--expected-native',type=Path,help='require the actual exported result.bin to match this accepted native capture byte-for-byte')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.reference/'manifest.json').read_text())
    report={'scope':'real body pose-branch UI QA; software WebGL presentation, native backend recorded by job','screenshots':[],'passed':False}
    with tempfile.TemporaryDirectory(prefix='sam3d-chrome-') as profile, (a.output/'chrome.log').open('w') as log:
        browser=subprocess.Popen([a.chrome,'--headless=new','--no-sandbox','--disable-dev-shm-usage','--use-angle=swiftshader','--enable-unsafe-swiftshader','--remote-debugging-port=0','--remote-debugging-address=127.0.0.1',f'--user-data-dir={profile}','--window-size=1500,1000','about:blank'],stdout=log,stderr=log)
        try:
            portfile=Path(profile)/'DevToolsActivePort';end=time.monotonic()+20
            while not portfile.exists():
                if browser.poll() is not None or time.monotonic()>end:raise RuntimeError('Chrome startup failed; see chrome.log')
                time.sleep(.1)
            port=int(portfile.read_text().splitlines()[0])
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list') as r:targets=json.load(r)
            c=CDP(next(t['webSocketDebuggerUrl'] for t in targets if t['type']=='page'))
            for method in ['Page.enable','Runtime.enable','Log.enable','Network.enable']:c.call(method)
            report['browser']=c.call('Browser.getVersion')
            c.call('Emulation.setDeviceMetricsOverride',dict(width=1500,height=1000,deviceScaleFactor=1,mobile=False))
            c.call('Page.navigate',dict(url=a.url+'/'));c.wait('window.sam3dQA?.state.ready')
            if a.expect_precision:
                cfg=c.evaluate('fetch("/api/config").then(r=>r.json())')
                assert cfg['precision']==a.expect_precision,cfg
                assert a.expect_precision.upper() in c.evaluate('document.querySelector("#model").textContent')
                report['server_precision']=cfg
            def shot(name):
                c.evaluate('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))');c.screenshot(a.output/name);report['screenshots'].append(name)
            shot('01-initial.png')
            if not a.reuse:
                # Client rejection must detach any previous result and disable
                # generation. These are real file-input events, not fake API jobs.
                invalid=Path(profile)/'invalid.png';invalid.write_bytes(b'not a PNG')
                c.file('#upload',invalid)
                c.wait('document.querySelector("#status").classList.contains("error")')
                assert c.evaluate('document.querySelector("#generate").disabled && !window.sam3dQA.state.result')
                # The malformed upload intentionally gets HTTP 422 from the
                # FFmpeg preparation endpoint. Retain its expected diagnostics.
                report['expected_invalid_upload_events']=[e for e in c.events if e.get('method') in ['Network.responseReceived','Log.entryAdded'] and '/api/prepare' in json.dumps(e)]
                c.events=[e for e in c.events if e not in report['expected_invalid_upload_events']]
                c.evaluate('(()=>{const dt=new DataTransfer();dt.items.add(new File([new Uint8Array(128*1024*1024+1)],"too-large.png",{type:"image/png"}));const el=document.querySelector("#upload");el.files=dt.files;el.dispatchEvent(new Event("change"));return true})()')
                c.wait('document.querySelector("#status").textContent.includes("128 MiB")')
                report['invalid_and_oversized_upload_rejected']=True
                # Exercise actual file chooser, not example-button-only/injected tensors.
                c.file('#upload',a.reference/'input.png');c.wait('window.sam3dQA.state.file && !document.querySelector("#generate").disabled')
                rect=c.evaluate('(()=>{const r=document.querySelector("#photo").getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}})()')
                x,y=rect['x']+rect['w']*.2,rect['y']+rect['h']*.1
                c.call('Input.dispatchMouseEvent',dict(type='mousePressed',x=x,y=y,button='left',clickCount=1))
                c.call('Input.dispatchMouseEvent',dict(type='mouseMoved',x=x+rect['w']*.5,y=y+rect['h']*.8,button='left',buttons=1))
                c.call('Input.dispatchMouseEvent',dict(type='mouseReleased',x=x+rect['w']*.5,y=y+rect['h']*.8,button='left',clickCount=1))
                assert c.evaluate('Number(document.querySelector("#x0").value)>0 && Number(document.querySelector("#y0").value)>0')
                report['person_box_drag']=True
                values={**dict(zip(['x0','y0','x1','y1'],manifest['settings']['box'])),**dict(zip(['fx','fy','cx','cy'],manifest['settings']['camera']))}
                c.evaluate('(()=>{const v='+json.dumps(values)+';for(const [id,x]of Object.entries(v)){const e=document.getElementById(id);e.value=x;e.dispatchEvent(new Event("input",{bubbles:true}))}return true})()')
                shot('02-selected-input.png')
                # Cancel a real worker, then recover through a second Generate.
                c.evaluate('document.querySelector("#generate").click()');c.wait('window.sam3dQA.state.job',30)
                report['cancelled_job']=c.evaluate('window.sam3dQA.state.job.id')
                c.evaluate('document.querySelector("#cancel").click()')
                c.wait('["cancelled","failed","complete"].includes(window.sam3dQA.state.job?.state)',30)
                cancelled=c.evaluate('window.sam3dQA.state.job')
                assert cancelled['state']=='cancelled',cancelled
                assert c.evaluate('!window.sam3dQA.state.result && document.querySelector("#downloads").hidden')
                report['cancellation']=True
                c.evaluate('document.querySelector("#generate").click()')
                c.wait('window.sam3dQA.state.job && window.sam3dQA.state.job.state!=="cancelled"',30)
                report['job_id']=c.evaluate('window.sam3dQA.state.job.id');print('Submitted real UI job '+report['job_id'],flush=True)
                end=time.monotonic()+600;stages=[];last=''
                while True:
                    j=c.evaluate('window.sam3dQA.state.job')
                    if j['stage']!=last:last=j['stage'];stages.append(last);print(last,flush=True)
                    if j['state'] in ['failed','cancelled']:raise RuntimeError(j.get('error',j['state']))
                    if j['state']=='complete':break
                    if time.monotonic()>end:raise TimeoutError('native job')
                    time.sleep(1)
                report['observed_stages']=stages
            else:
                report['job_id']=a.reuse
                c.evaluate('document.querySelector('+json.dumps('button[data-id="'+a.reuse+'"]')+').click()')
            if a.warm_repeat:
                cold=c.evaluate('window.sam3dQA.state.job')
                report['first_job']=cold
                c.wait('!document.querySelector("#generate").disabled',30)
                c.evaluate('document.querySelector("#generate").click()')
                c.wait('window.sam3dQA.state.job?.id!=='+json.dumps(cold['id']),30)
                c.wait('["complete","failed","cancelled"].includes(window.sam3dQA.state.job?.state)',60)
                warm=c.evaluate('window.sam3dQA.state.job')
                assert warm['state']=='complete',warm
                assert warm['result_sha256']==cold['result_sha256'],'resident UI result changed'
                report['warm_repeat']=True;report['job_id']=warm['id']
            c.wait('window.sam3dQA.rendered',30)
            report['job']=c.evaluate('window.sam3dQA.state.job')
            if a.expect_precision:
                assert report['job'].get('precision')==a.expect_precision,report['job']
                assert report['job']['provenance']['precision']==a.expect_precision
                assert a.expect_precision.upper() in c.evaluate('document.querySelector("#elapsed").textContent')
            if a.expected_native:
                expected=hashlib.sha256(a.expected_native.read_bytes()).hexdigest()
                assert report['job']['result_sha256']==expected,'actual UI output differs from accepted full native capture'
                report['accepted_native_sha256']=expected
            assert report['job']['native_input_sha256']==manifest['native_input_sha256'],'browser/native RGB/camera differ from original capture'
            assert c.evaluate('!document.querySelector("#reference").disabled'),'reference unexpectedly disabled'
            shot('03-native-oblique.png')
            before=c.evaluate('window.sam3dQA.camera.position')
            rect=c.evaluate('(()=>{const r=document.querySelector("#viewer").getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()')
            c.call('Input.dispatchMouseEvent',dict(type='mousePressed',**rect,button='left',clickCount=1))
            c.call('Input.dispatchMouseEvent',dict(type='mouseMoved',x=rect['x']+60,y=rect['y']+20,buttons=1))
            c.call('Input.dispatchMouseEvent',dict(type='mouseReleased',x=rect['x']+60,y=rect['y']+20,button='left',clickCount=1))
            c.wait('JSON.stringify(window.sam3dQA.camera.position)!=='+json.dumps(json.dumps(before,separators=(',',':'))))
            report['orbit_control']=True
            c.evaluate('document.querySelector("#front").click()');shot('04-native-front.png')
            c.evaluate('document.querySelector("#reference").click()');c.wait('window.sam3dQA.state.reference');shot('05-original-native-front.png')
            c.evaluate('document.querySelector("#side").click()');shot('06-original-native-side.png')
            report['geometry']=c.evaluate('(()=>{const s=window.sam3dQA;const gl=document.querySelector("#viewer canvas").getContext("webgl2");const ext=gl.getExtension("WEBGL_debug_renderer_info");return {vertices:s.state.result.tensors.vertices.length/3,faces:s.state.result.faces.length/3,joints:s.state.result.tensors.joints.length/3,camera:s.camera,draw_calls:s.drawCalls,gl_error:gl.getError(),renderer:ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER),errors:s.state.errors}})()')
            assert report['geometry']['gl_error']==0 and report['geometry']['draw_calls']>0 and not report['geometry']['errors']
            # Read the actual download, reload its glTF buffers, verify every
            # vertex/index and construct/render it with the bundled Three.js.
            report['export']=c.evaluate('''(async()=>{
              const buf=await (await fetch(document.querySelector('#glb').href)).arrayBuffer(),d=new DataView(buf);
              if(d.getUint32(0,true)!==0x46546c67||d.getUint32(4,true)!==2||d.getUint32(8,true)!==buf.byteLength)throw Error('Invalid GLB');
              const n=d.getUint32(12,true),doc=JSON.parse(new TextDecoder().decode(new Uint8Array(buf,20,n))),off=28+n;
              const pos=new Float32Array(buf,off,doc.accessors[0].count*3),ind=new Uint32Array(buf,off+doc.bufferViews[1].byteOffset,doc.accessors[1].count),b=window.sam3dQA.state.result;
              if(pos.length!==b.tensors.vertices.length||ind.length!==b.faces.length)throw Error('Export extent differs');
              for(let i=0;i<pos.length;i++)if(pos[i]!==Math.fround(b.tensors.vertices[i]*(i%3===0?1:-1)))throw Error('Export vertex differs');
              for(let i=0;i<ind.length;i++)if(ind[i]!==b.faces[i])throw Error('Export topology differs');
              const T=await import('/vendor/three.module.min.js'),g=new T.BufferGeometry();g.setAttribute('position',new T.BufferAttribute(pos,3));g.setIndex(new T.BufferAttribute(ind,1));g.computeVertexNormals();
              const scene=new T.Scene(),m=new T.MeshBasicMaterial({color:0x61b7d4,side:T.DoubleSide});scene.add(new T.Mesh(g,m));const cam=new T.PerspectiveCamera(38,1,.01,200);cam.position.fromArray(window.sam3dQA.camera.position);cam.lookAt(new T.Vector3().fromArray(window.sam3dQA.camera.target));
              const renderer=new T.WebGLRenderer();renderer.setSize(256,256);renderer.render(scene,cam);const calls=renderer.info.render.calls;renderer.dispose();g.dispose();m.dispose();
              return {bytes:buf.byteLength,all_vertices_exact:true,all_indices_exact:true,reloaded_draw_calls:calls};
            })()''')
            assert report['export']['reloaded_draw_calls']>0
            # Settings edits must immediately detach the old reconstruction.
            c.evaluate('(()=>{const e=document.querySelector("#fx");e.value=2000;e.dispatchEvent(new Event("input"));return true})()')
            assert c.evaluate('!window.sam3dQA.state.result && document.querySelector("#downloads").hidden')
            c.evaluate('document.querySelector('+json.dumps('button[data-id="'+report['job_id']+'"]')+').click()');c.wait('window.sam3dQA.rendered')
            # Settings are F32 in the C API. Go's shortest decimal JSON spelling
            # need not equal the binary value when read as a JavaScript F64.
            assert c.evaluate('Math.fround(Number(document.querySelector("#fx").value))')==manifest['settings']['camera'][0]
            c.call('Page.reload');c.wait('window.sam3dQA?.rendered')
            assert c.evaluate('window.sam3dQA.state.job.id')==report['job_id']
            report['history_reload']=True
            report['desktop_scroll']=c.evaluate('(()=>{const e=document.querySelector("aside");e.scrollTop=e.scrollHeight;return e.scrollTop>0})()');assert report['desktop_scroll']
            c.call('Emulation.setDeviceMetricsOverride',dict(width=390,height=844,deviceScaleFactor=1,mobile=True))
            c.evaluate('window.scrollTo(0,0)');shot('07-mobile-viewer.png')
            assert c.evaluate('document.documentElement.scrollWidth<=innerWidth'),'horizontal overflow'
            c.evaluate('document.querySelector("#history").scrollIntoView()');shot('08-mobile-history.png')
            report['mobile_scroll']=c.evaluate('scrollY>0');assert report['mobile_scroll']
            c.evaluate('document.querySelector("#example").click()')
            c.wait('window.sam3dQA.state.file && document.querySelector("#status").textContent.includes("Official example loaded")')
            assert c.evaluate('!window.sam3dQA.state.result && !window.sam3dQA.state.job && !document.querySelector("#generate").disabled')
            report['official_example_button']=True
            c.evaluate('document.querySelector('+json.dumps('button[data-id="'+report['job_id']+'"]')+').click()');c.wait('window.sam3dQA.rendered')
            report['page_errors']=c.evaluate('window.sam3dQA.state.errors')
            errors=[]
            for e in c.events:
                method=e.get('method');p=e.get('params',{})
                if method=='Runtime.exceptionThrown':errors.append(p)
                if method=='Log.entryAdded' and p.get('entry',{}).get('level')=='error':errors.append(p)
                if method=='Network.responseReceived' and p.get('response',{}).get('status',0)>=400:errors.append(p['response'])
            report['browser_errors']=errors
            assert not errors and not report['page_errors'],errors
            report['passed']=True
        finally:
            (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            browser.terminate()
            try:browser.wait(timeout=10)
            except subprocess.TimeoutExpired:browser.kill();browser.wait()
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
