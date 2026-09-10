#!/usr/bin/env python3
"""Render accepted saved original/native BF16 outputs in real headless Chrome.

Read-only captures, exact captured RGB and original model topology. No inference
or running-demo mutation; this is visual evidence, NOT demo upload/E2E QA.
Launch inside run_bounded.py. The temporary HTTP/CDP servers bind loopback only.
"""
import argparse,json,struct,subprocess,tempfile,threading,time,urllib.request
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import numpy as np
from safetensors import safe_open
from check_bf16_body import check
from check_body_api import read_output
from check_parity import read_rules,sha256_file
from prepare_demo_reference import png_rgb
from devtools import CDP

def image_source(original,image_record):
    # Older benchmarks did not store decoded RGB. A new original capture may
    # supply ONLY that artifact, with identical neural/pixel-source identities.
    def source(m):return {k:v for k,v in m['source'].items() if k not in {'script_sha256','reference_weight_residency'}}
    if source(original)!=source(image_record) or any(original[k]!=image_record[k] for k in ['torch','cuda','trained_state','device','precision','attention']):raise ValueError('image capture provenance differs')
    if image_record['status']!='complete' or 'image_input_sha256' not in image_record:raise ValueError('missing completed original RGB capture')
    return image_record['image_input_sha256']

def prepare(profile,index,reference,policy,safe_state,image_reference=None):
    run=json.loads((profile/'report.json').read_text());entry=run['worker_results'][index]
    if not run.get('complete_requests') or run['returncode']!=0 or '--bf16' not in run['command']:raise ValueError('requires completed native BF16 profile')
    native=Path(entry['output']);image=Path(entry['input'])
    if sha256_file(native)!=entry['sha256']:raise ValueError('native result identity mismatch')
    original=reference/'final-output.safetensors';m=json.loads((reference/'benchmark.json').read_text())
    image_manifest=(image_reference or reference)/'benchmark.json';im=json.loads(image_manifest.read_text())
    if m['status']!='complete' or m['precision']!='bf16' or m['attention']!='auto' or m['device']!='cuda' or sha256_file(image)!=image_source(m,im):raise ValueError('original/native scope or input mismatch')
    result=check(native,original,read_rules(policy))
    if not result['passed']:raise ValueError('final numerical policy did not pass')
    if sha256_file(safe_state)!='4c6b3f63ce8a050f6587cf833a036bad3f68377d86cfe69d591502c1373ba0a3':raise ValueError('unverified original topology')
    values=read_output(native)
    with safe_open(safe_state,framework='np') as f:faces=f.get_tensor('head_pose.faces')
    if not np.array_equal(faces,values['faces']):raise ValueError('native topology differs from original')
    if image.stat().st_size>52+16000000*3:raise ValueError('oversized captured image')
    packed=image.read_bytes()
    if packed[:8]!=b'S3DIMG01':raise ValueError('invalid captured image')
    w,h,stride=struct.unpack_from('<3I',packed,8)
    if not 1<=w<=32766 or not 1<=h<=32766 or w*h>16000000 or stride!=w*3 or len(packed)!=52+stride*h:raise ValueError('invalid captured image extent')
    if list(struct.unpack_from('<4f',packed,20))!=m['source']['bbox_xyxy'] or list(struct.unpack_from('<4f',packed,36))!=m['source']['intrinsics_fx_fy_cx_cy']:raise ValueError('captured camera/box differs')
    with safe_open(original,framework='np') as f:
        vertices=f.get_tensor('pred_vertices');pixels=f.get_tensor('pred_keypoints_2d_verts')
    distance=np.linalg.norm(vertices.astype(np.float64)-values['vertices'].astype(np.float64),axis=-1)*1000
    scene=dict(faces=faces.reshape(-1).tolist(),mean_mm=float(distance.mean()),max_mm=float(distance.max()),
        original=dict(vertices=vertices.reshape(-1).tolist(),pixels=pixels.reshape(-1).tolist()),
        native=dict(vertices=values['vertices'].reshape(-1).tolist(),pixels=values['vertices_pixels'].reshape(-1).tolist()))
    provenance=dict(native_sha256=sha256_file(native),original_sha256=sha256_file(original),policy_sha256=sha256_file(policy),
        input_sha256=sha256_file(image),topology_state_sha256=sha256_file(safe_state),native_profile_sha256=sha256_file(profile/'report.json'),
        original_benchmark_sha256=sha256_file(reference/'benchmark.json'),image_benchmark_sha256=sha256_file(image_manifest),numerical_checks=result)
    return scene,png_rgb(w,h,packed[52:]),provenance

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-profile',type=Path,required=True);p.add_argument('--request',type=int,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--safe-state',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--image-reference',type=Path,help='explicit newer original RGB capture for old benchmarks without image artifacts; never replaces reference geometry')
    p.add_argument('--name',required=True);p.add_argument('--chrome',default='chromium')
    a=p.parse_args()
    if not 0<=a.request<64:p.error('request must be 0–63')
    scene,png,provenance=prepare(a.native_profile,a.request,a.reference,a.policy,a.safe_state,a.image_reference);scene['name']=a.name
    a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    routes={'/':('text/html',(root/'scripts/body_visual_compare.html').read_bytes()),'/case.json':('application/json',json.dumps(scene,allow_nan=False).encode()),'/input.png':('image/png',png)}
    for name in ['three.module.min.js','three.core.min.js']:routes['/'+name]=('text/javascript',(root/'demo/web/vendor'/name).read_bytes())
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path=='/favicon.ico':self.send_response(204);self.end_headers();return
            if self.path not in routes:self.send_error(404);return
            mime,data=routes[self.path];self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    report=dict(scope=__doc__,passed=False,provenance=provenance,screenshots=[],views={},script_sha256=sha256_file(Path(__file__)),page_sha256=sha256_file(root/'scripts/body_visual_compare.html'))
    try:
        with tempfile.TemporaryDirectory(prefix='sam3d-visual-') as profile,(a.output/'chrome.log').open('x') as log:
            browser=subprocess.Popen([a.chrome,'--headless=new','--no-sandbox','--disable-dev-shm-usage','--use-angle=swiftshader','--enable-unsafe-swiftshader','--remote-debugging-address=127.0.0.1','--remote-debugging-port=0',f'--user-data-dir={profile}','about:blank'],stdout=log,stderr=log)
            try:
                portfile=Path(profile)/'DevToolsActivePort';deadline=time.monotonic()+20
                while not portfile.exists():
                    if browser.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Chrome startup failed')
                    time.sleep(.1)
                port=int(portfile.read_text().splitlines()[0])
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list') as r:targets=json.load(r)
                c=CDP(next(t['webSocketDebuggerUrl'] for t in targets if t['type']=='page'))
                for method in ['Page.enable','Runtime.enable','Log.enable','Network.enable']:c.call(method)
                report['browser']=c.call('Browser.getVersion');c.call('Emulation.setDeviceMetricsOverride',dict(width=1500,height=880,deviceScaleFactor=1,mobile=False))
                c.call('Page.navigate',dict(url=f'http://127.0.0.1:{server.server_port}/'));c.wait('window.visualQA?.ready',60)
                for view in ['front','side','oblique']:
                    frames=c.evaluate('window.visualQA.setView('+json.dumps(view)+')')
                    if len(frames)!=3 or any(f['gl_error'] or f['calls']<1 or f['occupied']<1000 for f in frames):raise ValueError('blank/failed mesh render')
                    report['views'][view]=dict(frames=frames,camera=c.evaluate('window.visualQA.camera'))
                    name=view+'.png';c.screenshot(a.output/name);report['screenshots'].append(dict(name=name,sha256=sha256_file(a.output/name)))
                report['errors']=c.evaluate('window.visualQA.errors')
                report['browser_errors']=[e for e in c.events if e['method']=='Runtime.exceptionThrown' or (e['method']=='Log.entryAdded' and e['params']['entry']['level']=='error') or (e['method']=='Network.responseReceived' and e['params']['response']['status']>=400)]
                if report['errors'] or report['browser_errors']:raise ValueError('browser errors')
                report['passed']=True
            finally:
                browser.terminate()
                try:browser.wait(timeout=10)
                except subprocess.TimeoutExpired:browser.kill();browser.wait()
    finally:
        server.shutdown();server.server_close();thread.join(timeout=5)
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],screenshots=report['screenshots'])))
if __name__=='__main__':main()
