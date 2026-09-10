#!/usr/bin/env python3
"""Real Chrome file-video + virtual-webcam QA against the actual native worker.
The virtual camera supplies decoded video pixels, never mocked inference output.
An insecure-origin camera override is TEST ONLY; users need HTTPS/localhost.
"""
import argparse
import json
from pathlib import Path
import subprocess
import struct
import tempfile
import time
import urllib.request
from devtools import CDP

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',required=True);p.add_argument('--video',type=Path,required=True)
    p.add_argument('--camera',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--live-hz',type=int,default=5)
    p.add_argument('--chrome',default='chromium');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    report={'passed':False,'scope':'real native inference on uploaded video and Chromium virtual webcam; not physical camera hardware'}
    with tempfile.TemporaryDirectory(prefix='sam3d-video-chrome-') as profile,(a.output/'chrome.log').open('w') as log:
        browser=subprocess.Popen([a.chrome,'--headless=new','--no-sandbox','--disable-dev-shm-usage','--use-angle=swiftshader','--enable-unsafe-swiftshader','--remote-debugging-port=0',f'--user-data-dir={profile}','--window-size=1500,1000','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream',f'--use-file-for-fake-video-capture={a.camera.resolve()}',f'--unsafely-treat-insecure-origin-as-secure={a.url}','about:blank'],stdout=log,stderr=log)
        c=None
        try:
            portfile=Path(profile)/'DevToolsActivePort';end=time.monotonic()+20
            while not portfile.exists():
                if browser.poll() is not None or time.monotonic()>end:raise RuntimeError('Chrome startup failed')
                time.sleep(.1)
            port=int(portfile.read_text().splitlines()[0])
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list') as r:targets=json.load(r)
            c=CDP(next(t['webSocketDebuggerUrl'] for t in targets if t['type']=='page'))
            for m in ['Page.enable','Runtime.enable','Network.enable']:c.call(m)
            c.call('Page.navigate',dict(url=a.url));c.wait('window.sam3dQA?.state.ready')
            report['browser']=c.call('Browser.getVersion')
            c.evaluate("document.querySelector('#input-mode').value='offline';document.querySelector('#input-mode').dispatchEvent(new Event('change'))")
            c.wait("window.sam3dQA.tracking.mode==='offline'")
            c.file('#video-upload',a.video);c.wait("!document.querySelector('#track-start').disabled")
            c.evaluate("document.querySelector('#track-hz').value=2;document.querySelector('#track-duration').value=2;document.querySelector('#follow-box').checked=false;document.querySelector('#track-start').click()")
            c.wait('window.sam3dQA.tracking.running');c.wait("!window.sam3dQA.tracking.running && !document.querySelector('#track-playback').hidden",120)
            report['offline']=c.evaluate("({diagnostics:window.sam3dQA.tracking,status:document.querySelector('#track-status').textContent})")
            assert report['offline']['diagnostics']['frames']==4,report['offline']
            assert c.evaluate('window.sam3dQA.rendered && window.sam3dQA.state.result.schema==="sam3d.body.track.v1"')
            report['tracks']=c.evaluate("fetch('/api/tracks').then(r=>r.json())")
            c.screenshot(a.output/'01-offline.png')
            before=c.evaluate('window.sam3dQA.tracking.interpolations')
            c.evaluate("document.querySelector('#track-play').click()")
            c.wait('window.sam3dQA.tracking.interpolations>'+str(before+3))
            c.evaluate("document.querySelector('#track-seek').value=.75;document.querySelector('#track-seek').dispatchEvent(new Event('input'))")
            c.wait("document.querySelector('#track-time').textContent==='0.75 s'")
            report['offline_play_scrub']=True
            c.evaluate("document.querySelector('#track-history button').click()")
            c.wait("document.querySelector('#input-name').textContent.includes('saved first frame')")
            c.wait('window.sam3dQA.rendered');report['history_reload']=True
            # Exercise cancellation during the first frame, then start again.
            c.evaluate("document.querySelector('#input-mode').value='live';document.querySelector('#input-mode').dispatchEvent(new Event('change'))")
            c.wait("window.sam3dQA.tracking.mode==='live'")
            c.evaluate("document.querySelector('#camera-open').click()")
            c.wait("!document.querySelector('#track-start').disabled",30)
            c.evaluate(f"document.querySelector('#track-hz').value={a.live_hz};document.querySelector('#track-start').click()")
            c.wait('window.sam3dQA.tracking.running');c.evaluate("document.querySelector('#track-stop').click()")
            c.wait('!window.sam3dQA.tracking.running');report['live_cancel']=True
            c.evaluate("document.querySelector('#camera-open').click()")
            c.wait("!document.querySelector('#track-start').disabled",30)
            c.evaluate("document.querySelector('#track-start').click()")
            c.wait('window.sam3dQA.tracking.frames>=8',120)
            report['live']=c.evaluate('window.sam3dQA.tracking')
            assert report['live']['actualHz']<=a.live_hz+.1,report['live']
            assert report['live']['maxInFlight']==1 and report['live']['interpolations']>report['live']['frames']
            report['finite']=c.evaluate("['vertices','joints','camera_translation'].every(k=>window.sam3dQA.state.result.tensors[k].every(x=>Number.isFinite(x)&&Math.abs(x)<=100))")
            assert report['finite'];c.screenshot(a.output/'02-live.png')
            c.evaluate("document.querySelector('#track-stop').click()");c.wait('!window.sam3dQA.tracking.running')
            report['stopped']=c.evaluate("({disabled:document.querySelector('#track-start').disabled,streams:document.querySelector('#track-status').textContent})")
            assert report['stopped']['disabled'],report['stopped']
            # Independently run the first exact decoded JPEG/settings through
            # the existing photo-job path; compare final native geometry bytes,
            # not screenshots or merely intermediate/model tensors.
            track=report['tracks'][0]
            expression='''(async()=>{const t=TRACK;const image=await (await fetch(`/tracks/${t.id}/preview.jpg`)).blob();const f=new FormData();f.append('image',image,'video-frame-parity.jpg');f.append('settings',JSON.stringify(t.samples[0].settings));const r=await fetch('/api/jobs',{method:'POST',body:f});if(!r.ok)throw Error(await r.text());return r.json()})()'''.replace('TRACK',json.dumps(track))
            photo=c.evaluate(expression);ident=photo['id']
            c.wait(f'fetch("/api/jobs/{ident}").then(r=>r.json()).then(j=>["complete","failed"].includes(j.state))',60)
            photo=c.evaluate(f'fetch("/api/jobs/{ident}").then(r=>r.json())');assert photo['state']=='complete',photo
            # Go correctly emits IEEE negative zero as JSON -0. Python's
            # default integer parser loses its sign; preserve it for byte QA.
            with urllib.request.urlopen(a.url+f'/files/{ident}/result.json') as r:result=json.load(r,parse_int=lambda s:-0.0 if s=='-0' else int(s))
            with urllib.request.urlopen(a.url+f'/tracks/{track["id"]}/000000.bin') as r:frame=r.read()
            offset=16
            for key in ['vertices','joints','camera_translation']:
                values=result['tensors'][key];raw=struct.pack('<'+'f'*len(values),*values)
                assert raw==frame[offset:offset+len(raw)],'frame/photo final output mismatch: '+key
                offset+=len(raw)
            with urllib.request.urlopen(a.url+f'/tracks/{track["id"]}/faces.bin') as r:faces=r.read()
            assert faces==struct.pack('<'+'I'*len(result['faces']),*result['faces'])
            report['final_geometry_exact_photo_path']={'photo_job':ident,'track':track['id'],'fields':['vertices','joints','camera_translation','faces']}
            c.call('Emulation.setDeviceMetricsOverride',dict(width=390,height=844,deviceScaleFactor=1,mobile=True))
            c.screenshot(a.output/'03-mobile.png')
            report['page_errors']=c.evaluate('window.sam3dQA.state.errors');report['tracking_errors']=c.evaluate('window.sam3dQA.tracking.errors')
            assert not report['page_errors'] and not report['tracking_errors'],report
            exceptions=[e for e in c.events if e.get('method')=='Runtime.exceptionThrown'];assert not exceptions,exceptions
            report['passed']=True
        finally:
            if c:
                try:report['last_status']=c.evaluate("({status:document.querySelector('#track-status').textContent,qa:window.sam3dQA.tracking,errors:window.sam3dQA.state.errors})")
                except Exception as e:report['diagnostic_error']=str(e)
            (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');browser.terminate()
            try:browser.wait(timeout=10)
            except subprocess.TimeoutExpired:browser.kill();browser.wait()
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
