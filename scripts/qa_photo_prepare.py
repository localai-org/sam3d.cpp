#!/usr/bin/env python3
"""Real-browser photo preparation QA. No neural inference needed."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import time
import urllib.request
import zlib
from devtools import CDP
from prepare_demo_reference import png_rgb

def fixtures(root):
    def chunk(name,data):return struct.pack('>I',len(data))+name+data+struct.pack('>I',zlib.crc32(name+data))
    # 17.5 MP, compressible grayscale; file is small, decoded dimensions are not.
    large=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',5000,3500,8,0,0,0,0))+chunk(b'IDAT',zlib.compress((b'\0'+b'\x80'*5000)*3500))+chunk(b'IEND',b'')
    (root/'large.png').write_bytes(large)
    pixels=b'\x20\x80\xe0'*(64*64)
    bmp=b'BM'+struct.pack('<IHHI',54+len(pixels),0,0,54)+struct.pack('<IiiHHIIiiII',40,64,64,1,24,0,len(pixels),2835,2835,0,0)+pixels
    (root/'photo.bmp').write_bytes(bmp)
    small=png_rgb(64,64,pixels)
    (root/'normal.png').write_bytes(small)
    (root/'large-file.png').write_bytes(small+b'\0'*(21*1024*1024-len(small)))
    (root/'invalid.png').write_bytes(b'not an image')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--chrome',default='chromium');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);fixtures(a.output)
    report={'passed':False,'cases':[]}
    with tempfile.TemporaryDirectory(prefix='sam3d-photo-chrome-') as profile,(a.output/'chrome.log').open('w') as log:
        browser=subprocess.Popen([a.chrome,'--headless=new','--no-sandbox','--disable-dev-shm-usage','--use-angle=swiftshader','--enable-unsafe-swiftshader','--remote-debugging-port=0',f'--user-data-dir={profile}','about:blank'],stdout=log,stderr=log)
        try:
            portfile=Path(profile)/'DevToolsActivePort';deadline=time.monotonic()+20
            while not portfile.exists():
                if browser.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Chrome startup failed')
                time.sleep(.1)
            port=int(portfile.read_text().splitlines()[0])
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list') as r:targets=json.load(r)
            c=CDP(next(t['webSocketDebuggerUrl'] for t in targets if t['type']=='page'))
            for m in ['Page.enable','Runtime.enable','Log.enable','Network.enable']:c.call(m)
            c.call('Emulation.setDeviceMetricsOverride',dict(width=1400,height=1000,deviceScaleFactor=1,mobile=False))
            c.call('Page.navigate',dict(url=a.url));c.wait('window.sam3dQA?.state.ready')
            report['browser']=c.call('Browser.getVersion')
            for filename,w,h,converted in [('photo.bmp',64,64,True),('large.png',4000,2800,True),('large-file.png',64,64,True),('normal.png',64,64,False)]:
                c.file('#upload',a.output/filename)
                c.wait('window.sam3dQA.state.name==='+json.dumps(filename)+' && window.sam3dQA.state.file && !window.sam3dQA.state.prepareAbort',60)
                state=c.evaluate('(()=>{const s=window.sam3dQA.state;return {width:s.image.width,height:s.image.height,note:s.preparation,enabled:!document.querySelector("#generate").disabled,has_old_result:!!s.result,notice_hidden:document.querySelector("#preparation").hidden}})()')
                assert (state['width'],state['height'])==(w,h),state
                assert state['enabled'] and not state['has_old_result']
                assert bool(state['note'])==converted and state['notice_hidden']!=converted,state
                if converted:assert 'FFmpeg' in state['note']
                state['file']=filename;report['cases'].append(state);print(state,flush=True)
                c.screenshot(a.output/(filename+'.png'))
            c.file('#upload',a.output/'invalid.png');c.wait('document.querySelector("#status").classList.contains("error")',60)
            failure=c.evaluate('({status:document.querySelector("#status").textContent,note:document.querySelector("#preparation").textContent,disabled:document.querySelector("#generate").disabled,empty:!window.sam3dQA.state.file&&!window.sam3dQA.state.result})')
            assert failure['disabled'] and failure['empty'] and 'FFmpeg' in failure['note'] and 'failed' in failure['note'];report['failure']=failure;c.screenshot(a.output/'failure.png')
            # Recover after a failed conversion; the stale failure notice clears.
            c.file('#upload',a.output/'normal.png');c.wait('window.sam3dQA.state.file && !window.sam3dQA.state.prepareAbort')
            assert c.evaluate('!document.querySelector("#generate").disabled && document.querySelector("#preparation").hidden')
            report['recovery']=True;report['page_errors']=c.evaluate('window.sam3dQA.state.errors');assert not report['page_errors']
            bad=[e for e in c.events if e.get('method')=='Runtime.exceptionThrown'];assert not bad,bad
            responses=[e['params']['response'] for e in c.events if e.get('method')=='Network.responseReceived' and e['params']['response']['status']>=400]
            assert len(responses)==1 and responses[0]['status']==422 and '/api/prepare' in responses[0]['url'],responses
            report['expected_failed_conversion_http_status']=422;report['passed']=True
        finally:
            (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');browser.terminate()
            try:browser.wait(timeout=10)
            except subprocess.TimeoutExpired:browser.kill();browser.wait()
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
