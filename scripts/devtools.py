# Trellis2cpp adaptation: Copyright (c) 2026 rms80; MIT.
# See LICENSES/Trellis2cpp-MIT.txt and THIRD_PARTY_NOTICES.md.
"""Minimal local Chrome DevTools client. Adapted from trellis2cpp's reviewed
scripts/headless_smoke.py, with bounded frames, continuation and ping handling.
Only connects to the disposable browser's loopback DevTools socket.
"""
import base64
import hashlib
import json
import secrets
import socket
import struct
import time
from urllib.parse import urlparse

class CDP:
    def __init__(self,url):
        p=urlparse(url)
        if p.scheme!='ws' or p.hostname not in ['127.0.0.1','localhost','::1']:raise ValueError('local browser socket required')
        self.sock=socket.create_connection((p.hostname,p.port),timeout=15)
        key=base64.b64encode(secrets.token_bytes(16)).decode()
        self.sock.sendall((f'GET {p.path} HTTP/1.1\r\nHost: {p.hostname}:{p.port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n').encode())
        response=b''
        while not response.endswith(b'\r\n\r\n'):
            response+=self.exact(1)
            if len(response)>16384:raise ValueError('oversized handshake')
        accept=base64.b64encode(hashlib.sha1((key+'258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest())
        if b' 101 ' not in response.split(b'\r\n')[0] or accept not in response:raise ValueError('invalid WebSocket upgrade')
        self.next_id=1;self.events=[]
    def exact(self,n):
        out=bytearray()
        while len(out)<n:
            data=self.sock.recv(n-len(out))
            if not data:raise EOFError('browser socket closed')
            out.extend(data)
        return bytes(out)
    def send(self,data,opcode=1):
        mask=secrets.token_bytes(4);n=len(data);header=bytes([0x80|opcode])
        header+=bytes([0x80|n]) if n<126 else bytes([0x80|126])+struct.pack('!H',n) if n<65536 else bytes([0x80|127])+struct.pack('!Q',n)
        self.sock.sendall(header+mask+bytes(v^mask[i%4] for i,v in enumerate(data)))
    def receive(self):
        data=bytearray()
        while True:
            b0,b1=self.exact(2);opcode=b0&15;n=b1&127
            if n==126:n=struct.unpack('!H',self.exact(2))[0]
            elif n==127:n=struct.unpack('!Q',self.exact(8))[0]
            if n+len(data)>32<<20 or b1&128:raise ValueError('invalid server frame')
            payload=self.exact(n)
            if opcode==8:raise EOFError('browser closed')
            if opcode==9:self.send(payload,10);continue
            if opcode==10:continue
            if opcode not in [0,1]:raise ValueError('unexpected browser frame')
            data.extend(payload)
            if b0&128:return json.loads(data)
    def call(self,method,params=None,timeout=30):
        ident=self.next_id;self.next_id+=1
        self.send(json.dumps(dict(id=ident,method=method,params=params or {})).encode())
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            self.sock.settimeout(max(.1,deadline-time.monotonic()));message=self.receive()
            if message.get('id')==ident:
                if 'error' in message:raise RuntimeError(message['error'])
                return message.get('result',{})
            self.events.append(message)
        raise TimeoutError(method)
    def evaluate(self,expression,timeout=30):
        r=self.call('Runtime.evaluate',dict(expression=expression,awaitPromise=True,returnByValue=True,userGesture=True),timeout)
        if 'exceptionDetails' in r:raise RuntimeError(r['exceptionDetails'])
        return r.get('result',{}).get('value')
    def wait(self,expression,timeout=30):
        end=time.monotonic()+timeout;last=None
        while time.monotonic()<end:
            last=self.evaluate(expression,timeout=min(5,timeout))
            if last:return last
            time.sleep(.1)
        raise TimeoutError(f'{expression}: {last}')
    def screenshot(self,path):
        r=self.call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))
        path.write_bytes(base64.b64decode(r['data']))
    def file(self,selector,path):
        root=self.call('DOM.getDocument')['root']['nodeId']
        node=self.call('DOM.querySelector',dict(nodeId=root,selector=selector))['nodeId']
        self.call('DOM.setFileInputFiles',dict(nodeId=node,files=[str(path.resolve())]))
