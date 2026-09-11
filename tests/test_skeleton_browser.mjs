// Optional browser integration test, driven by TestSkeletonBrowser in demo.
import {spawn} from 'node:child_process';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
const profile=await mkdtemp(join(tmpdir(),'sam3d-skeleton-browser-'));
const chrome=spawn(process.argv[3],['--headless','--no-sandbox','--enable-unsafe-swiftshader','--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{stdio:['ignore','ignore','pipe']});
let ws;const pending=new Map();let seq=0;const errors=[];
try{
 const endpoint=await new Promise((resolve,reject)=>{let log='';const timer=setTimeout(()=>reject(Error('browser startup timed out')),15000);chrome.stderr.on('data',v=>{log+=v;const m=log.match(/DevTools listening on (ws:\/\/[^\s]+)/);if(m){clearTimeout(timer);resolve(m[1])}});chrome.on('exit',code=>{clearTimeout(timer);reject(Error(`browser exited ${code}: ${log}`))})});
 ws=new WebSocket(endpoint);await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});
 ws.onmessage=({data})=>{const m=JSON.parse(data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result)}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.exception?.description||m.params.exceptionDetails.text)};
 const call=(method,params={},sessionId)=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params,sessionId}))});
 const {targetId}=await call('Target.createTarget',{url:'about:blank'});const {sessionId}=await call('Target.attachToTarget',{targetId,flatten:true});
 await call('Runtime.enable',{},sessionId);await call('Page.navigate',{url:process.argv[2]},sessionId);
 const result=await call('Runtime.evaluate',{awaitPromise:true,returnByValue:true,expression:`(async()=>{
 const sleep=ms=>new Promise(r=>setTimeout(r,ms));const wait=async(test,label)=>{for(let i=0;i<250;i++){if(test())return;await sleep(50)}throw Error(label+' timed out: '+document.body.innerText)};
 const $=id=>document.getElementById(id);await wait(()=>window.sam3dQA&&$('model')&&!$('model').textContent.includes('Connecting'),'app load');
 $('input-mode').value='live';$('input-mode').dispatchEvent(new Event('change'));await sleep(100);$('camera-open').click();await wait(()=>!$('track-start').disabled,'camera');
 $('track-start').click();await wait(()=>!$('track-record').hidden,'live session');$('track-record').click();await wait(()=>$('record-status').textContent.includes('Recording'),'record start');
 await wait(()=>/ · [2-9][0-9]* poses/.test($('record-status').textContent),'captured poses');$('track-record').click();await wait(()=>$('record-status').textContent.includes('Saved'),'record stop');
 if($('track-stop').hidden)throw Error('stopping recording stopped tracking');
 await wait(()=>$('track-history').querySelector('.history-item'),'saved take');$('track-history').querySelector('.history-item').click();
 await wait(()=>!$('track-playback').hidden&&window.sam3dQA.rendered,'take playback');
 if($('track-export').hidden)throw Error('export hidden');const response=await fetch($('track-export').href);if(!response.ok)throw Error(await response.text());const buffer=await response.arrayBuffer();const view=new DataView(buffer);if(view.getUint32(0,true)!==0x46546c67)throw Error('invalid GLB');const doc=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,20,view.getUint32(12,true))));if(doc.nodes.length!==128||doc.animations.length!==1)throw Error('incomplete animation');
 $('track-play').click();await sleep(200);$('track-play').click();
 return {nodes:doc.nodes.length,channels:doc.animations[0].channels.length,bytes:buffer.byteLength,viewer:window.sam3dQA.rendered,errors:window.sam3dQA.tracking.errors};
})()`},sessionId);
 if(result.exceptionDetails)throw Error(result.exceptionDetails.exception?.description||result.exceptionDetails.text);
 if(errors.length||result.result.value.errors.length)throw Error(JSON.stringify({errors,result:result.result.value}));console.log(JSON.stringify(result.result.value));
}finally{ws?.close();chrome.kill('SIGTERM');await new Promise(resolve=>{if(chrome.exitCode!==null)resolve();else chrome.once('exit',resolve)});await rm(profile,{recursive:true,force:true,maxRetries:10,retryDelay:100})}
