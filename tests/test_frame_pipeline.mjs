import {strict as assert} from 'node:assert';
import {LatestFrameSlot,createFrameEncoder,createCameraFrames} from '../demo/web/frame-pipeline.js';

let q=new LatestFrameSlot();
for(let i=0;i<10000;i++)q.put({id:i});
assert.equal(q.maximum,1);assert.equal(q.dropped,9999);
assert.equal((await q.take()).id,9999);assert.equal(q.frame,null);
let waiting=q.take();await assert.rejects(q.take(),/one frame consumer/);
q.put({id:10000});assert.equal((await waiting).id,10000);
waiting=q.take();q.close();assert.equal(await waiting,null);
q.put({id:10001});assert.equal(await q.take(),null);assert.equal(q.frame,null);
q=new LatestFrameSlot();waiting=q.take();q.close(Error('camera failed'));
await assert.rejects(waiting,/camera failed/);await assert.rejects(q.take(),/camera failed/);
q=new LatestFrameSlot();q.put({id:1});q.close(Error('encoding failed'));
await assert.rejects(q.take(),/encoding failed/);

// Test the real main-thread encoder ownership/protocol without browser APIs.
let worker;
const tick=()=>new Promise(setImmediate);
globalThis.OffscreenCanvas=class {};
globalThis.document={createElement:()=>({getContext:()=>({drawImage(){},getImageData(){return {data:new Uint8ClampedArray(16)}}})})};
globalThis.Worker=class {
 constructor(){worker=this}
 postMessage(data,transfer){this.data=data;assert.equal(transfer[0],data.frame||data.pixels)}
 terminate(){this.terminated=true}
 reply(data){this.onmessage({data})}
};
let encoder=createFrameEncoder();assert.equal(encoder.kind,'worker-offscreen');
let encoding=encoder.encode({},960,540);
await assert.rejects(encoder.encode({},960,540),/already busy/);
await tick();
worker.reply({blob:new Blob(['jpeg'],{type:'image/jpeg'}),encode_ms:4});
let result=await encoding;assert.equal(result.blob.size,4);assert.equal(result.timing.worker_encode_ms,4);
await assert.rejects(encoder.encode({},1000,540),/dimensions/);
encoding=encoder.encode({},960,540);await tick();encoder.close();
await assert.rejects(encoding,/stopped/);assert.ok(worker.terminated);
await assert.rejects(encoder.encode({},960,540),/stopped/);
encoder=createFrameEncoder();encoding=encoder.encode({},960,540);await tick();
worker.onerror({preventDefault(){},message:'worker failed'});
await assert.rejects(encoding,/worker failed/);assert.ok(worker.terminated);
encoder=createFrameEncoder();encoding=encoder.encode({},960,540);await tick();
worker.reply({blob:new Blob([],{type:'image/jpeg'})});await assert.rejects(encoding,/Invalid encoded/);encoder.close();
let framesClosed=0;
globalThis.VideoFrame=class {close(){framesClosed++}};
encoder=createFrameEncoder();encoding=encoder.encode({videoWidth:960},960,540);
assert.ok(worker.data.frame instanceof VideoFrame);assert.equal(worker.data.pixels,null);
worker.reply({blob:new Blob(['jpeg'],{type:'image/jpeg'})});await encoding;encoder.close();
encoder=createFrameEncoder();worker.postMessage=()=>{throw Error('transfer failed')};
await assert.rejects(encoder.encode({videoWidth:960},960,540),/transfer failed/);
assert.equal(framesClosed,1);encoder.close();delete globalThis.VideoFrame;

let callback;
globalThis.document={createElement:()=>({getContext:()=>({drawImage(){}}),toBlob(fn){callback=fn}})};
encoder=createFrameEncoder({forceFallback:true});assert.equal(encoder.kind,'canvas-fallback');
encoding=encoder.encode({},64,64);callback(new Blob(['jpeg'],{type:'image/jpeg'}));assert.equal((await encoding).blob.size,4);
encoding=encoder.encode({},64,64);encoder.close();await assert.rejects(encoding,/stopped/);callback(null);
assert.equal(createCameraFrames({}),null);
globalThis.VideoFrame=class {};
let resolveRead,cloneStopped=0,lockReleased=0;
const owned={stop(){cloneStopped++}},track={clone(){return owned}},closedFrames=[];
const sourceFrame=id=>({clone:()=>sourceFrame(id+'-clone'),close(){closedFrames.push(id)}});
globalThis.MediaStreamTrackProcessor=class {
 constructor(options){assert.equal(options.track,owned);assert.equal(options.maxBufferSize,1)}
 readable={getReader:()=>({read:()=>new Promise(resolve=>resolveRead=resolve),cancel(){resolveRead({done:true});return Promise.resolve()},releaseLock(){lockReleased++}})};
};
let camera=createCameraFrames(track);assert.equal(camera.take(),null);
resolveRead({value:sourceFrame('first')});await tick();
const retained=camera.take();assert.ok(retained.received>=0);
resolveRead({value:sourceFrame('second')});await tick();assert.deepEqual(closedFrames,['first']);
retained.frame.close();camera.close();await camera.done;
assert.deepEqual(closedFrames,['first','first-clone','second']);assert.ok(cloneStopped>0);assert.equal(lockReleased,1);
assert.equal(camera.take(),null);
camera=createCameraFrames(track);resolveRead({value:sourceFrame('late')});camera.close();await camera.done;assert.ok(closedFrames.includes('late'));
camera=createCameraFrames(track);resolveRead({done:true});await camera.done;assert.throws(()=>camera.take(),/source ended/);camera.close();
delete globalThis.MediaStreamTrackProcessor;delete globalThis.VideoFrame;
console.log('Latest-frame replacement, bounded ownership, worker/fallback encoding and cancellation tests passed.');
