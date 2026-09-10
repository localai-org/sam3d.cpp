// One consumer, one replaceable completed frame. Never queues camera history.
export class LatestFrameSlot {
 constructor(){this.frame=null;this.waiter=null;this.closed=false;this.error=null;this.dropped=0;this.maximum=0}
 put(frame){
  if(this.closed)return;
  if(this.waiter){const waiter=this.waiter;this.waiter=null;waiter.resolve(frame);return}
  if(this.frame)this.dropped++;
  this.frame=frame;this.maximum=1;
 }
 take(){
  if(this.closed)return this.error?Promise.reject(this.error):Promise.resolve(null);
  if(this.frame){const frame=this.frame;this.frame=null;return Promise.resolve(frame)}
  if(this.waiter)return Promise.reject(Error('Only one frame consumer allowed'));
  return new Promise((resolve,reject)=>this.waiter={resolve,reject});
 }
 close(error=null){
  if(this.closed)return;
  this.closed=true;this.error=error;this.frame=null;
  if(this.waiter){error?this.waiter.reject(error):this.waiter.resolve(null);this.waiter=null}
 }
}

// Use the camera's decoded frames before they pass through HTMLVideoElement's
// rendering path. The API is not exposed in every browser: those retain the
// video-element capture path. At most one source frame is retained; every
// replacement and every consumer clone has explicit ownership/close().
export function createCameraFrames(track){
 if(typeof MediaStreamTrackProcessor==='undefined'||typeof VideoFrame==='undefined')return null;
 const owned=track.clone();let reader;
 try{reader=new MediaStreamTrackProcessor({track:owned,maxBufferSize:1}).readable.getReader()}
 catch(e){owned.stop();throw e}
 let latest=null,received=0,closed=false,error=null;
 const done=(async()=>{
  try{
   while(!closed){
    const next=await reader.read();
    if(next.done){if(!closed)error=Error('Camera frame source ended');break}
    if(closed){next.value.close();break}
    latest?.close();latest=next.value;received=performance.now();
   }
  }catch(e){if(!closed)error=e}
  finally{latest?.close();latest=null;reader.releaseLock();owned.stop()}
 })();
 return {
  take(){if(error)throw error;return latest?{frame:latest.clone(),received}:null},
  close(){if(closed)return;closed=true;latest?.close();latest=null;reader.cancel().catch(()=>{});owned.stop()},
  done,
 };
}

// One encode at a time, independent of the display canvas and HTTP request.
// Same dimensions and JPEG quality as the original path. Missing worker APIs
// use a private main-thread canvas; worker failures are surfaced, not hidden.
export function createFrameEncoder({forceFallback=false}={}) {
 let worker=null,pending=null,busy=false,closed=false,canvas=null,context=null;
 const canWorker=!forceFallback&&typeof Worker!=='undefined'&&typeof OffscreenCanvas!=='undefined';
 if(canWorker){
  worker=new Worker(new URL('./frame-encoder-worker.js',import.meta.url),{type:'module'});
  worker.onmessage=({data})=>{
   if(!pending)return;
   const p=pending;pending=null;
   data.error?p.reject(Error(data.error)):p.resolve(data);
  };
  worker.onerror=e=>{e.preventDefault();close(Error(e.message||'Frame encoding worker failed'))};
  worker.onmessageerror=()=>close(Error('Invalid frame encoding worker reply'));
 }
 function close(error=new DOMException('Frame encoder stopped','AbortError')){
  closed=true;worker?.terminate();worker=null;
  if(pending){pending.reject(error);pending=null}
  canvas=null;context=null;
 }
 async function encode(source,width,height){
  if(closed)throw new DOMException('Frame encoder stopped','AbortError');
  if(busy)throw Error('Frame encoder already busy');
  if(!Number.isInteger(width)||!Number.isInteger(height)||width<8||height<8||width>960||height>960)throw Error('Invalid frame dimensions');
  busy=true;
  try{
   const started=performance.now();let captureMS,reply;
   let frame=null;
   if(worker&&typeof VideoFrame!=='undefined'&&(source.videoWidth>0||source instanceof VideoFrame))frame=new VideoFrame(source,{timestamp:0});
   if(!frame){
    if(!canvas){canvas=document.createElement('canvas');context=canvas.getContext('2d',{willReadFrequently:true})}
    if(canvas.width!==width)canvas.width=width;if(canvas.height!==height)canvas.height=height;
    context.drawImage(source,0,0,width,height);
   }
   if(worker){
    // Transfer CPU pixels, not a lazy GPU-backed ImageBitmap: the latter can
    // defer a costly cross-thread readback into convertToBlob. This also gives
    // worker and fallback exactly the same resize/color-conversion path.
    const pixels=frame?null:context.getImageData(0,0,width,height).data.buffer;
    captureMS=performance.now()-started;
    reply=await new Promise((resolve,reject)=>{
     const timer=setTimeout(()=>close(Error('Frame encoding timed out')),10000);
     pending={resolve:v=>{clearTimeout(timer);resolve(v)},reject:e=>{clearTimeout(timer);reject(e)}};
     try{worker.postMessage({frame,pixels,width,height},[frame||pixels])}catch(e){frame?.close();const p=pending;pending=null;p.reject(e)}
    });
   }else{
    captureMS=performance.now()-started;
    reply=await new Promise((resolve,reject)=>{
     const timer=setTimeout(()=>close(Error('Frame encoding timed out')),10000);
     pending={resolve:v=>{clearTimeout(timer);resolve(v)},reject:e=>{clearTimeout(timer);reject(e)}};
     try{canvas.toBlob(blob=>{if(pending){const p=pending;pending=null;p.resolve({blob})}},'image/jpeg',.94)}
     catch(e){const p=pending;pending=null;p.reject(e)}
    });
   }
   if(!reply.blob?.size||reply.blob.size>2*1024*1024||!['image/jpeg','image/png'].includes(reply.blob.type))throw Error('Invalid encoded frame');
   return {blob:reply.blob,timing:{capture_ms:captureMS,jpeg_ms:performance.now()-started-captureMS,
    ...(reply.encode_ms===undefined?{}:{worker_encode_ms:reply.encode_ms,worker_draw_ms:reply.draw_ms,worker_blob_ms:reply.blob_ms,capture_path:reply.capture_path}),upload_bytes:reply.blob.size}};
  }finally{busy=false}
 }
 return {encode,close,kind:canWorker?'worker-offscreen':'canvas-fallback'};
}
