// The main thread transfers exactly one immutable snapshot at a time.
let canvas=null,context=null,busy=false;
self.onmessage=async({data:{frame,pixels,width,height}})=>{
 if(busy){frame?.close();self.postMessage({error:'Concurrent frame encoding'});return}
 busy=true;
 try{
  if(!Number.isInteger(width)||!Number.isInteger(height)||width<8||height<8||width>960||height>960)throw Error('Invalid frame dimensions');
  const started=performance.now();
  // This canvas is read back for every JPEG, never displayed by WebGL. Prefer
  // CPU-backed pixels instead of paying a GPU readback on each conversion.
  if(!canvas){canvas=new OffscreenCanvas(width,height);context=canvas.getContext('2d',{willReadFrequently:true})}
  if(canvas.width!==width)canvas.width=width;if(canvas.height!==height)canvas.height=height;
  let capture_path='canvas-pixels';
  if(frame){
   capture_path='video-frame-draw';
   if(frame.visibleRect.width===width&&frame.visibleRect.height===height&&frame.displayWidth===width&&frame.displayHeight===height&&!frame.rotation&&!frame.flip){
    const rgba=new Uint8ClampedArray(width*height*4);
    try{
     await frame.copyTo(rgba,{format:'RGBA',colorSpace:'srgb',layout:[{offset:0,stride:width*4}]});
     context.putImageData(new ImageData(rgba,width,height),0,0);capture_path='video-frame-copy';
    }catch(e){if(e.name!=='NotSupportedError'&&e.name!=='TypeError')throw e;context.drawImage(frame,0,0,width,height)}
   }else context.drawImage(frame,0,0,width,height);
   frame.close();frame=null;
  }else{
   if(!(pixels instanceof ArrayBuffer)||pixels.byteLength!==width*height*4)throw Error('Invalid frame pixels');
   context.putImageData(new ImageData(new Uint8ClampedArray(pixels),width,height),0,0);
  }
  const drawn=performance.now(),blob=await canvas.convertToBlob({type:'image/jpeg',quality:.94});
  self.postMessage({blob,capture_path,draw_ms:drawn-started,blob_ms:performance.now()-drawn,encode_ms:performance.now()-started});
 }catch(e){self.postMessage({error:e.message||'Frame encoding failed'})}
 finally{frame?.close();busy=false}
};
