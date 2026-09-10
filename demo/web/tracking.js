import {bracket,decodeFrame,mixBodies,followBox} from './tracking-math.js';
import {LatestFrameSlot,createFrameEncoder,createCameraFrames} from './frame-pipeline.js';
import {LivePresentation} from './live-presentation.js';

export function initTracking(view) {
 const $=id=>document.getElementById(id),sleep=ms=>new Promise(r=>setTimeout(r,ms));
 const video=document.createElement('video');video.muted=true;video.playsInline=true;video.preload='auto';
 const canvas=document.createElement('canvas'),cx=canvas.getContext('2d');
 const diagnostics={mode:'photo',running:false,frames:0,actualHz:0,interpolations:0,maxInFlight:0,errors:[],buffer:0,timings:[],renderTimings:[]};
 let mode='photo',epoch=0,ready=false,url=null,stream=null,session=null,abort=null,task=null,name='',faces=null,sourceName='',cancelPipeline=null;
 const live=new LivePresentation(25);
 let presentation=null,renderedPresentation=null,previewTime=-1;
 let mixed=null,startClock=0,lastCapture=-1,playback=null,playing=false,playClock=0,playOrigin=0,fetching=false,cache=new Map(),lastDisplay=null;
 function message(text,error=false){$('track-status').textContent=text;$('track-status').className=error?'error':''}
 function rate(){const v=Number($('track-hz').value);if(!Number.isInteger(v)||v<1||v>30)throw Error('Choose a maximum rate from 1 to 30 Hz.');return v}
 function controls(){const running=!!task;$('track-start').disabled=!ready||running;$('track-stop').hidden=!running;for(const id of ['track-hz','track-duration','video-upload','camera-open','follow-box'])$(id).disabled=running;view.lock(running);diagnostics.running=running}
 function releaseSource(){ready=false;video.pause();stream?.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;video.removeAttribute('src');video.load();if(url)URL.revokeObjectURL(url);url=null;controls()}
 function resetPlayback(){playing=false;playback=null;cache.clear();live.reset();presentation=null;renderedPresentation=null;faces=null;mixed=null;lastDisplay=null;$('track-playback').hidden=true;$('track-play').textContent='Play';view.clear()}
 async function stop(){epoch++;abort?.abort();cancelPipeline?.();const pending=task;if(pending)await pending;task=null;abort=null;if(session){const id=session.id;session=null;try{await view.api(`/api/tracks/${id}/finish`,{method:'POST'})}catch(e){message(e.message,true)}}playing=false;controls();await history()}
 async function changeMode(next){await stop();releaseSource();resetPlayback();mode=next;diagnostics.mode=mode;$('input-mode').value=mode;view.mode(mode!=='photo');$('video-options').hidden=mode==='photo';$('video-upload-label').hidden=mode!=='offline';$('camera-open').hidden=mode!=='live';$('duration-label').hidden=mode!=='offline';$('camera-notice').hidden=mode!=='live'||!!navigator.mediaDevices?.getUserMedia;message('Choose a video or open the webcam.')}
 const guarded=fn=>async(...args)=>{try{await fn(...args)}catch(e){diagnostics.errors.push(e.message);message(e.message,true);controls()}};
 $('input-mode').onchange=guarded(()=>changeMode($('input-mode').value));
 function once(event,token){return new Promise((resolve,reject)=>{const timer=setTimeout(()=>finish(Error('Video decoding timed out. Use a browser-supported MP4/WebM video.')),15000);function finish(error){clearTimeout(timer);video.removeEventListener(event,ok);video.removeEventListener('error',bad);error?reject(error):resolve()}function ok(){finish(token===epoch?null:Error('Video selection changed'))}function bad(){finish(Error('Browser cannot decode this video; try H.264 MP4 or VP9 WebM.'))}video.addEventListener(event,ok,{once:true});video.addEventListener('error',bad,{once:true})})}
 function capture(initial=false){if(video.readyState<2)throw Error('No decoded video frame available');cx.drawImage(video,0,0,canvas.width,canvas.height);previewTime=video.currentTime;view.preview(canvas,name,initial)}
 function dimensions(){const scale=Math.min(1,960/video.videoWidth,960/video.videoHeight);canvas.width=Math.max(8,Math.round(video.videoWidth*scale));canvas.height=Math.max(8,Math.round(video.videoHeight*scale))}
 async function seek(time,token){if(Math.abs(video.currentTime-time)<.0001&&video.readyState>=2)return;const done=once('seeked',token);video.currentTime=time;await done}
 $('video-upload').onchange=guarded(async()=>{
  const file=$('video-upload').files[0];if(!file)return;
  await stop();releaseSource();resetPlayback();const token=epoch;name=sourceName=file.name;
  url=URL.createObjectURL(file);const loaded=once('loadeddata',token);video.src=url;await loaded;
  if(token!==epoch)return;
  if(!Number.isFinite(video.duration)||video.duration<=0)throw Error('Video must have a finite duration.');
  dimensions();capture(true);ready=true;$('track-duration').value=Math.min(10,video.duration).toFixed(2);controls();message(`${video.duration.toFixed(2)} seconds. Select one person, then start. Frames are resized to ${canvas.width} × ${canvas.height}.`);
 });
 $('camera-open').onclick=guarded(async()=>{
  if(!navigator.mediaDevices?.getUserMedia)throw Error('Webcam access requires HTTPS or localhost. Use HTTPS for the remote demo.');
  await stop();releaseSource();resetPlayback();const token=epoch;
  const acquired=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:960},height:{ideal:720}},audio:false});
  if(token!==epoch){acquired.getTracks().forEach(t=>t.stop());return}
  stream=acquired;name=sourceName='Webcam';const loaded=once('loadeddata',token);video.srcObject=stream;await video.play();await loaded;if(token!==epoch)return;
  dimensions();capture(true);ready=true;controls();message('Webcam ready. Select one person in the preview, then start.');
  stream.getVideoTracks()[0].addEventListener('ended',()=>{stop().then(()=>{releaseSource();message('Camera disconnected. Reopen it to continue.',true)})});
 });
 async function sendFrame(prepared,s,token){
  const {blob,stamp}=prepared,measured={...prepared.timing,queue_ms:performance.now()-prepared.completed};
  const query=new URLSearchParams({time:String(stamp),settings:JSON.stringify(s)});
  while(token===epoch){
   diagnostics.maxInFlight=Math.max(diagnostics.maxInFlight,1);
   const requestStarted=performance.now();
   measured.sent=requestStarted;
   const response=await fetch(`/api/tracks/${session.id}/frame?${query}`,{method:'POST',body:blob,signal:abort.signal});
   measured.headers_ms=performance.now()-requestStarted;
   if(response.status===429){if(mode==='live')return null;message('Waiting for the shared inference worker…');await sleep(100);continue}
   if(!response.ok){let e;try{e=(await response.json()).error}catch{e=response.statusText}throw Error(e)}
   const downloadStarted=performance.now(),buffer=await response.arrayBuffer();measured.download_ms=performance.now()-downloadStarted;
   const unpackStarted=performance.now(),frame=decodeFrame(buffer,faces);measured.unpack_ms=performance.now()-unpackStarted;
   measured.request_ms=performance.now()-requestStarted;measured.response_bytes=buffer.byteLength;
   measured.server={};for(const item of (response.headers.get('Server-Timing')||'').split(',')){const match=item.trim().match(/^([a-z_]+);dur=([0-9.]+)$/);if(match)measured.server[match[1]]=Number(match[2])}
   frame.timing=measured;faces=frame.body.faces;return frame;
  }
  return null;
 }
 async function run(token,hz,duration){
  const begin=performance.now();startClock=begin;lastCapture=-1;diagnostics.frames=0;diagnostics.actualHz=0;diagnostics.blend_ms=live.duration;
  let index=0,s=view.settings(),previousSent=null,nextSend=begin,encoder=null,producer=null,slot=null,cameraFrames=null;
  let captureNotBefore=begin,encodeEstimate=0,requestEstimate=0;
  diagnostics.timings=[];diagnostics.renderTimings=[];
  diagnostics.maxInFlight=0;diagnostics.maxPrepared=0;diagnostics.dropped=0;diagnostics.encoding=false;diagnostics.prepared=0;diagnostics.presentedFrame=0;diagnostics.settledFrame=0;
  try {
   encoder=createFrameEncoder();diagnostics.encoder=encoder.kind;
   if(mode==='live'&&encoder.kind==='worker-offscreen')cameraFrames=createCameraFrames(stream.getVideoTracks()[0]);
   diagnostics.cameraSource=cameraFrames?'camera-track':'video-element';
   slot=new LatestFrameSlot();
   cancelPipeline=()=>{slot.close();encoder.close();cameraFrames?.close()};
   const created=await view.api('/api/tracks',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode,name,hz,width:canvas.width,height:canvas.height})});
   session=created;
   if(token!==epoch)return;
   async function prepare(stamp){
    const selected=performance.now(),snapshot=cameraFrames?.take(),captured=snapshot?Math.min(selected,snapshot.received):selected;diagnostics.encoding=true;
    try{
     const encoded=await encoder.encode(snapshot?.frame||video,canvas.width,canvas.height),completed=performance.now();
     encoded.timing.source_age_ms=selected-captured;encoded.timing.frame_source=snapshot?'camera-track':'video-element';
     encodeEstimate=encodeEstimate?encodeEstimate*.75+(completed-selected)*.25:completed-selected;
     return {...encoded,stamp,captured,completed};
    }finally{snapshot?.frame.close();diagnostics.encoding=false}
   }
   if(mode==='live')producer=(async()=>{
    let next=performance.now();
    try{
     while(token===epoch&&!slot.closed&&!document.hidden){
      const now=performance.now(),wanted=Math.max(next,captureNotBefore);if(now<wanted){await sleep(Math.min(50,Math.ceil(wanted-now)));continue}
      if(video.currentTime===lastCapture){await sleep(10);continue}
      lastCapture=video.currentTime;next=now+1000/hz;
      slot.put(await prepare((now-startClock)/1000));
      diagnostics.maxPrepared=slot.maximum;diagnostics.prepared=slot.frame?1:0;diagnostics.dropped=slot.dropped;
     }
     slot.close();
    }catch(e){slot.close(e)}
   })();
   while(token===epoch){
    let prepared;
    if(mode==='offline'){
     const stamp=index/hz;if(stamp>=duration-1e-6)break;
     await seek(Math.min(stamp,video.duration-.001),token);
     if(token!==epoch)break;
     capture();prepared=await prepare(stamp);
    }else{
     if(document.hidden)break;
     const now=performance.now();if(now<nextSend){await sleep(Math.min(50,Math.ceil(nextSend-now)));continue}
     prepared=await slot.take();diagnostics.prepared=0;if(!prepared)break;
     // A stopped/paused camera must not send an arbitrarily old cached image.
     if(performance.now()-prepared.captured>Math.max(250,2000/hz))continue;
    }
    if(token!==epoch)break;
    nextSend=performance.now()+1000/hz;
    // Aim to finish the next encode just before this request returns. If it
    // stalls, the producer still refreshes the single slot at the Hz cap.
    // Do not learn the cold model-load duration as steady-state request time.
    captureNotBefore=performance.now()+Math.max(0,requestEstimate-encodeEstimate-10);
    const frame=await sendFrame(prepared,s,token);if(token!==epoch)break;
    if(!frame)continue;
    diagnostics.frames++;index++;
    const latency=(performance.now()-prepared.captured)/1000,sent=frame.timing.sent;
    const timing={...frame.timing,frame:index,capture_time:prepared.stamp,total_ms:latency*1000,interval_ms:previousSent===null?null:sent-previousSent};
    diagnostics.timings.push(timing);
    previousSent=sent;if(diagnostics.timings.length>240)diagnostics.timings.shift();
    if(index>=2)requestEstimate=Math.min(...diagnostics.timings.slice(-8).filter(t=>t.frame>1).map(t=>t.request_ms));
    diagnostics.sessionHz=diagnostics.frames/((performance.now()-begin)/1000);
    const intervals=diagnostics.timings.slice(-20).map(t=>t.interval_ms).filter(t=>t!==null);
    diagnostics.actualHz=intervals.length?1000*intervals.length/intervals.reduce((a,b)=>a+b,0):diagnostics.sessionHz;
    if(mode==='live'){
     const received=performance.now();live.push(frame.body,received);
     presentation={timing,captured:prepared.captured,received};diagnostics.buffer=1;
    }else display(frame.body);
    if($('follow-box').checked){s=followBox(frame.body,s,canvas.width,canvas.height);view.setSettings(s)}
    $('track-metrics').textContent=`${diagnostics.frames} frames · ${diagnostics.actualHz.toFixed(1)} Hz recent · ${hz} Hz cap · ${(latency*1000).toFixed(0)} ms frame latency${mode==='live'?` · ${live.duration} ms blend · no playback buffer`:''}`;
    message(mode==='offline'?`Processing ${Math.min(duration,prepared.stamp+1/hz).toFixed(1)} / ${duration.toFixed(1)} seconds…`:'Tracking live · latest frames only');
   }
   if(token===epoch&&session){
    const record=await view.api(`/api/tracks/${session.id}/finish`,{method:'POST'});session=null;
    if(token===epoch&&mode==='offline'){await loadTrack(record,false);message(`Saved ${record.count} frames. Play or scrub the result.`)}
    else if(token===epoch){releaseSource();message('Live tracking stopped.')}
   }
  }catch(e){if(token===epoch){diagnostics.errors.push(e.message);message(e.message,true)}}
  finally{
   slot?.close();encoder?.close();cameraFrames?.close();if(producer)await producer;if(cameraFrames)await cameraFrames.done;cancelPipeline=null;diagnostics.encoding=false;diagnostics.prepared=0;
   if(session&&token===epoch){const id=session.id;session=null;try{await view.api(`/api/tracks/${id}/finish`,{method:'POST'})}catch{}}
   if(token===epoch){task=null;controls();await history()}
  }
 }
 $('track-start').onclick=guarded(async()=>{
  if(task||!ready)return;const hz=rate();const duration=mode==='offline'?Math.min(Number($('track-duration').value),video.duration):0;
  if(mode==='offline'&&(!Number.isFinite(duration)||duration<=0||Math.ceil(duration*hz)>1800))throw Error('Choose a duration/rate giving at most 1,800 frames.');
  const s=view.settings(),b=s.box,c=s.camera;
  if(![...b,...c].every(Number.isFinite)||b[0]<0||b[1]<0||b[2]>canvas.width||b[3]>canvas.height||b[2]-b[0]<8||b[3]-b[1]<8||c[0]<1||c[1]<1)throw Error('Select a valid person box and camera first.');
  resetPlayback();abort=new AbortController();const token=++epoch;task=run(token,hz,duration);controls();message('Loading/resuming native model; first frame can take several seconds…');
 });
 $('track-stop').onclick=guarded(async()=>{await stop();if(mode==='live')releaseSource();message('Stopped. Completed offline frames are retained in history.')});
 function display(body){if(!mixed)mixed={schema:'sam3d.body.track.v1',faces:body.faces,tensors:{}};view.show(body);diagnostics.interpolations++}
 function interpolate(a,b,alpha){if(!mixed)mixed={schema:'sam3d.body.track.v1',faces:a.faces,tensors:{}};display(mixBodies(a,b,alpha,mixed))}
 async function history(){const list=await view.api('/api/tracks');$('track-history').replaceChildren();for(const t of list){const button=document.createElement('button');button.className='history-item';button.textContent=`${t.name} · ${t.count} frames · ${t.state}`;button.onclick=guarded(async()=>{await changeMode('offline');await loadTrack(t,true)});$('track-history').append(button)}}
 async function loadTrack(t,fromHistory){
  if(!t.samples.length){message('This sequence has no completed frames.',true);return}
  const token=epoch;resetPlayback();
  const response=await fetch(`/tracks/${t.id}/faces.bin`);if(!response.ok)throw Error('Saved topology unavailable');const buffer=await response.arrayBuffer();
  if(token!==epoch)return;
  if(buffer.byteLength!==36874*3*4)throw Error('Invalid saved topology');faces=new Uint32Array(buffer);
  playback=t;$('track-playback').hidden=false;$('track-seek').min=t.samples[0].time;$('track-seek').max=t.samples.at(-1).time;$('track-seek').value=t.samples[0].time;$('track-manifest').href=`/tracks/${t.id}/track.json`;
  if(fromHistory){
   const im=await createImageBitmap(await (await fetch(`/tracks/${t.id}/preview.jpg`)).blob());
   if(token!==epoch){im.close();return}
   canvas.width=t.width;canvas.height=t.height;cx.drawImage(im,0,0);im.close();view.preview(canvas,t.name+' (saved first frame; source video not retained)',false);sourceName='';
  }
  view.setSettings(t.samples[0].settings);await paintPlayback(Number($('track-seek').value));message(`${t.count} saved poses · ${t.hz} Hz sampling · ${t.precision.toUpperCase()}`);
 }
 async function getFrame(i){
  if(cache.has(i))return cache.get(i);const t=playback;if(!t)return null;
  const response=await fetch(`/tracks/${t.id}/${String(i).padStart(6,'0')}.bin`);if(!response.ok)throw Error('Saved pose unavailable');
  const frame=decodeFrame(await response.arrayBuffer(),faces);if(playback!==t)return null;
  if(Math.abs(frame.time-t.samples[i].time)>.00001)throw Error('Saved timestamp mismatch');
  cache.set(i,frame);while(cache.size>8)cache.delete(cache.keys().next().value);return frame;
 }
 async function paintPlayback(time){
  if(!playback||fetching)return;fetching=true;const t=playback;
  try{const [i,j,alpha]=bracket(t.samples,time);const [a,b]=await Promise.all([getFrame(i),getFrame(j)]);if(playback!==t||!a||!b)return;
   if(ready&&sourceName===t.name&&mode==='offline'&&Math.abs(video.currentTime-time)>.02){await seek(Math.min(time,video.duration-.001),epoch);if(playback!==t)return;capture()}
   view.setSettings(t.samples[i].settings);interpolate(a.body,b.body,alpha);$('track-time').textContent=`${time.toFixed(2)} s`;lastDisplay=time;
  }catch(e){playing=false;message(e.message,true);diagnostics.errors.push(e.message)}finally{fetching=false}
 }
 $('track-play').onclick=()=>{if(!playback)return;playing=!playing;if(playing){if(Number($('track-seek').value)>=playback.samples.at(-1).time)$('track-seek').value=playback.samples[0].time;playOrigin=Number($('track-seek').value);playClock=performance.now()}$('track-play').textContent=playing?'Pause':'Play'};
 $('track-seek').oninput=()=>{playing=false;$('track-play').textContent='Play';paintPlayback(Number($('track-seek').value))};
 function tick(now){
  const tickStarted=performance.now();
  if(mode==='live'&&ready&&video.readyState>=2&&video.currentTime!==previewTime)capture();
  if(mode==='live'&&task&&presentation&&(!live.finished||presentation!==renderedPresentation)){
   const value=live.sample(performance.now());display(value.body);
   renderedPresentation=presentation;renderedPresentation.finished=value.finished;
  }
  if(playback){
   let time=Number($('track-seek').value);
   if(playing){time=Math.min(playback.samples.at(-1).time,playOrigin+(now-playClock)/1000);$('track-seek').value=time;if(time>=playback.samples.at(-1).time){playing=false;$('track-play').textContent='Play'}}
   if(time!==lastDisplay)paintPlayback(time);
  }
  if(mode==='live'&&task){diagnostics.renderTimings.push(performance.now()-tickStarted);if(diagnostics.renderTimings.length>600)diagnostics.renderTimings.shift()}
 }
 // Called after Three.js submits this render. This is not a physical-display
 // timestamp: camera exposure and GPU/compositor/scanout delays are excluded.
 function afterRender(){
  if(mode!=='live'||!task||!renderedPresentation)return;
  const p=renderedPresentation,t=p.timing,now=performance.now();
  if(t.first_render_ms===undefined){t.first_render_ms=now-p.captured;t.response_to_render_ms=now-p.received;diagnostics.presentedFrame=t.frame}
  if(p.finished&&t.settled_render_ms===undefined){t.settled_render_ms=now-p.captured;diagnostics.settledFrame=t.frame}
 }
 document.addEventListener('visibilitychange',()=>{if(document.hidden&&mode==='live')stop().then(()=>{releaseSource();message('Webcam stopped while this tab was hidden.')})});
 window.addEventListener('pagehide',()=>{abort?.abort();cancelPipeline?.();stream?.getTracks().forEach(t=>t.stop());if(session)fetch(`/api/tracks/${session.id}/finish`,{method:'POST',keepalive:true}).catch(()=>{})});
 history().catch(e=>message(e.message,true));controls();
 return {tick,afterRender,diagnostics,leave:()=>mode==='photo'?Promise.resolve():changeMode('photo')};
}
