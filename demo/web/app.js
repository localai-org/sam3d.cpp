import * as THREE from './vendor/three.module.min.js';
import {initTracking} from './tracking.js';

const $=id=>document.getElementById(id), boxIDs=['x0','y0','x1','y1'], camIDs=['fx','fy','cx','cy'];
const state={image:null,file:null,name:'',job:null,result:null,reference:null,referenceManifest:null,epoch:0,history:[],ready:false,errors:[],kind:'body',maskPainted:false,config:null};
const objectMask=document.createElement('canvas'),objectMaskContext=objectMask.getContext('2d');
// Small observable diagnostics used by real-browser QA; never supplies inference results.
window.sam3dQA={state,rendered:false};
window.addEventListener('error',e=>state.errors.push(e.message));
window.addEventListener('unhandledrejection',e=>{state.errors.push(String(e.reason));status(String(e.reason),true)});
function status(s,error=false){const node=$('status'),className=error?'error':state.job&&['queued','running'].includes(state.job.state)?'busy':'';if(node.textContent!==s)node.textContent=s;if(node.className!==className)node.className=className;$('copy-status').hidden=!error}
$('copy-status').onclick=async()=>{const button=$('copy-status');try{await navigator.clipboard.writeText($('status').textContent);button.textContent='Copied';setTimeout(()=>button.textContent='Copy error',1500)}catch{button.textContent='Select the error and press Ctrl+C'}};
async function api(path,options){const r=await fetch(path,options);if(!r.ok){let e;try{e=(await r.json()).error}catch{e=r.statusText}throw new Error(e||`HTTP ${r.status}`)}return r.json()}
function settings(){return {box:boxIDs.map(id=>Number($(id).value)),camera:camIDs.map(id=>Number($(id).value))}}
function setSettings(s){boxIDs.forEach((id,i)=>$(id).value=s.box[i]);camIDs.forEach((id,i)=>$(id).value=s.camera[i])}
function validSettings(){if(!state.image||!state.file)return false;const {box:b,camera:c}=settings(),w=state.image.width,h=state.image.height;return [...b,...c].every(Number.isFinite)&&b[0]>=0&&b[1]>=0&&b[2]<=w&&b[3]<=h&&b[2]-b[0]>=8&&b[3]-b[1]>=8&&c[0]>=1&&c[1]>=1&&c[0]<=100000&&c[1]<=100000&&c[2]>=0&&c[2]<=w&&c[3]>=0&&c[3]<=h&&[...boxIDs,...camIDs].every(id=>$(id).value!=='')}
function busy(){return state.job&&['queued','running'].includes(state.job.state)}
function controls(){const valid=state.kind==='object'?state.image&&state.file&&state.maskPainted:validSettings();$('generate').disabled=state.videoActive||!valid||busy()||state.submitting||!state.ready;$('whole').disabled=!state.image||!!window.sam3dQA.tracking?.running;$('cancel').hidden=!busy();$('downloads').hidden=!state.result||state.videoActive}
const topology=await api('skeleton.json');
if(topology.parents.length!==127||topology.parents.some((p,i)=>i===0?p!==-1:p<0||p>=i))throw Error('Invalid MHR skeleton topology');

// Coordinate conversion matches exports: body (X,-Y,-Z). Both original and
// native share ONE camera/ground and origin; no independent pose alignment.
const scene=new THREE.Scene();scene.background=new THREE.Color('#10192c');
const camera=new THREE.PerspectiveCamera(38,1,.01,200);
const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;
$('viewer').appendChild(renderer.domElement);
scene.add(new THREE.HemisphereLight(0xcbeaff,0x384450,2.2));const key=new THREE.DirectionalLight(0xffffff,2.8);key.position.set(3,5,5);scene.add(key);
const grid=new THREE.GridHelper(10,40,0x526984,0x263b53);scene.add(grid);grid.visible=false;
let nativeGroup=null,originalGroup=null,mesh=null,bones=null,azimuth=.3,elevation=.1,distance=3,target=new THREE.Vector3(0,.8,0),frameCentre=new THREE.Vector3(),frameSize=1.8;
function toWorld(v){return [v[0],-v[1],-v[2]]}
function converted(v){const out=new Float32Array(v);for(let i=0;i<out.length;i+=3){out[i+1]*=-1;out[i+2]*=-1}return out}
function disposeGroup(group){if(!group)return;scene.remove(group);group.traverse(o=>{o.geometry?.dispose();if(Array.isArray(o.material))o.material.forEach(m=>m.dispose());else o.material?.dispose()})}
function validateBody(b){const t=b?.tensors,skeleton=b?.schema==='sam3d.body.skeleton.v1',track=skeleton||b?.schema==='sam3d.body.track.v1';if((!track&&b?.schema!=='sam3d.body.pose_branch.v1')||!t||t.vertices?.length!==(skeleton?0:18439*3)||t.joints?.length!==127*3||b.faces?.length!==(skeleton?0:36874*3)||t.camera_translation?.length!==3||(!track&&t.keypoints_pixels?.length!==140))throw Error('Invalid body result schema');for(const key of track?['vertices','joints','camera_translation']:['vertices','joints','camera_translation','keypoints_pixels','vertices_pixels'])if(!t[key]?.every(Number.isFinite))throw Error(`Nonfinite ${key}`);if(!b.faces.every(v=>Number.isInteger(v)&&v>=0&&v<18439))throw Error('Invalid mesh indices')}
function makeBody(b,reference=false){validateBody(b);const group=new THREE.Group();const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(converted(b.tensors.vertices),3));g.setIndex(new THREE.BufferAttribute(new Uint32Array(b.faces),1));g.computeVertexNormals();
 const m=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:reference?0xffad61:0x61b7d4,roughness:.75,metalness:0,side:THREE.DoubleSide,transparent:reference,opacity:reference?.36:1,depthWrite:!reference,wireframe:reference}));group.add(m);
 // Joint 0 is MHR's artificial body_world frame, not an anatomical bone.
 const j=converted(b.tensors.joints),edges=[];for(let i=2;i<127;i++){edges.push(...j.slice(i*3,i*3+3),...j.slice(topology.parents[i]*3,topology.parents[i]*3+3))}
 const line=new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position',new THREE.Float32BufferAttribute(edges,3)),new THREE.LineBasicMaterial({color:reference?0xffa34f:0x8cf1ff,depthTest:false,transparent:true,opacity:1}));line.renderOrder=3;group.add(line);
 const points=new THREE.Points(new THREE.BufferGeometry().setAttribute('position',new THREE.BufferAttribute(j.slice(3),3)),new THREE.PointsMaterial({color:reference?0xffbd75:0xe4fcff,size:.011,depthTest:false}));points.renderOrder=4;line.add(points);
 scene.add(group);return {group,mesh:m,bones:line};
}
function updateCamera(){camera.aspect=$('viewer').clientWidth/Math.max(1,$('viewer').clientHeight);camera.updateProjectionMatrix();camera.position.copy(target).add(new THREE.Vector3(Math.sin(azimuth)*Math.cos(elevation),Math.sin(elevation),Math.cos(azimuth)*Math.cos(elevation)).multiplyScalar(distance));camera.lookAt(target)}
function reset(view='oblique'){target.copy(frameCentre);azimuth=view==='front'?0:view==='side'?Math.PI/2:.3;elevation=view==='oblique'?.1:0;const aspect=$('viewer').clientWidth/Math.max(1,$('viewer').clientHeight),fov=THREE.MathUtils.degToRad(camera.fov);distance=frameSize*.68/Math.tan(fov/2)/Math.min(1,aspect);updateCamera()}
function clearResult(){state.result=null;disposeGroup(nativeGroup);disposeGroup(originalGroup);nativeGroup=originalGroup=mesh=bones=null;$('downloads').hidden=true;$('viewer-empty').hidden=false;$('geometry-info').textContent='';$('reference').checked=false;$('reference').disabled=true;$('reference-legend').hidden=true;$('ply').hidden=true;$('glb').textContent='Body mesh GLB';$('skeleton-glb').hidden=$('glb').hidden=$('obj').hidden=false;grid.visible=false;window.sam3dQA.rendered=false;controls()}
function showResult(b){clearResult();state.result=b;const parts=makeBody(b);nativeGroup=parts.group;mesh=parts.mesh;bones=parts.bones;
 const bounds=new THREE.Box3().setFromBufferAttribute(b.tensors.vertices.length?mesh.geometry.getAttribute('position'):new THREE.Float32BufferAttribute(converted(b.tensors.joints).slice(3),3));bounds.getCenter(frameCentre);const size=bounds.getSize(new THREE.Vector3());frameSize=Math.max(size.x,size.y,size.z);if(!Number.isFinite(frameSize)||frameSize<=0||frameSize>100)throw Error('Invalid or extreme mesh extent');
 grid.position.y=bounds.min.y-.006;grid.visible=true;reset();$('viewer-empty').hidden=true;$('geometry-info').textContent=`18,439 vertices · 127 joints · ${size.y.toFixed(2)} m high`;
 mesh.visible=$('mesh').checked&&state.result?.tensors.vertices.length>0;bones.visible=$('skeleton').checked;
 $('skeleton-glb').href=`/api/jobs/${state.job.id}/skeleton.glb?movement=${$('export-movement').value}`;
 for(const [id,file]of [['glb','body.glb'],['obj','body.obj'],['metadata','job.json']])$(id).href=`/files/${state.job.id}/${file}`;
 const matching=state.referenceManifest&&state.job.native_input_sha256===state.referenceManifest.native_input_sha256;$('reference').disabled=!matching;
 $('comparison').textContent=matching?'Original PyTorch body-branch output is available for this exact RGB, box and camera. Both use the same coordinates.':'Comparison requires the official example and unchanged box/camera settings.';
 drawPhoto();controls();window.sam3dQA.rendered=true;
}
function showObject(buffer){clearResult();const view=new DataView(buffer);if(view.byteLength<28||view.getUint32(0,true)!==0x46546c67||view.getUint32(4,true)!==2||view.getUint32(8,true)!==view.byteLength)throw Error('Invalid object GLB');const jsonLength=view.getUint32(12,true);if(view.getUint32(16,true)!==0x4e4f534a||20+jsonLength+8>view.byteLength)throw Error('Invalid object GLB JSON chunk');const doc=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,20,jsonLength)));const binHeader=20+jsonLength,binLength=view.getUint32(binHeader,true),binStart=binHeader+8;if(view.getUint32(binHeader+4,true)!==0x004e4942||binStart+binLength!==view.byteLength)throw Error('Invalid object GLB binary chunk');const primitive=doc.meshes?.[0]?.primitives?.[0],accessor=index=>{const a=doc.accessors?.[index],b=doc.bufferViews?.[a?.bufferView],width=a?.type==='VEC3'?3:1;if(!a||!b||b.buffer!==0||a.byteOffset||b.byteStride)throw Error('Unsupported object GLB layout');const at=binStart+(b.byteOffset||0);if(a.componentType===5126)return new Float32Array(new Float32Array(buffer,at,a.count*width));if(a.componentType===5125)return new Uint32Array(new Uint32Array(buffer,at,a.count*width));throw Error('Unsupported object GLB component type')};const pos=accessor(primitive?.attributes?.POSITION),col=accessor(primitive?.attributes?.COLOR_0),indices=accessor(primitive?.indices);if(!(pos instanceof Float32Array)||!(col instanceof Float32Array)||!(indices instanceof Uint32Array)||!pos.length||pos.length!==col.length||indices.length%3)throw Error('Invalid object mesh buffers');if(!pos.every(Number.isFinite)||!col.every(Number.isFinite)||indices.some(i=>i>=pos.length/3))throw Error('Invalid object mesh values');const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(pos,3));geometry.setAttribute('color',new THREE.BufferAttribute(col,3));geometry.setIndex(new THREE.BufferAttribute(indices,1));geometry.computeVertexNormals();mesh=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({vertexColors:true,roughness:1,metalness:0,side:THREE.DoubleSide}));nativeGroup=new THREE.Group();nativeGroup.add(mesh);scene.add(nativeGroup);const bounds=new THREE.Box3().setFromBufferAttribute(geometry.getAttribute('position'));bounds.getCenter(frameCentre);const size=bounds.getSize(new THREE.Vector3());frameSize=Math.max(size.x,size.y,size.z);if(!Number.isFinite(frameSize)||frameSize<=0||frameSize>100)throw Error('Invalid or extreme object mesh extent');grid.position.y=bounds.min.y-.006;grid.visible=true;reset();state.result={kind:'object'};$('viewer-empty').hidden=true;$('geometry-info').textContent=`${(pos.length/3).toLocaleString()} vertices · ${(indices.length/3).toLocaleString()} faces`;$('glb').href=`/files/${state.job.id}/object.glb`;$('glb').textContent='Object mesh GLB';$('glb').hidden=false;$('metadata').href=`/files/${state.job.id}/job.json`;$('skeleton-glb').hidden=$('obj').hidden=$('ply').hidden=true;$('downloads').hidden=false;controls();window.sam3dQA.rendered=true}
let pointer=null;
renderer.domElement.addEventListener('contextmenu',e=>e.preventDefault());
renderer.domElement.addEventListener('pointerdown',e=>{pointer={x:e.clientX,y:e.clientY,pan:e.button===2||e.shiftKey};renderer.domElement.setPointerCapture(e.pointerId)});
renderer.domElement.addEventListener('pointermove',e=>{if(!pointer)return;const dx=e.clientX-pointer.x,dy=e.clientY-pointer.y;pointer.x=e.clientX;pointer.y=e.clientY;if(pointer.pan){const right=new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld,0),up=new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld,1);target.addScaledVector(right,-dx*distance*.0015).addScaledVector(up,dy*distance*.0015)}else{azimuth-=dx*.006;elevation=THREE.MathUtils.clamp(elevation+dy*.006,-1.35,1.35)}updateCamera()});
renderer.domElement.addEventListener('pointerup',()=>pointer=null);renderer.domElement.addEventListener('pointercancel',()=>pointer=null);renderer.domElement.addEventListener('wheel',e=>{e.preventDefault();distance=THREE.MathUtils.clamp(distance*Math.exp(e.deltaY*.001),.1,100);updateCamera()},{passive:false});
new ResizeObserver(()=>{renderer.setSize($('viewer').clientWidth,$('viewer').clientHeight,false);updateCamera()}).observe($('viewer'));
let tracking=null;
function setKind(kind){state.kind=kind;$('reconstruction-kind').value=kind;$('tracking').hidden=kind==='object';$('body-selection').hidden=kind==='object';$('object-selection').hidden=kind!=='object';$('selection-title').textContent=kind==='object'?'Select an object':'Select a person';$('selection-help').textContent=kind==='object'?'Paint over one object in the source photo. Paint and erase refine the exact mask used for reconstruction.':'Drag a box around one person in the preview. For video, select the person before starting tracking.';$('generate').textContent=kind==='object'?'Reconstruct selected object':'Estimate 3D body';$('mode-notice').textContent=kind==='object'?'The native geometry decoder produces a vertex-coloured FlexiCubes mesh. Download the same GLB shown here.':'Body pose only. Detailed hand refinement is not yet enabled. Export a posed skeleton from a photo, or a skeleton animation from video and recorded live takes.';$('mesh').parentElement.hidden=kind==='object';$('skeleton').parentElement.hidden=kind==='object';$('projection').parentElement.hidden=kind==='object';$('reference').parentElement.hidden=kind==='object';$('comparison').hidden=kind==='object';$('movement-controls').hidden=kind==='object';$('view-help').textContent=kind==='object'?'Object mesh GLB: learned geometry and vertex colour, Y-up. Orbit: drag · zoom: wheel · pan: right-drag or Shift-drag.':'Skeleton GLB: named joint hierarchy, metres, Y-up. Mesh GLB / OBJ: static body. Orbit: drag · zoom: wheel · pan: right-drag or Shift-drag.';$('viewer-empty').textContent=kind==='object'?'Reconstruct an object to explore it in 3D.':'Estimate a body to explore it in 3D.';$('photo-empty').textContent=kind==='object'?'Your scene and painted object mask will appear here.':'Your photograph and projected skeleton will appear here.';if(state.config)$('model').textContent=`${kind==='object'?state.config.object_model:state.config.model} · ${state.config.backend}`;drawPhoto();controls()}
$('reconstruction-kind').onchange=async()=>{await tracking?.leave();detach();setKind($('reconstruction-kind').value)};
function render(now){tracking?.tick(now);renderer.render(scene,camera);tracking?.afterRender();window.sam3dQA.camera={position:camera.position.toArray(),target:target.toArray()};window.sam3dQA.drawCalls=renderer.info.render.calls;requestAnimationFrame(render)}render();
for(const id of ['reset','front','side'])$(id).onclick=()=>reset(id==='reset'?'oblique':id);
$('mesh').onchange=()=>{if(mesh)mesh.visible=$('mesh').checked&&state.result?.tensors.vertices.length>0};$('skeleton').onchange=()=>{if(bones)bones.visible=$('skeleton').checked};$('projection').onchange=drawPhoto;
$('reference').onchange=async()=>{try{if($('reference').checked&&!originalGroup){state.reference=state.reference||await api('/reference/result.json');originalGroup=makeBody(state.reference,true).group}if(originalGroup)originalGroup.visible=$('reference').checked;$('reference-legend').hidden=!$('reference').checked;drawPhoto()}catch(e){$('reference').checked=false;status(e.message,true)}};

const photo=$('photo'),pc=photo.getContext('2d');
function drawProjection(b,color){const j=b.tensors.joints,t=b.tensors.camera_translation,c=settings().camera;pc.strokeStyle=color;pc.fillStyle=color;pc.lineWidth=Math.max(1.5,photo.width/650);const pixels=[];
 for(let i=0;i<127;i++){const z=j[i*3+2]+t[2];pixels.push(z>1e-6?[(j[i*3]+t[0])/z*c[0]+c[2],(j[i*3+1]+t[1])/z*c[1]+c[3]]:null)}
 pc.beginPath();for(let i=2;i<127;i++){const p=pixels[i],q=pixels[topology.parents[i]];if(p&&q){pc.moveTo(...p);pc.lineTo(...q)}}pc.stroke();for(const p of pixels.slice(1)){if(p){pc.beginPath();pc.arc(...p,Math.max(2,photo.width/600),0,Math.PI*2);pc.fill()}}
}
function drawPhoto(){if(!state.image)return;pc.drawImage(state.image,0,0);if(state.kind==='object'){pc.save();pc.globalAlpha=.48;pc.drawImage(objectMask,0,0);pc.restore();return}const b=settings().box;pc.strokeStyle='#5fe5fc';pc.lineWidth=Math.max(2,photo.width/500);pc.setLineDash([photo.width/100,photo.width/160]);pc.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);pc.setLineDash([]);if(state.result&&$('projection').checked){drawProjection(state.result,'#5fe5fc');if($('reference').checked&&state.reference)drawProjection(state.reference,'#ffa74f')}}
function detach(){state.epoch++;state.job=null;clearResult();$('elapsed').textContent='';status(state.kind==='object'?'Input changed. Paint one object, then reconstruct it.':'Input changed. Estimate a new body to see its result.');drawPhoto()}
function preparation(note){state.preparation=note;$('preparation').textContent=note;$('preparation').hidden=!note}
function rememberJob(j){try{sessionStorage.setItem('sam3d-selected-job',j.id)}catch{}}
async function prepareUpload(file,force,signal){
 const form=new FormData();form.append('image',file);
 const response=await fetch('/api/prepare'+(force?'?force=1':''),{method:'POST',body:form,signal});
 if(!response.ok){let message;try{message=(await response.json()).error}catch{message=response.statusText}throw Error(message||'Photo preparation failed')}
 return {blob:await response.blob(),note:response.headers.get('X-Sam3d-Notice')||'',converted:response.headers.get('X-Sam3d-Converted')==='true'};
}
for(const id of [...boxIDs,...camIDs])$(id).oninput=()=>{detach();controls()};
async function loadImage(blob,name,s=null,epoch=state.epoch,canonicalized=false){const bitmap=await createImageBitmap(blob,{imageOrientation:'from-image',colorSpaceConversion:'none'});if(bitmap.width>32766||bitmap.height>32766||bitmap.width*bitmap.height>16000000||bitmap.width<8||bitmap.height<8){bitmap.close();throw Error('Image must be at most 16 megapixels and at least 8×8.')}
 // Normalize orientation and transparency ONCE in the browser. Upload this
 // exact opaque PNG and display it. FFmpeg output and saved inputs are already
 // canonical PNGs, so preserve their bounded encoding instead of expanding
 // them through another browser PNG encoder.
 let encoded=blob,preview=bitmap,width=bitmap.width,height=bitmap.height;
 if(!canonicalized){const canonical=document.createElement('canvas');canonical.width=width;canonical.height=height;const cx=canonical.getContext('2d');cx.fillStyle='black';cx.fillRect(0,0,width,height);cx.drawImage(bitmap,0,0);bitmap.close();encoded=await new Promise(resolve=>canonical.toBlob(resolve,'image/png'));if(!encoded||encoded.size>20*1024*1024)throw Error('Normalized image exceeds 20 MiB; retrying with FFmpeg.');preview=await createImageBitmap(encoded,{colorSpaceConversion:'none'})}
 if(epoch!==state.epoch){preview.close();return false}
 state.image?.close?.();state.image=preview;state.name=name;state.file=new File([encoded],name.replace(/\.[^.]+$/,'')+'.png',{type:'image/png'});
 photo.width=width;photo.height=height;objectMask.width=width;objectMask.height=height;state.maskPainted=false;$('photo-empty').hidden=true;photo.hidden=false;$('image-size').textContent=`${photo.width} × ${photo.height}`;$('input-name').textContent=name;
 const f=Math.hypot(photo.width,photo.height);setSettings(s||{box:[0,0,photo.width,photo.height],camera:[f,f,photo.width/2,photo.height/2]});drawPhoto();controls();return true;
}
$('upload').onchange=async()=>{
 const file=$('upload').files[0];if(!file)return;
 state.prepareAbort?.abort();detach();const epoch=state.epoch;state.file=null;state.image?.close?.();state.image=null;photo.hidden=true;$('photo-empty').hidden=false;$('image-size').textContent='';[...boxIDs,...camIDs].forEach(id=>$(id).value='');$('input-name').textContent=file.name;preparation('');controls();
 const controller=new AbortController();state.prepareAbort=controller;
 try{
  if(file.size>128*1024*1024)throw Error('Original photo exceeds the 128 MiB conversion limit.');
  status('Uploading and checking photo; local FFmpeg will convert it if needed…');
  let prepared=await prepareUpload(file,false,controller.signal);if(epoch!==state.epoch)return;
  try{if(!await loadImage(prepared.blob,file.name,null,epoch,prepared.converted))return}
  catch(error){
   if(prepared.note)throw Error('FFmpeg conversion returned an image the browser cannot use: '+error.message);
   status('Attempting local FFmpeg resize/re-encode: '+error.message);
   prepared=await prepareUpload(file,true,controller.signal);if(epoch!==state.epoch)return;
   if(!await loadImage(prepared.blob,file.name,null,epoch,prepared.converted))return;
  }
  preparation(prepared.note);status(state.kind==='object'?'Paint over one object, then reconstruct it.':'Drag a box around one person, then estimate their body.');
 }catch(error){if(epoch!==state.epoch)return;state.file=null;preparation(error.message);status(error.message,true);controls()}
 finally{if(state.prepareAbort===controller)state.prepareAbort=null}
};
$('whole').onclick=()=>{detach();const s=settings();s.box=[0,0,photo.width,photo.height];setSettings(s);drawPhoto();controls()};
let drag=null;function pixel(e){const r=photo.getBoundingClientRect();return [Math.round(THREE.MathUtils.clamp((e.clientX-r.left)/r.width*photo.width,0,photo.width)),Math.round(THREE.MathUtils.clamp((e.clientY-r.top)/r.height*photo.height,0,photo.height))]}
function paintMask(p){const radius=Number($('mask-brush').value)/2;objectMaskContext.globalCompositeOperation=$('mask-erase').checked?'destination-out':'source-over';objectMaskContext.fillStyle='#00dffc';objectMaskContext.beginPath();objectMaskContext.arc(p[0],p[1],radius,0,Math.PI*2);objectMaskContext.fill();if(!$('mask-erase').checked)state.maskPainted=true;drawPhoto();controls()}
photo.onpointerdown=e=>{if(!state.image)return;detach();drag=pixel(e);if(state.kind==='object')paintMask(drag);photo.setPointerCapture(e.pointerId)};
photo.onpointermove=e=>{if(!drag)return;const p=pixel(e);if(state.kind==='object'){const radius=Number($('mask-brush').value);objectMaskContext.lineWidth=radius;objectMaskContext.lineCap='round';objectMaskContext.strokeStyle='#00dffc';objectMaskContext.globalCompositeOperation=$('mask-erase').checked?'destination-out':'source-over';objectMaskContext.beginPath();objectMaskContext.moveTo(...drag);objectMaskContext.lineTo(...p);objectMaskContext.stroke();if(!$('mask-erase').checked)state.maskPainted=true;drag=p;drawPhoto();controls();return}const s=settings();s.box=[Math.min(drag[0],p[0]),Math.min(drag[1],p[1]),Math.max(drag[0],p[0]),Math.max(drag[1],p[1])];setSettings(s);drawPhoto();controls()};photo.onpointerup=()=>drag=null;photo.onpointercancel=()=>drag=null;
$('mask-clear').onclick=()=>{detach();objectMaskContext.clearRect(0,0,objectMask.width,objectMask.height);state.maskPainted=false;drawPhoto();controls()};
$('generate').onclick=async()=>{const valid=state.kind==='object'?state.image&&state.file&&state.maskPainted:validSettings();if(!valid||busy()||state.submitting)return;state.submitting=true;const epoch=state.epoch;try{clearResult();status('Uploading selected image…');const form=new FormData();form.append('image',state.file);form.append('preparation_note',state.preparation||'');let path='/api/jobs';if(state.kind==='object'){const blob=await new Promise(resolve=>objectMask.toBlob(resolve,'image/png'));form.append('mask',blob,'mask.png');path='/api/object-jobs'}else{form.append('settings',JSON.stringify(settings()))}const j=await api(path,{method:'POST',body:form});if(epoch===state.epoch){state.job=j;rememberJob(j);status(j.stage);controls()}await refresh()}catch(e){if(epoch===state.epoch)status(e.message,true)}finally{state.submitting=false;controls()}};
$('cancel').onclick=async()=>{if(state.job){try{await api(`/api/jobs/${state.job.id}`,{method:'DELETE'});status('Cancelling native worker…')}catch(e){status(e.message,true)}}};
async function selectJob(j){await tracking?.leave();setKind(j.kind==='object'?'object':'body');const epoch=++state.epoch;state.job=j;rememberJob(j);state.file=null;state.prepareAbort?.abort();preparation(j.preparation_note||'');clearResult();$('upload').value='';status('Loading saved input…');const r=await fetch(`/files/${j.id}/input.png`);if(!r.ok)throw Error('Saved image is unavailable');const blob=await r.blob();if(epoch!==state.epoch)return;await loadImage(blob,j.name,j.kind==='object'?null:(j.settings?.camera?.length?j.settings:null),epoch,true);if(epoch!==state.epoch)return;if(j.kind==='object'){const mr=await fetch(`/files/${j.id}/mask.png`);if(!mr.ok)throw Error('Saved object mask is unavailable');const mb=await createImageBitmap(await mr.blob());objectMaskContext.clearRect(0,0,objectMask.width,objectMask.height);objectMaskContext.drawImage(mb,0,0);mb.close();objectMaskContext.globalCompositeOperation='source-in';objectMaskContext.fillStyle='#00dffc';objectMaskContext.fillRect(0,0,objectMask.width,objectMask.height);objectMaskContext.globalCompositeOperation='source-over';state.maskPainted=true;drawPhoto()}await updateJob(j);renderHistory()}
async function updateJob(j){state.job=j;status(j.error||j.stage,j.state==='failed');controls();if(j.state==='complete'&&!state.result&&state.file){const epoch=state.epoch;if(j.kind==='object'){const r=await fetch(`/files/${j.id}/object.glb`);if(!r.ok)throw Error('Object GLB is unavailable');const b=await r.arrayBuffer();if(epoch!==state.epoch||state.job?.id!==j.id)return;showObject(b)}else{const b=await api(`/files/${j.id}/result.json`);if(epoch!==state.epoch||state.job?.id!==j.id)return;showResult(b)}}updateElapsed()}
function updateElapsed(){const j=state.job;if(!j||!j.started||j.started.startsWith('0001')){$('elapsed').textContent='';return}const end=['complete','failed','cancelled'].includes(j.state)?new Date(j.finished):new Date();const precision=j.kind==='object'?j.precision:(j.precision==='bf16'?'BF16 encoder / F32 decoder':'F32');$('elapsed').textContent=`${Math.max(0,(end-new Date(j.started))/1000).toFixed(1)} seconds · ${j.backend} · ${precision}${busy()?' · worker active':''}`}
function renderHistory(){const root=$('history');root.replaceChildren();if(!state.history.length){const p=document.createElement('p');p.className='small';p.textContent='No saved reconstructions yet.';root.append(p)}for(const j of state.history){const button=document.createElement('button');button.className='history-item'+(state.job?.id===j.id?' active':'');button.dataset.id=j.id;const im=document.createElement('img');im.src=`/files/${j.id}/input.png`;im.alt='';im.loading='lazy';const label=document.createElement('span');label.textContent=j.name;const small=document.createElement('small');small.textContent=`${j.kind==='object'?'object':'body'} · ${j.state} · ${new Date(j.created).toLocaleString()}`;label.append(small);button.append(im,label);button.onclick=()=>selectJob(j).catch(e=>status(e.message,true));root.append(button)}}
let polling=false;async function refresh(){if(polling)return;polling=true;try{state.history=await api('/api/jobs');renderHistory();const j=state.history.find(j=>j.id===state.job?.id);if(j)await updateJob(j)}finally{polling=false}}
tracking=initTracking({api,settings,setSettings,
 mode(active){state.videoActive=active;state.prepareAbort?.abort();detach();state.file=null;for(const id of ['upload','example'])$(id).disabled=active;for(const id of ['photo-upload-options','generate','status','elapsed'])$(id).hidden=active;controls()},
 preview(canvas,name,initial){state.image?.close?.();state.image=canvas;state.name=name;if(photo.width!==canvas.width)photo.width=canvas.width;if(photo.height!==canvas.height)photo.height=canvas.height;photo.hidden=false;$('photo-empty').hidden=true;$('input-name').textContent=name;$('image-size').textContent=`${canvas.width} × ${canvas.height}`;if(initial){const f=Math.hypot(canvas.width,canvas.height);setSettings({box:[0,0,canvas.width,canvas.height],camera:[f,f,canvas.width/2,canvas.height/2]})}drawPhoto()},
 lock(on){for(const id of [...boxIDs,...camIDs,'whole'])$(id).disabled=on;photo.style.pointerEvents=on?'none':''},
 clear:clearResult,
 show(b){if(!nativeGroup||state.result?.schema!==b.schema){
  clearResult();const parts=makeBody(b);nativeGroup=parts.group;mesh=parts.mesh;bones=parts.bones;
  const bounds=new THREE.Box3().setFromBufferAttribute(b.tensors.vertices.length?mesh.geometry.getAttribute('position'):new THREE.Float32BufferAttribute(converted(b.tensors.joints).slice(3),3));bounds.getCenter(frameCentre);frameSize=bounds.getSize(new THREE.Vector3()).length();grid.position.y=bounds.min.y-.006;grid.visible=true;reset();$('viewer-empty').hidden=true;
 }
 state.result=b;const v=b.tensors.vertices,j=b.tensors.joints,pos=mesh.geometry.attributes.position;
 for(let i=0;i<v.length;i++)pos.array[i]=v[i]*(i%3===0?1:-1);pos.needsUpdate=true;mesh.geometry.computeVertexNormals();mesh.geometry.computeBoundingSphere();
 const edges=bones.geometry.attributes.position;let k=0;for(let i=2;i<127;i++)for(const joint of [i,topology.parents[i]])for(let d=0;d<3;d++)edges.array[k++]=j[joint*3+d]*(d===0?1:-1);edges.needsUpdate=true;bones.geometry.computeBoundingSphere();
 const points=bones.children[0].geometry.attributes.position;for(let i=3;i<j.length;i++)points.array[i-3]=j[i]*(i%3===0?1:-1);points.needsUpdate=true;bones.children[0].geometry.computeBoundingSphere();
 mesh.visible=$('mesh').checked&&state.result?.tensors.vertices.length>0;bones.visible=$('skeleton').checked;$('downloads').hidden=true;$('geometry-info').textContent='Video · interpolated display · raw estimates saved unchanged';drawPhoto();window.sam3dQA.rendered=true;
 }});
window.sam3dQA.tracking=tracking.diagnostics;
try {
 const cfg=await api('/api/config');
 state.config=cfg;
 $('model').textContent=`${cfg.model} · ${cfg.backend}`;
 state.ready=true;
 if(cfg.reference) {
  state.referenceManifest=await api('/reference/manifest.json');
  $('example').hidden=false;
  $('example').onclick=async()=>{
   state.prepareAbort?.abort();detach();const epoch=state.epoch;state.file=null;preparation('');controls();$('upload').value='';
   try {
    const response=await fetch('/reference/input.png');
    if(!response.ok)throw Error('Example unavailable');
    const blob=await response.blob();
    if(epoch!==state.epoch)return;
    if(await loadImage(blob,'Official dancing example.png',state.referenceManifest.settings,epoch))
     status('Official example loaded. Estimate with GGML to compare against the original PyTorch body branch.');
   } catch(error) { if(epoch===state.epoch)status(error.message,true) }
  };
 }
 await refresh();
 let selectedID=null;try{selectedID=sessionStorage.getItem('sam3d-selected-job')}catch{}
 if(state.history.length)await selectJob(state.history.find(j=>j.id===selectedID)||state.history[0]);
 controls();
} catch(error) { status(error.message,true) }
setInterval(()=>refresh().catch(e=>status(e.message,true)),1500);setInterval(updateElapsed,250);

$('export-movement').addEventListener('change',()=>{if(state.job)$('skeleton-glb').href=`/api/jobs/${state.job.id}/skeleton.glb?movement=${$('export-movement').value}`});
