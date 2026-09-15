// Display-only interpolation. Never changes the saved raw estimates and never
// extrapolates through a tracking gap, restart or beyond the last sample.
export function bracket(samples, time) {
 if(!samples.length)return null;
 let lo=0,hi=samples.length-1;
 while(lo<hi){const m=Math.ceil((lo+hi)/2);if(samples[m].time<=time)lo=m;else hi=m-1}
 const next=Math.min(lo+1,samples.length-1),dt=samples[next].time-samples[lo].time;
 return [lo,next,dt>0?Math.max(0,Math.min(1,(time-samples[lo].time)/dt)):0];
}
export function decodeFrame(buffer,faces) {
 if(buffer.byteLength===16+(127*8+3)*4 && new TextDecoder().decode(new Uint8Array(buffer,0,8))==='S3DSKL01')return decodeSkeletonFrame(buffer);
 const floats=18439*3+127*3+3, size=16+floats*4;
 if(buffer.byteLength!==size&&buffer.byteLength!==size+36874*3*4)throw Error('Invalid video frame size');
 const view=new DataView(buffer);
 if(new TextDecoder().decode(new Uint8Array(buffer,0,8))!=='S3DTRK01')throw Error('Invalid video frame version');
 const time=view.getFloat64(8,true);
 if(!Number.isFinite(time)||time<0)throw Error('Invalid video timestamp');
 const v=new Float32Array(floats);for(let i=0;i<floats;i++){v[i]=view.getFloat32(16+i*4,true);if(!Number.isFinite(v[i])||Math.abs(v[i])>100)throw Error('Nonfinite/extreme video geometry')}
 if(buffer.byteLength>size){faces=new Uint32Array(36874*3);for(let i=0;i<faces.length;i++)faces[i]=view.getUint32(size+i*4,true)}
 if(!faces||faces.length!==36874*3||!faces.every(x=>Number.isInteger(x)&&x<18439))throw Error('Invalid video topology');
 return {time,body:{schema:'sam3d.body.track.v1',faces,tensors:{vertices:v.subarray(0,18439*3),joints:v.subarray(18439*3,18439*3+127*3),camera_translation:v.subarray(floats-3)}}};
}
export function mixBodies(a,b,alpha,out) {
 if(!Number.isFinite(alpha)||alpha<0||alpha>1)throw Error('Invalid interpolation');
 for(const key of ['vertices','joints','camera_translation']){
  const av=a.tensors[key],bv=b.tensors[key];
  let dest=out.tensors[key];if(!dest||dest.length!==av.length)dest=out.tensors[key]=new Float32Array(av.length);
  for(let i=0;i<av.length;i++)dest[i]=av[i]+(bv[i]-av[i])*alpha;
 }
 return out;
}
// Projected joints only guide the next crop, never fit/deform the skeleton.
// Bounded changes avoid a single poor frame throwing the crop off the image.
export function followBox(body,s,w,h) {
 const j=body.tensors.joints,t=body.tensors.camera_translation,c=s.camera,points=[];
 for(let i=1;i<127;i++){const z=j[3*i+2]+t[2];if(z<=.05)return s;points.push([(j[3*i]+t[0])/z*c[0]+c[2],(j[3*i+1]+t[1])/z*c[1]+c[3]])}
 const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
 if(![x0,x1,y0,y1].every(Number.isFinite)||x1-x0<8||y1-y0<8||x0< -w||x1>2*w||y0< -h||y1>2*h)return s;
 const want=[Math.max(0,x0-(x1-x0)*.2),Math.max(0,y0-(y1-y0)*.15),Math.min(w,x1+(x1-x0)*.2),Math.min(h,y1+(y1-y0)*.15)];
 const box=s.box.map((v,i)=>v+Math.max(-(i%2?h:w)*.05,Math.min((i%2?h:w)*.05,(want[i]-v)*.25)));
 if(box[2]-box[0]<8||box[3]-box[1]<8)return s;
 return {camera:[...c],box};
}

export function decodeSkeletonFrame(buffer){
 if(buffer.byteLength!==16+(127*8+3)*4)throw Error('Invalid skeleton frame size');
 const view=new DataView(buffer);if(new TextDecoder().decode(new Uint8Array(buffer,0,8))!=='S3DSKL01')throw Error('Invalid skeleton frame version');
 const time=view.getFloat64(8,true);if(!Number.isFinite(time)||time<0)throw Error('Invalid skeleton timestamp');
 const joints=new Float32Array(127*3),camera=new Float32Array(3);
 for(let j=0;j<127;j++){
  let norm=0;for(let k=0;k<8;k++){const x=view.getFloat32(16+(j*8+k)*4,true);if(!Number.isFinite(x))throw Error('Nonfinite skeleton');if(k<3){if(Math.abs(x)>10000)throw Error('Extreme skeleton');joints[j*3+k]=x*.01*(k===0?1:-1)}else if(k<7)norm+=x*x;else if(x<1e-5||x>1e5)throw Error('Invalid skeleton scale')}
  if(Math.abs(norm-1)>.01)throw Error('Invalid skeleton rotation');
 }
 for(let k=0;k<3;k++){camera[k]=view.getFloat32(16+(127*8+k)*4,true);if(!Number.isFinite(camera[k])||Math.abs(camera[k])>100)throw Error('Invalid skeleton camera')}
 return {time,body:{schema:'sam3d.body.skeleton.v1',faces:new Uint32Array(),tensors:{vertices:new Float32Array(),joints,camera_translation:camera}}};
}
