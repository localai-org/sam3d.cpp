import {mixBodies} from './tracking-math.js';

function copyBody(source,dest){
 dest.faces=source.faces;
 for(const key of ['vertices','joints','camera_translation']){
  const values=source.tensors[key];
  if(dest.tensors[key]?.length!==values.length)dest.tensors[key]=new Float32Array(values.length);
  dest.tensors[key].set(values);
 }
 return dest;
}
const body=()=>({schema:'sam3d.body.track.v1',faces:null,tensors:{}});

// No delayed source timeline, prediction or backlog. A new result starts a
// short blend from the CURRENT displayed pose, not from the previous target.
export class LivePresentation {
 constructor(duration=25){
  if(!Number.isFinite(duration)||duration<0||duration>50)throw Error('Invalid live blend duration');
  this.duration=duration;this.reset();
 }
 reset(){this.from=body();this.current=body();this.target=null;this.started=0;this.finished=true}
 push(target,now){
  if(!Number.isFinite(now))throw Error('Invalid presentation time');
  if(this.target&&now-this.started<750){this.sample(now);copyBody(this.current,this.from);this.finished=false}
  else{copyBody(target,this.from);copyBody(target,this.current);this.finished=true}
  this.target=target;this.started=now;
 }
 sample(now){
  if(!Number.isFinite(now))throw Error('Invalid presentation time');
  if(!this.target)return null;
  const alpha=this.finished||!this.duration?1:Math.max(0,Math.min(1,(now-this.started)/this.duration));
  if(alpha===1){copyBody(this.target,this.current);this.finished=true}
  else mixBodies(this.from,this.target,alpha,this.current);
  return {body:this.current,alpha,finished:this.finished};
 }
}
