import {strict as assert} from 'node:assert';
import {LivePresentation} from '../demo/web/live-presentation.js';
const body=x=>({faces:new Uint32Array([0,0,0]),tensors:{vertices:new Float32Array([x]),joints:new Float32Array([x]),camera_translation:new Float32Array([x])}});
const value=(p,t)=>p.sample(t).body.tensors.vertices[0];
const p=new LivePresentation(25),first=body(0),second=body(10),third=body(-10);
assert.equal(p.sample(0),null);p.push(first,100);assert.equal(value(p,100),0);assert.ok(p.finished);
p.push(second,200);assert.equal(value(p,200),0);
assert.equal(value(p,205),2);assert.equal(value(p,212.5),5);
// Retarget a blend already in progress, without jumping to its old target.
p.push(third,212.5);assert.equal(value(p,212.5),5);
assert.equal(value(p,225),-2.5);assert.equal(value(p,237.5),-10);assert.ok(p.finished);
assert.equal(first.tensors.vertices[0],0);assert.equal(second.tensors.vertices[0],10);
for(let t=240;t<1000;t+=10)assert.equal(value(p,t),-10); // no extrapolation
p.push(second,2000);assert.equal(value(p,2000),10); // gaps snap, not slow recovery
p.reset();p.push(first,3000);assert.equal(value(p,3000),0);
const instant=new LivePresentation(0);instant.push(first,0);instant.push(second,1);assert.equal(value(instant,1),10);
for(const duration of [-1,51,NaN,Infinity])assert.throws(()=>new LivePresentation(duration));
// Step response begins on the next render and reaches target in <=25 ms,
// independent of the inference/sample period or the preceding cold load.
for(const interval of [100,140,333]){
 const p=new LivePresentation();p.push(first,0);p.push(second,interval);
 assert.ok(value(p,interval+1)>0);assert.equal(value(p,interval+25),10);
}
console.log('Immediate live step response, interrupted blends, exact settling, gaps and no-extrapolation tests passed.');
