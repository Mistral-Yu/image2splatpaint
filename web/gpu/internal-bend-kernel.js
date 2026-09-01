// Analytic single-Splat bend v2. Shared optimizer/loss/alpha remain unchanged.
// Signal-sRGB; positions and scales use the existing NDC image coordinates.
(() => {
// CPU reference only. Signal-sRGB; positions/scales are NDC, not metric units.
const EXTENT=1.5, BEND_RANGE=1.6;
const FAMILIES=Object.freeze([
  Object.freeze({length:.90,width:.94,shoulder:.10,bias:.02,fixedBend:.025}),
  Object.freeze({length:1,width:.82,shoulder:.17,bias:-.025,fixedBend:.055}),
  Object.freeze({length:.74,width:.68,shoulder:.26,bias:-.08,fixedBend:.035}),
]);
function assertRow(row) {
  if(![0,1].includes(row.axis)||![0,1,2].includes(row.family))throw Error('Missing owned axis/family');
  for(const key of ['x','y','sx','sy','theta','amount'])if(!Number.isFinite(row[key]))throw Error('Non-finite '+key);
  if(row.sx<=0||row.sy<=0||row.amount<0||row.amount>1)throw Error('Invalid scale/bend');
}
function normalizedControlPointPositions(values=[.5]) {
  const positions=Array.from(values||[],Number);
  if(!positions.length||positions.length>6||positions.some(value=>!Number.isFinite(value)||value<0||value>1))
    throw Error('Invalid internal bend control-point positions');
  positions.sort((a,b)=>a-b);
  return Object.freeze(positions);
}
function bendProfileWGSL(values=[.5]) {
  const positions=normalizedControlPointPositions(values);
  if(positions.length===1&&Math.abs(positions[0]-.5)<1e-7)return `
fn owned_bend_profile(t:f32,ell:f32,family:f32)->vec2<f32>{
  return vec2<f32>(ell*ell/3.0-t*t,-2.0*t);
}`;
  const terms=positions.map((position,index)=>{
    const previous=index?positions[index-1]:0;
    const next=index+1<positions.length?positions[index+1]:1;
    const gap=Math.max(.08,Math.min(.45,.55*Math.max(.08,Math.min(position-previous,next-position))));
    return `
  let q${index}=(u-${position.toFixed(8)})/${gap.toFixed(8)};
  let w${index}=exp(-0.5*q${index}*q${index});
  let dw${index}=w${index}*(-q${index}/${gap.toFixed(8)})*du;
  let sign${index}=select(1.0,-1.0,((u32(family)+${index}u)&1u)==1u);
  weighted+=sign${index}*w${index}; weight+=w${index};
  dWeighted+=sign${index}*dw${index}; dWeight+=dw${index};`;
  }).join('');
  return `
fn owned_bend_profile(t:f32,ell:f32,family:f32)->vec2<f32>{
  let safeEll=max(ell,0.000001);let u=clamp(0.5+0.5*t/safeEll,0.0,1.0);let du=0.5/safeEll;
  var weighted=0.0;var weight=0.0;var dWeighted=0.0;var dWeight=0.0;${terms}
  let invWeight=1.0/max(weight,0.000001);let blend=weighted*invWeight;
  let dBlend=(dWeighted*weight-weighted*dWeight)*invWeight*invWeight;
  let v=t/safeEll;let insideSupport=abs(v)<1.0;let envelope=select(0.0,1.0-v*v,insideSupport);
  let dEnvelope=select(0.0,-2.0*t/(safeEll*safeEll),insideSupport);
  let magnitude=ell*ell/3.0;
  return vec2<f32>(magnitude*envelope*blend,magnitude*(dEnvelope*blend+envelope*dBlend));
}`;
}
function conservativeBounds(row,feather=.18) {
  const f=FAMILIES[row.family],L=row.axis===0?row.sx:row.sy,W=row.axis===0?row.sy:row.sx;
  const limit=(1+feather)**.25,tMax=f.length*limit;
  const maxH=Math.max(f.length*f.length/3,Math.abs(f.length*f.length/3-tMax*tMax));
  // The owned footprint adds a pressure envelope to the accepted Brush body.
  // 1.18 bounds its widest center/side bias without narrowing tile coverage.
  const widthBound=1.18*f.width*(1+f.shoulder*limit**2+Math.abs(f.bias)*limit)*limit;
  const fixedBound=f.fixedBend*limit*(1+limit**2);
  const long=EXTENT*L*tMax,trans=EXTENT*(W*(widthBound+fixedBound)+Math.abs(BEND_RANGE*(row.amount-.5))*L*maxH);
  const ex=row.axis===0?long:trans,ey=row.axis===0?trans:long,c=Math.abs(Math.cos(row.theta)),s=Math.abs(Math.sin(row.theta));
  return {x:c*ex+s*ey,y:s*ex+c*ey};
}

// Reuse the accepted Brush body via its WGSL
// string; supply immutable axis/family from row metadata, never infer from scale.
function ownedKernelWGSL(acceptedBrushWGSL) {
  if(!acceptedBrushWGSL.includes('fn illustrative_oil_kernel_sample('))throw Error('Accepted Brush kernel missing');
  return acceptedBrushWGSL+ownedSampleWGSL();
}
// Also usable after an unmodified shared shader already containing the base kernel.
function ownedSampleWGSL(controlPointPositions=[.5]) {
  return bendProfileWGSL(controlPointPositions)+`
struct OwnedBrushSample {
  kernel:f32, dCenter:vec2<f32>, dLogScale:vec2<f32>, dTheta:f32, dAmount:f32,
};
fn owned_brush_sample(delta:vec2<f32>, angle:f32, scale:vec2<f32>, amount:f32,
  axisX:bool, family:f32, feather:f32) -> OwnedBrushSample {
  return owned_brush_sample_cs(delta,cos(angle),sin(angle),scale,amount,axisX,family,feather);
}
fn owned_brush_sample_cs(delta:vec2<f32>, c:f32, s:f32, scale:vec2<f32>, amount:f32,
  axisX:bool, family:f32, feather:f32) -> OwnedBrushSample {
  let r=vec2<f32>(c*delta.x+s*delta.y,-s*delta.x+c*delta.y);
  let n=r/(1.5*scale);
  let ratio=select(scale.y/scale.x,scale.x/scale.y,axisX);
  let t=select(n.y,n.x,axisX);
  let ell=select(select(0.90,1.0,family>0.5),0.74,family>1.5);
  // Reject before the accepted width polynomial degenerates outside support.
  let ownedU=t/ell;let ownedU2=ownedU*ownedU;
  if (ownedU2*ownedU2>=1.0+feather) {
    return OwnedBrushSample(0.0,vec2<f32>(0.0),vec2<f32>(0.0),0.0,0.0);
  }
  let b=1.6*(amount-0.5);let bendProfile=owned_bend_profile(t,ell,family);
  let h=bendProfile.x;let dh=bendProfile.y;
  let shift=b*ratio*h;
  // A curved Splat also carries a smooth pressure profile. Broad rear marks
  // vary gently; middle and accent marks taper more strongly toward their
  // ends. The learned bend scalar adds a small asymmetric pressure component,
  // so curvature and width can evolve together without another GPU buffer.
  let pressureShoulder=select(select(0.20,0.38,family>0.5),0.54,family>1.5);
  let pressureBias=0.22*(amount-0.5);
  let pressure=max(0.34,1.0-pressureShoulder*ownedU2+pressureBias*ownedU);
  let dPressureDt=(-2.0*pressureShoulder*ownedU+pressureBias)/ell;
  let dPressureAmount=0.22*ownedU;
  let transverse=select(n.x,n.y,axisX);
  let warpedTransverse=(transverse-shift)/pressure;
  let warped=select(vec2<f32>(warpedTransverse,t),vec2<f32>(t,warpedTransverse),axisX);
  let base=illustrative_oil_kernel_sample(warped,axisX,feather,family,
    0.0,false,false,1.0,1.0,1.0,1.0,1.0,1.0);
  let baseLong=select(base.gradient.y,base.gradient.x,axisX);
  let baseTransverse=select(base.gradient.x,base.gradient.y,axisX);
  let gTransverse=baseTransverse/pressure;
  let dWarpDt=-b*ratio*dh/pressure-warpedTransverse*dPressureDt/pressure;
  let gLong=baseLong+baseTransverse*dWarpDt;
  let g=select(vec2<f32>(gTransverse,gLong),vec2<f32>(gLong,gTransverse),axisX);
  let gr=g/(1.5*scale); let aspect=-gTransverse*shift;
  let logs=-g*n+select(vec2<f32>(-aspect,aspect),vec2<f32>(aspect,-aspect),axisX);
  let dWarpAmount=-1.6*ratio*h/pressure-warpedTransverse*dPressureAmount/pressure;
  return OwnedBrushSample(base.kernel,vec2<f32>(-c*gr.x+s*gr.y,-s*gr.x-c*gr.y),logs,
    gr.x*r.y-gr.y*r.x,baseTransverse*dWarpAmount);
}
`;
}

// Strict source adapters for one fixed-count, Brush-only session.

const METADATA_BINDING=10;
const metadata=`@group(0) @binding(${METADATA_BINDING}) var<storage,read> ownedShapes:array<vec2<u32>>;`;

// Conservative preflight counts every declaration, including unused entry-point
// resources. Do not request weaker limits or silently drop metadata on mobile.
function assertStorageLimit(source,limit) {
  const count=(source.match(/var<storage\b/g)||[]).length;
  if(!Number.isInteger(limit)||limit<count)throw Error(`Owned shader requires up to ${count} storage buffers; limit ${limit}`);
  return count;
}

function replaceFunction(source,name,replacement) {
  const token=`fn ${name}(`,start=source.indexOf(token);
  if(start<0||source.indexOf(token,start+token.length)>=0)throw Error(`Expected one ${name}`);
  const open=source.indexOf('{',start);let depth=1,end=open+1;
  for(;depth&&end<source.length;end++) {if(source[end]==='{')depth++;if(source[end]==='}')depth--;}
  if(depth)throw Error(`Unclosed ${name}`);
  return source.slice(0,start)+replacement+source.slice(end);
}
function replaceCount(source,pattern,replacement,expected,label) {
  let count=0;const result=source.replace(pattern,(...args)=>{count++;return typeof replacement==='function'?replacement(...args):replacement;});
  if(count!==expected)throw Error(`${label}: expected ${expected}, got ${count}`);
  return result;
}
function attach(source,{kernel=true,controlPointPositions=[.5]}={}) {
  if(/@binding\(\s*10\s*\)/.test(source)||/var[^;]*ownedShapes:/.test(source))throw Error('Metadata binding collision / already adapted');
  if(!source.includes('fn illustrative_oil_kernel_sample('))throw Error('Accepted Brush body missing');
  return source+'\n'+metadata+(kernel?ownedSampleWGSL(controlPointPositions):'');
}
function adaptTrainingShaders(original,{controlPointPositions=[.5]}={}) {
  const result={...original};
  result.renderShader=replaceFunction(original.renderShader,'training_kernel',`fn training_kernel(
 d:vec2<f32>,c:f32,s:f32,scale:vec2<f32>,packedTag:f32,amount:f32,shape:vec2<u32>
) -> f32 {
 return owned_brush_sample_cs(d,c,s,scale,amount,shape.x==0u,f32(shape.y),clamp(cfg(41u),0.01,0.49)).kernel;
}`);
  result.renderShader=replaceCount(result.renderShader,/(training_kernel\([^\n]+t\.w, xy\[g\]\.rawDepth)\)/g,
    (_,prefix)=>prefix+', ownedShapes[g])',5,'linear forward samples');
  result.renderShader=replaceCount(result.renderShader,/tileSharedPackedOrder\[j\], tileSharedTaper\[j\]\)/g,
    'tileSharedPackedOrder[j], tileSharedTaper[j], ownedShapes[tileSharedIndex[j]])',5,'cooperative forward samples');
  result.renderShader=attach(result.renderShader,{controlPointPositions});
  result.exactBackwardShader=replaceFunction(original.exactBackwardShader,'kernel_sample',`fn kernel_sample(
 d:vec2<f32>,c:f32,s:f32,baseScale:vec2<f32>,sampleScale:vec2<f32>,includeMipGradient:bool,
 packedTag:f32,amount:f32,shape:vec2<u32>
) -> KernelSample {
 let q=owned_brush_sample_cs(d,c,s,sampleScale,amount,shape.x==0u,f32(shape.y),clamp(cfg(41u),0.01,0.49));
 if(q.kernel<=0.00000001){return KernelSample(0.0,vec2<f32>(0.0),vec2<f32>(0.0),0.0,0.0);}
 // The shared evaluator expects log-kernel derivatives, multiplying by kernel
 // before EWA averaging. Apply the MIP chain to BOTH normalization and bend aspect.
 let inv=1.0/max(q.kernel,0.000001);
 let ratio=(baseScale*baseScale)/max(sampleScale*sampleScale,vec2<f32>(0.00000001));
 return KernelSample(q.kernel,q.dCenter*inv,q.dLogScale*ratio*inv,q.dTheta*inv,q.dAmount*inv);
}`);
  result.exactBackwardShader=replaceCount(result.exactBackwardShader,/(kernel_sample\([^\n]+t\.w, rawDepth)\);/g,
    (_,prefix)=>prefix+', ownedShapes[g]);',10,'backward samples');
  result.exactBackwardShader=attach(result.exactBackwardShader,{controlPointPositions});
  // Optimizer, alpha, loss, sorting and all other returned shader strings intact.
  return result;
}
function adaptPreviewShader(source,{controlPointPositions=[.5]}={}) {
  source=replaceFunction(source,'preview_kernel',`fn preview_kernel(
 d:vec2<f32>,c:f32,s:f32,scale:vec2<f32>,packedTag:f32,amount:f32,worldPoint:vec2<f32>,shape:vec2<u32>
) -> f32 {
 return owned_brush_sample_cs(d,c,s,scale,amount,shape.x==0u,f32(shape.y),0.18).kernel;
}`);
  source=source.replace(/xy\[i\]\.rawDepth, (p(?:00|10|01|11)?)\)/g,
    'xy[i].rawDepth, $1, ownedShapes[i])');
  if((source.match(/ownedShapes\[i\]/g)||[]).length!==5)throw Error('Preview samples changed');
  return attach(source,{controlPointPositions});
}
const boundsWGSL=`
fn owned_local_bounds(scale:vec2<f32>,amount:f32,shape:vec2<u32>,feather:f32)->vec2<f32>{
 let axisX=shape.x==0u;let family=shape.y;
 let ell=select(select(0.90,1.0,family==1u),0.74,family==2u);
 let w=select(select(0.94,0.82,family==1u),0.68,family==2u);
 let shoulder=select(select(0.10,0.17,family==1u),0.26,family==2u);
 let bias=select(select(0.02,0.025,family==1u),0.08,family==2u);
 let fixed=select(select(0.025,0.055,family==1u),0.035,family==2u);
 let limit=pow(1.0+feather,0.25);let tMax=ell*limit;
 let maxH=max(ell*ell/3.0,abs(ell*ell/3.0-tMax*tMax));
 let widthBound=w*(1.0+shoulder*limit*limit+bias*limit)*limit;
 let fixedBound=fixed*limit*(1.0+limit*limit);
 let L=select(scale.y,scale.x,axisX);let W=select(scale.x,scale.y,axisX);
 let longitudinal=1.5*L*tMax;
 let transverse=1.5*(W*(1.18*widthBound+fixedBound)+abs(1.6*(amount-0.5))*L*maxH);
 return select(vec2<f32>(transverse,longitudinal),vec2<f32>(longitudinal,transverse),axisX);
}`;
function adaptTileShader(source) {
  source=replaceCount(source,/let boundsScale = select\(effective, effective \* finite_paint_extent\(g\), finitePaint\);/g,
    'let boundsScale = owned_local_bounds(effective,xy[g].rawDepth,ownedShapes[g],clamp(cfg(41u),0.01,0.49));',1,'tile bounds');
  source=replaceCount(source,/let boundsSigma = select\([^;]+;/g,'let boundsSigma = 1.0;',1,'finite support');
  // The straight ellipse may reject an occupied bent corner. Keep tile AABB
  // filtering; capacity/overflow detection and painter order remain unchanged.
  if(source.includes('fn tile_intersects_footprint('))source=replaceFunction(source,'tile_intersects_footprint',
    'fn tile_intersects_footprint(g:u32,tx:u32,ty:u32)->bool{return true;}');
  return attach(source,{kernel:false})+boundsWGSL;
}

// Immutable per-Splat axis/family metadata, shared by training and preview.
function assertOwnedShape(row) {
  if(![0,1].includes(row?.axis)||![0,1,2].includes(row?.family))throw Error('Missing owned axis/family');
}

class OwnedMetadata {
  #catalog; #rows; #buffer=null; #disposed=false;
  constructor(rows) {
    if(!rows.length)throw Error('Empty owned shape session');
    this.#catalog=rows.map(row=>{assertOwnedShape(row);return Object.freeze({axis:row.axis,family:row.family});});
    this.#rows=this.#catalog.slice();
  }
  get count(){return this.#rows.length;}
  assertCompatible(rows) {
    if(this.#disposed)throw Error('Owned metadata disposed; fresh session required');
    if(rows.length>this.count)throw Error('Owned metadata capacity exceeded');
    rows.forEach((row,i)=>{
      assertOwnedShape(row);
      if(row.axis!==this.#rows[i].axis||row.family!==this.#rows[i].family)throw Error('Owned shape changed');
    });
  }
  packed() {
    if(this.#disposed)throw Error('Owned metadata disposed');
    return Uint32Array.from(this.#rows.flatMap(row=>[row.axis,row.family]));
  }
  upload(device,usage) {
    if(this.#disposed||this.#buffer)throw Error('Fresh metadata upload required');
    // Pass GPUBufferUsage.STORAGE explicitly; immutable mapped-at-creation data.
    // No per-iteration queue.writeBuffer and no rawDepth/layer/tag overloading.
    const data=this.packed();
    const buffer=device.createBuffer({label:'owned-axis-family-v2',size:data.byteLength,usage,mappedAtCreation:true});
    try {new Uint32Array(buffer.getMappedRange()).set(data);buffer.unmap();}
    catch(error){buffer.destroy();throw error;}
    this.#buffer=buffer;return buffer;
  }
  compact(device,usage,keepIndices,oldCount) {
    if(this.#disposed||!this.#buffer)throw Error('Owned metadata not live');
    if(!keepIndices?.length||keepIndices.length>=oldCount||oldCount>this.count)
      throw Error('Invalid owned metadata compaction');
    let previous=-1;
    const kept=[];
    for(const index of keepIndices) {
      if(index<=previous||index<0||index>=oldCount)throw Error('Owned metadata keep order changed');
      kept.push(this.#rows[index]);previous=index;
    }
    // Active rows follow the same stable keep order as parameter compaction.
    // Inactive capacity slots return to their deterministic catalog entry so
    // later GPU growth can append rows without reallocating metadata.
    const next=this.#catalog.slice();
    for(let i=0;i<kept.length;i++)next[i]=kept[i];
    const data=Uint32Array.from(next.flatMap(row=>[row.axis,row.family]));
    const buffer=device.createBuffer({label:'owned-axis-family-v2-compacted',size:data.byteLength,usage,mappedAtCreation:true});
    try {new Uint32Array(buffer.getMappedRange()).set(data);buffer.unmap();}
    catch(error){buffer.destroy();throw error;}
    const previousBuffer=this.#buffer;
    this.#rows=next;this.#buffer=buffer;previousBuffer.destroy();
  }
  entries(stage,entries) {
    if(this.#disposed||!this.#buffer)throw Error('Owned metadata not live');
    if(!['render','exact-backward','preview','tile'].includes(stage))throw Error('Unsupported metadata stage');
    if(entries.some(e=>e.binding===METADATA_BINDING))throw Error('Owned metadata binding collision');
    return [...entries,{binding:METADATA_BINDING,resource:{buffer:this.#buffer}}];
  }
  dispose(){if(this.#disposed)return;this.#buffer?.destroy();this.#buffer=null;this.#disposed=true;}
}

// Deliberately narrower than product controls: same sRGB/alpha contract and no
// silently ignored opacity/taper/depth/shape modes. Call before generating shaders.
function assertFixtureContract(config) {
  const expected={kernel:'owned-brush-v2',fixedCount:false,virtual:false,split:true,clone:true,
    fusion:false,directionalTaper:false,opacityGradientMin:1,opacityGradientMax:1,
    centerOpacityMin:1,centerOpacityMax:1,feather:.18,coarseLoss:false};
  for(const [key,value] of Object.entries(expected))if(config[key]!==value)throw Error(`Unsupported owned fixture config: ${key}`);
  return true;
}

// A caller-owned shared renderer/device, not an alternative GPU backend.

const pipelineStages={
  pipeline:'preview',renderStatePipeline:'render',tileCooperativeRenderPipeline:'render',
  exactAlphaBackwardPipeline:'exact-backward',sourceDomainBackwardPipeline:'exact-backward',
  tileCountPipeline:'tile',tileFillPipeline:'tile',
};

class OwnedHostSession {
  #metadata; #renderer=null; #closed=false; #closing=false; #storageUsage; #controlPointPositions;
  constructor(rows,config,{storageUsage}={}) {
    assertFixtureContract(config);
    if(storageUsage!==128)throw Error('Owned metadata requires storage-only usage');
    this.#metadata=new OwnedMetadata(rows);this.#storageUsage=storageUsage;
    this.#controlPointPositions=normalizedControlPointPositions(config.controlPointPositions||[.5]);
  }
  get count(){return this.#metadata.count;}
  attach(renderer) {
    if(this.#closed||this.#renderer)throw Error('Fresh owned session required');
    if(renderer.trainState||renderer.resultRenderState||renderer.vertexBuffer||
      Object.entries(renderer).some(([key,value])=>(key==='pipeline'||key.endsWith('Pipeline'))&&value))
      throw Error('Fresh renderer required; cached pipelines cannot be reused');
    const limits=renderer.device.limits,bytes=this.count*8;
    if(!Number.isInteger(limits?.maxStorageBuffersPerShaderStage)||limits.maxStorageBuffersPerShaderStage<9)
      throw Error('Owned exact backward requires at least 9 storage buffers');
    for(const key of ['maxStorageBufferBindingSize','maxBufferSize'])
      if(!Number.isFinite(limits[key])||bytes>limits[key])throw Error(`Owned metadata exceeds ${key}`);
    this.#metadata.upload(renderer.device,this.#storageUsage);
    this.#renderer=renderer;
  }
  #live(renderer=this.#renderer) {
    if(this.#closed||this.#closing||!renderer||renderer!==this.#renderer)
      throw Error('Owned session not live on this renderer');
    for(const value of [renderer.trainState,renderer.resultRenderState])
      if(value&&value.count>this.count)throw Error('Owned metadata capacity exceeded');
    return renderer;
  }
  assertRows(rows){this.#live();this.#metadata.assertCompatible(rows);}
  compact(keepIndices,oldCount) {
    const renderer=this.#live();
    this.#metadata.compact(renderer.device,this.#storageUsage,keepIndices,oldCount);
    renderer.trainState?.bindGroupCache?.clear?.();
  }
  #check(source){assertStorageLimit(source,this.#live().device.limits.maxStorageBuffersPerShaderStage);return source;}
  training(factory,receiver,options) {
    this.#live(receiver);
    // The product factory reads subgroup flags from `this`; do not detach it.
    const shaders=adaptTrainingShaders(factory.create.call(receiver,options),{
      controlPointPositions:this.#controlPointPositions,
    });
    this.#check(shaders.renderShader);this.#check(shaders.exactBackwardShader);
    return shaders; // All optimizer/loss strings remain unmodified.
  }
  preview(factory){return this.#check(adaptPreviewShader(factory.renderPreview(),{
    controlPointPositions:this.#controlPointPositions,
  }));}
  tile(factory,options){return this.#check(adaptTileShader(factory.create(options)));}
  createBindGroup(renderer,pipeline,descriptor) {
    this.#live(renderer);
    if(!pipeline)throw Error('Owned bind group requires a pipeline');
    const stages=Object.entries(pipelineStages).filter(([key])=>renderer[key]===pipeline).map(([,stage])=>stage);
    if(new Set(stages).size>1)throw Error('Ambiguous owned pipeline');
    const stage=stages[0];
    // In particular, optimizer binding 10 is tile control, not metadata.
    // Tile prefix/sort do not read the owned footprint; auto layout omits it.
    return renderer.device.createBindGroup(stage
      ? {...descriptor,entries:this.#metadata.entries(stage,descriptor.entries)}
      : descriptor);
  }
  dispose() {
    if (this.#renderer?.trainState || this.#renderer?.resultRenderState) throw Error("Release owned surfaces first");
    this.#metadata.dispose(); this.#closed = true;
  }
  async close() {
    if(this.#closed)return;
    if(this.#closing)throw Error('Owned session close already pending');
    const renderer=this.#live();
    if(renderer.trainState||renderer.resultRenderState)
      throw Error('Release training AND cached result before owned metadata');
    this.#closing=true;
    try {
      await renderer.device.queue.onSubmittedWorkDone();
    } finally {
      // Device-loss rejection must still release local ownership; propagate
      // the fence error so cleanup is not misreported as successful GPU work.
      this.#metadata.dispose();this.#closed=true;this.#closing=false;
    }
  }
}


 globalThis.Image2SplatPaintInternalBendKernel = Object.freeze({OwnedHostSession, conservativeBounds, adaptTrainingShaders, adaptPreviewShader, adaptTileShader, ownedSampleWGSL});
})();
