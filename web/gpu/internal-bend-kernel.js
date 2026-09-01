// Analytic single-Splat bend v2. Shared optimizer/loss/alpha remain unchanged.
// Signal-sRGB; positions and scales use the existing NDC image coordinates.
(() => {
// CPU reference only. Signal-sRGB; positions/scales are NDC, not metric units.
const EXTENT=1.5, BEND_RANGE=1.2;
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
function conservativeBounds(row,feather=.18) {
  const f=FAMILIES[row.family],L=row.axis===0?row.sx:row.sy,W=row.axis===0?row.sy:row.sx;
  const limit=(1+feather)**.25,tMax=f.length*limit;
  const maxH=Math.max(f.length*f.length/3,Math.abs(f.length*f.length/3-tMax*tMax));
  const widthBound=f.width*(1+f.shoulder*limit**2+Math.abs(f.bias)*limit)*limit;
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
function ownedSampleWGSL() {
  return `
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
  let b=1.2*(amount-0.5); let h=ell*ell/3.0-t*t;
  let shift=b*ratio*h;
  let warped=n-select(vec2<f32>(shift,0.0),vec2<f32>(0.0,shift),axisX);
  let base=illustrative_oil_kernel_sample(warped,axisX,feather,family,
    0.0,false,false,1.0,1.0,1.0,1.0,1.0,1.0);
  let gm=select(base.gradient.x,base.gradient.y,axisX);
  let slope=gm*2.0*b*ratio*t;
  let g=base.gradient+select(vec2<f32>(0.0,slope),vec2<f32>(slope,0.0),axisX);
  let gr=g/(1.5*scale); let aspect=-gm*shift;
  let logs=-g*n+select(vec2<f32>(-aspect,aspect),vec2<f32>(aspect,-aspect),axisX);
  return OwnedBrushSample(base.kernel,vec2<f32>(-c*gr.x+s*gr.y,-s*gr.x-c*gr.y),logs,
    gr.x*r.y-gr.y*r.x,-gm*1.2*ratio*h);
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
function attach(source,{kernel=true}={}) {
  if(/@binding\(\s*10\s*\)/.test(source)||/var[^;]*ownedShapes:/.test(source))throw Error('Metadata binding collision / already adapted');
  if(!source.includes('fn illustrative_oil_kernel_sample('))throw Error('Accepted Brush body missing');
  return source+'\n'+metadata+(kernel?ownedSampleWGSL():'');
}
function adaptTrainingShaders(original) {
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
  result.renderShader=attach(result.renderShader);
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
  result.exactBackwardShader=attach(result.exactBackwardShader);
  // Optimizer, alpha, loss, sorting and all other returned shader strings intact.
  return result;
}
function adaptPreviewShader(source) {
  source=replaceFunction(source,'preview_kernel',`fn preview_kernel(
 d:vec2<f32>,c:f32,s:f32,scale:vec2<f32>,packedTag:f32,amount:f32,worldPoint:vec2<f32>,shape:vec2<u32>
) -> f32 {
 return owned_brush_sample_cs(d,c,s,scale,amount,shape.x==0u,f32(shape.y),0.18).kernel;
}`);
  source=source.replace(/xy\[i\]\.rawDepth, (p(?:00|10|01|11)?)\)/g,
    'xy[i].rawDepth, $1, ownedShapes[i])');
  if((source.match(/ownedShapes\[i\]/g)||[]).length!==5)throw Error('Preview samples changed');
  return attach(source);
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
 let transverse=1.5*(W*(widthBound+fixedBound)+abs(1.2*(amount-0.5))*L*maxH);
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
class OwnedMetadata {
  #rows; #buffer=null; #disposed=false;
  constructor(rows) {
    if(!rows.length)throw Error('Empty owned shape session');
    this.#rows=rows.map(row=>{assertRow(row);return Object.freeze({axis:row.axis,family:row.family});});
  }
  get count(){return this.#rows.length;}
  assertCompatible(rows) {
    if(this.#disposed)throw Error('Owned metadata disposed; fresh session required');
    if(rows.length!==this.count)throw Error('Split / clone / compaction not enabled in fixed-count gate');
    rows.forEach((row,i)=>{
      assertRow(row);
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
  const expected={kernel:'owned-brush-v2',fixedCount:true,virtual:false,split:false,clone:false,
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
  #metadata; #renderer=null; #closed=false; #closing=false; #storageUsage;
  constructor(rows,config,{storageUsage}={}) {
    assertFixtureContract(config);
    if(storageUsage!==128)throw Error('Owned metadata requires storage-only usage');
    this.#metadata=new OwnedMetadata(rows);this.#storageUsage=storageUsage;
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
      if(value&&value.count!==this.count)throw Error('Owned fixed-count session changed count');
    return renderer;
  }
  assertRows(rows){this.#live();this.#metadata.assertCompatible(rows);}
  #check(source){assertStorageLimit(source,this.#live().device.limits.maxStorageBuffersPerShaderStage);return source;}
  training(factory,receiver,options) {
    this.#live(receiver);
    // The product factory reads subgroup flags from `this`; do not detach it.
    const shaders=adaptTrainingShaders(factory.create.call(receiver,options));
    this.#check(shaders.renderShader);this.#check(shaders.exactBackwardShader);
    return shaders; // All optimizer/loss strings remain unmodified.
  }
  preview(factory){return this.#check(adaptPreviewShader(factory.renderPreview()));}
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
