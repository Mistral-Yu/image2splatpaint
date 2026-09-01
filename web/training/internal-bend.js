// Progressive-count internal bend path. The shared density allocator adds
// independent rows while each row keeps one owned analytic bend footprint.
(() => {
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function seeded(seed) {let a=seed>>>0;return ()=>{a=(Math.imul(a,1664525)+1013904223)>>>0;return a/4294967296;};}
function initialize(target,width,height,count=512,seed=20260831) {
  const rng=seeded(seed),rows=[];
  const budgets=[Math.floor(count*.2),Math.floor(count*.4)];budgets.push(count-budgets[0]-budgets[1]);
  for(let layer=0;layer<3;layer++) {
    const n=budgets[layer],cols=Math.max(1,Math.round(Math.sqrt(n*width/height))),nr=Math.ceil(n/cols);
    for(let i=0;i<n;i++) {
      // Stratified, source-independent locations/directions. No curve extraction.
      const x=clamp(-1+2*((i%cols)+.5+(rng()-.5)*.45)/cols,-1,1);
      const y=clamp(-1+2*(Math.floor(i/cols)+.5+(rng()-.5)*.45)/nr,-1,1);
      const p=(Math.round((y+1)*(height-1)/2)*width+Math.round((x+1)*(width-1)/2))*3;
      const size=.8+.4*rng(),sx=2/cols*size*(layer===0?1.10:.95),sy=2/nr*size*(layer===0?1.10:.32+.18*rng());
      rows.push({x,y,sx,sy,theta:layer===0?0:rng()*Math.PI,amount:.5,
        axis:0,family:layer===0?0:1,layer,locked:layer===0,rgb:Array.from(target.slice(p,p+3))});
    }
  }
  return rows;
}
// Backcoat protection only. No graph, connections, fusion or topology changes.
const shader=`struct Frozen{p:vec4<f32>,t:vec4<f32>,c:vec4<f32>};
@group(0) @binding(0) var<storage,read> frozen:array<Frozen>;
@group(0) @binding(1) var<storage,read_write> xy:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read_write> transform:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read_write> color:array<vec4<f32>>;
@compute @workgroup_size(64) fn restore(@builtin(global_invocation_id) id:vec3<u32>){
let i=id.x;if(i>=arrayLength(&frozen)){return;}xy[i]=frozen[i].p;transform[i]=frozen[i].t;color[i]=frozen[i].c;}`;
class FixedCountBendRuntime{
 constructor(renderer,params){this.renderer=renderer;this.capacity=params.internalBendCapacity||params.count;this.count=params.count;this.fixedCount=params.flowBackcoatCount||0;this.frozen=null;if(this.fixedCount){const p=packPositions(params),t=packTransforms(params),c=packColors(params),data=new Float32Array(this.fixedCount*12);for(let i=0;i<this.fixedCount;i++){data.set(p.subarray(i*4,i*4+4),i*12);data.set(t.subarray(i*4,i*4+4),i*12+4);data.set(c.subarray(i*4,i*4+4),i*12+8);}this.frozen=renderer.device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});renderer.device.queue.writeBuffer(this.frozen,0,data);}}
 buffers(){return this.frozen?[this.frozen]:[];}
 async prepare(image,p){if(p.count>this.capacity)throw Error('Internal bend capacity exceeded');this.count=p.count;if(!this.frozen||this.pipeline)return;const d=this.renderer.device,module=d.createShaderModule({code:shader});const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');if(errors.length)throw Error(errors.map(m=>m.message).join('\n'));this.pipeline=await d.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'restore'}});}
 encode(){} // All image gradients go directly to the existing per-Splat Adam.
 restore(encoder,front){if(!this.frozen)return;const r=this.renderer,bind=r.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[this.frozen,r.trainState.xyBuffers[front],r.trainState.transformBuffers[front],r.trainState.colorBuffers[front]].map((buffer,binding)=>({binding,resource:{buffer}}))});const pass=encoder.beginComputePass();pass.setPipeline(this.pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(Math.ceil(this.fixedCount/64));pass.end();}
 restoreNow(){if(!this.frozen)return;const r=this.renderer,e=r.device.createCommandEncoder();for(let i=0;i<r.trainState.xyBuffers.length;i++)this.restore(e,i);r.device.queue.submit([e.finish()]);}
 grow(oldCount,newCount){if(oldCount!==this.count||newCount>this.capacity)throw Error('Internal bend growth contract changed');this.count=newCount;}
 compact(){throw Error('Pruning not implemented: bend-only gate');}
 relocate(){throw Error('Relocation not implemented: bend-only gate');}
 summary(){return {count:this.count,capacity:this.capacity,backcoat:this.fixedCount,fusion:false,edges:0};}
}

let generation = 0;
function initializeParams(image, count, capacity = count) {
  const rows = initialize(image.rgb, image.width, image.height, count);
  const safeCapacity = Math.max(count, Math.round(capacity));
  const catalog = initialize(image.rgb, image.width, image.height, safeCapacity);
  for (let i = 0; i < rows.length; i++) {
    rows[i].axis = catalog[i].axis;
    rows[i].family = catalog[i].family;
  }
  const p = initLayeredOpaqueBrush(image, count);
  Object.assign(p, {
    internalBendKey: `internal-bend-${++generation}`,
    internalBendShapes: Uint32Array.from(rows.flatMap(row => [row.axis, row.family])),
    internalBendCapacity: safeCapacity,
    internalBendCapacityShapes: Uint32Array.from(catalog.flatMap(row => [row.axis, row.family])),
    kernelShape: "opaque-brush", opaqueLayered: true,
    minimumOpacityEnabled: true, minimumOpacity: .995, maximumOpacity: .995,
    brushWidthTaperEnabled: true, brushWidthTaperStart: 1, brushWidthTaperEnd: 1,
    brushOpacityGradientEnabled: false, brushOpacityGradientStart: 1, brushOpacityGradientEnd: 1,
    brushCenterOpacityGradientMin: 1, brushCenterOpacityGradientMax: 1,
    brushMinAspectRatio: 1, brushMaxAspectRatio: 8,
    virtualDepthEnabled: false, virtualCameraSamplingEnabled: false,
    flowBirthLinksEnabled: true, flowBirthLinkStrength: 0, flowBackcoatCount: Math.floor(count * .2),
    flowTrainingSize: [image.width, image.height],
    tileCullingEnabled: true, layerOrderEnabled: true, surfaceLayerPriorEnabled: false,
    brushLocalColorFlowEnabled: false, brushStrokePersistenceEnabled: false,
    discreteLayersEnabled: false, discreteLayerMoveRadius: 0,
    monochromeUnderpaintingEnabled: false, bg: new Float32Array([0, 0, 0]),
  });
  rows.forEach((row, i) => {
    p.xy.set([row.x, row.y], i * 2); p.scale.set([row.sx, row.sy], i * 2);
    p.theta[i] = row.theta; p.rgb.set(row.rgb, i * 3); p.opacity[i] = .995;
    p.brushTaper[i] = .5; p.detailTags[i] = 1; p.depthOrder[i] = row.layer / 3;
  });
  return p;
}

function rowAt(p, i, sx = p.scale[i * 2], sy = p.scale[i * 2 + 1]) {
  if (p.internalBendShapes?.length !== p.count * 2) throw Error("Missing internal bend axis/family metadata");
  return {x: p.xy[i * 2], y: p.xy[i * 2 + 1], sx, sy, theta: p.theta[i],
    amount: p.brushTaper[i], axis: p.internalBendShapes[i * 2], family: p.internalBendShapes[i * 2 + 1]};
}

function createSession(params) {
  const shapes = params.internalBendCapacityShapes || params.internalBendShapes;
  const rows = Array.from({length: shapes.length / 2}, (_, i) => ({axis: shapes[i * 2], family: shapes[i * 2 + 1]}));
  return new Image2SplatPaintInternalBendKernel.OwnedHostSession(rows, {
    kernel: "owned-brush-v2", fixedCount: false, virtual: false, split: true, clone: true,
    fusion: false, directionalTaper: false, opacityGradientMin: 1, opacityGradientMax: 1,
    centerOpacityMin: 1, centerOpacityMax: 1, feather: .18, coarseLoss: false,
  }, {storageUsage: GPUBufferUsage.STORAGE});
}

function growParams(params, targetCount) {
  const capacity = params.internalBendCapacity || params.count;
  if (targetCount <= params.count || targetCount > capacity) throw Error("Invalid internal bend growth target");
  const next = growParamPlaceholders(params, targetCount);
  next.internalBendKey = params.internalBendKey;
  next.internalBendCapacity = capacity;
  next.internalBendCapacityShapes = params.internalBendCapacityShapes;
  next.internalBendShapes = params.internalBendCapacityShapes.slice(0, targetCount * 2);
  return next;
}

function extent(params, index, sx, sy) {
  return Image2SplatPaintInternalBendKernel.conservativeBounds(rowAt(params, index, sx, sy));
}

globalThis.Image2SplatPaintInternalBend = Object.freeze({initializeRows: initialize, initialize: initializeParams,
  growParams, FixedCountBendRuntime, createSession, extent});
})();
