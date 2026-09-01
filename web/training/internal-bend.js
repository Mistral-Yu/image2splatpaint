// Progressive-count internal bend path. The shared density allocator adds
// independent rows while each row keeps one owned analytic bend footprint.
(() => {
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function seeded(seed) {let a=seed>>>0;return ()=>{a=(Math.imul(a,1664525)+1013904223)>>>0;return a/4294967296;};}
function defaultControlPointPositions(count) {
  return Array.from({length: count}, (_, index) => (index + 1) / (count + 1));
}
function normalizeControlPointConfig(countValue, positionsValue) {
  const count = Math.max(1, Math.min(6, Math.round(Number(countValue) || 1)));
  const normalizedInput = Array.isArray(positionsValue);
  const parsed = normalizedInput
    ? positionsValue.map(Number)
    : String(positionsValue || "").split(/[\s,;]+/).filter(Boolean).map(Number);
  const positions = parsed.length === count && parsed.every(Number.isFinite)
    ? parsed.map((value) => clamp(normalizedInput ? value : value / 100, 0, 1)).sort((a, b) => a - b)
    : defaultControlPointPositions(count);
  return Object.freeze({count, positions: Object.freeze(positions)});
}
function selectedControlPointConfig() {
  return normalizeControlPointConfig(
    els.flowInternalBendControlPointCount?.value,
    els.flowInternalBendControlPointPositions?.value,
  );
}
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
      // The coverage layer stays straight. Visible paint starts with a bounded,
      // deterministic arc in both directions so the optimizer does not need to
      // escape the zero-curvature saddle before a curved footprint is visible.
      const bendMagnitude=layer===0?0:(layer===1?.15:.22)+rng()*(layer===1?.12:.16);
      const amount=layer===0?.5:.5+(rng()<.5?-bendMagnitude:bendMagnitude);
      rows.push({x,y,sx,sy,theta:layer===0?0:rng()*Math.PI,amount,
        axis:0,family:layer,layer,locked:layer===0,rgb:Array.from(target.slice(p,p+3))});
    }
  }
  return rows;
}
function capacityCatalog(target, width, height, activeRows, capacity, seed = 20260831) {
  const safeCapacity = Math.max(activeRows.length, Math.round(capacity));
  if (safeCapacity === activeRows.length) return activeRows.slice();
  // Capacity metadata must begin with the exact active family/bend catalog.
  // Rebuilding the full 8192-row catalog and copying its prefix made a
  // 128-Splat start entirely family 0, erasing the intended 20/40/40 hierarchy.
  return [
    ...activeRows,
    ...initialize(target, width, height, safeCapacity - activeRows.length, seed ^ 0x9e3779b9),
  ];
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
 compact(keep){this.renderer.ownedBendSession.compact(keep,this.count);this.count=keep.length;}
 relocate(){throw Error('Relocation not implemented: bend-only gate');}
 summary(){return {count:this.count,capacity:this.capacity,backcoat:this.fixedCount,fusion:false,edges:0};}
}

let generation = 0;
function initializeParams(image, count, capacity = count) {
  const rows = initialize(image.rgb, image.width, image.height, count);
  const safeCapacity = Math.max(count, Math.round(capacity));
  const catalog = capacityCatalog(image.rgb, image.width, image.height, rows, safeCapacity);
  const p = initLayeredOpaqueBrush(image, count);
  const fixedBackcoatCount = Math.floor(count * .2);
  const brushMinAspectRatio = selectedBrushMinAspectRatio();
  const brushMaxAspectRatio = selectedBrushMaxAspectRatio();
  const brushAspectFloors = selectedBrushAspectFloors();
  const maximumPlanarScale = selectedLearningRates().maxPlanarScale;
  const layerSettings = discreteLayerSettings();
  const surfaceLayerPrior = scaleBiasedSurfaceLayerPriorSettings();
  const frontFootprintRefinement = harmfulRectangleParentSplitSettings();
  const controlPoints = selectedControlPointConfig();
  Object.assign(p, {
    internalBendKey: `internal-bend-${++generation}`,
    internalBendShapes: Uint32Array.from(rows.flatMap(row => [row.axis, row.family])),
    internalBendCapacity: safeCapacity,
    internalBendCapacityShapes: Uint32Array.from(catalog.flatMap(row => [row.axis, row.family])),
    internalBendControlPoints: new Float32Array(controlPoints.positions),
    kernelShape: "opaque-brush", opaqueLayered: true,
    minimumOpacityEnabled: true, minimumOpacity: .995, maximumOpacity: .995,
    brushWidthTaperEnabled: true, brushWidthTaperStart: 1, brushWidthTaperEnd: 1,
    brushOpacityGradientEnabled: false, brushOpacityGradientStart: 1, brushOpacityGradientEnd: 1,
    brushCenterOpacityGradientMin: 1, brushCenterOpacityGradientMax: 1,
    brushMinAspectRatio, brushMaxAspectRatio,
    virtualDepthEnabled: false, virtualCameraSamplingEnabled: false,
    brushStrokePersistenceEnabled: opaqueBrushStrokePersistenceEnabled(),
    brushRibbonAspectFloor: brushAspectFloors.ribbon,
    brushAccentAspectFloor: brushAspectFloors.accent,
    flowBirthLinksEnabled: true, flowBirthLinkStrength: 0, flowBackcoatCount: fixedBackcoatCount,
    flowTrainingSize: [image.width, image.height],
    tileCullingEnabled: true, layerOrderEnabled: true,
    layerAwareAccumulationEnabled: layerSettings.accumulationEnabled,
    currentVisibilityChildPolicyEnabled: layerSettings.currentVisibilityChildPolicyEnabled,
    currentVisibilityCompactionEnabled: layerSettings.currentVisibilityCompactionEnabled,
    discreteLayersEnabled: layerSettings.enabled,
    discreteLayerCount: layerSettings.count,
    discreteLayerMoveRadius: layerSettings.moveRadius,
    surfaceLayerPriorEnabled: surfaceLayerPrior.enabled,
    surfaceLayerPriorColorAwarePromotion: surfaceLayerPrior.colorAwarePromotion,
    surfaceLayerPriorLayers: surfaceLayerPrior.layers,
    surfaceLayerPriorP1Interval: surfaceLayerPrior.p1Interval,
    surfaceLayerPriorP2Interval: surfaceLayerPrior.p2Interval,
    surfaceLayerPriorP3Interval: surfaceLayerPrior.p3Interval,
    surfaceLayerPriorUntilFraction: surfaceLayerPrior.untilFraction,
    trainLayerColorGuardEnabled: trainLayerColorGuardEnabled(),
    harmfulRectangleParentSplitEnabled: frontFootprintRefinement.enabled,
    harmfulRectangleParentSplitTransitionOnly: false,
    frontSplitChildrenEnabled: false,
    brushLocalColorFlowEnabled: false,
    monochromeUnderpaintingEnabled: false, bg: new Float32Array([0, 0, 0]),
  });
  rows.forEach((row, i) => {
    const sx = i < fixedBackcoatCount ? Math.min(row.sx, maximumPlanarScale) : row.sx;
    const sy = i < fixedBackcoatCount ? Math.min(row.sy, maximumPlanarScale) : row.sy;
    p.xy.set([row.x, row.y], i * 2); p.scale.set([sx, sy], i * 2);
    p.theta[i] = row.theta; p.rgb.set(row.rgb, i * 3); p.opacity[i] = .995;
    p.brushTaper[i] = row.amount; p.detailTags[i] = row.family; p.depthOrder[i] = row.layer / 3;
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
    controlPointPositions: Array.from(params.internalBendControlPoints || [0.5]),
  }, {storageUsage: GPUBufferUsage.STORAGE});
}

function growParams(params, targetCount) {
  const capacity = params.internalBendCapacity || params.count;
  if (targetCount <= params.count || targetCount > capacity) throw Error("Invalid internal bend growth target");
  const next = growParamPlaceholders(params, targetCount);
  next.internalBendKey = params.internalBendKey;
  next.internalBendCapacity = capacity;
  next.internalBendCapacityShapes = params.internalBendCapacityShapes;
  next.internalBendControlPoints = params.internalBendControlPoints;
  const shapes = new Uint32Array(targetCount * 2);
  shapes.set(params.internalBendShapes);
  shapes.set(params.internalBendCapacityShapes.subarray(params.count * 2, targetCount * 2), params.count * 2);
  next.internalBendShapes = shapes;
  return next;
}

function compactParams(params, keepIndices) {
  const next = compactSplatParams(params, keepIndices);
  const shapes = new Uint32Array(keepIndices.length * 2);
  for (let target = 0; target < keepIndices.length; target += 1) {
    const source = keepIndices[target];
    shapes[target * 2] = params.internalBendShapes[source * 2];
    shapes[target * 2 + 1] = params.internalBendShapes[source * 2 + 1];
  }
  next.internalBendShapes = shapes;
  next.internalBendControlPoints = params.internalBendControlPoints;
  return next;
}

function extent(params, index, sx, sy) {
  return Image2SplatPaintInternalBendKernel.conservativeBounds(rowAt(params, index, sx, sy));
}

globalThis.Image2SplatPaintInternalBend = Object.freeze({initializeRows: initialize, initialize: initializeParams,
  normalizeControlPointConfig, selectedControlPointConfig, capacityCatalog,
  growParams, compactParams, FixedCountBendRuntime, createSession, extent});
})();
