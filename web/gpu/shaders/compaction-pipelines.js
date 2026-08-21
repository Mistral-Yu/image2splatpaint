(function installCompactionShaderFactories(global) {
  function parameterShader() {
    return `
struct CompactConfig { oldCount: u32, newCount: u32, capacity: u32, _padding: u32, };
struct SplatPosition { center: vec2<f32>, rawDepth: f32, depthGradient: f32, };
@group(0) @binding(0) var<uniform> config: CompactConfig;
@group(0) @binding(1) var<storage, read> keepIndices: array<u32>;
@group(0) @binding(2) var<storage, read> sourceXy: array<SplatPosition>;
@group(0) @binding(3) var<storage, read> sourceTransform: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> sourceColor: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> outputXy: array<SplatPosition>;
@group(0) @binding(6) var<storage, read_write> outputTransform: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> outputColor: array<vec4<f32>>;

@compute @workgroup_size(64)
fn compact_parameters(@builtin(global_invocation_id) id: vec3u) {
  let destination = id.x;
  if (destination >= config.newCount) { return; }
  let source = keepIndices[destination];
  if (source >= config.oldCount) { return; }
  outputXy[destination] = sourceXy[source];
  outputTransform[destination] = sourceTransform[source];
  outputColor[destination] = sourceColor[source];
}`;
  }

  function stateShader() {
    return `
struct CompactConfig { oldCount: u32, newCount: u32, capacity: u32, _padding: u32, };
struct AdamState {
  mGeom: vec4<f32>,
  vGeom: vec4<f32>,
  mColor: vec4<f32>,
  vColor: vec4<f32>,
  mTheta: vec4<f32>,
  vTheta: vec4<f32>,
};
@group(0) @binding(0) var<uniform> config: CompactConfig;
@group(0) @binding(1) var<storage, read> keepIndices: array<u32>;
@group(0) @binding(2) var<storage, read> sourceAdam: array<AdamState>;
@group(0) @binding(3) var<storage, read> sourceStats: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> outputAdam: array<AdamState>;
@group(0) @binding(5) var<storage, read_write> outputStats: array<vec4<f32>>;

@compute @workgroup_size(64)
fn compact_state(@builtin(global_invocation_id) id: vec3u) {
  let destination = id.x;
  if (destination >= config.newCount) { return; }
  let source = keepIndices[destination];
  if (source >= config.oldCount) { return; }
  outputAdam[destination] = sourceAdam[source];
  outputStats[destination] = sourceStats[source];
  outputStats[config.newCount + destination] = sourceStats[config.capacity + source];
}`;
  }

  global.Image2SplatPaintCompactionShaders = Object.freeze({ parameterShader, stateShader });
})(globalThis);
