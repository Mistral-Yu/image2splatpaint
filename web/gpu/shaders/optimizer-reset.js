(function installOptimizerResetShaderFactory(global) {
  function create() {
    return `
struct AdamState {
  mGeom: vec4<f32>,
  vGeom: vec4<f32>,
  mColor: vec4<f32>,
  vColor: vec4<f32>,
  mTheta: vec4<f32>,
  vTheta: vec4<f32>,
};
@group(0) @binding(0) var<storage, read> config: array<f32>;
@group(0) @binding(1) var<storage, read_write> control: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> adam: array<AdamState>;
const SOURCE_MASK = 0x3fffffffu;
const ROLE_SOURCE_MASK = 0x60000000u;

fn reset_state(adcResetStep: f32) -> AdamState {
  return AdamState(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0, 0.0, 0.0, adcResetStep), vec4<f32>(0.0));
}

@compute @workgroup_size(64)
fn reset_selected(@builtin(global_invocation_id) id: vec3u) {
  let g = id.x;
  let oldCount = u32(config[2]);
  let targetCount = u32(config[3]);
  let capacity = u32(config[10]);
  let mode = u32(config[11]);
  var destination = g;
  var selection = capacity + g;
  if (mode == 1u) {
    if (g >= targetCount - oldCount) { return; }
    destination = oldCount + g;
  } else if (g >= oldCount) {
    return;
  }

  let encoded = atomicLoad(&control[selection]);
  if ((encoded & SOURCE_MASK) == 0u) { return; }
  let adcResetStep = select(0.0, config[36], mode == 3u);
  adam[destination] = reset_state(adcResetStep);
}

@compute @workgroup_size(64)
fn reset_sources(@builtin(global_invocation_id) id: vec3u) {
  let source = id.x;
  let oldCount = u32(config[2]);
  let mode = u32(config[11]);
  if (source >= oldCount) { return; }
  let packed = atomicLoad(&control[source]);
  let roleSelected = (packed & ROLE_SOURCE_MASK) != 0u;
  let legacySelected = select(packed > 0u, (packed >> 16u) > 0u, mode == 1u);
  let selected = select(legacySelected, roleSelected, config[42] > 0.5);
  if (!selected) { return; }
  let adcResetStep = select(0.0, config[36], mode == 3u);
  adam[source] = reset_state(adcResetStep);
}`;
  }

  global.Image2SplatPaintOptimizerResetShader = Object.freeze({ create });
})(globalThis);
