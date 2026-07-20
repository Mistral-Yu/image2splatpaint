import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const app = await readFile(new URL("../web/app.js", import.meta.url), "utf8");
const html = await readFile(new URL("../web/index.html", import.meta.url), "utf8");

const thickness = 0.005;
for (const raw of [-100, -4, -1, 0, 1, 4, 100]) {
  const z = thickness * Math.tanh(raw);
  assert.ok(Number.isFinite(z));
  assert.ok(Math.abs(z) <= thickness + 1e-12);
}
assert.equal(thickness * Math.tanh(0), 0);

const contracts = {
  uiOptIn:
    html.includes('data-testid="virtual-bounded-depth"') &&
    html.includes("Bounded virtual depth"),
  gofDensityOptIn:
    html.includes('data-testid="virtual-gof-density"') &&
    html.includes("GOF density ranking") &&
    app.includes("gofDensity: typeof overrides.gofDensity === \"boolean\"") &&
    app.includes("densityGradientMode == 2u && cfg(78u) > 0.5") &&
    app.includes("els.virtualGofDensity.disabled = disabled"),
  independentPositionStorage:
    app.includes("struct SplatPosition { center: vec2<f32>, rawDepth: f32, depthGradient: f32, };") &&
    app.includes("const positions = new Float32Array(params.count * 4)") &&
    app.includes('add("xy-depth", capacity * 4 * 4, true)'),
  boundedSlab:
    app.includes("cfg(68u) * tanh(rawDepth)") &&
    app.includes("let depthChain = cfg(68u) * (1.0 - tanh(nextVirtualDepthRaw) * tanh(nextVirtualDepthRaw))") &&
    app.includes("nextVirtualDepthRaw = clamp(nextVirtualDepthRaw - depthAdam * cfg(71u)"),
  separateOrderAndVirtualDepth:
    app.includes("return virtual_layer_depth(packedTag) + virtual_bounded_depth(rawDepth)") &&
    app.includes("let packedLayer = layerTag + select(0.0, layerOrder * ${LAYER_CODE_RANGE}") &&
    !app.includes("packed_virtual_depth_raw"),
  intervalAccumulation:
    app.includes("accumulatedDepthGradient += gradDepth * normalizer * depthChain") &&
    app.includes("depthObservationCount += 1.0") &&
    app.includes("if (cfg(76u) > 0.5)") &&
    app.includes("accumulatedDepthGradient = 0.0") &&
    app.includes("opt.vTheta.w = depthObservationCount"),
  regularization:
    app.includes("2.0 * cfg(69u) * boundedDepth * depthChain") &&
    app.includes("2.0 * cfg(70u) * spatialWeight * (boundedDepth - neighborDepth) * depthChain"),
  exactBackwardDepth:
    app.includes("dDepth: f32") &&
    app.includes("depth += dWeightedKernel * evaluation.dDepth") &&
    app.includes("add_subtile_gradient(localIndex${this.subgroupExactBackwardEnabled") &&
    app.includes("g, geom, appearance, misc, density, depth)") &&
    app.includes("let gradDepth = densitySum.w"),
  densityLifecycle:
    app.includes("xy[index].rawDepth = 0.0") &&
    app.includes("xy[index].rawDepth = xy[source].rawDepth") &&
    app.includes("xy[g].rawDepth = xy[source].rawDepth") &&
    (app.match(/depthGradient = 0\.0/g) || []).length >= 5,
  readbackLifecycle:
    app.includes("const GPU_READBACK_ROW_BYTES = 12 * 4") &&
    app.includes("params.virtualDepth[i] = params.virtualDepthEnabled ? positions[i * 4 + 2] : 0") &&
    app.includes("virtualDepth: params.virtualDepth ? new Float32Array(params.virtualDepth)") &&
    app.includes("state.metrics.virtual_depth_stats = virtualDepthDistribution(state.params)") &&
    app.includes("virtual_depth_stats: m.virtual_depth_stats ? structuredClone(m.virtual_depth_stats) : null"),
  plyCompatibility:
    app.includes("const PLY_ROW_BYTES = 17 * 4") &&
    app.includes("return layerDepth + boundedVirtualDepth(params, index)") &&
    app.includes("plyLayerDepth(i, params, layerOrderEnabled)") &&
    app.includes("plyContract.row_bytes === 68") &&
    app.includes("row_bytes: rowBytes"),
  offPathHardPlane:
    app.includes("Hard-plane baseline: virtual teachers describe one physical board.") &&
    app.includes("select(virtual_layer_depth(packedTag), 0.0, virtual_tilt_enabled() && cfg(65u) > 0.5)") &&
    app.includes("config[67] = boundedDepthEnabled ? 1 : 0") &&
    app.includes("return Math.abs(delta) > 1e-7 ? delta : a - b"),
  safety:
    app.includes("virtualDepthEnabled: Boolean(params.virtualDepthEnabled)") &&
    app.includes("params?.virtualDepth") &&
    app.includes("assertFiniteParams(params, \"ply-export\")"),
};

assert.ok(Object.values(contracts).every(Boolean), JSON.stringify(contracts));
console.log(JSON.stringify({ ok: true, contracts, bounded_depth_max: thickness }));
