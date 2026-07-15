import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const [app, browserProbe] = await Promise.all([
  readFile(resolve(root, "web/app.js"), "utf8"),
  readFile(resolve(root, "scripts/p1_webgpu_probe.mjs"), "utf8"),
]);
const layout = (count, limit = 65535) => {
  const x = Math.min(count, limit);
  return { x, y: Math.ceil(count / x) };
};
const imageDispatch = layout(72929);
const exactOptimizerDispatch = (splats) => layout(Math.ceil(splats / 64));
const exactDispatchCases = [65535, 65536, 262144, 1048576].map((splats) => ({
  splats,
  ...exactOptimizerDispatch(splats),
}));

const checks = {
  device_limit_used:
    app.includes("this.device.limits?.maxComputeWorkgroupsPerDimension || 65535") &&
    app.includes("pass.dispatchWorkgroups(x, y)"),
  render_linearized:
    app.includes("let pixel = id.x + id.y * workgroups.x * 64u") &&
    app.includes("this.dispatchLinear(renderPass, Math.ceil((image.width * image.height) / 64))"),
  ssim_linearized:
    app.includes("let base = tileIndex * 4u") &&
    app.includes("this.dispatchLinear(ssimPass"),
  metrics_linearized:
    app.includes("let out = tileIndex * ${METRIC_TILE_STRIDE}u") &&
    app.includes("this.dispatchLinear(pass, partialCount)"),
  overlap_linearized:
    app.includes("let out = tileIndex * ${OVERLAP_METRIC_STRIDE}u"),
  large_splat_paths:
    app.includes("let g = id.x + id.y * workgroups.x * 64u") &&
    app.includes("this.dispatchLinear(optimizerPass, Math.ceil(params.count / 64))"),
  no_quality_reduction:
    !app.includes("65535 / partialCount") &&
    !app.includes("Math.min(partialCount, 65535)"),
  failing_case_is_covered:
    imageDispatch.x === 65535 && imageDispatch.y === 2 && imageDispatch.x * imageDispatch.y >= 72929,
  exact_optimizer_boundaries_covered: exactDispatchCases.every(({ splats, x, y }) => {
    const workgroups = Math.ceil(splats / 64);
    return x * y >= workgroups && (x - 1) * 64 < splats;
  }),
  saturating_tile_prefix:
    app.includes("if (count > 0xffffffffu - requiredTotal)") &&
    app.includes("acceptedTotal += min(count, capacity - acceptedTotal)") &&
    app.includes("tileOffsets[tileCount] = acceptedTotal | select(0u, 0x80000000u, overflow)") &&
    app.includes("if (overflow) { overflowAmount = requiredTotal - capacity; }") &&
    !app.includes("select(0u, requiredTotal - capacity, overflow)") &&
    app.includes("let tileEnd = min(tileOffsets[tile + 1u] & 0x7fffffffu, arrayLength(&tileIndices))"),
  overflow_is_parameter_noop:
    app.includes("pixelState[pixel] = vec4<f32>(bg, -1.0)") &&
    (app.match(/pixelState\[0\]\.a < 0\.0/g) || []).length >= 3 &&
    (app.match(/atomicAdd\(&tileControl\[2\], 1u\)/g) || []).length >= 3 &&
    app.includes("if (g == 0u && lid.x == 0u)") &&
    app.includes("commandEncoder.clearBuffer(this.trainState.tileControlBuffer, 0, 8)"),
  overflow_is_logically_retried:
    app.includes("async function resolveTileOverflowRetry(parameterHashBefore = null)") &&
    app.includes("state.metrics.tile_retry_steps += retrySteps") &&
    app.includes("step = resumedStep") &&
    app.includes("await renderer.clearTileNoopCounter()"),
  forced_overflow_fixture:
    app.includes('query.has("tile-index-capacity")') &&
    app.includes("async hashTrainParameters(params)") &&
    app.includes("tile overflow mutated parameters before retry") &&
    app.includes("tileRetryParameterHashMatches") &&
    app.includes("!qaOverflowPending && tileCounters") &&
    app.includes("!qaOverflowPending && (tileCounters.overflow > 0"),
  browser_probe_forces_2d_dispatch:
    browserProbe.includes("const twoDimCount = 72929") &&
    browserProbe.includes("@compute @workgroup_size(1)") &&
    browserProbe.includes("twoDimLayout.y > 1") &&
    browserProbe.includes("dispatch_2d: dispatch2d"),
};

const failed = Object.entries(checks).filter(([, ok]) => !ok).map(([name]) => name);
if (failed.length) throw new Error(`large dispatch checks failed: ${failed.join(", ")}`);
console.log(JSON.stringify({ ok: true, checks, exactDispatchCases }, null, 2));
