(function installMetricShaderFactories(global) {
  function trainingResidualMap() {
    return `
struct Config { values: array<vec4<f32>, 8>, };
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> targetRgb: array<f32>;
@group(0) @binding(2) var<storage, read> pixelState: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> errorMap: array<f32>;
fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }
fn mean_rgb(value: vec3<f32>) -> f32 {
  return dot(value, vec3<f32>(1.0 / 3.0));
}
fn target_mean(pixel: u32) -> f32 {
  let base = pixel * 3u;
  return (targetRgb[base] + targetRgb[base + 1u] + targetRgb[base + 2u]) / 3.0;
}

@compute @workgroup_size(64)
fn update_training_residual(@builtin(global_invocation_id) id: vec3<u32>) {
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let pixel = id.x;
  if (pixel >= width * height) { return; }
  let px = pixel % width;
  let py = pixel / width;
  let rendered = pixelState[pixel].rgb;
  let targetBase = pixel * 3u;
  let targetColor = vec3<f32>(
    targetRgb[targetBase],
    targetRgb[targetBase + 1u],
    targetRgb[targetBase + 2u]
  );
  let loss = (
    abs(rendered.r - targetColor.r) +
    abs(rendered.g - targetColor.g) +
    abs(rendered.b - targetColor.b)
  ) / 3.0;
  var gradientError = 0.0;
  var gradientCount = 0.0;
  let renderedMean = mean_rgb(rendered);
  let targetMean = mean_rgb(targetColor);
  if (px + 1u < width) {
    gradientError += abs(
      (mean_rgb(pixelState[pixel + 1u].rgb) - renderedMean) -
      (target_mean(pixel + 1u) - targetMean)
    );
    gradientCount += 1.0;
  }
  if (py + 1u < height) {
    gradientError += abs(
      (mean_rgb(pixelState[pixel + width].rgb) - renderedMean) -
      (target_mean(pixel + width) - targetMean)
    );
    gradientCount += 1.0;
  }
  errorMap[pixel] = loss + select(
    0.0,
    0.2 * gradientError / max(1.0, gradientCount),
    cfg(20u) > 0.5
  );
}`;
  }

  function pixelMetrics() {
    return `
struct Config { values: array<vec4<f32>, 8>, };
struct AlphaState { compositeAlpha: f32, acceptedEnd: u32, pad0: f32, pad1: u32, };
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> targetRgb: array<f32>;
@group(0) @binding(2) var<storage, read> pixelState: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> metricsOut: array<f32>;
@group(0) @binding(4) var<storage, read> ssimData: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> targetAlpha: array<f32>;
@group(0) @binding(6) var<storage, read> alphaState: array<AlphaState>;
var<workgroup> wgLoss: array<f32, 64>;
var<workgroup> wgSquaredError: array<f32, 64>;
var<workgroup> wgX: array<f32, 64>;
var<workgroup> wgY: array<f32, 64>;
var<workgroup> wgX2: array<f32, 64>;
var<workgroup> wgY2: array<f32, 64>;
var<workgroup> wgXY: array<f32, 64>;
var<workgroup> wgMax: array<f32, 64>;
var<workgroup> wgCount: array<f32, 64>;
var<workgroup> wgCoverage: array<f32, 64>;
var<workgroup> wgCoverageMin: array<f32, 64>;
var<workgroup> wgCoverageUnder: array<f32, 64>;
var<workgroup> wgBackgroundExposure: array<f32, 64>;
var<workgroup> wgGradientError: array<f32, 64>;
var<workgroup> wgTargetGradientEnergy: array<f32, 64>;
var<workgroup> wgGradientCount: array<f32, 64>;
var<workgroup> wgAlphaError: array<f32, 64>;
var<workgroup> wgAlphaDark: array<vec4<f32>, 64>;
var<workgroup> wgAlphaMid: array<vec4<f32>, 64>;
var<workgroup> wgAlphaLight: array<vec4<f32>, 64>;
var<workgroup> wgAlphaMoments: array<vec4<f32>, 64>;
var<workgroup> wgAlphaCross: array<f32, 64>;
var<workgroup> wgSsim: array<vec2<f32>, 64>;
fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }

@compute @workgroup_size(64)
fn metrics(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>
) {
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let tileCols = (width + 7u) / 8u;
  let tileIndex = wid.y * workgroups.x + wid.x;
  let tileCount = tileCols * ((height + 7u) / 8u);
  if (tileIndex >= tileCount) { return; }
  let tileX = tileIndex % tileCols;
  let tileY = tileIndex / tileCols;
  let px = tileX * 8u + lid.x % 8u;
  let py = tileY * 8u + lid.x / 8u;
  var loss = 0.0;
  var squaredError = 0.0;
  var x = 0.0;
  var y = 0.0;
  var valid = 0.0;
  var coverage = 0.0;
  var coverageUnder = 0.0;
  var backgroundExposure = 0.0;
  var gradientError = 0.0;
  var targetGradientEnergy = 0.0;
  var gradientCount = 0.0;
  var alphaError = 0.0;
  var alphaDark = vec4<f32>(0.0);
  var alphaMid = vec4<f32>(0.0);
  var alphaLight = vec4<f32>(0.0);
  var alphaMoments = vec4<f32>(0.0);
  var alphaCross = 0.0;
  var ssim = vec2<f32>(0.0);
  if (px < width && py < height) {
    let pixel = py * width + px;
    let rendered = pixelState[pixel].rgb;
    alphaError = abs(alphaState[pixel].compositeAlpha - targetAlpha[pixel]);
    coverage = pixelState[pixel].a;
    alphaMoments = vec4<f32>(coverage, targetAlpha[pixel], coverage * coverage, targetAlpha[pixel] * targetAlpha[pixel]);
    alphaCross = coverage * targetAlpha[pixel];
    let ssimChannels = ssimData[pixel * 4u + 3u];
    ssim = vec2<f32>((ssimChannels.r + ssimChannels.g + ssimChannels.b) / 3.0, ssimChannels.a);
    coverageUnder = select(0.0, 1.0, coverage < cfg(22u));
    backgroundExposure = select(0.0, 1.0, coverage < ${DEFAULT_ALPHA_TARGET});
    let targetIndex = pixel * 3u;
    let targetColor = vec3<f32>(targetRgb[targetIndex], targetRgb[targetIndex + 1u], targetRgb[targetIndex + 2u]);
    let residual = rendered - targetColor;
    loss = (abs(rendered.r - targetColor.r) + abs(rendered.g - targetColor.g) + abs(rendered.b - targetColor.b)) / 3.0;
    squaredError = dot(residual, residual);
    x = dot(rendered, vec3<f32>(1.0 / 3.0));
    y = dot(targetColor, vec3<f32>(1.0 / 3.0));
    let alphaBucket = vec4<f32>(coverage, alphaError, backgroundExposure, 1.0);
    if (y < 0.25) {
      alphaDark = alphaBucket;
    } else if (y < 0.75) {
      alphaMid = alphaBucket;
    } else {
      alphaLight = alphaBucket;
    }
    if (px + 1u < width) {
      let rightPixel = pixel + 1u;
      let rightRendered = dot(pixelState[rightPixel].rgb, vec3<f32>(1.0 / 3.0));
      let rightTargetIndex = rightPixel * 3u;
      let rightTarget = (targetRgb[rightTargetIndex] + targetRgb[rightTargetIndex + 1u] + targetRgb[rightTargetIndex + 2u]) / 3.0;
      gradientError += abs((rightRendered - x) - (rightTarget - y));
      targetGradientEnergy += abs(rightTarget - y);
      gradientCount += 1.0;
    }
    if (py + 1u < height) {
      let downPixel = pixel + width;
      let downRendered = dot(pixelState[downPixel].rgb, vec3<f32>(1.0 / 3.0));
      let downTargetIndex = downPixel * 3u;
      let downTarget = (targetRgb[downTargetIndex] + targetRgb[downTargetIndex + 1u] + targetRgb[downTargetIndex + 2u]) / 3.0;
      gradientError += abs((downRendered - x) - (downTarget - y));
      targetGradientEnergy += abs(downTarget - y);
      gradientCount += 1.0;
    }
    valid = 1.0;
  }
  wgLoss[lid.x] = loss;
  wgSquaredError[lid.x] = squaredError;
  wgX[lid.x] = x;
  wgY[lid.x] = y;
  wgX2[lid.x] = x * x;
  wgY2[lid.x] = y * y;
  wgXY[lid.x] = x * y;
  wgMax[lid.x] = loss;
  wgCount[lid.x] = valid;
  wgCoverage[lid.x] = coverage;
  wgCoverageMin[lid.x] = select(1000000000.0, coverage, valid > 0.5);
  wgCoverageUnder[lid.x] = coverageUnder;
  wgBackgroundExposure[lid.x] = backgroundExposure;
  wgGradientError[lid.x] = gradientError;
  wgTargetGradientEnergy[lid.x] = targetGradientEnergy;
  wgGradientCount[lid.x] = gradientCount;
  wgAlphaError[lid.x] = alphaError;
  wgAlphaDark[lid.x] = alphaDark;
  wgAlphaMid[lid.x] = alphaMid;
  wgAlphaLight[lid.x] = alphaLight;
  wgAlphaMoments[lid.x] = alphaMoments;
  wgAlphaCross[lid.x] = alphaCross;
  wgSsim[lid.x] = ssim;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lid.x < stride) {
      wgLoss[lid.x] += wgLoss[lid.x + stride];
      wgSquaredError[lid.x] += wgSquaredError[lid.x + stride];
      wgX[lid.x] += wgX[lid.x + stride];
      wgY[lid.x] += wgY[lid.x + stride];
      wgX2[lid.x] += wgX2[lid.x + stride];
      wgY2[lid.x] += wgY2[lid.x + stride];
      wgXY[lid.x] += wgXY[lid.x + stride];
      wgMax[lid.x] = max(wgMax[lid.x], wgMax[lid.x + stride]);
      wgCount[lid.x] += wgCount[lid.x + stride];
      wgCoverage[lid.x] += wgCoverage[lid.x + stride];
      wgCoverageMin[lid.x] = min(wgCoverageMin[lid.x], wgCoverageMin[lid.x + stride]);
      wgCoverageUnder[lid.x] += wgCoverageUnder[lid.x + stride];
      wgBackgroundExposure[lid.x] += wgBackgroundExposure[lid.x + stride];
      wgGradientError[lid.x] += wgGradientError[lid.x + stride];
      wgTargetGradientEnergy[lid.x] += wgTargetGradientEnergy[lid.x + stride];
      wgGradientCount[lid.x] += wgGradientCount[lid.x + stride];
      wgAlphaError[lid.x] += wgAlphaError[lid.x + stride];
      wgAlphaDark[lid.x] += wgAlphaDark[lid.x + stride];
      wgAlphaMid[lid.x] += wgAlphaMid[lid.x + stride];
      wgAlphaLight[lid.x] += wgAlphaLight[lid.x + stride];
      wgAlphaMoments[lid.x] += wgAlphaMoments[lid.x + stride];
      wgAlphaCross[lid.x] += wgAlphaCross[lid.x + stride];
      wgSsim[lid.x] += wgSsim[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    let out = tileIndex * ${METRIC_TILE_STRIDE}u;
    metricsOut[out] = wgLoss[0];
    metricsOut[out + 1u] = wgX[0];
    metricsOut[out + 2u] = wgY[0];
    metricsOut[out + 3u] = wgX2[0];
    metricsOut[out + 4u] = wgY2[0];
    metricsOut[out + 5u] = wgXY[0];
    metricsOut[out + 6u] = wgMax[0];
    metricsOut[out + 7u] = wgCount[0];
    metricsOut[out + 8u] = wgCoverage[0];
    metricsOut[out + 9u] = wgCoverageMin[0];
    metricsOut[out + 10u] = wgCoverageUnder[0];
    metricsOut[out + 11u] = wgBackgroundExposure[0];
    metricsOut[out + 12u] = wgGradientError[0];
    metricsOut[out + 13u] = wgTargetGradientEnergy[0];
    metricsOut[out + 14u] = wgGradientCount[0];
    metricsOut[out + 15u] = wgAlphaError[0];
    metricsOut[out + 16u] = wgAlphaDark[0].x;
    metricsOut[out + 17u] = wgAlphaDark[0].y;
    metricsOut[out + 18u] = wgAlphaDark[0].z;
    metricsOut[out + 19u] = wgAlphaDark[0].w;
    metricsOut[out + 20u] = wgAlphaMid[0].x;
    metricsOut[out + 21u] = wgAlphaMid[0].y;
    metricsOut[out + 22u] = wgAlphaMid[0].z;
    metricsOut[out + 23u] = wgAlphaMid[0].w;
    metricsOut[out + 24u] = wgAlphaLight[0].x;
    metricsOut[out + 25u] = wgAlphaLight[0].y;
    metricsOut[out + 26u] = wgAlphaLight[0].z;
    metricsOut[out + 27u] = wgAlphaLight[0].w;
    metricsOut[out + 28u] = wgAlphaMoments[0].x;
    metricsOut[out + 29u] = wgAlphaMoments[0].y;
    metricsOut[out + 30u] = wgAlphaMoments[0].z;
    metricsOut[out + 31u] = wgAlphaMoments[0].w;
    metricsOut[out + 32u] = wgAlphaCross[0];
    metricsOut[out + 33u] = wgSquaredError[0];
    metricsOut[out + 34u] = wgSsim[0].x;
    metricsOut[out + 35u] = wgSsim[0].y;
  }
}`;
  }

  function virtualCameraMetrics() {
    return `
struct Config { values: array<vec4<f32>, 19>, };
struct AlphaState { compositeAlpha: f32, acceptedEnd: u32, pad0: f32, pad1: u32, };
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> targetRgb: array<f32>;
@group(0) @binding(2) var<storage, read> pixelState: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> metricsOut: array<f32>;
@group(0) @binding(4) var<storage, read> targetAlpha: array<f32>;
@group(0) @binding(5) var<storage, read> alphaState: array<AlphaState>;
@group(0) @binding(6) var<storage, read> ssimData: array<vec4<f32>>;
var<workgroup> wgLoss: array<f32, 64>;
var<workgroup> wgSquaredError: array<f32, 64>;
var<workgroup> wgX: array<f32, 64>;
var<workgroup> wgY: array<f32, 64>;
var<workgroup> wgX2: array<f32, 64>;
var<workgroup> wgY2: array<f32, 64>;
var<workgroup> wgXY: array<f32, 64>;
var<workgroup> wgCount: array<f32, 64>;
var<workgroup> wgAlphaL1: array<f32, 64>;
var<workgroup> wgAlphaX: array<f32, 64>;
var<workgroup> wgAlphaY: array<f32, 64>;
var<workgroup> wgAlphaX2: array<f32, 64>;
var<workgroup> wgAlphaY2: array<f32, 64>;
var<workgroup> wgAlphaXY: array<f32, 64>;
var<workgroup> wgCoverage: array<f32, 64>;
var<workgroup> wgBackground: array<f32, 64>;
var<workgroup> wgRenderedChroma: array<f32, 64>;
var<workgroup> wgTargetChroma: array<f32, 64>;
var<workgroup> wgSsim: array<vec2<f32>, 64>;
fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }
${VIRTUAL_TILT_WGSL}

fn target_rgb_at(point: vec2<f32>, width: u32, height: u32) -> vec3<f32> {
  let source = clamp((point * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u)), vec2<f32>(0.0), vec2<f32>(f32(width - 1u), f32(height - 1u)));
  let p0 = vec2<u32>(floor(source));
  let p1 = min(p0 + vec2<u32>(1u), vec2<u32>(width - 1u, height - 1u));
  let f = fract(source);
  let i00 = (p0.y * width + p0.x) * 3u;
  let i10 = (p0.y * width + p1.x) * 3u;
  let i01 = (p1.y * width + p0.x) * 3u;
  let i11 = (p1.y * width + p1.x) * 3u;
  let c00 = vec3<f32>(targetRgb[i00], targetRgb[i00 + 1u], targetRgb[i00 + 2u]);
  let c10 = vec3<f32>(targetRgb[i10], targetRgb[i10 + 1u], targetRgb[i10 + 2u]);
  let c01 = vec3<f32>(targetRgb[i01], targetRgb[i01 + 1u], targetRgb[i01 + 2u]);
  let c11 = vec3<f32>(targetRgb[i11], targetRgb[i11 + 1u], targetRgb[i11 + 2u]);
  return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

fn target_alpha_at(point: vec2<f32>, width: u32, height: u32) -> f32 {
  let source = clamp((point * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u)), vec2<f32>(0.0), vec2<f32>(f32(width - 1u), f32(height - 1u)));
  let p0 = vec2<u32>(floor(source));
  let p1 = min(p0 + vec2<u32>(1u), vec2<u32>(width - 1u, height - 1u));
  let f = fract(source);
  return mix(mix(targetAlpha[p0.y * width + p0.x], targetAlpha[p0.y * width + p1.x], f.x), mix(targetAlpha[p1.y * width + p0.x], targetAlpha[p1.y * width + p1.x], f.x), f.y);
}

@compute @workgroup_size(64)
fn metrics(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>
) {
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let tileCols = (width + 7u) / 8u;
  let tileIndex = wid.y * workgroups.x + wid.x;
  let tileCount = tileCols * ((height + 7u) / 8u);
  if (tileIndex >= tileCount) { return; }
  let tileX = tileIndex % tileCols;
  let tileY = tileIndex / tileCols;
  let px = tileX * 8u + lid.x % 8u;
  let py = tileY * 8u + lid.x / 8u;
  var loss = 0.0;
  var squaredError = 0.0;
  var x = 0.0;
  var y = 0.0;
  var valid = 0.0;
  var alphaL1 = 0.0;
  var alphaX = 0.0;
  var alphaY = 0.0;
  var coverage = 0.0;
  var background = 0.0;
  var renderedChroma = 0.0;
  var targetChroma = 0.0;
  var ssim = vec2<f32>(0.0);
  if (px < width && py < height) {
    let gridPoint = vec2<f32>(
      select(0.0, f32(px) / f32(width - 1u) * 2.0 - 1.0, width > 1u),
      select(0.0, f32(py) / f32(height - 1u) * 2.0 - 1.0, height > 1u)
    );
    let sourcePoint = virtual_inverse_point(gridPoint);
    if (sourcePoint.z > 0.5) {
      let pixel = py * width + px;
      let rendered = pixelState[pixel].rgb;
      let targetColor = target_rgb_at(sourcePoint.xy, width, height);
      coverage = alphaState[pixel].compositeAlpha;
      alphaY = target_alpha_at(sourcePoint.xy, width, height);
      alphaX = coverage;
      alphaL1 = abs(alphaX - alphaY);
      background = select(0.0, 1.0, alphaX < ${DEFAULT_ALPHA_TARGET});
      let residual = rendered - targetColor;
      loss = (abs(rendered.r - targetColor.r) + abs(rendered.g - targetColor.g) + abs(rendered.b - targetColor.b)) / 3.0;
      squaredError = dot(residual, residual);
      x = dot(rendered, vec3<f32>(1.0 / 3.0));
      y = dot(targetColor, vec3<f32>(1.0 / 3.0));
      renderedChroma = max(rendered.r, max(rendered.g, rendered.b)) - min(rendered.r, min(rendered.g, rendered.b));
      targetChroma = max(targetColor.r, max(targetColor.g, targetColor.b)) - min(targetColor.r, min(targetColor.g, targetColor.b));
      let ssimChannels = ssimData[pixel * 4u + 3u];
      ssim = vec2<f32>((ssimChannels.r + ssimChannels.g + ssimChannels.b) / 3.0, ssimChannels.a);
      valid = 1.0;
    }
  }
  wgLoss[lid.x] = loss;
  wgSquaredError[lid.x] = squaredError;
  wgX[lid.x] = x;
  wgY[lid.x] = y;
  wgX2[lid.x] = x * x;
  wgY2[lid.x] = y * y;
  wgXY[lid.x] = x * y;
  wgCount[lid.x] = valid;
  wgAlphaL1[lid.x] = alphaL1;
  wgAlphaX[lid.x] = alphaX;
  wgAlphaY[lid.x] = alphaY;
  wgAlphaX2[lid.x] = alphaX * alphaX;
  wgAlphaY2[lid.x] = alphaY * alphaY;
  wgAlphaXY[lid.x] = alphaX * alphaY;
  wgCoverage[lid.x] = coverage;
  wgBackground[lid.x] = background;
  wgRenderedChroma[lid.x] = renderedChroma;
  wgTargetChroma[lid.x] = targetChroma;
  wgSsim[lid.x] = ssim;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lid.x < stride) {
      wgLoss[lid.x] += wgLoss[lid.x + stride];
      wgSquaredError[lid.x] += wgSquaredError[lid.x + stride];
      wgX[lid.x] += wgX[lid.x + stride];
      wgY[lid.x] += wgY[lid.x + stride];
      wgX2[lid.x] += wgX2[lid.x + stride];
      wgY2[lid.x] += wgY2[lid.x + stride];
      wgXY[lid.x] += wgXY[lid.x + stride];
      wgCount[lid.x] += wgCount[lid.x + stride];
      wgAlphaL1[lid.x] += wgAlphaL1[lid.x + stride];
      wgAlphaX[lid.x] += wgAlphaX[lid.x + stride];
      wgAlphaY[lid.x] += wgAlphaY[lid.x + stride];
      wgAlphaX2[lid.x] += wgAlphaX2[lid.x + stride];
      wgAlphaY2[lid.x] += wgAlphaY2[lid.x + stride];
      wgAlphaXY[lid.x] += wgAlphaXY[lid.x + stride];
      wgCoverage[lid.x] += wgCoverage[lid.x + stride];
      wgBackground[lid.x] += wgBackground[lid.x + stride];
      wgRenderedChroma[lid.x] += wgRenderedChroma[lid.x + stride];
      wgTargetChroma[lid.x] += wgTargetChroma[lid.x + stride];
      wgSsim[lid.x] += wgSsim[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    let out = tileIndex * ${VIRTUAL_CAMERA_METRIC_TILE_STRIDE}u;
    metricsOut[out] = wgLoss[0];
    metricsOut[out + 1u] = wgX[0];
    metricsOut[out + 2u] = wgY[0];
    metricsOut[out + 3u] = wgX2[0];
    metricsOut[out + 4u] = wgY2[0];
    metricsOut[out + 5u] = wgXY[0];
    metricsOut[out + 6u] = wgCount[0];
    metricsOut[out + 7u] = wgAlphaL1[0];
    metricsOut[out + 8u] = wgAlphaX[0];
    metricsOut[out + 9u] = wgAlphaY[0];
    metricsOut[out + 10u] = wgAlphaX2[0];
    metricsOut[out + 11u] = wgAlphaY2[0];
    metricsOut[out + 12u] = wgAlphaXY[0];
    metricsOut[out + 13u] = wgCoverage[0];
    metricsOut[out + 14u] = wgBackground[0];
    metricsOut[out + 15u] = wgSquaredError[0];
    metricsOut[out + 16u] = wgRenderedChroma[0];
    metricsOut[out + 17u] = wgTargetChroma[0];
    metricsOut[out + 18u] = wgSsim[0].x;
    metricsOut[out + 19u] = wgSsim[0].y;
  }
}`;
  }

  function overlapMetrics({ hiddenRgbBinding, hiddenRgbShader }) {
    return `
struct Config { values: array<vec4<f32>, 19>, };
@group(0) @binding(0) var<uniform> config: Config;
struct SplatPosition { center: vec2<f32>, rawDepth: f32, depthGradient: f32, };
@group(0) @binding(1) var<storage, read> xy: array<SplatPosition>;
@group(0) @binding(2) var<storage, read> transform: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> color: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> targetRgb: array<f32>;
@group(0) @binding(5) var<storage, read> targetAlpha: array<f32>;
@group(0) @binding(6) var<storage, read> tileOffsets: array<u32>;
@group(0) @binding(7) var<storage, read> tileIndices: array<u32>;
@group(0) @binding(8) var<storage, read_write> metricsOut: array<f32>;
${hiddenRgbBinding}
const HIDDEN_RGB_ATTRIBUTION_STRIDE = 5u;
const HIDDEN_RGB_ATTRIBUTION_QUANTIZATION = 4096.0;
var<workgroup> reduceA: array<vec4<f32>, 64>;
var<workgroup> reduceB: array<vec4<f32>, 64>;
var<workgroup> reduceC: array<vec4<f32>, 64>;
var<workgroup> reduceD: array<vec4<f32>, 64>;
var<workgroup> reduceOrder: array<vec4<f32>, 64>;

fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }
${VIRTUAL_TILT_WGSL}

fn gaussian_kernel(d: vec2<f32>, c: f32, s: f32, scale: vec2<f32>) -> f32 {
  let r = vec2<f32>(c * d.x + s * d.y, -s * d.x + c * d.y);
  let q = dot(r / scale, r / scale);
  return exp(-0.5 * q);
}

fn target_rgb_at(point: vec2<f32>, width: u32, height: u32) -> vec3<f32> {
  let source = clamp((point * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u)), vec2<f32>(0.0), vec2<f32>(f32(width - 1u), f32(height - 1u)));
  let p0 = vec2<u32>(floor(source));
  let p1 = min(p0 + vec2<u32>(1u), vec2<u32>(width - 1u, height - 1u));
  let f = fract(source);
  let i00 = (p0.y * width + p0.x) * 3u;
  let i10 = (p0.y * width + p1.x) * 3u;
  let i01 = (p1.y * width + p0.x) * 3u;
  let i11 = (p1.y * width + p1.x) * 3u;
  let c00 = vec3<f32>(targetRgb[i00], targetRgb[i00 + 1u], targetRgb[i00 + 2u]);
  let c10 = vec3<f32>(targetRgb[i10], targetRgb[i10 + 1u], targetRgb[i10 + 2u]);
  let c01 = vec3<f32>(targetRgb[i01], targetRgb[i01 + 1u], targetRgb[i01 + 2u]);
  let c11 = vec3<f32>(targetRgb[i11], targetRgb[i11 + 1u], targetRgb[i11 + 2u]);
  return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

fn target_alpha_at(point: vec2<f32>, width: u32, height: u32) -> f32 {
  let source = clamp((point * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u)), vec2<f32>(0.0), vec2<f32>(f32(width - 1u), f32(height - 1u)));
  let p0 = vec2<u32>(floor(source));
  let p1 = min(p0 + vec2<u32>(1u), vec2<u32>(width - 1u, height - 1u));
  let f = fract(source);
  return mix(mix(targetAlpha[p0.y * width + p0.x], targetAlpha[p0.y * width + p1.x], f.x), mix(targetAlpha[p1.y * width + p0.x], targetAlpha[p1.y * width + p1.x], f.x), f.y);
}

@compute @workgroup_size(64)
fn overlap_metrics(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>
) {
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let tileCols8 = (width + 7u) / 8u;
  let tileIndex = wid.y * workgroups.x + wid.x;
  let tileCount = tileCols8 * ((height + 7u) / 8u);
  if (tileIndex >= tileCount) { return; }
  let tileX8 = tileIndex % tileCols8;
  let tileY8 = tileIndex / tileCols8;
  let px = tileX8 * 8u + lid.x % 8u;
  let py = tileY8 * 8u + lid.x / 8u;
  var a = vec4<f32>(0.0);
  var b = vec4<f32>(0.0);
  var cc = vec4<f32>(0.0);
  var dOut = vec4<f32>(0.0);
  var orderOut = vec4<f32>(0.0);
  if (px < width && py < height) {
    let pixel = py * width + px;
    let outputPoint = vec2<f32>(
      select(0.0, f32(px) / f32(width - 1u) * 2.0 - 1.0, width > 1u),
      select(0.0, f32(py) / f32(height - 1u) * 2.0 - 1.0, height > 1u)
    );
    let inversePoint = virtual_inverse_point(outputPoint);
    let targetColor = target_rgb_at(inversePoint.xy, width, height);
    let targetAlphaValue = target_alpha_at(inversePoint.xy, width, height);
    let useTiles = cfg(19u) > 0.5;
    let tileCols = (width + ${TILE_SIZE - 1}u) / ${TILE_SIZE}u;
    let tile = (py / ${TILE_SIZE}u) * tileCols + (px / ${TILE_SIZE}u);
    let capacity = arrayLength(&tileIndices);
    let start = select(0u, min(tileOffsets[tile] & 0x7fffffffu, capacity), useTiles);
    let end = select(u32(cfg(2u)), min(tileOffsets[tile + 1u] & 0x7fffffffu, capacity), useTiles);
    let pixelSigma = ${MIP_PIXEL_SIGMA} * 2.0 / max(cfg(0u), cfg(1u));
    let scaleFactor = clamp(cfg(63u), 0.01, 1.0);
    var numerator = vec3<f32>(0.0);
    var colorSecond = vec3<f32>(0.0);
    var denom = 0.0;
    var sumW2 = 0.0;
    var maxW = 0.0;
    var sumWLogW = 0.0;
    var targetDistance = 0.0;
    var transmittance = 1.0;
    var compositedRgb = vec3<f32>(0.0);
    var previousFrontOrder = 0.0;
    var hasPreviousFrontOrder = false;
    var adjacentOrderPairs = 0.0;
    var adjacentOrderFlips = 0.0;
    var acceptedEnd = start;
    var cursor = start;
    loop {
      if (cursor >= end) { break; }
      var g = cursor;
      if (useTiles) { g = tileIndices[cursor]; }
      let t = transform[g];
      if (t.w >= 0.5) {
        let center = xy[g].center;
        let c = cos(t.z);
        let s = sin(t.z);
        let samplePoint = virtual_inverse_point_at_z(outputPoint, virtual_pass_layer_depth(t.w, xy[g].rawDepth)).xy;
        let baseScale = max(t.xy * scaleFactor, vec2<f32>(0.0001));
        let effective = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));
        var kernel = gaussian_kernel(samplePoint - center, c, s, effective);
        var mip = sqrt((baseScale.x * baseScale.y) / max(effective.x * effective.y, 0.00000001));
        if (cfg(26u) > 0.5) {
          let sampleOffset = select(0.5, 0.28867513459481287, cfg(31u) > 0.5);
          let ox = select(0.0, sampleOffset / f32(width - 1u), width > 1u);
          let oy = select(0.0, sampleOffset / f32(height - 1u), height > 1u);
          kernel = 0.25 * (
            gaussian_kernel(virtual_inverse_point_at_z(clamp(outputPoint + vec2<f32>(-ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0)), virtual_pass_layer_depth(t.w, xy[g].rawDepth)).xy - center, c, s, baseScale) +
            gaussian_kernel(virtual_inverse_point_at_z(clamp(outputPoint + vec2<f32>( ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0)), virtual_pass_layer_depth(t.w, xy[g].rawDepth)).xy - center, c, s, baseScale) +
            gaussian_kernel(virtual_inverse_point_at_z(clamp(outputPoint + vec2<f32>(-ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0)), virtual_pass_layer_depth(t.w, xy[g].rawDepth)).xy - center, c, s, baseScale) +
            gaussian_kernel(virtual_inverse_point_at_z(clamp(outputPoint + vec2<f32>( ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0)), virtual_pass_layer_depth(t.w, xy[g].rawDepth)).xy - center, c, s, baseScale)
          );
          mip = 1.0;
        }
        let weight = clamp(kernel * color[g].a * mip, 0.0, 0.99);
        if (kernel >= 0.0003354626 && weight >= 0.0039215686) {
          let rgb = color[g].rgb;
          numerator += weight * rgb;
          colorSecond += weight * rgb * rgb;
          denom += weight;
          sumW2 += weight * weight;
          maxW = max(maxW, weight);
          sumWLogW += weight * log(max(weight, 0.00000001));
          targetDistance += weight * dot(abs(rgb - targetColor), vec3<f32>(1.0 / 3.0));
          if (transmittance >= 0.0001) {
            let frontOrder = fract(t.w);
            if (hasPreviousFrontOrder) {
              adjacentOrderPairs += 1.0;
              adjacentOrderFlips += select(0.0, 1.0, frontOrder > previousFrontOrder + 0.0000001);
            }
            previousFrontOrder = frontOrder;
            hasPreviousFrontOrder = true;
            compositedRgb += transmittance * weight * rgb;
            transmittance *= 1.0 - weight;
            acceptedEnd = cursor + 1u;
          }
        }
      }
      cursor += 1u;
    }
    let validPixel = inversePoint.z > 0.5;
    let covered = denom > ${BACKGROUND_EXPOSURE_EPSILON} && validPixel;
    let weightedMean = select(vec3<f32>(0.0), numerator / max(denom, ${BACKGROUND_EXPOSURE_EPSILON}), covered);
    let rendered = compositedRgb + transmittance * vec3<f32>(cfg(3u), cfg(4u), cfg(5u));
    let variance = max(vec3<f32>(0.0), colorSecond / max(denom, ${BACKGROUND_EXPOSURE_EPSILON}) - weightedMean * weightedMean);
    let effectiveContributors = select(0.0, denom * denom / max(sumW2, 0.0000000000000001), covered);
    let maxShare = select(0.0, maxW / max(denom, ${BACKGROUND_EXPOSURE_EPSILON}), covered);
    let entropy = select(0.0, max(0.0, log(max(denom, 0.00000001)) - sumWLogW / max(denom, ${BACKGROUND_EXPOSURE_EPSILON})), covered);
    let alpha = 1.0 - transmittance;
    let rgbError = select(vec3<f32>(0.0), abs(rendered - targetColor), validPixel);
    let l1 = dot(rgbError, vec3<f32>(1.0 / 3.0));
    let maxChannel = max(rgbError.r, max(rgbError.g, rgbError.b));
${hiddenRgbShader}
    a = vec4<f32>(select(0.0, 1.0, validPixel), select(0.0, denom, validPixel), effectiveContributors, maxShare);
    b = vec4<f32>(entropy, select(0.0, alpha, validPixel), select(0.0, abs(alpha - targetAlphaValue), validPixel), dot(variance, vec3<f32>(1.0 / 3.0)));
    cc = select(vec4<f32>(0.0), vec4<f32>(select(0.0, targetDistance / max(denom, ${BACKGROUND_EXPOSURE_EPSILON}), covered), l1, maxChannel, select(0.0, 1.0, maxChannel > 0.10)), validPixel);
    dOut = select(vec4<f32>(0.0), vec4<f32>(select(0.0, 1.0, alpha < ${DEFAULT_ALPHA_TARGET}), sumW2, maxW, maxChannel), validPixel);
    orderOut = select(vec4<f32>(0.0), vec4<f32>(adjacentOrderPairs, adjacentOrderFlips, select(0.0, 1.0, covered), 0.0), validPixel);
  }
  reduceA[lid.x] = a;
  reduceB[lid.x] = b;
  reduceC[lid.x] = cc;
  reduceD[lid.x] = dOut;
  reduceOrder[lid.x] = orderOut;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lid.x < stride) {
      reduceA[lid.x] += reduceA[lid.x + stride];
      reduceB[lid.x] += reduceB[lid.x + stride];
      reduceC[lid.x] += reduceC[lid.x + stride];
      reduceD[lid.x] = vec4<f32>(
        reduceD[lid.x].xyz + reduceD[lid.x + stride].xyz,
        max(reduceD[lid.x].w, reduceD[lid.x + stride].w)
      );
      reduceOrder[lid.x] += reduceOrder[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    let out = tileIndex * ${OVERLAP_METRIC_STRIDE}u;
    metricsOut[out] = reduceA[0].x;
    metricsOut[out + 1u] = reduceA[0].y;
    metricsOut[out + 2u] = reduceA[0].z;
    metricsOut[out + 3u] = reduceA[0].w;
    metricsOut[out + 4u] = reduceB[0].x;
    metricsOut[out + 5u] = reduceB[0].y;
    metricsOut[out + 6u] = reduceB[0].z;
    metricsOut[out + 7u] = reduceB[0].w;
    metricsOut[out + 8u] = reduceC[0].x;
    metricsOut[out + 9u] = reduceC[0].y;
    metricsOut[out + 10u] = reduceC[0].z;
    metricsOut[out + 11u] = reduceC[0].w;
    metricsOut[out + 12u] = reduceD[0].x;
    metricsOut[out + 13u] = reduceD[0].y;
    metricsOut[out + 14u] = reduceD[0].z;
    metricsOut[out + 15u] = reduceD[0].w;
    metricsOut[out + 16u] = reduceOrder[0].x;
    metricsOut[out + 17u] = reduceOrder[0].y;
    metricsOut[out + 18u] = reduceOrder[0].z;
  }
}`;
  }

  function alphaLoss() {
    return `
struct Config { values: array<vec4<f32>, 12>, };
struct AlphaState { compositeAlpha: f32, acceptedEnd: u32, pad0: f32, pad1: u32, };
@group(0) @binding(0) var<uniform> config: Config;
struct SplatPosition { center: vec2<f32>, rawDepth: f32, depthGradient: f32, };
@group(0) @binding(1) var<storage, read> xy: array<SplatPosition>;
@group(0) @binding(2) var<storage, read> transform: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> color: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> targetAlpha: array<f32>;
@group(0) @binding(5) var<storage, read> alphaState: array<AlphaState>;
var<workgroup> reduceGradient: array<f32, 64>;
var<workgroup> reduceWeight: array<f32, 64>;
fn cfg(i: u32) -> f32 { return config.values[i / 4u][i % 4u]; }
fn gaussian_kernel(d: vec2<f32>, c: f32, s: f32, scale: vec2<f32>) -> f32 {
  let r = vec2<f32>(c * d.x + s * d.y, -s * d.x + c * d.y);
  return exp(-0.5 * dot(r / scale, r / scale));
}
@compute @workgroup_size(64)
fn alpha_loss(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>
) {
  let g = wid.y * workgroups.x + wid.x;
  let width = u32(cfg(0u));
  let height = u32(cfg(1u));
  let alphaActive = cfg(46u) > 0.0;
  let isActive = g < u32(cfg(2u)) && transform[g].w >= 0.5 && alphaActive;
  var gradient = 0.0;
  var weightSum = 0.0;
  if (isActive) {
    let center = xy[g].center;
    let t = transform[g];
    let rgba = color[g];
    let c = cos(t.z);
    let s = sin(t.z);
    let baseScale = max(t.xy, vec2<f32>(0.0001));
    let pixelSigma = ${MIP_PIXEL_SIGMA} * 2.0 / max(cfg(0u), cfg(1u));
    let effective = sqrt(baseScale * baseScale + vec2<f32>(pixelSigma * pixelSigma));
    let useEwa = cfg(26u) > 0.5;
    let radius = vec2<f32>(
      ${RENDER_SIGMA} * (abs(c) * effective.x + abs(s) * effective.y),
      ${RENDER_SIGMA} * (abs(s) * effective.x + abs(c) * effective.y)
    );
    let minNorm = max(vec2<f32>(-1.0), center - radius);
    let maxNorm = min(vec2<f32>(1.0), center + radius);
    let minPx = vec2<u32>(floor((minNorm * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u))));
    let maxPx = vec2<u32>(ceil((maxNorm * 0.5 + 0.5) * vec2<f32>(f32(width - 1u), f32(height - 1u))));
    let spanX = maxPx.x - minPx.x + 1u;
    let pixelCount = (maxPx.y - minPx.y + 1u) * spanX;
    for (var offset = lid.x; offset < pixelCount; offset += 64u) {
      let px = minPx.x + offset % spanX;
      let py = minPx.y + offset / spanX;
      let p = vec2<f32>(
        select(0.0, f32(px) / f32(width - 1u) * 2.0 - 1.0, width > 1u),
        select(0.0, f32(py) / f32(height - 1u) * 2.0 - 1.0, height > 1u)
      );
      var kernel = gaussian_kernel(p - center, c, s, effective);
      var mip = sqrt((baseScale.x * baseScale.y) / max(effective.x * effective.y, 0.00000001));
      if (useEwa) {
        let sampleOffset = select(0.5, 0.28867513459481287, cfg(31u) > 0.5);
        let ox = select(0.0, sampleOffset / f32(width - 1u), width > 1u);
        let oy = select(0.0, sampleOffset / f32(height - 1u), height > 1u);
        kernel = 0.25 * (
          gaussian_kernel(clamp(p + vec2<f32>(-ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - center, c, s, baseScale) +
          gaussian_kernel(clamp(p + vec2<f32>( ox, -oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - center, c, s, baseScale) +
          gaussian_kernel(clamp(p + vec2<f32>(-ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - center, c, s, baseScale) +
          gaussian_kernel(clamp(p + vec2<f32>( ox,  oy), vec2<f32>(-1.0), vec2<f32>(1.0)) - center, c, s, baseScale)
        );
        mip = 1.0;
      }
      let rawWeight = clamp(kernel * rgba.a * mip, 0.0, 0.99);
      if (kernel >= 0.0003354626 && rawWeight >= 0.0039215686) {
        let pixel = py * width + px;
        let alpha = alphaState[pixel].compositeAlpha;
        let alphaGoal = targetAlpha[pixel];
        let derivative = (1.0 - alpha) * rawWeight * (1.0 - rgba.a) / max(0.01, 1.0 - rawWeight);
        gradient += sign(alpha - alphaGoal) * derivative;
        weightSum += rawWeight;
      }
    }
  }
  reduceGradient[lid.x] = gradient;
  reduceWeight[lid.x] = weightSum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lid.x < stride) {
      reduceGradient[lid.x] += reduceGradient[lid.x + stride];
      reduceWeight[lid.x] += reduceWeight[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u && isActive) {
    let rgba = color[g];
    let logit = log(clamp(rgba.a, 0.005, 0.995) / (1.0 - clamp(rgba.a, 0.005, 0.995)));
    let learningRate = min(0.05, cfg(12u) * cfg(46u));
    let nextOpacity = 1.0 / (1.0 + exp(-(logit - learningRate * reduceGradient[0] / max(reduceWeight[0], 0.01))));
    color[g] = vec4<f32>(rgba.rgb, clamp(nextOpacity, 0.005, 0.995));
  }
}`;
  }

  global.Image2SplatPaintMetricShaders = Object.freeze({
    trainingResidualMap,
    pixelMetrics,
    virtualCameraMetrics,
    overlapMetrics,
    alphaLoss,
  });
})(globalThis);
