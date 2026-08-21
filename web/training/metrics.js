function ssimFromMoments(meanA, meanB, varA, varB, cov) {
  const c1 = 0.01 ** 2;
  const c2 = 0.03 ** 2;
  return ((2 * meanA * meanB + c1) * (2 * cov + c2)) / ((meanA ** 2 + meanB ** 2 + c1) * (varA + varB + c2));
}

function percentileSorted(values, fraction) {
  if (!values.length) return null;
  const position = clampNumber(fraction, 0, 1, 0) * (values.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const mix = position - lower;
  return values[lower] * (1 - mix) + values[upper] * mix;
}

function psnrFromRgbMse(mse) {
  if (!Number.isFinite(mse) || mse < 0) return Number.NaN;
  return 10 * Math.log10(1 / Math.max(PSNR_MSE_FLOOR, mse));
}

function ssimWorkingBufferBytes(image) {
  return image.width * image.height * SSIM_WORKING_BYTES_PER_PIXEL;
}

function summarizeVirtualCameraMetricSet(entries) {
  const finite = entries.filter((entry) => Number.isFinite(entry.metrics?.ssim));
  const values = (key) => finite
    .map((entry) => Number(entry.metrics?.[key]))
    .filter(Number.isFinite);
  const average = (key) => {
    const set = values(key);
    return set.length ? set.reduce((sum, value) => sum + value, 0) / set.length : null;
  };
  const sortedSsim = values("ssim").sort((a, b) => a - b);
  return {
    camera_count: finite.length,
    valid_pixel_count: finite.reduce((sum, entry) => sum + (entry.metrics.valid_pixel_count || 0), 0),
    valid_pixel_ratio_mean: average("valid_pixel_ratio"),
    rgb_ssim_macro: average("ssim"),
    rgb_ssim_p10: percentileSorted(sortedSsim, 0.1),
    rgb_ssim_min: sortedSsim[0] ?? null,
    rgb_l1_macro: average("loss"),
    rgb_mse_macro: average("mse"),
    rgb_psnr_macro: average("psnr"),
    alpha_ssim_macro: average("alphaSsim"),
    alpha_l1_macro: average("alphaL1"),
    coverage_mean: average("coverage_mean"),
    background_exposure_ratio_mean: average("background_exposure_ratio"),
    rendered_mean_srgb_signal: average("rendered_mean_srgb_signal"),
    target_mean_srgb_signal: average("target_mean_srgb_signal"),
    rendered_minus_target_signal: average("rendered_minus_target_signal"),
    rendered_signal_stddev: average("rendered_signal_stddev"),
    target_signal_stddev: average("target_signal_stddev"),
    rendered_mean_srgb_chroma: average("rendered_mean_srgb_chroma"),
    target_mean_srgb_chroma: average("target_mean_srgb_chroma"),
    rendered_minus_target_chroma: average("rendered_minus_target_chroma"),
  };
}

function targetTangentAt(image, x, y) {
  const px = Math.max(1, Math.min(image.width - 2, Math.round((x * 0.5 + 0.5) * (image.width - 1))));
  const py = Math.max(1, Math.min(image.height - 2, Math.round((y * 0.5 + 0.5) * (image.height - 1))));
  const luma = (ix, iy) => {
    const offset = (iy * image.width + ix) * 3;
    return image.rgb[offset] * 0.299 + image.rgb[offset + 1] * 0.587 + image.rgb[offset + 2] * 0.114;
  };
  const gx = 0.5 * (luma(px + 1, py) - luma(px - 1, py));
  const gy = 0.5 * (luma(px, py + 1) - luma(px, py - 1));
  return { angle: Math.atan2(gy, gx) + Math.PI * 0.5, energy: gx * gx + gy * gy };
}

function phase39PixelLengths(width, height, sx, sy, theta) {
  const pixelScale = (angle) => 0.5 * Math.hypot(
    Math.cos(angle) * Math.max(1, width - 1),
    Math.sin(angle) * Math.max(1, height - 1),
  );
  const xLength = sx * pixelScale(theta);
  const yLength = sy * pixelScale(theta + Math.PI * 0.5);
  return {
    xLength,
    yLength,
    major: Math.max(xLength, yLength),
    minor: Math.min(xLength, yLength),
    majorAxis: xLength >= yLength ? "x" : "y",
  };
}

function phase39ContractProbe(width = 1024, height = 512, sx = 0.02, sy = 0.004, theta = Math.PI * 0.25) {
  const lengths = phase39PixelLengths(width, height, sx, sy, theta);
  const maxSearchPx = Math.min(64, Math.max(width, height) * 0.02);
  return {
    ...lengths,
    maxSearchPx,
    maxCellRadius: 8,
    maxBucketsPerSide: 17,
    singleSourceClaim: phase39Variants().singleSourceClaim,
  };
}

function computeThinLineMetrics(image, params) {
  const tags = params.detailTags;
  if (!tags || !params.count) return null;
  const sampling = finalDiagnosticSampling(params.count, MAX_THIN_LINE_DIAGNOSTIC_SAMPLES);
  const points = [];
  const cellSize = 8;
  const cells = new Map();
  const key = (x, y) => `${Math.floor(x / cellSize)},${Math.floor(y / cellSize)}`;
  for (let i = 0; i < params.count; i += sampling.stride) {
    if (tags[i] <= 1.5) continue;
    const sx = params.scale[i * 2];
    const sy = params.scale[i * 2 + 1];
    const theta = params.theta[i];
    const lengths = phase39PixelLengths(image.width, image.height, sx, sy, theta);
    const useX = lengths.majorAxis === "x";
    const angle = theta + (useX ? 0 : Math.PI * 0.5);
    const point = {
      i,
      x: (params.xy[i * 2] * 0.5 + 0.5) * (image.width - 1),
      y: (params.xy[i * 2 + 1] * 0.5 + 0.5) * (image.height - 1),
      angle,
      majorPx: lengths.major,
      minorPx: lengths.minor,
    };
    points.push(point);
    const bucketKey = key(point.x, point.y);
    if (!cells.has(bucketKey)) cells.set(bucketKey, []);
    cells.get(bucketKey).push(point);
  }
  if (!points.length) {
    return {
      detail_count: 0,
      sampled_splats: sampling.sampleCount,
      source_splats: sampling.sourceCount,
      sample_stride: sampling.stride,
      gap_ratio: null,
      isolated_detail_ratio: null,
      off_ridge_streak_ratio: null,
    };
  }
  const hasSupport = (point, side) => {
    const maxSearchPx = Math.min(64, Math.max(image.width, image.height) * 0.02);
    const reach = Math.min(maxSearchPx, Math.max(2, point.majorPx * 1.25));
    const targetX = point.x + Math.cos(point.angle) * reach * side;
    const targetY = point.y + Math.sin(point.angle) * reach * side;
    const radius = Math.min(maxSearchPx, Math.max(3, point.minorPx * 3, point.majorPx * 0.45));
    const cellRadius = Math.min(8, Math.max(1, Math.ceil(radius / cellSize)));
    const cx = Math.floor(targetX / cellSize);
    const cy = Math.floor(targetY / cellSize);
    let checkedCandidates = 0;
    const maxCandidateChecks = 256;
    for (let oy = -cellRadius; oy <= cellRadius; oy += 1) {
      for (let ox = -cellRadius; ox <= cellRadius; ox += 1) {
        for (const candidate of cells.get(`${cx + ox},${cy + oy}`) || []) {
          checkedCandidates += 1;
          if (checkedCandidates > maxCandidateChecks) return false;
          if (candidate.i === point.i || Math.abs(Math.cos(candidate.angle - point.angle)) < 0.8) continue;
          if (Math.hypot(candidate.x - targetX, candidate.y - targetY) <= radius) return true;
        }
      }
    }
    return false;
  };
  let missingSides = 0;
  let isolated = 0;
  let offRidge = 0;
  for (const point of points) {
    const forward = hasSupport(point, 1);
    const backward = hasSupport(point, -1);
    missingSides += Number(!forward) + Number(!backward);
    if (!forward && !backward) isolated += 1;
    const target = targetTangentAt(image, params.xy[point.i * 2], params.xy[point.i * 2 + 1]);
    if (target.energy < 0.0004 || Math.abs(Math.cos(target.angle - point.angle)) < 0.8) offRidge += 1;
  }
  return {
    detail_count: points.length,
    sampled_splats: sampling.sampleCount,
    source_splats: sampling.sourceCount,
    sample_stride: sampling.stride,
    gap_ratio: missingSides / (points.length * 2),
    isolated_detail_ratio: isolated / points.length,
    off_ridge_streak_ratio: offRidge / points.length,
  };
}

function regionalSsimFromTileMetrics(values, width, height, tileSize = 8, columns = 4, rows = 4) {
  const regions = Array.from({ length: columns * rows }, (_, index) => ({
    index,
    column: index % columns,
    row: Math.floor(index / columns),
    count: 0,
    loss: 0,
    renderedY: 0,
    targetY: 0,
    renderedY2: 0,
    targetY2: 0,
    renderedTargetY: 0,
    gradientError: 0,
    targetGradientEnergy: 0,
    gradientCount: 0,
    ssimSum: 0,
  }));
  const tileColumns = Math.ceil(width / tileSize);
  const tileCount = Math.ceil(height / tileSize) * tileColumns;
  for (let tileIndex = 0; tileIndex < tileCount; tileIndex += 1) {
    const source = tileIndex * METRIC_TILE_STRIDE;
    const count = values[source + 7];
    if (count <= 0) continue;
    const tileX = tileIndex % tileColumns;
    const tileY = Math.floor(tileIndex / tileColumns);
    const centerX = Math.min(width - 1, tileX * tileSize + (Math.min(tileSize, width - tileX * tileSize) - 1) * 0.5);
    const centerY = Math.min(height - 1, tileY * tileSize + (Math.min(tileSize, height - tileY * tileSize) - 1) * 0.5);
    const column = Math.min(columns - 1, Math.floor((centerX / Math.max(1, width)) * columns));
    const row = Math.min(rows - 1, Math.floor((centerY / Math.max(1, height)) * rows));
    const region = regions[row * columns + column];
    region.count += count;
    region.loss += values[source];
    region.renderedY += values[source + 1];
    region.targetY += values[source + 2];
    region.renderedY2 += values[source + 3];
    region.targetY2 += values[source + 4];
    region.renderedTargetY += values[source + 5];
    region.gradientError += values[source + 12];
    region.targetGradientEnergy += values[source + 13];
    region.gradientCount += values[source + 14];
    region.ssimSum += values[source + 34];
  }

  const measured = regions.filter((region) => region.count > 0).map((region) => {
    const ssim = region.ssimSum / region.count;
    return {
      index: region.index,
      column: region.column,
      row: region.row,
      bounds: [
        Math.floor((region.column * width) / columns),
        Math.floor((region.row * height) / rows),
        Math.floor(((region.column + 1) * width) / columns),
        Math.floor(((region.row + 1) * height) / rows),
      ],
      pixels: region.count,
      ssim,
      l1: region.loss / region.count,
      gradient_l1: region.gradientError / Math.max(1, region.gradientCount),
      target_gradient_energy: region.targetGradientEnergy / Math.max(1, region.gradientCount),
      gradient_fidelity: 1 - region.gradientError / Math.max(0.000001, region.targetGradientEnergy),
    };
  });
  const sorted = measured.map((region) => region.ssim).sort((a, b) => a - b);
  const worst = measured.reduce((current, region) => (!current || region.ssim < current.ssim ? region : current), null);
  const highFrequencyRegions = measured
    .slice()
    .sort((a, b) => b.target_gradient_energy - a.target_gradient_energy)
    .slice(0, Math.max(1, Math.ceil(measured.length * 0.25)));
  return {
    grid: [columns, rows],
    tile_size: tileSize,
    mean: sorted.reduce((sum, value) => sum + value, 0) / Math.max(1, sorted.length),
    minimum: sorted[0] ?? null,
    p10: percentileSorted(sorted, 0.1),
    median: percentileSorted(sorted, 0.5),
    p90: percentileSorted(sorted, 0.9),
    maximum: sorted[sorted.length - 1] ?? null,
    worst_region: worst,
    high_frequency_regions: highFrequencyRegions,
    regions: measured,
  };
}

