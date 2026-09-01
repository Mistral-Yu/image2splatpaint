function rgbToImageData(rgb, width, height, alpha = null) {
  const imageData = new ImageData(width, height);
  for (let i = 0, j = 0, p = 0; j < rgb.length; i += 4, j += 3, p += 1) {
    imageData.data[i] = clampByte(rgb[j] * 255);
    imageData.data[i + 1] = clampByte(rgb[j + 1] * 255);
    imageData.data[i + 2] = clampByte(rgb[j + 2] * 255);
    imageData.data[i + 3] = clampByte((alpha?.[p] ?? 1) * 255);
  }
  return imageData;
}

function resizeFloatImageBilinear(image, width, height) {
  const sourceWidth = Math.max(1, Math.round(image.width));
  const sourceHeight = Math.max(1, Math.round(image.height));
  const targetWidth = Math.max(1, Math.round(width));
  const targetHeight = Math.max(1, Math.round(height));
  if (sourceWidth === targetWidth && sourceHeight === targetHeight) {
    return {
      width: targetWidth,
      height: targetHeight,
      rgb: image.rgb,
      alpha: image.alpha || null,
    };
  }
  const rgb = new Float32Array(targetWidth * targetHeight * 3);
  const alpha = new Float32Array(targetWidth * targetHeight);
  const scaleX = sourceWidth / targetWidth;
  const scaleY = sourceHeight / targetHeight;
  for (let y = 0; y < targetHeight; y += 1) {
    const sourceY = Math.max(0, Math.min(sourceHeight - 1, (y + 0.5) * scaleY - 0.5));
    const y0 = Math.floor(sourceY);
    const y1 = Math.min(sourceHeight - 1, y0 + 1);
    const fy = sourceY - y0;
    for (let x = 0; x < targetWidth; x += 1) {
      const sourceX = Math.max(0, Math.min(sourceWidth - 1, (x + 0.5) * scaleX - 0.5));
      const x0 = Math.floor(sourceX);
      const x1 = Math.min(sourceWidth - 1, x0 + 1);
      const fx = sourceX - x0;
      const p00 = y0 * sourceWidth + x0;
      const p10 = y0 * sourceWidth + x1;
      const p01 = y1 * sourceWidth + x0;
      const p11 = y1 * sourceWidth + x1;
      const target = y * targetWidth + x;
      for (let channel = 0; channel < 3; channel += 1) {
        const top =
          image.rgb[p00 * 3 + channel] * (1 - fx) +
          image.rgb[p10 * 3 + channel] * fx;
        const bottom =
          image.rgb[p01 * 3 + channel] * (1 - fx) +
          image.rgb[p11 * 3 + channel] * fx;
        rgb[target * 3 + channel] = top * (1 - fy) + bottom * fy;
      }
      const a00 = image.alpha?.[p00] ?? 1;
      const a10 = image.alpha?.[p10] ?? 1;
      const a01 = image.alpha?.[p01] ?? 1;
      const a11 = image.alpha?.[p11] ?? 1;
      alpha[target] =
        (a00 * (1 - fx) + a10 * fx) * (1 - fy) +
        (a01 * (1 - fx) + a11 * fx) * fy;
    }
  }
  return { width: targetWidth, height: targetHeight, rgb, alpha };
}

function makeCoarseTrainingImage(image, maxSide) {
  const currentSide = Math.max(image.width, image.height);
  if (currentSide <= maxSide) return null;
  const [width, height] = resizedSize(image.width, image.height, maxSide);
  return resizeFloatImageBilinear(image, width, height);
}

function curriculumCoarseMaxSide(fullSide, variants = phase33Variants()) {
  const boundedFullSide = Math.max(1, Math.round(fullSide));
  if (!variants.adaptiveCurriculum) return Math.min(boundedFullSide, variants.coarseMaxSide);
  // The curriculum is defined relative to the effective full training image,
  // including a 512px run: full / 4 -> full / 2 -> full.
  return Math.min(
    boundedFullSide,
    Math.max(CURRICULUM_COARSE_MIN_SIDE, Math.round(boundedFullSide / CURRICULUM_COARSE_DIVISOR)),
  );
}

function curriculumMidMaxSideForFullSide(fullSide, coarseMaxSide = PHASE33_COARSE_MAX_SIDE) {
  const boundedFullSide = Math.max(1, Math.round(fullSide));
  const coarseSide = Math.min(boundedFullSide, Math.max(1, Math.round(coarseMaxSide)));
  return Math.min(boundedFullSide, Math.max(coarseSide + 1, Math.round(boundedFullSide / 2)));
}

function curriculumStageDimensions(width, height, variants = phase33Variants()) {
  const full = { width: Math.max(1, Math.round(width)), height: Math.max(1, Math.round(height)) };
  const fullSide = Math.max(full.width, full.height);
  if (!variants.coarseToFull) {
    return { full, coarse: null, mid: null, coarseMaxSide: fullSide, midMaxSide: fullSide };
  }
  const coarseMaxSide = curriculumCoarseMaxSide(fullSide, variants);
  const midMaxSide = curriculumMidMaxSideForFullSide(fullSide, coarseMaxSide);
  const dimensionsAt = (maxSide) => {
    if (maxSide >= fullSide) return null;
    const [stageWidth, stageHeight] = resizedSize(full.width, full.height, maxSide);
    return { width: stageWidth, height: stageHeight };
  };
  return {
    full,
    coarse: dimensionsAt(coarseMaxSide),
    mid: variants.threeStageCurriculum ? dimensionsAt(midMaxSide) : null,
    coarseMaxSide,
    midMaxSide,
  };
}

function makeCurriculumImages(image, variants = phase33Variants()) {
  if (!variants.coarseToFull) return { coarseImage: null, midImage: null };
  const dimensions = curriculumStageDimensions(image.width, image.height, variants);
  const coarseImage = dimensions.coarse ? makeCoarseTrainingImage(image, dimensions.coarseMaxSide) : null;
  const midImage = dimensions.mid ? makeCoarseTrainingImage(image, dimensions.midMaxSide) : null;
  return { coarseImage, midImage, coarseMaxSide: dimensions.coarseMaxSide, midMaxSide: dimensions.midMaxSide };
}

function clampByte(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function splatGridLayout(image, count) {
  const cols = Math.max(1, Math.round(Math.sqrt((count * image.width) / image.height)));
  const rows = Math.max(1, Math.ceil(count / cols));
  const baseScale = INITIAL_SPLAT_COVERAGE_MULTIPLIER / Math.max(rows, cols);
  // Parameters remain in per-axis image NDC; this only matches initial coverage to each grid cell.
  const baseScaleX = aspectAwareGridEnabled() ? INITIAL_SPLAT_COVERAGE_MULTIPLIER / cols : baseScale;
  const baseScaleY = aspectAwareGridEnabled() ? INITIAL_SPLAT_COVERAGE_MULTIPLIER / rows : baseScale;
  return {
    rows,
    cols,
    baseScaleX,
    baseScaleY,
    baseScale,
  };
}

function stageMinimumScale(image, initialCount, trainingStage, ratio) {
  if (trainingStage === "full" || ratio <= 0) return MIN_SPLAT_SCALE;
  const referenceLayout = splatGridLayout(image, Math.max(1, initialCount));
  return Math.max(
    MIN_SPLAT_SCALE,
    Math.min(referenceLayout.baseScaleX, referenceLayout.baseScaleY) * ratio,
  );
}

function stageBaseScaleFloorRatio(trainingStage) {
  // Growth/reseed birth size is independent of the optional optimizer guard.
  // The optimizer may shrink useful detail later, but a phase transition must
  // not make new children start at the absolute minimum scale.
  if (trainingStage === "coarse") return DEFAULT_P1_BASE_SCALE_FLOOR_RATIO;
  if (trainingStage === "mid") return DEFAULT_P2_BASE_SCALE_FLOOR_RATIO;
  return DEFAULT_P3_BASE_SCALE_FLOOR_RATIO;
}

function stageRelativeScaleFloorRatio(trainingStage, variants = phase33Variants()) {
  if (trainingStage === "coarse") return variants.p1RelativeScaleFloorRatio;
  if (trainingStage === "mid") return variants.p2RelativeScaleFloorRatio;
  return variants.p3RelativeScaleFloorRatio;
}

function geometricMeanScaleMedian(params) {
  const values = [];
  for (let i = 0; i < (params?.count || 0); i += 1) {
    const sx = Math.max(MIN_SPLAT_SCALE, Number(params.scale[i * 2]) || MIN_SPLAT_SCALE);
    const sy = Math.max(MIN_SPLAT_SCALE, Number(params.scale[i * 2 + 1]) || MIN_SPLAT_SCALE);
    values.push(Math.sqrt(sx * sy));
  }
  values.sort((a, b) => a - b);
  return values[Math.floor(values.length / 2)] || MIN_SPLAT_SCALE;
}

function splatGridAt(layout, index) {
  const col = index % layout.cols;
  const row = Math.floor(index / layout.cols);
  return constrainSplat(
    layout.cols === 1 ? 0 : -0.95 + (1.9 * col) / (layout.cols - 1),
    layout.rows === 1 ? 0 : -0.95 + (1.9 * row) / (layout.rows - 1),
    layout.baseScaleX,
    layout.baseScaleY,
  );
}

function initialSplatOrientation(index, columns = 1) {
  const safeIndex = Math.max(0, Math.round(index));
  const safeColumns = Math.max(1, Math.round(columns));
  const column = safeIndex % safeColumns;
  const row = Math.floor(safeIndex / safeColumns);
  let bits = (Math.imul(column + 1, 0x9e3779b1) ^ Math.imul(row + 1, 0x85ebca6b)) >>> 0;
  bits = ((bits >>> 16) | (bits << 16)) >>> 0;
  bits = (((bits & 0x55555555) << 1) | ((bits >>> 1) & 0x55555555)) >>> 0;
  bits = (((bits & 0x33333333) << 2) | ((bits >>> 2) & 0x33333333)) >>> 0;
  bits = (((bits & 0x0f0f0f0f) << 4) | ((bits >>> 4) & 0x0f0f0f0f)) >>> 0;
  bits = (((bits & 0x00ff00ff) << 8) | ((bits >>> 8) & 0x00ff00ff)) >>> 0;
  const unit = (bits + 0.5) / 4294967296;
  return (unit - 0.5) * Math.PI;
}

function initialSplatShape(image, layout, index) {
  const longSide = Math.max(1, image.width, image.height);
  const frameX = image.width / longSide;
  const frameY = image.height / longSide;
  const radiusX = frameX * layout.baseScaleX;
  const radiusY = frameY * layout.baseScaleY;
  const major = Math.max(radiusX, radiusY);
  const minor = Math.max(MIN_SPLAT_SCALE * Math.min(frameX, frameY), Math.min(radiusX, radiusY));
  const worldTheta = initialSplatOrientation(index, layout.cols);
  const c = Math.cos(worldTheta);
  const s = Math.sin(worldTheta);
  const major2 = major * major;
  const minor2 = minor * minor;
  const covarianceWorldX = c * c * major2 + s * s * minor2;
  const covarianceWorldY = s * s * major2 + c * c * minor2;
  const covarianceWorldXY = c * s * (major2 - minor2);
  const covarianceX = covarianceWorldX / (frameX * frameX);
  const covarianceY = covarianceWorldY / (frameY * frameY);
  const covarianceXY = -covarianceWorldXY / (frameX * frameY);
  const trace = covarianceX + covarianceY;
  const delta = Math.hypot(covarianceX - covarianceY, 2 * covarianceXY);
  return {
    sx: Math.sqrt(Math.max(MIN_SPLAT_SCALE ** 2, 0.5 * (trace + delta))),
    sy: Math.sqrt(Math.max(MIN_SPLAT_SCALE ** 2, 0.5 * (trace - delta))),
    theta: 0.5 * Math.atan2(2 * covarianceXY, covarianceX - covarianceY),
  };
}

function adaptiveBspImportanceGrid(image, count) {
  const longSide = Math.max(1, image.width, image.height);
  const targetLongSide = Math.min(
    longSide,
    Math.max(64, Math.min(384, Math.ceil(Math.sqrt(Math.max(1, count)) * 2))),
  );
  const width = Math.max(1, Math.round(image.width * targetLongSide / longSide));
  const height = Math.max(1, Math.round(image.height * targetLongSide / longSide));
  const luma = new Float32Array(width * height);
  const color = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    const sourceY = Math.min(image.height - 1, Math.round((y + 0.5) * image.height / height - 0.5));
    for (let x = 0; x < width; x += 1) {
      const sourceX = Math.min(image.width - 1, Math.round((x + 0.5) * image.width / width - 0.5));
      const source = (sourceY * image.width + sourceX) * 3;
      const destination = (y * width + x) * 3;
      const r = image.rgb[source];
      const g = image.rgb[source + 1];
      const b = image.rgb[source + 2];
      color[destination] = r;
      color[destination + 1] = g;
      color[destination + 2] = b;
      luma[y * width + x] = r * 0.299 + g * 0.587 + b * 0.114;
    }
  }
  const importance = new Float32Array(width * height);
  let detailMaximum = 1e-6;
  for (let y = 0; y < height; y += 1) {
    const ym = Math.max(0, y - 1);
    const yp = Math.min(height - 1, y + 1);
    for (let x = 0; x < width; x += 1) {
      const xm = Math.max(0, x - 1);
      const xp = Math.min(width - 1, x + 1);
      const center = y * width + x;
      const gx = 0.5 * (luma[y * width + xp] - luma[y * width + xm]);
      const gy = 0.5 * (luma[yp * width + x] - luma[ym * width + x]);
      const laplacian = Math.abs(
        luma[y * width + xm] +
        luma[y * width + xp] +
        luma[ym * width + x] +
        luma[yp * width + x] -
        4 * luma[center],
      );
      let colorDifference = 0;
      const colorCenter = center * 3;
      for (const neighbor of [y * width + xm, y * width + xp, ym * width + x, yp * width + x]) {
        const offset = neighbor * 3;
        colorDifference += Math.hypot(
          color[colorCenter] - color[offset],
          color[colorCenter + 1] - color[offset + 1],
          color[colorCenter + 2] - color[offset + 2],
        );
      }
      const detail = 0.5 * Math.hypot(gx, gy) + 0.3 * laplacian + 0.05 * colorDifference;
      importance[center] = detail;
      detailMaximum = Math.max(detailMaximum, detail);
    }
  }
  for (let i = 0; i < importance.length; i += 1) {
    const normalized = Math.min(1, importance[i] / detailMaximum);
    importance[i] = 0.12 + 0.88 * Math.sqrt(normalized);
  }
  return { width, height, importance };
}

const structureGuidedRegionProfileCache = new WeakMap();

// Project-native allocator: it keeps the existing standard-alpha renderer,
// layer model, growth schedule, and optimizer, and only adds a soft regional
// destination budget when the user enables the checkbox.
function computeStructureGuidedRegionProfile(
  image,
  { lumaSpace = "srgb-baseline", regionGrid = DEFAULT_STRUCTURE_REGION_GRID } = {},
) {
  const started = performance.now();
  const lumaAt = (x, y) => {
    const px = Math.max(0, Math.min(image.width - 1, x));
    const py = Math.max(0, Math.min(image.height - 1, y));
    const offset = (py * image.width + px) * 3;
    if (lumaSpace === "srgb-baseline") {
      return 0.299 * image.rgb[offset] +
        0.587 * image.rgb[offset + 1] +
        0.114 * image.rgb[offset + 2];
    }
    return 0.2126729 * srgbSignalToLinear(image.rgb[offset]) +
      0.7151522 * srgbSignalToLinear(image.rgb[offset + 1]) +
      0.0721750 * srgbSignalToLinear(image.rgb[offset + 2]);
  };
  const smoothLumaAt = (x, y) => (
    lumaAt(x - 1, y - 1) + 2 * lumaAt(x, y - 1) + lumaAt(x + 1, y - 1) +
    2 * lumaAt(x - 1, y) + 4 * lumaAt(x, y) + 2 * lumaAt(x + 1, y) +
    lumaAt(x - 1, y + 1) + 2 * lumaAt(x, y + 1) + lumaAt(x + 1, y + 1)
  ) / 16;
  const measureGrid = (gridSize) => {
    const raw = new Float64Array(gridSize * gridSize);
    const sampleGrid = 8;
    for (let regionY = 0; regionY < gridSize; regionY += 1) {
      const y0 = Math.floor(regionY * image.height / gridSize);
      const y1 = Math.max(y0 + 1, Math.floor((regionY + 1) * image.height / gridSize));
      for (let regionX = 0; regionX < gridSize; regionX += 1) {
        const x0 = Math.floor(regionX * image.width / gridSize);
        const x1 = Math.max(x0 + 1, Math.floor((regionX + 1) * image.width / gridSize));
        let sum = 0;
        let sumSquared = 0;
        let samples = 0;
        for (let sy = 0; sy < sampleGrid; sy += 1) {
          const y = Math.min(image.height - 1, Math.floor(y0 + (sy + 0.5) * (y1 - y0) / sampleGrid));
          for (let sx = 0; sx < sampleGrid; sx += 1) {
            const x = Math.min(image.width - 1, Math.floor(x0 + (sx + 0.5) * (x1 - x0) / sampleGrid));
            const gx = 0.5 * (smoothLumaAt(x + 1, y) - smoothLumaAt(x - 1, y));
            const gy = 0.5 * (smoothLumaAt(x, y + 1) - smoothLumaAt(x, y - 1));
            const magnitude = Math.hypot(gx, gy);
            sum += magnitude;
            sumSquared += magnitude * magnitude;
            samples += 1;
          }
        }
        const mean = sum / Math.max(1, samples);
        raw[regionY * gridSize + regionX] = Math.sqrt(Math.max(
          0,
          sumSquared / Math.max(1, samples) - mean * mean,
        ));
      }
    }
    const sorted = Array.from(raw).sort((a, b) => a - b);
    const low = percentileSorted(sorted, 0.1);
    const high = percentileSorted(sorted, 0.9);
    const demand = Float64Array.from(raw, (value) => Math.max(
      0,
      Math.min(1, (Math.log1p(value * 4096) - Math.log1p(low * 4096)) /
        Math.max(1e-9, Math.log1p(high * 4096) - Math.log1p(low * 4096))),
    ));
    return { raw, demand, percentile10: low, percentile90: high };
  };
  const child = measureGrid(regionGrid);
  const profile = {
    ...child,
    lumaSpace,
    regionGrid,
    regionMode: `${regionGrid}x${regionGrid}`,
    processingMs: performance.now() - started,
  };
  return profile;
}

function structureGuidedRegionProfile(image, options = null) {
  const variants = options || phase39Variants();
  const lumaSpace = variants.structureLumaSpace || "srgb-baseline";
  const regionGrid = variants.structureRegionGrid || DEFAULT_STRUCTURE_REGION_GRID;
  let cachedByVariant = structureGuidedRegionProfileCache.get(image);
  if (!cachedByVariant) {
    cachedByVariant = new Map();
    structureGuidedRegionProfileCache.set(image, cachedByVariant);
  }
  const key = `${lumaSpace}:${regionGrid}`;
  if (!cachedByVariant.has(key)) {
    cachedByVariant.set(key, computeStructureGuidedRegionProfile(image, { lumaSpace, regionGrid }));
  }
  return cachedByVariant.get(key);
}

function structureGuidedRegionQuotas(image, targetCount, finalCount, options = null) {
  const profile = structureGuidedRegionProfile(image, options);
  const regionCount = profile.regionGrid * profile.regionGrid;
  const progress = Math.max(0, Math.min(1, targetCount / Math.max(1, finalCount)));
  const structureStrength = 0.88 - 0.06 * progress;
  const detailMass = Array.from(profile.demand, (value) => 0.02 + value ** 1.75);
  const detailTotal = detailMass.reduce((sum, value) => sum + value, 0);
  const exact = detailMass.map((value) => targetCount * (
    (1 - structureStrength) / regionCount +
    structureStrength * value / Math.max(1e-9, detailTotal)
  ));
  const quotas = Uint32Array.from(exact, Math.floor);
  let remaining = targetCount - quotas.reduce((sum, value) => sum + value, 0);
  const fractions = exact.map((value, region) => ({ region, fraction: value - Math.floor(value) }))
    .sort((a, b) => b.fraction - a.fraction || a.region - b.region);
  for (let i = 0; i < remaining; i += 1) quotas[fractions[i].region] += 1;
  return { ...profile, quotas, structureStrength };
}

function structureGuidedRegionControl(image, targetCount, finalCount) {
  const allocation = structureGuidedRegionQuotas(image, targetCount, finalCount);
  const control = new Uint32Array(PHASE45_REGION_COUNT * PHASE45_REGION_STRIDE);
  const regionCount = allocation.regionGrid * allocation.regionGrid;
  for (let region = 0; region < regionCount; region += 1) {
    const base = region * PHASE45_REGION_STRIDE;
    control[base + 1] = Math.round(Math.min(1, allocation.demand[region]) * 65535);
    control[base + 9] = allocation.quotas[region];
  }
  return { control, allocation };
}

function structureGuidedProfileBenchmark(image = state.image, repetitions = 25) {
  if (!image) throw new Error("Load an image before benchmarking structure profiles.");
  const variants = ["linear-srgb", "srgb-baseline"].map((lumaSpace) => ({ lumaSpace }));
  return variants.map((variant) => {
    const samples = [];
    for (let i = 0; i < Math.max(1, Math.round(repetitions)); i += 1) {
      samples.push(computeStructureGuidedRegionProfile(image, variant).processingMs);
    }
    samples.sort((a, b) => a - b);
    return {
      ...variant,
      repetitions: samples.length,
      median_ms: percentileSorted(samples, 0.5),
      p10_ms: percentileSorted(samples, 0.1),
      p90_ms: percentileSorted(samples, 0.9),
    };
  });
}

function sourceDetailSplatDistribution(image, params) {
  const profile = structureGuidedRegionProfile(image);
  const regionGrid = profile.regionGrid;
  const regionCount = regionGrid * regionGrid;
  const demand = Array.from(profile.demand);
  const counts = new Uint32Array(regionCount);
  for (let index = 0; index < params.count; index += 1) {
    const x = Math.max(0, Math.min(0.999999, params.xy[index * 2] * 0.5 + 0.5));
    const y = Math.max(0, Math.min(0.999999, params.xy[index * 2 + 1] * 0.5 + 0.5));
    const regionX = Math.min(regionGrid - 1, Math.floor(x * regionGrid));
    const regionY = Math.min(regionGrid - 1, Math.floor(y * regionGrid));
    counts[regionY * regionGrid + regionX] += 1;
  }
  const regions = demand.map((value, region) => ({ region, demand: value, count: counts[region] }))
    .sort((a, b) => a.demand - b.demand || a.region - b.region);
  const quartileSize = Math.max(1, Math.floor(regions.length / 4));
  const bottom = regions.slice(0, quartileSize);
  const top = regions.slice(-quartileSize);
  const meanCount = (items) => items.reduce((sum, item) => sum + item.count, 0) / items.length;
  const flatMean = meanCount(bottom);
  const detailMean = meanCount(top);
  const countMean = params.count / regionCount;
  const demandMean = demand.reduce((sum, value) => sum + value, 0) / demand.length;
  let covariance = 0;
  let countVariance = 0;
  let demandVariance = 0;
  for (let region = 0; region < regionCount; region += 1) {
    const countDelta = counts[region] - countMean;
    const demandDelta = demand[region] - demandMean;
    covariance += countDelta * demandDelta;
    countVariance += countDelta * countDelta;
    demandVariance += demandDelta * demandDelta;
  }
  return {
    grid: [regionGrid, regionGrid],
    count: params.count,
    flat_quartile_mean_count: flatMean,
    detail_quartile_mean_count: detailMean,
    detail_to_flat_count_ratio: detailMean / Math.max(1e-9, flatMean),
    demand_count_correlation: covariance / Math.max(1e-9, Math.sqrt(countVariance * demandVariance)),
    minimum_region_count: Math.min(...counts),
    maximum_region_count: Math.max(...counts),
  };
}

function adaptiveBspIntegral(grid, valueAt) {
  const stride = grid.width + 1;
  const values = new Float64Array(stride * (grid.height + 1));
  for (let y = 0; y < grid.height; y += 1) {
    let row = 0;
    for (let x = 0; x < grid.width; x += 1) {
      row += valueAt(x, y, grid.importance[y * grid.width + x]);
      values[(y + 1) * stride + x + 1] = values[y * stride + x + 1] + row;
    }
  }
  return { values, stride };
}

function adaptiveBspSum(integral, x0, y0, x1, y1) {
  const { values, stride } = integral;
  return values[y1 * stride + x1] -
    values[y0 * stride + x1] -
    values[y1 * stride + x0] +
    values[y0 * stride + x0];
}

function adaptiveBspRegions(image, count) {
  const grid = adaptiveBspImportanceGrid(image, count);
  const massIntegral = adaptiveBspIntegral(grid, (_x, _y, importance) => importance);
  const xIntegral = adaptiveBspIntegral(grid, (x, _y, importance) => importance * (x + 0.5));
  const yIntegral = adaptiveBspIntegral(grid, (_x, y, importance) => importance * (y + 0.5));
  const x2Integral = adaptiveBspIntegral(grid, (x, _y, importance) => importance * (x + 0.5) ** 2);
  const y2Integral = adaptiveBspIntegral(grid, (_x, y, importance) => importance * (y + 0.5) ** 2);
  let serial = 0;
  const describe = (x0, y0, x1, y1) => {
    const mass = Math.max(1e-8, adaptiveBspSum(massIntegral, x0, y0, x1, y1));
    const meanX = adaptiveBspSum(xIntegral, x0, y0, x1, y1) / mass;
    const meanY = adaptiveBspSum(yIntegral, x0, y0, x1, y1) / mass;
    const varianceX = Math.max(0, adaptiveBspSum(x2Integral, x0, y0, x1, y1) / mass - meanX ** 2);
    const varianceY = Math.max(0, adaptiveBspSum(y2Integral, x0, y0, x1, y1) / mass - meanY ** 2);
    const splittable = x1 - x0 > 1 || y1 - y0 > 1;
    return {
      x0, y0, x1, y1, mass, meanX, meanY, varianceX, varianceY,
      serial: serial += 1,
      priority: splittable ? mass * Math.sqrt((x1 - x0) * (y1 - y0)) : -Infinity,
    };
  };
  const heap = [];
  const push = (region) => {
    heap.push(region);
    let index = heap.length - 1;
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);
      if (heap[parent].priority >= region.priority) break;
      heap[index] = heap[parent];
      index = parent;
    }
    heap[index] = region;
  };
  const pop = () => {
    const root = heap[0];
    const tail = heap.pop();
    if (heap.length && tail) {
      let index = 0;
      while (true) {
        const left = index * 2 + 1;
        const right = left + 1;
        if (left >= heap.length) break;
        const child = right < heap.length && heap[right].priority > heap[left].priority ? right : left;
        if (heap[child].priority <= tail.priority) break;
        heap[index] = heap[child];
        index = child;
      }
      heap[index] = tail;
    }
    return root;
  };
  push(describe(0, 0, grid.width, grid.height));
  while (heap.length < count) {
    const region = pop();
    if (!region || !Number.isFinite(region.priority)) {
      if (region) push(region);
      break;
    }
    const width = region.x1 - region.x0;
    const height = region.y1 - region.y0;
    const splitX = width > 1 && (height <= 1 || region.varianceX / Math.max(1, width ** 2) >= region.varianceY / Math.max(1, height ** 2));
    const start = splitX ? region.x0 : region.y0;
    const end = splitX ? region.x1 : region.y1;
    const targetMass = region.mass * 0.5;
    let cut = start + 1;
    let bestDifference = Infinity;
    for (let candidate = start + 1; candidate < end; candidate += 1) {
      const partialMass = splitX
        ? adaptiveBspSum(massIntegral, region.x0, region.y0, candidate, region.y1)
        : adaptiveBspSum(massIntegral, region.x0, region.y0, region.x1, candidate);
      const difference = Math.abs(partialMass - targetMass);
      if (difference < bestDifference) {
        bestDifference = difference;
        cut = candidate;
      }
    }
    if (splitX) {
      push(describe(region.x0, region.y0, cut, region.y1));
      push(describe(cut, region.y0, region.x1, region.y1));
    } else {
      push(describe(region.x0, region.y0, region.x1, cut));
      push(describe(region.x0, cut, region.x1, region.y1));
    }
  }
  const regions = heap.sort((a, b) => a.meanY - b.meanY || a.meanX - b.meanX || a.serial - b.serial);
  return { grid, regions };
}

function applyAdaptiveBspPaintInitialization(image, params, kernelShape) {
  const { grid, regions } = adaptiveBspRegions(image, params.count);
  const extent = kernelShape === "rectangle" ? RECTANGLE_KERNEL_EXTENT : LAYERED_OPAQUE_BRUSH_KERNEL_EXTENT;
  let regionAreaTotal = 0;
  for (let i = 0; i < params.count; i += 1) {
    const region = regions[i % Math.max(1, regions.length)] || {
      x0: 0, y0: 0, x1: grid.width, y1: grid.height,
      meanX: grid.width * 0.5, meanY: grid.height * 0.5,
    };
    const x = -1 + 2 * region.meanX / grid.width;
    const y = -1 + 2 * region.meanY / grid.height;
    const regionWidth = 2 * (region.x1 - region.x0) / grid.width;
    const regionHeight = 2 * (region.y1 - region.y0) / grid.height;
    const structure = strokeStructureAt(image, x, y, params.theta[i]);
    const theta = kernelShape === "rectangle"
      ? rectangleDirectedTaperTheta(
          image,
          x,
          y,
          structure.theta,
          params.rectangleEdgeDirectedTaper &&
            params.rectangleTopRatio < 1 - 0.000001,
        )
      : structure.theta;
    const areaScale = Math.sqrt(Math.max(MIN_SPLAT_SCALE ** 2, regionWidth * regionHeight)) * 0.62 / extent;
    const anisotropy = 1.25 + 3.25 * structure.coherence;
    const stretch = Math.sqrt(anisotropy);
    const constrained = constrainSplat(
      x,
      y,
      areaScale * stretch,
      areaScale / stretch,
      theta,
      params.boundarySigma,
      Math.max(4, anisotropy),
    );
    params.xy[i * 2] = constrained.x;
    params.xy[i * 2 + 1] = constrained.y;
    params.scale[i * 2] = constrained.sx;
    params.scale[i * 2 + 1] = constrained.sy;
    params.theta[i] = theta;
    if (kernelShape === "rectangle" && params.rectangleStructureAwareRatio) {
      params.detailTags[i] =
        structure.coherence >= RECTANGLE_STRUCTURE_MIN_COHERENCE &&
        structure.energy >= RECTANGLE_STRUCTURE_MIN_ENERGY
          ? 2
          : 1;
    }
    regionAreaTotal += regionWidth * regionHeight;
  }
  params.initializationScheme = "image-importance-bsp";
  params.initializationStats = {
    map_width: grid.width,
    map_height: grid.height,
    unique_region_count: regions.length,
    splat_count: params.count,
    mean_region_area_ndc: regionAreaTotal / Math.max(1, regions.length),
  };
  return params;
}

function initialOrientationStats(params, image) {
  const bins = new Array(8).fill(0);
  let anisotropyTotal = 0;
  let anisotropyMax = 1;
  let finite = true;
  for (let index = 0; index < params.count; index += 1) {
    const world = transformPlanarSplatForPly(
      params.xy[index * 2],
      params.xy[index * 2 + 1],
      params.scale[index * 2],
      params.scale[index * 2 + 1],
      params.theta[index],
      image,
    );
    const angle = ((world.theta % Math.PI) + Math.PI) % Math.PI;
    bins[Math.min(bins.length - 1, Math.floor(angle / Math.PI * bins.length))] += 1;
    const ratio = world.sx / Math.max(1e-12, world.sy);
    anisotropyTotal += ratio;
    anisotropyMax = Math.max(anisotropyMax, ratio);
    finite &&= Number.isFinite(angle + world.sx + world.sy + ratio);
  }
  return {
    scheme: "deterministic-spatially-decorrelated-world-angle",
    bins,
    bin_spread: Math.max(...bins) - Math.min(...bins),
    world_anisotropy_mean: anisotropyTotal / Math.max(1, params.count),
    world_anisotropy_max: anisotropyMax,
    finite,
  };
}

function selectedBoundarySigma() {
  return clampNumber(
    els.boundarySigma?.value,
    LIMITS.boundarySigmaMin,
    LIMITS.boundarySigmaMax,
    DEFAULT_BOUNDARY_SIGMA,
  );
}

function clampSplatCenter(value, margin = selectedBoundarySigma() * MIN_SPLAT_SCALE) {
  return Math.max(-1 + margin, Math.min(1 - margin, value));
}

function currentMaxAnisotropy() {
  return clampNumber(els.maxAnisotropy?.value, LIMITS.maxAnisotropyMin, LIMITS.maxAnisotropyMax, DEFAULT_MAX_ANISOTROPY);
}

function anisotropyLimitsForParams(params = null) {
  const detail = clampNumber(
    params?.maxAnisotropy ?? state.metrics?.learning_rates?.maxAnisotropy ?? currentMaxAnisotropy(),
    LIMITS.maxAnisotropyMin,
    LIMITS.maxAnisotropyMax,
    DEFAULT_MAX_ANISOTROPY,
  );
  const surface = Math.min(
    detail,
    clampNumber(
      params?.surfaceAnisotropy ?? state.metrics?.learning_rates?.surfaceAnisotropy ?? qualityRecoveryVariants().surfaceAnisotropy,
      LIMITS.maxAnisotropyMin,
      LIMITS.maxAnisotropyMax,
      DEFAULT_SURFACE_ANISOTROPY,
    ),
  );
  return { surface, detail };
}

function anisotropyLimitForTag(tag, params = null) {
  const limits = anisotropyLimitsForParams(params);
  return Math.floor(Number(tag) || 1) >= 2 ? limits.detail : limits.surface;
}

function capScaleAnisotropy(sx, sy, maxRatio = currentMaxAnisotropy()) {
  const safeRatio = Math.max(1, maxRatio);
  const safeSx = Math.max(MIN_SPLAT_SCALE, sx);
  const safeSy = Math.max(MIN_SPLAT_SCALE, sy);
  const major = Math.max(safeSx, safeSy);
  const minor = Math.max(MIN_SPLAT_SCALE, Math.min(safeSx, safeSy));
  if (major / minor <= safeRatio) return { sx: safeSx, sy: safeSy };
  const cappedMajor = minor * safeRatio;
  return safeSx >= safeSy ? { sx: cappedMajor, sy: minor } : { sx: minor, sy: cappedMajor };
}

function rotatedExtentAtSigma(sx, sy, theta = 0, sigma = BOUNDARY_SIGMA) {
  const c = Math.abs(Math.cos(theta));
  const s = Math.abs(Math.sin(theta));
  return {
    x: sigma * Math.hypot(c * sx, s * sy),
    y: sigma * Math.hypot(s * sx, c * sy),
  };
}

function rotatedSplatExtent(sx, sy, theta = 0, sigma = selectedBoundarySigma()) {
  return rotatedExtentAtSigma(sx, sy, theta, sigma);
}

function previewPaddingSpec(image, params, enabled = els.outsidePreviewToggle.checked) {
  if (!enabled || !image || !params?.count) {
    return { x: 0, y: 0, width: image?.width || 1, height: image?.height || 1, scaleX: 1, scaleY: 1, bytes: 0 };
  }
  let outsideX = 0;
  let outsideY = 0;
  for (let i = 0; i < params.count; i += 1) {
    const extent = params.internalBendKey ? Image2SplatPaintInternalBend.extent(params, i)
      : rotatedExtentAtSigma(params.scale[i * 2], params.scale[i * 2 + 1], params.theta?.[i] || 0, RENDER_SIGMA);
    outsideX = Math.max(outsideX, Math.abs(params.xy[i * 2]) + extent.x - 1);
    outsideY = Math.max(outsideY, Math.abs(params.xy[i * 2 + 1]) + extent.y - 1);
  }
  const limitX = Math.min(MAX_PREVIEW_PADDING_PX, Math.round(image.width * MAX_PREVIEW_PADDING_FRACTION));
  const limitY = Math.min(MAX_PREVIEW_PADDING_PX, Math.round(image.height * MAX_PREVIEW_PADDING_FRACTION));
  const x = Math.min(limitX, Math.max(0, Math.ceil(outsideX * image.width * 0.5)));
  const y = Math.min(limitY, Math.max(0, Math.ceil(outsideY * image.height * 0.5)));
  const width = image.width + x * 2;
  const height = image.height + y * 2;
  return {
    x,
    y,
    width,
    height,
    scaleX: image.width / width,
    scaleY: image.height / height,
    bytes: Math.max(0, (width * height - image.width * image.height) * 4),
  };
}

function buildPreviewTileIndexData(image, params, options = {}) {
  const preview = previewPaddingSpec(image, params, Boolean(options.outside));
  const tileCols = Math.ceil(preview.width / TILE_SIZE);
  const tileRows = Math.ceil(preview.height / TILE_SIZE);
  const tileCount = tileCols * tileRows;
  const maxTileReferences = Number.isFinite(Number(options.maxTileReferences))
    ? Math.max(1, Math.floor(Number(options.maxTileReferences)))
    : Number.MAX_SAFE_INTEGER;
  const counts = new Uint32Array(tileCount);
  const bounds = new Int32Array(params.count * 4);
  const aspectStretch = Math.sqrt(Math.max(0.000001, Number(options.localAspectRatio) || 1));
  const scaleMultiplier = Number.isFinite(Number(options.splatScaleMultiplier))
    ? Math.max(0, Number(options.splatScaleMultiplier))
    : 1;
  const useEwa = phase33Variants().ewa2x2;
  const pixelSigma = MIP_PIXEL_SIGMA * 2 / Math.max(image.width, image.height);
  const pixelPadX = useEwa && image.width > 1 ? 0.5 / (image.width - 1) : 0;
  const pixelPadY = useEwa && image.height > 1 ? 0.5 / (image.height - 1) : 0;
  let referenceCount = 0;
  for (let index = 0; index < params.count; index += 1) {
    const theta = params.theta?.[index] || 0;
    const c = Math.abs(Math.cos(theta));
    const s = Math.abs(Math.sin(theta));
    const baseX = Math.max(0.0001, params.scale[index * 2] * scaleMultiplier * aspectStretch);
    const baseY = Math.max(0.0001, params.scale[index * 2 + 1] * scaleMultiplier / aspectStretch);
    const effectiveX = useEwa ? baseX : Math.hypot(baseX, pixelSigma);
    const effectiveY = useEwa ? baseY : Math.hypot(baseY, pixelSigma);
    const ownedExtent = params.internalBendKey
      ? Image2SplatPaintInternalBend.extent(params, index, effectiveX, effectiveY) : null;
    const radiusX = ((ownedExtent?.x ?? RENDER_SIGMA * (c * effectiveX + s * effectiveY)) + pixelPadX) * preview.scaleX;
    const radiusY = ((ownedExtent?.y ?? RENDER_SIGMA * (s * effectiveX + c * effectiveY)) + pixelPadY) * preview.scaleY;
    const centerX = params.xy[index * 2] * preview.scaleX;
    const centerY = params.xy[index * 2 + 1] * preview.scaleY;
    const minX = Math.max(0, Math.min(preview.width - 1, Math.floor(((centerX - radiusX) * 0.5 + 0.5) * Math.max(0, preview.width - 1))));
    const maxX = Math.max(0, Math.min(preview.width - 1, Math.ceil(((centerX + radiusX) * 0.5 + 0.5) * Math.max(0, preview.width - 1))));
    const minY = Math.max(0, Math.min(preview.height - 1, Math.floor(((centerY - radiusY) * 0.5 + 0.5) * Math.max(0, preview.height - 1))));
    const maxY = Math.max(0, Math.min(preview.height - 1, Math.ceil(((centerY + radiusY) * 0.5 + 0.5) * Math.max(0, preview.height - 1))));
    const minTileX = Math.floor(minX / TILE_SIZE);
    const maxTileX = Math.floor(maxX / TILE_SIZE);
    const minTileY = Math.floor(minY / TILE_SIZE);
    const maxTileY = Math.floor(maxY / TILE_SIZE);
    bounds.set([minTileX, maxTileX, minTileY, maxTileY], index * 4);
    for (let tileY = minTileY; tileY <= maxTileY; tileY += 1) {
      for (let tileX = minTileX; tileX <= maxTileX; tileX += 1) {
        referenceCount += 1;
        if (referenceCount > maxTileReferences) {
          throw new Error("Preview creates too many tile references; reduce Splat scale or Local aspect ratio.");
        }
        counts[tileY * tileCols + tileX] += 1;
      }
    }
  }
  const offsets = new Uint32Array(tileCount + 1);
  for (let tile = 0; tile < tileCount; tile += 1) offsets[tile + 1] = offsets[tile] + counts[tile];
  const indices = new Uint32Array(offsets[tileCount]);
  const cursors = offsets.slice(0, tileCount);
  for (let index = 0; index < params.count; index += 1) {
    const offset = index * 4;
    for (let tileY = bounds[offset + 2]; tileY <= bounds[offset + 3]; tileY += 1) {
      for (let tileX = bounds[offset]; tileX <= bounds[offset + 1]; tileX += 1) {
        const tile = tileY * tileCols + tileX;
        indices[cursors[tile]++] = index;
      }
    }
  }
  const comparator = options.splatSmallFirstOrder
    ? (a, b) => splatPreviewOrderComparator(a, b, params)
    : params.layerOrderEnabled
      ? (a, b) => layerOrderComparator(a, b, params)
      : null;
  if (comparator) {
    for (let tile = 0; tile < tileCount; tile += 1) {
      if (offsets[tile + 1] - offsets[tile] > 1) {
        indices.subarray(offsets[tile], offsets[tile + 1]).sort(comparator);
      }
    }
  }
  return { preview, offsets, indices };
}

function constrainSplat(
  x,
  y,
  sx,
  sy,
  theta = 0,
  boundarySigma = selectedBoundarySigma(),
  maxAnisotropy = currentMaxAnisotropy(),
) {
  if (boundarySigma <= 0) {
    const capped = capScaleAnisotropy(sx, sy, maxAnisotropy);
    return {
      x: clampSplatCenter(x, 0),
      y: clampSplatCenter(y, 0),
      sx: capped.sx,
      sy: capped.sy,
    };
  }
  const minimumExtent = rotatedSplatExtent(MIN_SPLAT_SCALE, MIN_SPLAT_SCALE, theta, boundarySigma);
  let cx = clampSplatCenter(x, minimumExtent.x);
  let cy = clampSplatCenter(y, minimumExtent.y);
  const capped = capScaleAnisotropy(sx, sy, maxAnisotropy);
  const extent = rotatedSplatExtent(capped.sx, capped.sy, theta, boundarySigma);
  const fit = Math.min(
    1,
    (1 - Math.abs(cx)) / Math.max(extent.x, 1e-8),
    (1 - Math.abs(cy)) / Math.max(extent.y, 1e-8),
  );
  let fitted = capScaleAnisotropy(
    Math.max(MIN_SPLAT_SCALE, capped.sx * fit),
    Math.max(MIN_SPLAT_SCALE, capped.sy * fit),
    maxAnisotropy,
  );
  let finalExtent = rotatedSplatExtent(fitted.sx, fitted.sy, theta, boundarySigma);
  const globalFit = Math.min(1, 0.999 / Math.max(finalExtent.x, finalExtent.y));
  fitted = capScaleAnisotropy(
    Math.max(MIN_SPLAT_SCALE, fitted.sx * globalFit),
    Math.max(MIN_SPLAT_SCALE, fitted.sy * globalFit),
    maxAnisotropy,
  );
  finalExtent = rotatedSplatExtent(fitted.sx, fitted.sy, theta, boundarySigma);
  cx = clampSplatCenter(cx, finalExtent.x);
  cy = clampSplatCenter(cy, finalExtent.y);
  return {
    x: cx,
    y: cy,
    sx: fitted.sx,
    sy: fitted.sy,
  };
}

function snapshotParams(params) {
  const anisotropyLimits = anisotropyLimitsForParams(params);
  return {
    ...(params.internalBendKey ? {internalBendKey: params.internalBendKey,
      internalBendShapes: params.internalBendShapes.slice(),
      internalBendControlPoints: params.internalBendControlPoints?.slice()} : {}),
    ...(params.flowBirthLinksEnabled ? {
      flowBirthLinksEnabled: true,
      flowBirthLinkStrength: params.flowBirthLinkStrength,
      flowLinkedSplatMin: params.flowLinkedSplatMin,
      flowLinkedSplatMax: params.flowLinkedSplatMax,
      flowBackcoatCount: params.flowBackcoatCount,
      flowTrainingSize: params.flowTrainingSize?.slice(),
    } : {}),
    kernelShape: normalizedKernelShape(params.kernelShape),
    rectangleTopRatio: clampNumber(
      params.rectangleTopRatio,
      MIN_RECTANGLE_TOP_RATIO,
      MAX_RECTANGLE_TOP_RATIO,
      DEFAULT_RECTANGLE_TOP_RATIO,
    ),
    rectangleTopRatioMax: clampNumber(
      params.rectangleTopRatioMax,
      params.rectangleTopRatio,
      MAX_RECTANGLE_TOP_RATIO,
      DEFAULT_RECTANGLE_TOP_RATIO_MAX,
    ),
    rectangleOpacityGradientMin: clampNumber(
      params.rectangleOpacityGradientMin,
      0,
      1,
      1,
    ),
    rectangleOpacityGradientMax: clampNumber(
      params.rectangleOpacityGradientMax,
      0,
      1,
      1,
    ),
    rectangleCenterOpacityGradientMin: clampNumber(
      params.rectangleCenterOpacityGradientMin,
      0,
      1,
      1,
    ),
    rectangleCenterOpacityGradientMax: clampNumber(
      params.rectangleCenterOpacityGradientMax,
      clampNumber(params.rectangleCenterOpacityGradientMin, 0, 1, 1),
      1,
      1,
    ),
    rectangleMinAspectRatio: clampNumber(
      params.rectangleMinAspectRatio,
      MIN_RECTANGLE_ASPECT_RATIO,
      params.rectangleMaxAspectRatio,
      DEFAULT_RECTANGLE_MIN_ASPECT_RATIO,
    ),
    rectangleMaxAspectRatio: clampNumber(
      params.rectangleMaxAspectRatio,
      MIN_RECTANGLE_ASPECT_RATIO,
      MAX_RECTANGLE_ASPECT_RATIO,
      DEFAULT_RECTANGLE_ASPECT_RATIO,
    ),
    rectangleOrientation: normalizedRectangleOrientation(params.rectangleOrientation),
    rectangleOrientationTolerance: normalizedRectangleOrientationTolerance(params.rectangleOrientationTolerance),
    rectanglePreserveArea:
      params.rectanglePreserveArea ?? DEFAULT_RECTANGLE_PRESERVE_AREA,
    rectangleEdgeDirectedTaper:
      params.rectangleEdgeDirectedTaper ?? DEFAULT_RECTANGLE_EDGE_DIRECTED_TAPER,
    rectangleStructureAwareRatio:
      params.rectangleStructureAwareRatio ?? DEFAULT_RECTANGLE_STRUCTURE_AWARE_RATIO,
    rectangleAsymmetricSoftness:
      params.rectangleAsymmetricSoftness ?? DEFAULT_RECTANGLE_ASYMMETRIC_SOFTNESS,
    opaqueLayered: Boolean(params.opaqueLayered),
    minimumOpacityEnabled: Boolean(params.minimumOpacityEnabled),
    minimumOpacity: clampNumber(
      params.minimumOpacity,
      MIN_LEARNED_PAINT_OPACITY,
      MAX_LEARNED_PAINT_OPACITY,
      MAX_LEARNED_PAINT_OPACITY,
    ),
    maximumOpacity: clampNumber(
      params.maximumOpacity,
      params.minimumOpacity,
      MAX_LEARNED_PAINT_OPACITY,
      MAX_LEARNED_PAINT_OPACITY,
    ),
    brushMinAspectRatio: clampNumber(
      params.brushMinAspectRatio,
      LIMITS.maxAnisotropyMin,
      params.brushMaxAspectRatio,
      LIMITS.maxAnisotropyMin,
    ),
    brushMaxAspectRatio: clampNumber(
      params.brushMaxAspectRatio,
      LIMITS.maxAnisotropyMin,
      LIMITS.maxAnisotropyMax,
      DEFAULT_MAX_ANISOTROPY,
    ),
    illustrativeOilVersion: Math.max(0, Math.round(Number(params.illustrativeOilVersion) || 0)),
    brushLocalColorFlowEnabled: Boolean(params.brushLocalColorFlowEnabled),
    brushStrokePersistenceEnabled: Boolean(params.brushStrokePersistenceEnabled),
    brushRibbonAspectFloor: clampNumber(
      params.brushRibbonAspectFloor,
      1,
      LIMITS.maxAnisotropyMax,
      BRUSH_STROKE_PERSISTENCE_RIBBON_MIN_RATIO,
    ),
    brushAccentAspectFloor: clampNumber(
      params.brushAccentAspectFloor,
      1,
      LIMITS.maxAnisotropyMax,
      BRUSH_STROKE_PERSISTENCE_ACCENT_MIN_RATIO,
    ),
    surfaceLayerPriorEnabled: Boolean(params.surfaceLayerPriorEnabled),
    surfaceLayerPriorColorAwarePromotion:
      params.surfaceLayerPriorColorAwarePromotion !== false,
    trainLayerColorGuardEnabled: Boolean(params.trainLayerColorGuardEnabled),
    surfaceLayerPriorLayers: Math.round(clampNumber(
      params.surfaceLayerPriorLayers,
      MIN_DISCRETE_LAYER_COUNT,
      MAX_DISCRETE_LAYER_COUNT,
      DEFAULT_SCALE_BIASED_SURFACE_LAYER_SORT_LAYERS,
    )),
    surfaceLayerPriorP1Interval: Math.max(0, Math.round(params.surfaceLayerPriorP1Interval || 0)),
    surfaceLayerPriorP2Interval: Math.max(0, Math.round(params.surfaceLayerPriorP2Interval || 0)),
    surfaceLayerPriorP3Interval: Math.max(0, Math.round(params.surfaceLayerPriorP3Interval || 0)),
    surfaceLayerPriorUntilFraction: clampNumber(
      params.surfaceLayerPriorUntilFraction,
      0,
      1,
      DEFAULT_SCALE_BIASED_SURFACE_LAYER_SORT_UNTIL,
    ),
    harmfulRectangleParentSplitEnabled: Boolean(params.harmfulRectangleParentSplitEnabled),
    harmfulRectangleParentSplitTransitionOnly: Boolean(
      params.harmfulRectangleParentSplitTransitionOnly,
    ),
    frontSplitChildrenEnabled: Boolean(params.frontSplitChildrenEnabled),
    brushOpacityGradientEnabled: Boolean(params.brushOpacityGradientEnabled),
    brushOpacityGradientStart: clampNumber(params.brushOpacityGradientStart, 0, 1, 0),
    brushOpacityGradientEnd: clampNumber(params.brushOpacityGradientEnd, 0, 1, 1),
    brushCenterOpacityGradientMin: clampNumber(params.brushCenterOpacityGradientMin, 0, 1, 1),
    brushCenterOpacityGradientMax: clampNumber(
      params.brushCenterOpacityGradientMax,
      clampNumber(params.brushCenterOpacityGradientMin, 0, 1, 1),
      1,
      1,
    ),
    brushWidthTaperEnabled: Boolean(params.brushWidthTaperEnabled),
    brushWidthTaperStart: clampNumber(params.brushWidthTaperStart, 0, 1, 1),
    brushWidthTaperEnd: clampNumber(params.brushWidthTaperEnd, 0, 1, 0),
    monochromeUnderpaintingEnabled: Boolean(params.monochromeUnderpaintingEnabled),
    colorFinishStartPercent: clampNumber(
      params.colorFinishStartPercent,
      MIN_COLOR_FINISH_START_PERCENT,
      MAX_COLOR_FINISH_START_PERCENT,
      DEFAULT_COLOR_FINISH_START_PERCENT,
    ),
    colorFinishStartStep: Math.max(
      0,
      Math.round(Number(params.colorFinishStartStep) || 0),
    ),
    currentVisibilityChildPolicyEnabled: params.currentVisibilityChildPolicyEnabled !== false,
    currentVisibilityCompactionEnabled: params.currentVisibilityCompactionEnabled !== false,
    illustrativeOilFamilyStats: params.illustrativeOilFamilyStats
      ? structuredClone(params.illustrativeOilFamilyStats)
      : null,
    count: params.count,
    xy: new Float32Array(params.xy),
    scale: new Float32Array(params.scale),
    rgb: new Float32Array(params.rgb),
    opacity: new Float32Array(params.opacity),
    theta: new Float32Array(params.theta),
    depthOrder: params.depthOrder ? new Float32Array(params.depthOrder) : initialDepthOrder(params.count),
    virtualDepth: params.virtualDepth ? new Float32Array(params.virtualDepth) : new Float32Array(params.count),
    brushTaper: params.brushTaper
      ? new Float32Array(params.brushTaper)
      : new Float32Array(params.count).fill(DEFAULT_LAYERED_BRUSH_TAPER),
    virtualDepthEnabled: Boolean(params.virtualDepthEnabled),
    virtualDepthThickness: Number(params.virtualDepthThickness) || DEFAULT_VIRTUAL_DEPTH_THICKNESS,
    virtualDepthSoftConstraintEnabled: params.virtualDepthSoftConstraintEnabled !== false,
    virtualDepthPriorDelta: Number(params.virtualDepthPriorDelta) || DEFAULT_VIRTUAL_DEPTH_PRIOR_DELTA,
    detailTags: params.detailTags ? new Float32Array(params.detailTags) : new Float32Array(params.count).fill(1),
    boundarySigma: Number.isFinite(params.boundarySigma) ? params.boundarySigma : selectedBoundarySigma(),
    layerOrderEnabled: Boolean(params.layerOrderEnabled),
    layerAwareAccumulationEnabled: Boolean(params.layerAwareAccumulationEnabled),
    discreteLayersEnabled: Boolean(params.discreteLayersEnabled),
    discreteLayerCount: Math.max(MIN_DISCRETE_LAYER_COUNT, Math.min(MAX_DISCRETE_LAYER_COUNT, Math.round(params.discreteLayerCount || DEFAULT_DISCRETE_LAYER_COUNT))),
    discreteLayerMoveRadius: Math.max(0, Math.round(params.discreteLayerMoveRadius ?? DEFAULT_DISCRETE_LAYER_MOVE_RADIUS)),
    maxAnisotropy: anisotropyLimits.detail,
    surfaceAnisotropy: anisotropyLimits.surface,
    rows: params.rows,
    cols: params.cols,
    bg: params.bg ? new Float32Array(params.bg) : new Float32Array([0, 0, 0]),
  };
}

function nonfiniteParamCount(params) {
  let count = 0;
  for (const values of [params?.xy, params?.scale, params?.rgb, params?.opacity, params?.theta, params?.depthOrder, params?.virtualDepth, params?.brushTaper, params?.detailTags]) {
    if (!values) continue;
    for (let i = 0; i < values.length; i += 1) {
      if (!Number.isFinite(values[i])) count += 1;
    }
  }
  return count;
}

function assertFiniteParams(params, context) {
  const count = nonfiniteParamCount(params);
  if (count > 0) throw runtimeSafetyError("safety_stop_nonfinite_params", context, { nonfinite_values: count });
}

function finalSplatInspectorNonfiniteCount(params, metrics) {
  const finalReadbackStep = metrics?.final_readback_step;
  const immutableFinal = Boolean(
    params &&
    metrics?.cpu_mirror_current &&
    Number.isFinite(finalReadbackStep) &&
    finalReadbackStep === metrics?.steps_done,
  );
  if (!immutableFinal) return nonfiniteParamCount(params);
  const cache = state.splatInspectorNonfiniteCache;
  if (
    cache?.params === params &&
    cache.finalReadbackStep === finalReadbackStep &&
    cache.count === params.count
  ) {
    return cache.value;
  }
  const value = nonfiniteParamCount(params);
  state.splatInspectorNonfiniteCache = { params, finalReadbackStep, count: params.count, value };
  return value;
}

function meanAbsDelta(a, b, length) {
  let total = 0;
  for (let i = 0; i < length; i += 1) total += Math.abs(b[i] - a[i]);
  return length > 0 ? total / length : 0;
}

function paramDeltaFromSnapshot(snapshot, params) {
  if (!snapshot) return null;
  const count = Math.min(snapshot.count, params.count);
  return {
    scope: "initial-prefix",
    count,
    position: meanAbsDelta(snapshot.xy, params.xy, count * 2),
    color: meanAbsDelta(snapshot.rgb, params.rgb, count * 3),
    opacity: meanAbsDelta(snapshot.opacity, params.opacity, count),
    scale: meanAbsDelta(snapshot.scale, params.scale, count * 2),
    rotation: meanAbsDelta(snapshot.theta, params.theta, count),
    layerOrder: meanAbsDelta(snapshot.depthOrder, params.depthOrder, count),
    virtualDepth: meanAbsDelta(snapshot.virtualDepth, params.virtualDepth, count),
  };
}

function boundaryLeakStats(params, sigma = params?.boundarySigma ?? selectedBoundarySigma()) {
  let count = 0;
  let maxLeak = 0;
  for (let i = 0; i < params.count; i += 1) {
    const extent = rotatedSplatExtent(params.scale[i * 2], params.scale[i * 2 + 1], params.theta?.[i] || 0, sigma);
    const leakX = Math.max(0, Math.abs(params.xy[i * 2]) + extent.x - 1);
    const leakY = Math.max(0, Math.abs(params.xy[i * 2 + 1]) + extent.y - 1);
    const leak = Math.max(leakX, leakY);
    if (leak > 1e-6) count += 1;
    maxLeak = Math.max(maxLeak, leak);
  }
  return { count, maxLeak };
}

function outsideRenderFootprintStats(params) {
  return boundaryLeakStats(params, RENDER_SIGMA);
}

function renderFootprintSupportFrame(image, params) {
  const frame = plyFrameScale(image);
  let supportX = 1;
  let supportY = 1;
  for (let i = 0; i < (params?.count || 0); i += 1) {
    const extent = rotatedExtentAtSigma(
      params.scale[i * 2],
      params.scale[i * 2 + 1],
      params.theta?.[i] || 0,
      RENDER_SIGMA,
    );
    supportX = Math.max(supportX, Math.abs(params.xy[i * 2]) + extent.x);
    supportY = Math.max(supportY, Math.abs(params.xy[i * 2 + 1]) + extent.y);
  }
  return {
    x: frame.x * supportX,
    y: frame.y * supportY,
    normalized_x: supportX,
    normalized_y: supportY,
  };
}

function optimizerFootprintHistogram(image = state.image, params = state.params) {
  if (!image || !params) return null;
  const labels = ["1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-127", "128-255", "256-511", "512-1023", "1024-2047", "2048-4095", "4096+"];
  const bins = Object.fromEntries(labels.map((label) => [label, 0]));
  const areas = [];
  let totalWaves = 0;
  let totalLaneCapacity = 0;
  const useEwa = phase33Variants().ewa2x2;
  const pixelSigma = MIP_PIXEL_SIGMA * 2 / Math.max(image.width, image.height);
  for (let i = 0; i < params.count; i += 1) {
    const theta = params.theta[i] || 0;
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const baseX = Math.max(0.0001, params.scale[i * 2]);
    const baseY = Math.max(0.0001, params.scale[i * 2 + 1]);
    const scaleX = useEwa ? baseX : Math.sqrt(baseX * baseX + pixelSigma * pixelSigma);
    const scaleY = useEwa ? baseY : Math.sqrt(baseY * baseY + pixelSigma * pixelSigma);
    const padX = useEwa && image.width > 1 ? 0.5 / (image.width - 1) : 0;
    const padY = useEwa && image.height > 1 ? 0.5 / (image.height - 1) : 0;
    const radiusX = RENDER_SIGMA * (Math.abs(c) * scaleX + Math.abs(s) * scaleY) + padX;
    const radiusY = RENDER_SIGMA * (Math.abs(s) * scaleX + Math.abs(c) * scaleY) + padY;
    const centerX = params.xy[i * 2];
    const centerY = params.xy[i * 2 + 1];
    const minX = Math.floor((Math.max(-1, centerX - radiusX) * 0.5 + 0.5) * Math.max(0, image.width - 1));
    const maxX = Math.ceil((Math.min(1, centerX + radiusX) * 0.5 + 0.5) * Math.max(0, image.width - 1));
    const minY = Math.floor((Math.max(-1, centerY - radiusY) * 0.5 + 0.5) * Math.max(0, image.height - 1));
    const maxY = Math.ceil((Math.min(1, centerY + radiusY) * 0.5 + 0.5) * Math.max(0, image.height - 1));
    const area = Math.max(1, maxX - minX + 1) * Math.max(1, maxY - minY + 1);
    areas.push(area);
    const exponent = Math.max(0, Math.ceil(Math.log2(area + 1)) - 1);
    const label = exponent >= labels.length - 1 ? labels[labels.length - 1] : labels[exponent];
    bins[label] += 1;
    const waves = Math.ceil(area / 64);
    totalWaves += waves;
    totalLaneCapacity += waves * 64;
  }
  areas.sort((a, b) => a - b);
  const percentile = (q) => areas[Math.min(areas.length - 1, Math.floor(Math.max(0, areas.length - 1) * q))] ?? null;
  return {
    source: "qa-final-cpu-mirror",
    count: areas.length,
    bins,
    minimum: areas[0] ?? null,
    median: percentile(0.5),
    p90: percentile(0.9),
    p99: percentile(0.99),
    maximum: areas[areas.length - 1] ?? null,
    total_64_lane_waves: totalWaves,
    lane_utilization: areas.reduce((sum, area) => sum + area, 0) / Math.max(1, totalLaneCapacity),
  };
}

function distributionStats(values, binCount = 8, fixedRange = null) {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!finite.length) {
    return { count: 0, min: 0, median: 0, mean: 0, max: 0, bins: new Array(binCount).fill(0), range: [0, 0] };
  }
  const min = fixedRange?.[0] ?? finite[0];
  const max = fixedRange?.[1] ?? finite[finite.length - 1];
  const span = Math.max(max - min, Number.EPSILON);
  const bins = new Array(binCount).fill(0);
  let sum = 0;
  for (const value of finite) {
    sum += value;
    const normalized = Math.max(0, Math.min(1, (value - min) / span));
    bins[Math.min(binCount - 1, Math.floor(normalized * binCount))] += 1;
  }
  return {
    count: finite.length,
    min: finite[0],
    median: finite[Math.floor(finite.length / 2)],
    mean: sum / finite.length,
    max: finite[finite.length - 1],
    bins,
    range: [min, max],
  };
}

function splatShapeStats(params, image) {
  const count = params?.count || 0;
  if (!count) return null;
  const sampling = finalDiagnosticSampling(count);
  const maxSide = Math.max(image?.width || 1, image?.height || 1);
  const pixelScale = maxSide * 0.5 * RENDER_SIGMA;
  const values = [];
  const bins = [0, 0, 0, 0, 0];
  let minScale = Infinity;
  let maxScale = 0;
  let sumScale = 0;
  let tinySplatCount = 0;
  let boundaryTinySplatCount = 0;
  let interiorTinySplatCount = 0;
  let anisotropySum = 0;
  let anisotropyMax = 1;
  let elongatedCount = 0;
  let boundarySplatCount = 0;
  let nonfiniteCount = 0;
  const opacityValues = [];
  const sxValues = [];
  const syValues = [];
  const geometricMeanScaleValues = [];
  const radiusValues = [];
  const anisotropyValues = [];
  const rotationValues = [];
  for (let i = 0; i < count; i += 1) {
    const rawValues = [
      params.xy[i * 2],
      params.xy[i * 2 + 1],
      params.scale[i * 2],
      params.scale[i * 2 + 1],
      params.opacity[i],
      params.theta?.[i] || 0,
    ];
    if (rawValues.some((value) => !Number.isFinite(value))) nonfiniteCount += 1;
    const sx = Math.max(MIN_SPLAT_SCALE, params.scale[i * 2]);
    const sy = Math.max(MIN_SPLAT_SCALE, params.scale[i * 2 + 1]);
    const opacity = params.opacity[i];
    const rotation = Math.atan2(Math.sin(params.theta?.[i] || 0), Math.cos(params.theta?.[i] || 0));
    const scale = (sx + sy) * 0.5;
    const major = Math.max(sx, sy);
    const minor = Math.max(MIN_SPLAT_SCALE, Math.min(sx, sy));
    const ratio = major / minor;
    const radiusPx = major * pixelScale;
    const areaScale = Math.sqrt(sx * sy);
    const edgeBandX = 8 * 2 / Math.max(1, image?.width || 1);
    const edgeBandY = 8 * 2 / Math.max(1, image?.height || 1);
    const boundaryAnchored =
      Math.abs(params.xy[i * 2]) >= 1 - edgeBandX ||
      Math.abs(params.xy[i * 2 + 1]) >= 1 - edgeBandY;
    if (i % sampling.stride === 0) {
      opacityValues.push(opacity);
      sxValues.push(sx);
      syValues.push(sy);
      geometricMeanScaleValues.push(areaScale);
      radiusValues.push(radiusPx);
      anisotropyValues.push(ratio);
      rotationValues.push(rotation);
      values.push(scale);
    }
    if (boundaryAnchored) boundarySplatCount += 1;
    minScale = Math.min(minScale, scale);
    maxScale = Math.max(maxScale, scale);
    sumScale += scale;
    anisotropySum += ratio;
    anisotropyMax = Math.max(anisotropyMax, ratio);
    if (areaScale <= MIN_SPLAT_SCALE * 1.05) {
      tinySplatCount += 1;
      if (boundaryAnchored) boundaryTinySplatCount += 1;
      else interiorTinySplatCount += 1;
    }
    if (ratio >= 1.8) elongatedCount += 1;
    if (radiusPx < 1) bins[0] += 1;
    else if (radiusPx < 2) bins[1] += 1;
    else if (radiusPx < 4) bins[2] += 1;
    else if (radiusPx < 8) bins[3] += 1;
    else bins[4] += 1;
  }
  values.sort((a, b) => a - b);
  const medianScale = values[Math.floor(values.length / 2)] || 0;
  return {
    count,
    inspection_sample_count: values.length,
    inspection_sample_stride: sampling.stride,
    min_scale: minScale,
    median_scale: medianScale,
    mean_scale: sumScale / count,
    max_scale: maxScale,
    scale_histogram: {
      radius_px_lt_1: bins[0],
      radius_px_1_2: bins[1],
      radius_px_2_4: bins[2],
      radius_px_4_8: bins[3],
      radius_px_gte_8: bins[4],
    },
    tiny_splat_count: tinySplatCount,
    tiny_splat_ratio: tinySplatCount / count,
    boundary_tiny_splat_count: boundaryTinySplatCount,
    boundary_tiny_splat_ratio: boundaryTinySplatCount / count,
    interior_tiny_splat_count: interiorTinySplatCount,
    interior_tiny_splat_ratio: interiorTinySplatCount / count,
    anisotropy_ratio_mean: anisotropySum / count,
    anisotropy_ratio_max: anisotropyMax,
    elongated_splat_count: elongatedCount,
    boundary_splat_count: boundarySplatCount,
    nonfinite_splat_count: nonfiniteCount,
    inspection: {
      opacity: distributionStats(opacityValues, 8, [0, 1]),
      scale_x: distributionStats(sxValues),
      scale_y: distributionStats(syValues),
      geometric_mean_scale: distributionStats(geometricMeanScaleValues),
      radius_px: distributionStats(radiusValues),
      anisotropy: distributionStats(anisotropyValues, 8, [1, Math.max(1, anisotropyMax)]),
      rotation: distributionStats(rotationValues, 8, [-Math.PI, Math.PI]),
    },
  };
}
