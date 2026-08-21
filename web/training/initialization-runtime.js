function initGaussians(image, count) {
  count = normalizeActiveSplatCount(count, DEFAULT_INITIAL_SPLATS);
  const boundarySigma = selectedBoundarySigma();
  const layout = splatGridLayout(image, count);
  const bg = new Float32Array([0, 0, 0]);
  const xy = new Float32Array(count * 2);
  const scale = new Float32Array(count * 2);
  const rgb = new Float32Array(count * 3);
  const opacity = new Float32Array(count);
  const theta = new Float32Array(count);
  const depthOrder = initialDepthOrder(count);
  const virtualDepth = new Float32Array(count);
  const detailTags = new Float32Array(count).fill(1);

  for (let i = 0; i < count; i += 1) {
    const grid = splatGridAt(layout, i);
    const shape = initialSplatShape(image, layout, i);
    const c = constrainSplat(grid.x, grid.y, shape.sx, shape.sy, shape.theta, boundarySigma);
    xy[i * 2] = c.x;
    xy[i * 2 + 1] = c.y;
    scale[i * 2] = c.sx;
    scale[i * 2 + 1] = c.sy;
    sampleImageAt(image, xy[i * 2], xy[i * 2 + 1], rgb, i * 3);
    opacity[i] = 0.98;
    theta[i] = shape.theta;
  }
  return {
    kernelShape: "gaussian",
    xy, scale, rgb, opacity, theta, depthOrder, virtualDepth, detailTags, count,
    rows: layout.rows, cols: layout.cols, bg,
    boundarySigma,
    layerOrderEnabled: Boolean(els.trainLayerOrder?.checked),
    discreteLayersEnabled: discreteLayerSettings().enabled,
    discreteLayerCount: discreteLayerSettings().count,
    virtualDepthEnabled: false,
    virtualDepthThickness: DEFAULT_VIRTUAL_DEPTH_THICKNESS,
    virtualDepthSoftConstraintEnabled: DEFAULT_VIRTUAL_DEPTH_SOFT_CONSTRAINT,
    virtualDepthPriorDelta: DEFAULT_VIRTUAL_DEPTH_PRIOR_DELTA,
  };
}

function initRectangles(image, count) {
  return initOpaqueLayeredPaint(image, count, "rectangle");
}

function strokeTensorAt(image, x, y, radius = 1) {
  const sampleRadius = Math.max(1, Math.round(radius));
  const dx = 2 * sampleRadius / Math.max(1, image.width - 1);
  const dy = 2 * sampleRadius / Math.max(1, image.height - 1);
  const sample = new Float32Array(3);
  const lumaAt = (px, py) => {
    sampleImageRgbBilinear(image, px, py, sample);
    return sample[0] * 0.299 + sample[1] * 0.587 + sample[2] * 0.114;
  };
  let jxx = 0;
  let jxy = 0;
  let jyy = 0;
  const derivativeScale = 0.5 / sampleRadius;
  for (let oy = -1; oy <= 1; oy += 1) {
    for (let ox = -1; ox <= 1; ox += 1) {
      const px = x + ox * dx;
      const py = y + oy * dy;
      const gx = derivativeScale * (lumaAt(px + dx, py) - lumaAt(px - dx, py));
      const gy = derivativeScale * (lumaAt(px, py + dy) - lumaAt(px, py - dy));
      jxx += gx * gx;
      jxy += gx * gy;
      jyy += gy * gy;
    }
  }
  return { jxx: jxx / 9, jxy: jxy / 9, jyy: jyy / 9 };
}

function strokeStructureFromTensor(tensor, image, fallbackTheta = 0) {
  const { jxx, jxy, jyy } = tensor;
  const dx = 2 / Math.max(1, image.width - 1);
  const dy = 2 / Math.max(1, image.height - 1);
  const trace = jxx + jyy;
  if (!Number.isFinite(trace) || trace < 1e-10) {
    return { theta: fallbackTheta, coherence: 0, energy: 0 };
  }
  const separation = Math.hypot(jxx - jyy, 2 * jxy);
  const normalPixelTheta = 0.5 * Math.atan2(2 * jxy, jxx - jyy);
  const tangentPixelX = -Math.sin(normalPixelTheta);
  const tangentPixelY = Math.cos(normalPixelTheta);
  return {
    theta: Math.atan2(tangentPixelY * dy, tangentPixelX * dx),
    coherence: Math.max(0, Math.min(1, separation / Math.max(trace, 1e-10))),
    energy: trace,
  };
}

function strokeStructureAt(image, x, y, fallbackTheta = 0) {
  return strokeStructureFromTensor(strokeTensorAt(image, x, y, 1), image, fallbackTheta);
}

function rectangleDirectedTaperTheta(image, x, y, theta, enabled = true) {
  if (!enabled) return theta;
  const normalX = -Math.sin(theta);
  const normalY = Math.cos(theta);
  const offset = 4 / Math.max(1, Math.max(image.width, image.height) - 1);
  const positive = strokeStructureAt(
    image,
    x + normalX * offset,
    y + normalY * offset,
    theta,
  ).energy;
  const negative = strokeStructureAt(
    image,
    x - normalX * offset,
    y - normalY * offset,
    theta,
  ).energy;
  // Local -Y is the narrow edge. A pi flip preserves the axial direction
  // while putting that narrow edge on the stronger-detail side.
  return positive > negative + 1e-10 ? theta + Math.PI : theta;
}

function oilStrokeStructureAt(image, x, y, fallbackTheta = 0) {
  const fine = strokeTensorAt(image, x, y, 1);
  const medium = strokeTensorAt(image, x, y, 3);
  const coarse = strokeTensorAt(image, x, y, 7);
  return strokeStructureFromTensor({
    jxx: fine.jxx * 0.55 + medium.jxx * 0.30 + coarse.jxx * 0.15,
    jxy: fine.jxy * 0.55 + medium.jxy * 0.30 + coarse.jxy * 0.15,
    jyy: fine.jyy * 0.55 + medium.jyy * 0.30 + coarse.jyy * 0.15,
  }, image, fallbackTheta);
}

function blendAxialAngle(from, to, amount) {
  const t = Math.max(0, Math.min(1, amount));
  const x = (1 - t) * Math.cos(2 * from) + t * Math.cos(2 * to);
  const y = (1 - t) * Math.sin(2 * from) + t * Math.sin(2 * to);
  return 0.5 * Math.atan2(y, x);
}

function oilStrokeHierarchy(structure, fallbackTheta) {
  const detail = Math.max(0, Math.min(1, Math.sqrt(Math.max(0, structure.energy) / 0.0025)));
  const directionWeight = Math.max(0, Math.min(1, (structure.coherence - 0.08) / 0.37));
  return {
    detail,
    scaleMultiplier: 1.65 + (0.62 - 1.65) * detail,
    anisotropy: 1.45 + 5.55 * structure.coherence * (0.70 + 0.30 * detail),
    theta: blendAxialAngle(fallbackTheta, structure.theta, directionWeight * directionWeight * (3 - 2 * directionWeight)),
  };
}

function illustrativeOilFamily(detail, anisotropy) {
  if (detail >= ILLUSTRATIVE_OIL_DETAIL_THRESHOLD) return 2;
  return anisotropy >= ILLUSTRATIVE_OIL_RIBBON_ANISOTROPY ? 1 : 0;
}

function gaussianBlurScalar(values, width, height, sigma) {
  const boundedSigma = Math.max(0.6, Math.min(4, Number(sigma) || 1));
  const radius = Math.max(1, Math.min(10, Math.ceil(boundedSigma * 3)));
  const kernel = new Float32Array(radius * 2 + 1);
  let kernelSum = 0;
  for (let offset = -radius; offset <= radius; offset += 1) {
    const value = Math.exp(-(offset * offset) / (2 * boundedSigma * boundedSigma));
    kernel[offset + radius] = value;
    kernelSum += value;
  }
  for (let index = 0; index < kernel.length; index += 1) kernel[index] /= kernelSum;
  const horizontal = new Float32Array(values.length);
  const result = new Float32Array(values.length);
  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    for (let x = 0; x < width; x += 1) {
      let sum = 0;
      for (let offset = -radius; offset <= radius; offset += 1) {
        sum += values[row + Math.max(0, Math.min(width - 1, x + offset))] * kernel[offset + radius];
      }
      horizontal[row + x] = sum;
    }
  }
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let sum = 0;
      for (let offset = -radius; offset <= radius; offset += 1) {
        const sampleY = Math.max(0, Math.min(height - 1, y + offset));
        sum += horizontal[sampleY * width + x] * kernel[offset + radius];
      }
      result[y * width + x] = sum;
    }
  }
  return result;
}

function initLayeredOpaqueBrush(image, count) {
  return initOpaqueLayeredPaint(image, count, "opaque-brush");
}

function initOpaqueLayeredPaint(image, count, kernelShape) {
  const params = initGaussians(image, count);
  const seededTheta = params.theta.slice();
  params.kernelShape = kernelShape;
  params.rectangleTopRatio = kernelShape === "rectangle"
    ? selectedRectangleTopRatio()
    : DEFAULT_RECTANGLE_TOP_RATIO;
  params.rectangleTopRatioMax = kernelShape === "rectangle"
    ? selectedRectangleTopRatioMax(params.rectangleTopRatio)
    : DEFAULT_RECTANGLE_TOP_RATIO_MAX;
  const rectangleOpacityGradient = selectedRectangleOpacityGradient();
  params.rectangleOpacityGradientMin = rectangleOpacityGradient.min;
  params.rectangleOpacityGradientMax = rectangleOpacityGradient.max;
  const rectangleCenterOpacityGradient = selectedRectangleCenterOpacityGradient();
  params.rectangleCenterOpacityGradientMin = rectangleCenterOpacityGradient.min;
  params.rectangleCenterOpacityGradientMax = rectangleCenterOpacityGradient.max;
  const learnedOpacity = kernelShape === "rectangle"
    ? selectedRectangleLearnedOpacity()
    : selectedLayeredBrushLearnedOpacity();
  params.rectangleMinAspectRatio = kernelShape === "rectangle"
    ? selectedRectangleMinAspectRatio()
    : DEFAULT_RECTANGLE_MIN_ASPECT_RATIO;
  params.rectangleMaxAspectRatio = kernelShape === "rectangle"
    ? selectedRectangleMaxAspectRatio(params.rectangleMinAspectRatio)
    : DEFAULT_RECTANGLE_ASPECT_RATIO;
  params.rectangleOrientation = kernelShape === "rectangle"
    ? selectedRectangleOrientation()
    : DEFAULT_RECTANGLE_ORIENTATION;
  const rectangleShape = selectedRectangleShapeSettings();
  params.rectanglePreserveArea =
    kernelShape === "rectangle" ? rectangleShape.preserveArea : DEFAULT_RECTANGLE_PRESERVE_AREA;
  params.rectangleEdgeDirectedTaper =
    kernelShape === "rectangle"
      ? rectangleShape.edgeDirectedTaper
      : DEFAULT_RECTANGLE_EDGE_DIRECTED_TAPER;
  params.rectangleStructureAwareRatio =
    kernelShape === "rectangle"
      ? rectangleShape.structureAwareRatio
      : DEFAULT_RECTANGLE_STRUCTURE_AWARE_RATIO;
  params.rectangleAsymmetricSoftness =
    kernelShape === "rectangle"
      ? rectangleShape.asymmetricSoftness
      : DEFAULT_RECTANGLE_ASYMMETRIC_SOFTNESS;
  params.opaqueLayered = true;
  params.currentVisibilityChildPolicyEnabled = opaquePaintCurrentVisibilityChildPolicyEnabled();
  params.currentVisibilityCompactionEnabled = opaquePaintCurrentVisibilityCompactionEnabled();
  const surfaceLayerPrior = scaleBiasedSurfaceLayerPriorSettings();
  params.surfaceLayerPriorEnabled = surfaceLayerPrior.enabled;
  params.surfaceLayerPriorColorAwarePromotion = surfaceLayerPrior.colorAwarePromotion;
  params.trainLayerColorGuardEnabled = trainLayerColorGuardEnabled();
  params.surfaceLayerPriorLayers = surfaceLayerPrior.layers;
  params.surfaceLayerPriorP1Interval = surfaceLayerPrior.p1Interval;
  params.surfaceLayerPriorP2Interval = surfaceLayerPrior.p2Interval;
  params.surfaceLayerPriorP3Interval = surfaceLayerPrior.p3Interval;
  params.surfaceLayerPriorUntilFraction = surfaceLayerPrior.untilFraction;
  params.harmfulRectangleParentSplitEnabled =
    harmfulRectangleParentSplitSettings().enabled;
  params.frontSplitChildrenEnabled = frontSplitChildrenSettings().enabled;
  const directionalEffects = selectedLayeredBrushDirectionalEffects();
  const brushCenterOpacityGradient = selectedLayeredBrushCenterOpacityGradient();
  params.minimumOpacityEnabled = true;
  params.minimumOpacity = learnedOpacity.min;
  params.maximumOpacity = learnedOpacity.max;
  params.boundarySigma = Math.min(
    params.boundarySigma,
    kernelShape === "rectangle" ? RECTANGLE_KERNEL_EXTENT : LAYERED_OPAQUE_BRUSH_KERNEL_EXTENT,
  );
  params.layerOrderEnabled = true;
  params.layerAwareAccumulationEnabled = true;
  params.discreteLayersEnabled = true;
  params.discreteLayerMoveRadius = LAYERED_OPAQUE_BRUSH_LAYER_MOVE_RADIUS;
  applyAdaptiveBspPaintInitialization(image, params, kernelShape);
  const baseCount = params.count;
  params.opacity.fill(params.maximumOpacity);
  const illustrativeOil = kernelShape === "opaque-brush";
  params.brushOpacityGradientEnabled = illustrativeOil && directionalEffects.opacity;
  params.brushOpacityGradientStart = directionalEffects.opacityStart;
  params.brushOpacityGradientEnd = directionalEffects.opacityEnd;
  params.brushCenterOpacityGradientMin = brushCenterOpacityGradient.min;
  params.brushCenterOpacityGradientMax = brushCenterOpacityGradient.max;
  params.brushWidthTaperEnabled = illustrativeOil && directionalEffects.widthTaper;
  params.brushWidthTaperStart = directionalEffects.widthStart;
  params.brushWidthTaperEnd = directionalEffects.widthEnd;
  params.brushTaper = new Float32Array(baseCount).fill(DEFAULT_LAYERED_BRUSH_TAPER);
  params.illustrativeOilVersion = illustrativeOil ? 1 : 0;
  params.brushLocalColorFlowEnabled = illustrativeOil && opaqueBrushLocalColorFlowEnabled();
  params.brushStrokePersistenceEnabled = illustrativeOil && opaqueBrushStrokePersistenceEnabled();
  params.brushMinAspectRatio = illustrativeOil
    ? selectedBrushMinAspectRatio()
    : LIMITS.maxAnisotropyMin;
  params.brushMaxAspectRatio = illustrativeOil
    ? selectedBrushMaxAspectRatio()
    : DEFAULT_MAX_ANISOTROPY;
  const brushAspectFloors = selectedBrushAspectFloors();
  params.brushRibbonAspectFloor = brushAspectFloors.ribbon;
  params.brushAccentAspectFloor = brushAspectFloors.accent;
  const illustrativeOilFamilyCounts = [0, 0, 0];
  for (let i = 0; i < baseCount; i += 1) {
    const x = params.xy[i * 2];
    const y = params.xy[i * 2 + 1];
    const structure = illustrativeOil
      ? oilStrokeStructureAt(image, x, y, seededTheta[i])
      : strokeStructureAt(image, x, y, seededTheta[i]);
    const areaScale = Math.sqrt(Math.max(MIN_SPLAT_SCALE ** 2, params.scale[i * 2] * params.scale[i * 2 + 1]));
    const oilHierarchy = illustrativeOil
      ? oilStrokeHierarchy(structure, seededTheta[i])
      : null;
    const anisotropy = oilHierarchy?.anisotropy ?? (1.35 + 2.65 * structure.coherence);
    // Preserve Rectangle's established image-structure initialization. The
    // configurable minimum is an optimizer constraint, not a reason to change
    // the starting footprint distribution.
    const stretch = Math.sqrt(anisotropy);
    const scaleMultiplier = oilHierarchy?.scaleMultiplier ?? 1;
    const unconstrainedTheta = illustrativeOil
      ? oilHierarchy.theta
      : rectangleDirectedTaperTheta(
          image,
          x,
          y,
          structure.theta,
          params.rectangleEdgeDirectedTaper &&
            params.rectangleTopRatio < 1 - 0.000001,
        );
    const nextSx = areaScale * scaleMultiplier * stretch;
    const nextSy = areaScale * scaleMultiplier / stretch;
    const theta = kernelShape === "rectangle"
      ? constrainedRectangleTheta(
          unconstrainedTheta,
          nextSx,
          nextSy,
          params.rectangleOrientation,
        )
      : unconstrainedTheta;
    const constrained = constrainSplat(
      x,
      y,
      nextSx,
      nextSy,
      theta,
      params.boundarySigma,
      kernelShape === "rectangle"
        ? Math.min(params.rectangleMaxAspectRatio, Math.max(4, anisotropy))
        : Math.max(7, anisotropy),
    );
    params.xy[i * 2] = constrained.x;
    params.xy[i * 2 + 1] = constrained.y;
    params.scale[i * 2] = constrained.sx;
    params.scale[i * 2 + 1] = constrained.sy;
    params.theta[i] = theta;
    if (illustrativeOil) {
      const constrainedAnisotropy =
        Math.max(constrained.sx, constrained.sy) / Math.max(MIN_SPLAT_SCALE, Math.min(constrained.sx, constrained.sy));
      const family = illustrativeOilFamily(oilHierarchy.detail, constrainedAnisotropy);
      params.detailTags[i] = family === 2 ? 2 : 1;
      illustrativeOilFamilyCounts[family] += 1;
    } else if (params.rectangleStructureAwareRatio) {
      params.detailTags[i] =
        structure.coherence >= RECTANGLE_STRUCTURE_MIN_COHERENCE &&
        structure.energy >= RECTANGLE_STRUCTURE_MIN_ENERGY
          ? 2
          : 1;
    }
  }
  params.illustrativeOilFamilyStats = illustrativeOil
    ? {
        version: 1,
        selection: "deterministic-local-structure",
        base_patch: illustrativeOilFamilyCounts[0],
        directional_ribbon: illustrativeOilFamilyCounts[1],
        edge_accent: illustrativeOilFamilyCounts[2],
      }
    : null;
  for (let i = 0; i < baseCount; i += 1) {
    footprintWeightedTargetColor(image, params, i, params.rgb.subarray(i * 3, i * 3 + 3));
  }
  return params;
}

