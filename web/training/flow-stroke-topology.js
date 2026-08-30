(function installFlowStrokeTopology(global) {
  "use strict";

  const PARAM_STRIDE = 16;
  const DEFAULT_OPTIONS = Object.freeze({
    minimumWidthFactor: 0.55,
    maximumWidthFactor: 1.65,
    frontWidthMaximumFactor: 3,
    fixedOpacity: 0.995,
    splitFraction: 0.04,
    mergeFraction: 0.01,
    maximumSplitsPerEvent: 24,
    maximumMergesPerEvent: 6,
    splitApplyUntil: 0.75,
    paintCurriculumEnabled: true,
    startingWidthDivisor: 32,
    startingLengthPercent: 160,
    residualMovePerEventPx: 1.5,
    residualMoveFraction: 0.02,
    maximumMovesPerEvent: 48,
    minimumMoveGain: 0.0025,
    maximumTotalMovementPx: 12,
    textureGuidedAllocation: false,
    scaleMatchedResidualRepaint: false,
  });

  function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, Number(value) || 0));
  }

  function clamp01(value) {
    return clamp(value, 0, 1);
  }

  function fract(value) {
    return value - Math.floor(value);
  }

  function hash01(value) {
    return fract(Math.sin((Number(value) || 0) * 91.3458 + 12.345) * 47453.5453);
  }

  function sigmoid(value) {
    return 1 / (1 + Math.exp(-Number(value)));
  }

  function fixedStrokeOpacity(options) {
    return clamp(options.fixedOpacity, 0.05, 0.995);
  }

  function widthFactorRangeForLayer(layer, options) {
    const front = Math.round(Number(layer) || 0) === 2;
    return {
      minimum: options.minimumWidthFactor,
      maximum: front
        ? Math.max(options.minimumWidthFactor, options.frontWidthMaximumFactor)
        : options.maximumWidthFactor,
    };
  }

  function signalSrgbToLinear(value) {
    const channel = clamp01(value);
    return channel <= 0.04045
      ? channel / 12.92
      : ((channel + 0.055) / 1.055) ** 2.4;
  }

  function point(stroke, name) {
    return [Number(stroke[`${name}_x`]) || 0, Number(stroke[`${name}_y`]) || 0];
  }

  function mixPoint(a, b, t) {
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  }

  function normalize(vector, fallback = [1, 0]) {
    const length = Math.hypot(vector[0], vector[1]);
    return length > 1e-6 ? [vector[0] / length, vector[1] / length] : fallback.slice();
  }

  function cubicPoint(stroke, t) {
    const p0 = point(stroke, "start");
    const p1 = point(stroke, "control_1");
    const p2 = point(stroke, "control_2");
    const p3 = point(stroke, "end");
    const s = 1 - t;
    return [
      s * s * s * p0[0] + 3 * s * s * t * p1[0] + 3 * s * t * t * p2[0] + t * t * t * p3[0],
      s * s * s * p0[1] + 3 * s * s * t * p1[1] + 3 * s * t * t * p2[1] + t * t * t * p3[1],
    ];
  }

  function cubicArcLength(stroke, segments = 12) {
    let previous = cubicPoint(stroke, 0);
    let length = 0;
    for (let index = 1; index <= segments; index += 1) {
      const next = cubicPoint(stroke, index / segments);
      length += Math.hypot(next[0] - previous[0], next[1] - previous[1]);
      previous = next;
    }
    return length;
  }

  function withDerivedGeometry(stroke) {
    const center = cubicPoint(stroke, 0.5);
    return {
      ...stroke,
      center_x: center[0],
      center_y: center[1],
      path_length_px: cubicArcLength(stroke),
    };
  }

  function withTopologyOrigin(stroke) {
    const derived = withDerivedGeometry(stroke);
    return {
      ...derived,
      topology_origin_center_x: Number.isFinite(Number(derived.topology_origin_center_x))
        ? Number(derived.topology_origin_center_x)
        : derived.center_x,
      topology_origin_center_y: Number.isFinite(Number(derived.topology_origin_center_y))
        ? Number(derived.topology_origin_center_y)
        : derived.center_y,
      topology_target_half_width_px: Math.max(
        0.55,
        Number(derived.topology_target_half_width_px) || Number(derived.half_width_px) || 0.55,
      ),
      topology_curriculum_length_scale: Math.max(
        1e-3,
        Number(derived.topology_curriculum_length_scale) || 1,
      ),
    };
  }

  function validTextureGuide(guide) {
    const width = Math.max(1, Math.round(Number(guide?.width) || 0));
    const height = Math.max(1, Math.round(Number(guide?.height) || 0));
    const count = width * height;
    return guide?.score instanceof Float32Array
      && guide.score.length === count
      && guide.dabVisibility instanceof Float32Array
      && guide.dabVisibility.length === count
      && guide.darkFlat instanceof Float32Array
      && guide.darkFlat.length === count;
  }

  function guideValueAt(guide, field, x, y, fallback) {
    if (!validTextureGuide(guide) || !(guide[field] instanceof Float32Array)) return fallback;
    const px = Math.max(0, Math.min(guide.width - 1, Math.round(Number(x) || 0)));
    const py = Math.max(0, Math.min(guide.height - 1, Math.round(Number(y) || 0)));
    return clamp01(guide[field][py * guide.width + px]);
  }

  function strokeGuideStats(stroke, guide) {
    if (!validTextureGuide(guide)) {
      return { texture: 1, edge: 1, visibility: 1, darkFlat: 0 };
    }
    const sampleTs = [0.10, 0.30, 0.50, 0.70, 0.90];
    let textureTotal = 0;
    let textureMaximum = 0;
    let edgeTotal = 0;
    let edgeMaximum = 0;
    let visibilityTotal = 0;
    let darkFlatTotal = 0;
    for (const t of sampleTs) {
      const position = cubicPoint(stroke, t);
      const texture = guideValueAt(guide, "score", position[0], position[1], 1);
      textureTotal += texture;
      textureMaximum = Math.max(textureMaximum, texture);
      const edge = guideValueAt(guide, "edgeScore", position[0], position[1], texture);
      edgeTotal += edge;
      edgeMaximum = Math.max(edgeMaximum, edge);
      visibilityTotal += guideValueAt(
        guide,
        "dabVisibility",
        position[0],
        position[1],
        texture,
      );
      darkFlatTotal += guideValueAt(guide, "darkFlat", position[0], position[1], 0);
    }
    const sampleCount = sampleTs.length;
    return {
      texture: clamp01(textureTotal / sampleCount * 0.65 + textureMaximum * 0.35),
      // A crossing of one strong source boundary is enough to justify the two
      // thin-long physical Splats, so edge maximum carries more weight than
      // the path mean. Texture allocation remains a separate statistic.
      edge: clamp01(edgeTotal / sampleCount * 0.35 + edgeMaximum * 0.65),
      visibility: clamp01(visibilityTotal / sampleCount * 0.65 + textureMaximum * 0.35),
      darkFlat: clamp01(darkFlatTotal / sampleCount),
    };
  }

  function annotateTextureGuide(stroke, guide) {
    if (!validTextureGuide(guide)) return stroke;
    const stats = strokeGuideStats(stroke, guide);
    return {
      ...stroke,
      texture_score: stats.texture,
      edge_score: stats.edge,
      dab_visibility_score: stats.visibility,
      dark_flat_score: stats.darkFlat,
    };
  }

  function translateStroke(stroke, deltaX, deltaY) {
    return withDerivedGeometry({
      ...stroke,
      start_x: Number(stroke.start_x) + deltaX,
      start_y: Number(stroke.start_y) + deltaY,
      control_1_x: Number(stroke.control_1_x) + deltaX,
      control_1_y: Number(stroke.control_1_y) + deltaY,
      control_2_x: Number(stroke.control_2_x) + deltaX,
      control_2_y: Number(stroke.control_2_y) + deltaY,
      end_x: Number(stroke.end_x) + deltaX,
      end_y: Number(stroke.end_y) + deltaY,
    });
  }

  function scaleStrokeAroundCenter(stroke, scale) {
    const boundedScale = Math.max(0.05, Math.min(4, Number(scale) || 1));
    const center = cubicPoint(stroke, 0.5);
    const scaleCoordinate = (value, axis) => center[axis] + (Number(value) - center[axis]) * boundedScale;
    return withDerivedGeometry({
      ...stroke,
      start_x: scaleCoordinate(stroke.start_x, 0),
      start_y: scaleCoordinate(stroke.start_y, 1),
      control_1_x: scaleCoordinate(stroke.control_1_x, 0),
      control_1_y: scaleCoordinate(stroke.control_1_y, 1),
      control_2_x: scaleCoordinate(stroke.control_2_x, 0),
      control_2_y: scaleCoordinate(stroke.control_2_y, 1),
      end_x: scaleCoordinate(stroke.end_x, 0),
      end_y: scaleCoordinate(stroke.end_y, 1),
    });
  }

  function varyStroke(stroke, sourceIndex, options) {
    if (stroke.underpaint_splat === true) return { ...stroke };
    const random = Number(stroke.random) || hash01(sourceIndex + 1);
    const widthRandom = hash01(random + sourceIndex * 0.754877666);
    const widthRange = widthFactorRangeForLayer(stroke.layer, options);
    const widthFactor = widthRange.minimum
      + (widthRange.maximum - widthRange.minimum) * widthRandom;
    const varied = withDerivedGeometry({
      ...stroke,
      half_width_px: Math.max(0.55, Number(stroke.half_width_px) * widthFactor),
      opacity: fixedStrokeOpacity(options),
      topology_source_index: sourceIndex,
      topology_generation: 0,
      topology_kind: "source",
    });
    return withTopologyOrigin({
      ...varied,
      topology_target_half_width_px: varied.half_width_px,
      topology_curriculum_length_scale: 1,
    });
  }

  function strokeFromParams(template, params, index) {
    const base = index * PARAM_STRIDE;
    if (!(params instanceof Float32Array) || base + 15 >= params.length) {
      return withDerivedGeometry({ ...template });
    }
    return withDerivedGeometry({
      ...template,
      start_x: params[base],
      start_y: params[base + 1],
      control_1_x: params[base + 2],
      control_1_y: params[base + 3],
      control_2_x: params[base + 4],
      control_2_y: params[base + 5],
      end_x: params[base + 6],
      end_y: params[base + 7],
      half_width_px: Math.max(0.55, params[base + 8]),
      opacity: clamp(sigmoid(params[base + 9]), 0.01, 0.995),
      paint_linear_r: clamp01(params[base + 10]),
      paint_linear_g: clamp01(params[base + 11]),
      paint_linear_b: clamp01(params[base + 12]),
      random: params[base + 13],
      layer: Math.max(0, Math.min(3, Math.round(params[base + 14]))),
    });
  }

  function scaleMatchedResidualRadius(image, progress) {
    const width = Math.max(1, Math.round(Number(image?.width) || 1));
    const height = Math.max(1, Math.round(Number(image?.height) || 1));
    const maximumRadius = Math.max(2, Math.min(16, Math.round(Math.min(width, height) / 32)));
    const remaining = Math.max(0, 1 - clamp01(progress) / 0.90);
    return remaining > 0 ? Math.max(1, Math.round(maximumRadius * remaining)) : 0;
  }

  function buildScaleMatchedResidualEvidence(image, renderedLinearRgba, progress) {
    if (!(image?.rgb instanceof Float32Array) || !(renderedLinearRgba instanceof Float32Array)) {
      return null;
    }
    const width = Math.max(1, Math.round(Number(image.width) || 1));
    const height = Math.max(1, Math.round(Number(image.height) || 1));
    const radius = scaleMatchedResidualRadius(image, progress);
    const values = new Float32Array(width * height);
    for (let pixel = 0; pixel < values.length; pixel += 1) {
      const targetBase = pixel * 3;
      const renderBase = pixel * 4;
      values[pixel] = (
        Math.abs(signalSrgbToLinear(image.rgb[targetBase]) - renderedLinearRgba[renderBase])
        + Math.abs(signalSrgbToLinear(image.rgb[targetBase + 1]) - renderedLinearRgba[renderBase + 1])
        + Math.abs(signalSrgbToLinear(image.rgb[targetBase + 2]) - renderedLinearRgba[renderBase + 2])
      ) / 3 + clamp01(renderedLinearRgba[renderBase + 3]) * 0.18;
    }
    if (radius === 0) return { width, height, radius, values };

    const stride = width + 1;
    const integral = new Float32Array((width + 1) * (height + 1));
    for (let y = 0; y < height; y += 1) {
      let rowSum = 0;
      const sourceRow = y * width;
      const integralRow = (y + 1) * stride;
      const previousRow = y * stride;
      for (let x = 0; x < width; x += 1) {
        rowSum += values[sourceRow + x];
        integral[integralRow + x + 1] = integral[previousRow + x + 1] + rowSum;
      }
    }
    const blurred = new Float32Array(values.length);
    for (let y = 0; y < height; y += 1) {
      const y0 = Math.max(0, y - radius);
      const y1 = Math.min(height - 1, y + radius);
      for (let x = 0; x < width; x += 1) {
        const x0 = Math.max(0, x - radius);
        const x1 = Math.min(width - 1, x + radius);
        const sum = integral[(y1 + 1) * stride + x1 + 1]
          - integral[y0 * stride + x1 + 1]
          - integral[(y1 + 1) * stride + x0]
          + integral[y0 * stride + x0];
        blurred[y * width + x] = sum / ((x1 - x0 + 1) * (y1 - y0 + 1));
      }
    }
    return { width, height, radius, values: blurred };
  }

  function residualEvidenceAt(evidence, x, y) {
    if (!evidence) return null;
    const sampleX = Math.max(0, Math.min(evidence.width - 1, Math.round(x)));
    const sampleY = Math.max(0, Math.min(evidence.height - 1, Math.round(y)));
    return evidence.values[sampleY * evidence.width + sampleX];
  }

  function sampleResidual(stroke, image, renderedLinearRgba, residualEvidence = null) {
    if (!(image?.rgb instanceof Float32Array) || !(renderedLinearRgba instanceof Float32Array)) {
      return 0;
    }
    const width = Math.max(1, Math.round(Number(image.width) || 1));
    const height = Math.max(1, Math.round(Number(image.height) || 1));
    let total = 0;
    const sampleTs = [0.10, 0.30, 0.50, 0.70, 0.90];
    for (const t of sampleTs) {
      const position = cubicPoint(stroke, t);
      const x = Math.max(0, Math.min(width - 1, Math.round(position[0])));
      const y = Math.max(0, Math.min(height - 1, Math.round(position[1])));
      const pixel = y * width + x;
      const matchedResidual = residualEvidenceAt(residualEvidence, x, y);
      if (matchedResidual !== null) {
        total += matchedResidual;
        continue;
      }
      const targetBase = pixel * 3;
      const renderBase = pixel * 4;
      let rgbResidual = 0;
      for (let channel = 0; channel < 3; channel += 1) {
        rgbResidual += Math.abs(
          signalSrgbToLinear(image.rgb[targetBase + channel])
          - renderedLinearRgba[renderBase + channel],
        );
      }
      total += rgbResidual / 3 + clamp01(renderedLinearRgba[renderBase + 3]) * 0.18;
    }
    return total / sampleTs.length;
  }

  function samplePlacementScore(stroke, image, renderedLinearRgba, options, residualEvidence = null) {
    if (!(image?.rgb instanceof Float32Array) || !(renderedLinearRgba instanceof Float32Array)) {
      return 0;
    }
    const width = Math.max(1, Math.round(Number(image.width) || 1));
    const height = Math.max(1, Math.round(Number(image.height) || 1));
    const pigment = strokeColor(stroke);
    const sampleTs = [0.10, 0.30, 0.50, 0.70, 0.90];
    let total = 0;
    for (const t of sampleTs) {
      const position = cubicPoint(stroke, t);
      const x = Math.max(0, Math.min(width - 1, Math.round(position[0])));
      const y = Math.max(0, Math.min(height - 1, Math.round(position[1])));
      const pixel = y * width + x;
      const targetBase = pixel * 3;
      const renderBase = pixel * 4;
      const target = [
        signalSrgbToLinear(image.rgb[targetBase]),
        signalSrgbToLinear(image.rgb[targetBase + 1]),
        signalSrgbToLinear(image.rgb[targetBase + 2]),
      ];
      const matchedResidual = residualEvidenceAt(residualEvidence, x, y);
      const residual = matchedResidual ?? ((
        Math.abs(target[0] - renderedLinearRgba[renderBase])
        + Math.abs(target[1] - renderedLinearRgba[renderBase + 1])
        + Math.abs(target[2] - renderedLinearRgba[renderBase + 2])
      ) / 3 + clamp01(renderedLinearRgba[renderBase + 3]) * 0.18);
      const pigmentDistanceSquared = (
        (pigment[0] - target[0]) ** 2
        + (pigment[1] - target[1]) ** 2
        + (pigment[2] - target[2]) ** 2
      ) / 3;
      const colorCompatibility = Math.exp(-8 * pigmentDistanceSquared);
      total += residual * (0.18 + 0.82 * colorCompatibility);
    }
    const residualScore = total / sampleTs.length;
    if (!options?.textureGuidedAllocation) return residualScore;
    const guide = strokeGuideStats(stroke, options.textureGuide);
    return residualScore * (0.30 + guide.texture * 0.70)
      + guide.texture * 0.012
      - guide.darkFlat * 0.004;
  }

  function movementBoundedTranslation(stroke, deltaX, deltaY, options) {
    const originX = Number(stroke.topology_origin_center_x);
    const originY = Number(stroke.topology_origin_center_y);
    const center = cubicPoint(stroke, 0.5);
    const maximum = Math.max(0, Number(options.maximumTotalMovementPx) || 0);
    if (!(maximum > 0) || !Number.isFinite(originX) || !Number.isFinite(originY)) {
      return maximum > 0 ? [deltaX, deltaY] : [0, 0];
    }
    const requestedX = center[0] + deltaX - originX;
    const requestedY = center[1] + deltaY - originY;
    const distance = Math.hypot(requestedX, requestedY);
    if (distance <= maximum) return [deltaX, deltaY];
    const scale = maximum / Math.max(1e-6, distance);
    return [
      originX + requestedX * scale - center[0],
      originY + requestedY * scale - center[1],
    ];
  }

  function clampStrokeToMovementLimit(stroke, options) {
    const prepared = withTopologyOrigin(stroke);
    const center = cubicPoint(prepared, 0.5);
    const originX = Number(prepared.topology_origin_center_x);
    const originY = Number(prepared.topology_origin_center_y);
    const maximum = Math.max(0, Number(options.maximumTotalMovementPx) || 0);
    const distance = Math.hypot(center[0] - originX, center[1] - originY);
    if (!(maximum > 0) || distance <= maximum) return prepared;
    const scale = maximum / Math.max(1e-6, distance);
    return translateStroke(
      prepared,
      originX + (center[0] - originX) * scale - center[0],
      originY + (center[1] - originY) * scale - center[1],
    );
  }

  function optimizeResidualPlacement(
    plan,
    image,
    renderedLinearRgba,
    progress,
    options,
    residualEvidence = null,
  ) {
    const baseStep = Math.max(0, Number(options.residualMovePerEventPx) || 0);
    if (!(baseStep > 0)
      || !(image?.rgb instanceof Float32Array)
      || !(renderedLinearRgba instanceof Float32Array)) {
      return { plan, movedCount: 0, meanMovePx: 0, meanGain: 0 };
    }
    const boundedProgress = clamp01(progress);
    const localStep = baseStep * (0.20 + 0.80 * (1 - boundedProgress));
    const step = options.scaleMatchedResidualRepaint && residualEvidence
      ? Math.min(
        Math.max(localStep, residualEvidence.radius * 0.85),
        Math.max(localStep, options.maximumTotalMovementPx),
      )
      : localStep;
    const halfStep = step * 0.5;
    const diagonal = step / Math.sqrt(2);
    const halfDiagonal = halfStep / Math.sqrt(2);
    const offsets = [
      [step, 0], [-step, 0], [0, step], [0, -step],
      [diagonal, diagonal], [diagonal, -diagonal],
      [-diagonal, diagonal], [-diagonal, -diagonal],
      ...(options.scaleMatchedResidualRepaint && step > localStep + 1e-4 ? [
        [halfStep, 0], [-halfStep, 0], [0, halfStep], [0, -halfStep],
        [halfDiagonal, halfDiagonal], [halfDiagonal, -halfDiagonal],
        [-halfDiagonal, halfDiagonal], [-halfDiagonal, -halfDiagonal],
      ] : []),
    ];
    const proposals = [];
    for (let index = 0; index < plan.length; index += 1) {
      const stroke = clampStrokeToMovementLimit(plan[index], options);
      if (stroke.underpaint_splat === true) continue;
      const currentScore = samplePlacementScore(
        stroke,
        image,
        renderedLinearRgba,
        options,
        residualEvidence,
      );
      let best = null;
      for (const offset of offsets) {
        const bounded = movementBoundedTranslation(stroke, offset[0], offset[1], options);
        if (Math.hypot(bounded[0], bounded[1]) < 1e-4) continue;
        const candidate = translateStroke(stroke, bounded[0], bounded[1]);
        const score = samplePlacementScore(
          candidate,
          image,
          renderedLinearRgba,
          options,
          residualEvidence,
        );
        const gain = score - currentScore;
        if (!best || gain > best.gain) {
          best = { index, candidate, gain, move: Math.hypot(bounded[0], bounded[1]) };
        }
      }
      if (best
        && best.gain >= options.minimumMoveGain
        && best.gain >= Math.max(0, currentScore) * 0.04) {
        proposals.push(best);
      }
    }
    proposals.sort((a, b) => b.gain - a.gain || a.index - b.index);
    const moveBudget = Math.min(
      options.maximumMovesPerEvent,
      Math.max(0, Math.round(plan.length * options.residualMoveFraction)),
      proposals.length,
    );
    const accepted = new Map(proposals.slice(0, moveBudget).map((proposal) => [
      proposal.index,
      proposal,
    ]));
    let moveTotal = 0;
    let gainTotal = 0;
    const movedPlan = plan.map((stroke, index) => {
      const proposal = accepted.get(index);
      if (!proposal) return clampStrokeToMovementLimit(stroke, options);
      moveTotal += proposal.move;
      gainTotal += proposal.gain;
      return {
        ...annotateTextureGuide(proposal.candidate, options.textureGuide),
        topology_kind: stroke.topology_kind === "source"
          ? (options.scaleMatchedResidualRepaint
            ? "scale-matched-residual-repaint"
            : "residual-move")
          : stroke.topology_kind,
        topology_residual_move_count: (Number(stroke.topology_residual_move_count) || 0) + 1,
      };
    });
    return {
      plan: movedPlan,
      movedCount: moveBudget,
      meanMovePx: moveTotal / Math.max(1, moveBudget),
      meanGain: gainTotal / Math.max(1, moveBudget),
    };
  }

  function scheduledHalfWidth(targetWidth, progress, options) {
    const eased = clamp01(progress) ** 0.75;
    const imageLongSide = Math.max(1, Number(options.imageLongSide) || 1);
    const startingHalfWidth = imageLongSide / Math.max(1, options.startingWidthDivisor) * 0.5;
    return options.paintCurriculumEnabled
      ? targetWidth + (Math.max(targetWidth, startingHalfWidth) - targetWidth) * (1 - eased)
      : targetWidth;
  }

  function applyPaintCurriculum(plan, progress, options) {
    const boundedProgress = clamp01(progress);
    const imageLongSide = Math.max(1, Number(options.imageLongSide) || 1);
    const startingHalfWidth = imageLongSide / Math.max(1, options.startingWidthDivisor) * 0.5;
    const startingLengthScale = Math.max(1, options.startingLengthPercent / 100);
    const eased = boundedProgress ** 0.75;
    const desiredLengthScale = options.paintCurriculumEnabled
      ? 1 + (startingLengthScale - 1) * (1 - eased)
      : 1;
    let widthTotal = 0;
    let arcTotal = 0;
    const nextPlan = plan.map((sourceStroke) => {
      let stroke = withTopologyOrigin(sourceStroke);
      const targetWidth = Math.max(0.55, Number(stroke.topology_target_half_width_px));
      const scheduledWidth = options.paintCurriculumEnabled
        ? scheduledHalfWidth(targetWidth, boundedProgress, options)
        : Number(stroke.half_width_px);
      const currentLengthScale = Math.max(
        1e-3,
        Number(stroke.topology_curriculum_length_scale) || 1,
      );
      let appliedLengthScale = currentLengthScale;
      if (options.paintCurriculumEnabled) {
        const arcBefore = Math.max(1e-6, stroke.path_length_px);
        stroke = scaleStrokeAroundCenter(stroke, desiredLengthScale / currentLengthScale);
        if (options.maximumCurveArcPx > 0 && stroke.path_length_px > options.maximumCurveArcPx) {
          stroke = scaleStrokeAroundCenter(
            stroke,
            options.maximumCurveArcPx / Math.max(1e-6, stroke.path_length_px),
          );
        }
        appliedLengthScale = currentLengthScale * stroke.path_length_px / arcBefore;
      }
      stroke = withDerivedGeometry({
        ...stroke,
        half_width_px: Math.min(options.maximumWidthPx, Math.max(0.55, scheduledWidth)),
        topology_curriculum_length_scale: appliedLengthScale,
        topology_curriculum_progress: boundedProgress,
      });
      widthTotal += stroke.half_width_px;
      arcTotal += stroke.path_length_px;
      return stroke;
    });
    return {
      plan: nextPlan,
      progress: boundedProgress,
      startingFullWidthPx: startingHalfWidth * 2,
      desiredLengthPercent: desiredLengthScale * 100,
      meanHalfWidthPx: widthTotal / Math.max(1, nextPlan.length),
      meanArcPx: arcTotal / Math.max(1, nextPlan.length),
    };
  }

  function removePaintCurriculum(stroke, options) {
    let target = withTopologyOrigin(stroke);
    if (!options.paintCurriculumEnabled) return target;
    const previousTargetWidth = Math.max(
      0.55,
      Number(target.topology_target_half_width_px) || Number(target.half_width_px),
    );
    const previousProgress = clamp01(target.topology_curriculum_progress ?? 1);
    const scheduledWidth = scheduledHalfWidth(previousTargetWidth, previousProgress, options);
    const learnedWidthDelta = Number(target.half_width_px) - scheduledWidth;
    const frontLayer = Math.round(Number(target.layer) || 0) === 2;
    const nextTargetWidth = frontLayer
      ? clamp(previousTargetWidth + learnedWidthDelta, 0.55, options.maximumWidthPx)
      : previousTargetWidth;
    const currentLengthScale = Math.max(
      1e-3,
      Number(target.topology_curriculum_length_scale) || 1,
    );
    if (Math.abs(currentLengthScale - 1) > 1e-4) {
      target = scaleStrokeAroundCenter(target, 1 / currentLengthScale);
    }
    return withDerivedGeometry({
      ...target,
      half_width_px: nextTargetWidth,
      topology_target_half_width_px: nextTargetWidth,
      topology_curriculum_length_scale: 1,
    });
  }

  function strokeColor(stroke) {
    return [
      clamp01(stroke.paint_linear_r),
      clamp01(stroke.paint_linear_g),
      clamp01(stroke.paint_linear_b),
    ];
  }

  function colorDistance(a, b) {
    const ca = strokeColor(a);
    const cb = strokeColor(b);
    return Math.hypot(ca[0] - cb[0], ca[1] - cb[1], ca[2] - cb[2]);
  }

  function reverseStroke(stroke) {
    return withDerivedGeometry({
      ...stroke,
      start_x: stroke.end_x,
      start_y: stroke.end_y,
      control_1_x: stroke.control_2_x,
      control_1_y: stroke.control_2_y,
      control_2_x: stroke.control_1_x,
      control_2_y: stroke.control_1_y,
      end_x: stroke.start_x,
      end_y: stroke.start_y,
    });
  }

  function closestJoin(first, second) {
    const variants = [
      [first, second],
      [first, reverseStroke(second)],
      [reverseStroke(first), second],
      [reverseStroke(first), reverseStroke(second)],
    ];
    let best = null;
    for (const [a, b] of variants) {
      const aEnd = point(a, "end");
      const bStart = point(b, "start");
      const distance = Math.hypot(aEnd[0] - bStart[0], aEnd[1] - bStart[1]);
      if (!best || distance < best.distance) best = { a, b, distance };
    }
    return best;
  }

  function mergePair(first, second, options) {
    const joined = closestJoin(first, second);
    const a = joined.a;
    const b = joined.b;
    const start = point(a, "start");
    const end = point(b, "end");
    const chord = [end[0] - start[0], end[1] - start[1]];
    const fallbackDirection = normalize(chord);
    const startDirection = normalize([
      Number(a.control_1_x) - start[0],
      Number(a.control_1_y) - start[1],
    ], fallbackDirection);
    const endDirection = normalize([
      end[0] - Number(b.control_2_x),
      end[1] - Number(b.control_2_y),
    ], fallbackDirection);
    const firstArc = cubicArcLength(a);
    const secondArc = cubicArcLength(b);
    const handle = Math.max(1, Math.min((firstArc + secondArc + joined.distance) / 5, Math.hypot(...chord) / 2));
    const firstWeight = firstArc / Math.max(1e-6, firstArc + secondArc);
    const secondWeight = 1 - firstWeight;
    const firstColor = strokeColor(a);
    const secondColor = strokeColor(b);
    const targetWidth = Math.min(
      options.maximumWidthPx,
      Math.max(
        Number(a.topology_target_half_width_px) || Number(a.half_width_px),
        Number(b.topology_target_half_width_px) || Number(b.half_width_px),
      ) * 1.25,
    );
    const merged = withDerivedGeometry({
      ...a,
      start_x: start[0],
      start_y: start[1],
      control_1_x: start[0] + startDirection[0] * handle,
      control_1_y: start[1] + startDirection[1] * handle,
      control_2_x: end[0] - endDirection[0] * handle,
      control_2_y: end[1] - endDirection[1] * handle,
      end_x: end[0],
      end_y: end[1],
      half_width_px: Math.min(
        options.maximumWidthPx,
        Math.max(Number(a.half_width_px), Number(b.half_width_px)) * 1.25,
      ),
      opacity: fixedStrokeOpacity(options),
      paint_linear_r: firstColor[0] * firstWeight + secondColor[0] * secondWeight,
      paint_linear_g: firstColor[1] * firstWeight + secondColor[1] * secondWeight,
      paint_linear_b: firstColor[2] * firstWeight + secondColor[2] * secondWeight,
      random: fract((Number(a.random) || 0) * 0.618 + (Number(b.random) || 0) * 0.382 + 0.13),
      topology_generation: Math.max(
        Number(a.topology_generation) || 0,
        Number(b.topology_generation) || 0,
      ) + 1,
      topology_kind: "merge",
      topology_merged_source_indices: [a.topology_source_index, b.topology_source_index],
      topology_target_half_width_px: targetWidth,
      topology_curriculum_length_scale: 1,
    });
    return withTopologyOrigin({
      ...merged,
      topology_origin_center_x: merged.center_x,
      topology_origin_center_y: merged.center_y,
    });
  }

  function splitStroke(stroke, options) {
    const p0 = point(stroke, "start");
    const p1 = point(stroke, "control_1");
    const p2 = point(stroke, "control_2");
    const p3 = point(stroke, "end");
    const p01 = mixPoint(p0, p1, 0.5);
    const p12 = mixPoint(p1, p2, 0.5);
    const p23 = mixPoint(p2, p3, 0.5);
    const p012 = mixPoint(p01, p12, 0.5);
    const p123 = mixPoint(p12, p23, 0.5);
    const middle = mixPoint(p012, p123, 0.5);
    const swap = hash01((Number(stroke.random) || 0) + (Number(stroke.topology_generation) || 0)) > 0.5;
    const widthFactors = swap ? [0.78, 1.15] : [1.15, 0.78];
    const pigmentFactors = swap ? [1.035, 0.985] : [0.985, 1.035];
    // A connected half split and a full-length parallel split both preserve too
    // much parent coverage and converge toward a smooth fill. Separate the two
    // half strokes slightly around their join so the topology event leaves a
    // visible fork/gap while keeping both original endpoints anchored.
    const tangent = normalize([p123[0] - p012[0], p123[1] - p012[1]]);
    const sideSign = hash01((Number(stroke.random) || 0) * 1.414213562 + 0.37) > 0.5 ? 1 : -1;
    const normal = [-tangent[1] * sideSign, tangent[0] * sideSign];
    const separation = Math.max(0.55, Math.min(3.5, Number(stroke.half_width_px) * 0.48));
    const offset = (value, amount) => [
      value[0] + normal[0] * amount,
      value[1] + normal[1] * amount,
    ];
    const geometries = [
      [
        p0,
        offset(p01, separation * 0.18),
        offset(p012, separation * 0.82),
        offset(middle, separation * 0.68),
      ],
      [
        offset(middle, -separation * 0.68),
        offset(p123, -separation * 0.82),
        offset(p23, -separation * 0.18),
        p3,
      ],
    ];
    const generation = (Number(stroke.topology_generation) || 0) + 1;
    const createChild = (geometry, child) => {
      const targetWidth = Math.max(0.55, Math.min(
        options.maximumWidthPx,
        (Number(stroke.topology_target_half_width_px) || Number(stroke.half_width_px))
          * widthFactors[child],
      ));
      const created = withDerivedGeometry({
        ...stroke,
        start_x: geometry[0][0],
        start_y: geometry[0][1],
        control_1_x: geometry[1][0],
        control_1_y: geometry[1][1],
        control_2_x: geometry[2][0],
        control_2_y: geometry[2][1],
        end_x: geometry[3][0],
        end_y: geometry[3][1],
        half_width_px: Math.max(0.55, Math.min(
          options.maximumWidthPx,
          Number(stroke.half_width_px) * widthFactors[child],
        )),
        opacity: fixedStrokeOpacity(options),
        paint_linear_r: clamp01(Number(stroke.paint_linear_r) * pigmentFactors[child]),
        paint_linear_g: clamp01(Number(stroke.paint_linear_g) * pigmentFactors[child]),
        paint_linear_b: clamp01(Number(stroke.paint_linear_b) * pigmentFactors[child]),
        random: fract((Number(stroke.random) || 0) + 0.271828 * (child + 1)),
        topology_generation: generation,
        topology_kind: "split-fork",
        topology_split_child: child,
        topology_target_half_width_px: targetWidth,
        topology_curriculum_length_scale: 1,
      });
      return withTopologyOrigin({
        ...created,
        topology_origin_center_x: created.center_x,
        topology_origin_center_y: created.center_y,
      });
    };
    return [
      createChild(geometries[0], 0),
      createChild(geometries[1], 1),
    ];
  }

  function summarizeDistribution(plan) {
    const widths = plan.map((stroke) => Number(stroke.half_width_px)).filter(Number.isFinite);
    const opacities = plan.map((stroke) => Number(stroke.opacity)).filter(Number.isFinite);
    const arcs = plan.map((stroke) => cubicArcLength(stroke)).filter(Number.isFinite);
    const centerDisplacements = plan.map((stroke) => {
      const center = cubicPoint(stroke, 0.5);
      const originX = Number(stroke.topology_origin_center_x);
      const originY = Number(stroke.topology_origin_center_y);
      return Number.isFinite(originX) && Number.isFinite(originY)
        ? Math.hypot(center[0] - originX, center[1] - originY)
        : 0;
    });
    const textureScores = plan
      .map((stroke) => Number(stroke.texture_score))
      .filter(Number.isFinite);
    const edgeScores = plan
      .map((stroke) => Number(stroke.edge_score))
      .filter(Number.isFinite);
    const dabVisibilityScores = plan
      .map((stroke) => Number(stroke.dab_visibility_score))
      .filter(Number.isFinite);
    const darkFlatScores = plan
      .map((stroke) => Number(stroke.dark_flat_score))
      .filter(Number.isFinite);
    const summarize = (values) => ({
      minimum: values.length ? Math.min(...values) : 0,
      maximum: values.length ? Math.max(...values) : 0,
      mean: values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0,
    });
    const opacityByLayer = {};
    for (let layer = 0; layer <= 3; layer += 1) {
      opacityByLayer[layer] = summarize(plan
        .filter((stroke) => Math.round(Number(stroke.layer) || 0) === layer)
        .map((stroke) => Number(stroke.opacity))
        .filter(Number.isFinite));
    }
    return {
      width_px: summarize(widths),
      opacity: summarize(opacities),
      opacity_by_layer: opacityByLayer,
      arc_px: summarize(arcs),
      center_displacement_px: summarize(centerDisplacements),
      texture_score: summarize(textureScores),
      edge_score: summarize(edgeScores),
      dab_visibility_score: summarize(dabVisibilityScores),
      dark_flat_score: summarize(darkFlatScores),
      dark_flat_parent_fraction: darkFlatScores.length
        ? darkFlatScores.filter((value) => value >= 0.60).length / darkFlatScores.length
        : 0,
    };
  }

  function normalizedOptions(options = {}) {
    const minimumWidthFactor = clamp(
      options.minimumWidthFactor ?? DEFAULT_OPTIONS.minimumWidthFactor,
      0.25,
      3,
    );
    const maximumWidthFactor = Math.max(minimumWidthFactor, clamp(
      options.maximumWidthFactor ?? DEFAULT_OPTIONS.maximumWidthFactor,
      0.25,
      3,
    ));
    const frontWidthMaximumFactor = Math.max(minimumWidthFactor, clamp(
      options.frontWidthMaximumFactor ?? DEFAULT_OPTIONS.frontWidthMaximumFactor,
      0.25,
      4,
    ));
    const fixedOpacity = clamp(
      options.fixedOpacity ?? DEFAULT_OPTIONS.fixedOpacity,
      0.05,
      0.995,
    );
    const textureGuide = options.textureGuidedAllocation === true
      && validTextureGuide(options.textureGuide)
      ? options.textureGuide
      : null;
    return {
      ...DEFAULT_OPTIONS,
      ...options,
      minimumWidthFactor,
      maximumWidthFactor,
      frontWidthMaximumFactor,
      fixedOpacity,
      splitFraction: clamp(options.splitFraction ?? DEFAULT_OPTIONS.splitFraction, 0, 0.25),
      mergeFraction: clamp(options.mergeFraction ?? DEFAULT_OPTIONS.mergeFraction, 0, 0.10),
      maximumSplitsPerEvent: Math.max(0, Math.round(
        options.maximumSplitsPerEvent ?? DEFAULT_OPTIONS.maximumSplitsPerEvent,
      )),
      maximumMergesPerEvent: Math.max(0, Math.round(
        options.maximumMergesPerEvent ?? DEFAULT_OPTIONS.maximumMergesPerEvent,
      )),
      splitApplyUntil: clamp01(
        options.splitApplyUntil ?? DEFAULT_OPTIONS.splitApplyUntil,
      ),
      maximumWidthPx: Math.max(2, Number(options.maximumWidthPx) || 64),
      maximumCurveArcPx: Math.max(0, Number(options.maximumCurveArcPx) || 0),
      imageLongSide: Math.max(1, Number(options.imageLongSide) || 1),
      paintCurriculumEnabled: options.paintCurriculumEnabled !== false,
      startingWidthDivisor: clamp(
        options.startingWidthDivisor ?? DEFAULT_OPTIONS.startingWidthDivisor,
        8,
        256,
      ),
      startingLengthPercent: clamp(
        options.startingLengthPercent ?? DEFAULT_OPTIONS.startingLengthPercent,
        100,
        300,
      ),
      residualMovePerEventPx: clamp(
        options.residualMovePerEventPx ?? DEFAULT_OPTIONS.residualMovePerEventPx,
        0,
        8,
      ),
      residualMoveFraction: clamp(
        options.residualMoveFraction ?? DEFAULT_OPTIONS.residualMoveFraction,
        0,
        0.10,
      ),
      maximumMovesPerEvent: Math.max(0, Math.round(
        options.maximumMovesPerEvent ?? DEFAULT_OPTIONS.maximumMovesPerEvent,
      )),
      minimumMoveGain: clamp(
        options.minimumMoveGain ?? DEFAULT_OPTIONS.minimumMoveGain,
        0,
        0.10,
      ),
      maximumTotalMovementPx: clamp(
        options.maximumTotalMovementPx ?? DEFAULT_OPTIONS.maximumTotalMovementPx,
        0,
        64,
      ),
      textureGuide,
      textureGuidedAllocation: Boolean(textureGuide),
      scaleMatchedResidualRepaint: options.scaleMatchedResidualRepaint === true,
      curriculumProgress: clamp01(options.curriculumProgress ?? 1),
    };
  }

  function initialize(referencePlan, targetCount, options = {}) {
    const normalized = normalizedOptions(options);
    const count = Math.max(1, Math.min(referencePlan.length, Math.round(targetCount)));
    const sourcePlan = referencePlan.slice(0, count).map((stroke, index) => (
      annotateTextureGuide(varyStroke(stroke, index, normalized), normalized.textureGuide)
    ));
    const curriculum = applyPaintCurriculum(
      sourcePlan,
      normalized.curriculumProgress,
      normalized,
    );
    return {
      plan: curriculum.plan,
      nextSourceIndex: count,
      eventIndex: 0,
      totals: { splits: 0, merges: 0, sourceAdded: count, residualMoves: 0 },
      initialDistribution: summarizeDistribution(curriculum.plan),
      initialCurriculum: {
        progress: curriculum.progress,
        startingFullWidthPx: curriculum.startingFullWidthPx,
        desiredLengthPercent: curriculum.desiredLengthPercent,
        meanHalfWidthPx: curriculum.meanHalfWidthPx,
        meanArcPx: curriculum.meanArcPx,
      },
      events: [],
    };
  }

  function evolve(state, params, image, renderedLinearRgba, targetCount, referencePlan, options = {}) {
    const normalized = normalizedOptions(options);
    const residualEvidence = normalized.scaleMatchedResidualRepaint
      ? buildScaleMatchedResidualEvidence(
        image,
        renderedLinearRgba,
        normalized.curriculumProgress,
      )
      : null;
    const requestedTarget = Math.max(1, Math.min(referencePlan.length, Math.round(targetCount)));
    let plan = state.plan.map((stroke, index) => {
      const prepared = clampStrokeToMovementLimit(
        removePaintCurriculum(strokeFromParams(stroke, params, index), normalized),
        normalized,
      );
      return annotateTextureGuide({
        ...prepared,
        opacity: fixedStrokeOpacity(normalized),
      }, normalized.textureGuide);
    });
    const residualScores = plan.map((stroke) => sampleResidual(
      stroke,
      image,
      renderedLinearRgba,
      residualEvidence,
    ));
    const textureScores = plan.map((stroke) => clamp01(stroke.texture_score ?? 1));
    const originalCount = plan.length;
    const growthSlots = Math.max(0, requestedTarget - originalCount);

    const mergeBudget = Math.min(
      normalized.maximumMergesPerEvent,
      Math.max(0, Math.round(originalCount * normalized.mergeFraction)),
      Math.max(0, originalCount - 1),
    );
    const used = new Set();
    const mergePairs = [];
    const lowResidualOrder = residualScores
      .map((score, index) => ({
        score: normalized.textureGuidedAllocation
          ? score * (0.45 + textureScores[index] * 0.55) + textureScores[index] * 0.035
          : score,
        index,
      }))
      .sort((a, b) => a.score - b.score || a.index - b.index);
    for (let cursor = 0; cursor < lowResidualOrder.length && mergePairs.length < mergeBudget; cursor += 1) {
      const firstIndex = lowResidualOrder[cursor].index;
      if (used.has(firstIndex)) continue;
      const first = plan[firstIndex];
      let best = null;
      const scanEnd = Math.min(lowResidualOrder.length, cursor + 65);
      for (let otherCursor = cursor + 1; otherCursor < scanEnd; otherCursor += 1) {
        const secondIndex = lowResidualOrder[otherCursor].index;
        if (used.has(secondIndex)) continue;
        const second = plan[secondIndex];
        if (Math.round(Number(first.layer) || 0) !== Math.round(Number(second.layer) || 0)) continue;
        if (colorDistance(first, second) > 0.28) continue;
        const joined = closestJoin(first, second);
        const distanceLimit = Math.max(
          4,
          Math.min(18, (Number(first.half_width_px) + Number(second.half_width_px)) * 2.5),
        );
        if (joined.distance > distanceLimit) continue;
        const tangentA = normalize([
          Number(joined.a.end_x) - Number(joined.a.control_2_x),
          Number(joined.a.end_y) - Number(joined.a.control_2_y),
        ]);
        const tangentB = normalize([
          Number(joined.b.control_1_x) - Number(joined.b.start_x),
          Number(joined.b.control_1_y) - Number(joined.b.start_y),
        ]);
        if (tangentA[0] * tangentB[0] + tangentA[1] * tangentB[1] < 0.25) continue;
        const combinedArc = cubicArcLength(first) + joined.distance + cubicArcLength(second);
        if (normalized.maximumCurveArcPx > 0 && combinedArc > normalized.maximumCurveArcPx * 1.15) continue;
        const score = joined.distance + colorDistance(first, second) * 16
          + (residualScores[firstIndex] + residualScores[secondIndex]) * 8
          + (normalized.textureGuidedAllocation
            ? (textureScores[firstIndex] + textureScores[secondIndex]) * 4
            : 0);
        if (!best || score < best.score) best = { firstIndex, secondIndex, score };
      }
      if (!best) continue;
      used.add(best.firstIndex);
      used.add(best.secondIndex);
      mergePairs.push(best);
    }
    if (mergePairs.length > 0) {
      const mergeByFirst = new Map(mergePairs.map((pair) => [pair.firstIndex, pair]));
      const removed = new Set(mergePairs.map((pair) => pair.secondIndex));
      plan = plan.flatMap((stroke, index) => {
        if (removed.has(index)) return [];
        const pair = mergeByFirst.get(index);
        return pair ? [mergePair(plan[pair.firstIndex], plan[pair.secondIndex], normalized)] : [stroke];
      });
    }

    const splitEnabled = normalized.curriculumProgress <= normalized.splitApplyUntil;
    const splitBudget = splitEnabled
      ? Math.min(
          normalized.maximumSplitsPerEvent,
          Math.max(0, Math.round(originalCount * normalized.splitFraction)),
          Math.max(0, requestedTarget - plan.length),
          Math.max(0, Math.round(growthSlots * 0.45) + mergePairs.length),
        )
      : 0;
    const splitScores = plan.map((stroke, index) => {
      const residual = sampleResidual(stroke, image, renderedLinearRgba, residualEvidence);
      const arc = cubicArcLength(stroke);
      const texture = clamp01(stroke.texture_score ?? 1);
      const residualArc = residual * Math.sqrt(Math.max(1, arc));
      return {
        index,
        residual,
        arc,
        texture,
        priority: normalized.textureGuidedAllocation
          ? residualArc * (0.35 + texture * 1.15)
            + texture * Math.sqrt(Math.max(1, arc)) * 0.01
          : residualArc,
      };
    }).sort((a, b) => b.priority - a.priority || a.index - b.index);
    const spatialCellSize = Math.max(24, Math.min(Number(image?.width) || 256, Number(image?.height) || 256) / 8);
    const splitCellCounts = new Map();
    const splitIndices = new Set();
    for (const candidate of splitScores) {
      if (splitIndices.size >= splitBudget) break;
      const stroke = plan[candidate.index];
      if (candidate.arc < Math.max(6, Number(stroke.half_width_px) * 3)) continue;
      const center = cubicPoint(stroke, 0.5);
      const key = `${Math.floor(center[0] / spatialCellSize)},${Math.floor(center[1] / spatialCellSize)}`;
      const splitCellCapacity = normalized.textureGuidedAllocation
        ? 1 + Math.floor(candidate.texture * 2.999)
        : 1;
      const splitCellCount = splitCellCounts.get(key) || 0;
      if (splitCellCount >= splitCellCapacity) continue;
      splitCellCounts.set(key, splitCellCount + 1);
      splitIndices.add(candidate.index);
    }
    if (splitIndices.size > 0) {
      plan = plan.flatMap((stroke, index) => (
        splitIndices.has(index) ? splitStroke(stroke, normalized) : [stroke]
      ));
    }

    let nextSourceIndex = state.nextSourceIndex;
    let sourceAdded = 0;
    while (plan.length < requestedTarget && nextSourceIndex < referencePlan.length) {
      plan.push(annotateTextureGuide(
        varyStroke(referencePlan[nextSourceIndex], nextSourceIndex, normalized),
        normalized.textureGuide,
      ));
      nextSourceIndex += 1;
      sourceAdded += 1;
    }
    while (plan.length < requestedTarget && plan.length > 0) {
      const sourceIndex = plan.length - 1;
      const source = plan[sourceIndex];
      const children = splitStroke(source, normalized);
      plan[sourceIndex] = annotateTextureGuide(children[0], normalized.textureGuide);
      plan.push(annotateTextureGuide(children[1], normalized.textureGuide));
    }
    if (plan.length > requestedTarget) plan = plan.slice(0, requestedTarget);

    const placement = optimizeResidualPlacement(
      plan,
      image,
      renderedLinearRgba,
      normalized.curriculumProgress,
      normalized,
      residualEvidence,
    );
    const curriculum = applyPaintCurriculum(
      placement.plan.map((stroke) => annotateTextureGuide(stroke, normalized.textureGuide)),
      normalized.curriculumProgress,
      normalized,
    );
    plan = curriculum.plan;

    const event = {
      event: state.eventIndex + 1,
      count_before: originalCount,
      count_after: plan.length,
      split_count: splitIndices.size,
      split_enabled: splitEnabled,
      split_apply_until: normalized.splitApplyUntil,
      merge_count: mergePairs.length,
      source_added: sourceAdded,
      residual_move_count: placement.movedCount,
      residual_move_mean_px: placement.meanMovePx,
      residual_move_mean_gain: placement.meanGain,
      scale_matched_residual_repaint: normalized.scaleMatchedResidualRepaint,
      scale_matched_residual_radius_px: residualEvidence?.radius || 0,
      curriculum_progress: curriculum.progress,
      curriculum_starting_full_width_px: curriculum.startingFullWidthPx,
      curriculum_length_percent: curriculum.desiredLengthPercent,
      curriculum_mean_half_width_px: curriculum.meanHalfWidthPx,
      curriculum_mean_arc_px: curriculum.meanArcPx,
      mean_residual_score: residualScores.length
        ? residualScores.reduce((sum, value) => sum + value, 0) / residualScores.length
        : 0,
      distribution: summarizeDistribution(plan),
    };
    return {
      plan,
      nextSourceIndex,
      eventIndex: state.eventIndex + 1,
      totals: {
        splits: state.totals.splits + splitIndices.size,
        merges: state.totals.merges + mergePairs.length,
        sourceAdded: state.totals.sourceAdded + sourceAdded,
        residualMoves: (state.totals.residualMoves || 0) + placement.movedCount,
      },
      initialDistribution: state.initialDistribution,
      events: [...state.events, event],
    };
  }

  global.Image2SplatPaintFlowStrokeTopology = Object.freeze({
    initialize,
    evolve,
    strokeFromParams,
    summarizeDistribution,
    applyPaintCurriculum,
    constants: Object.freeze({
      PARAM_STRIDE,
      ...DEFAULT_OPTIONS,
    }),
  });
})(globalThis);
