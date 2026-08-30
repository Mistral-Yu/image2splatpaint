(function installFlowRibbonTrainer(global) {
  "use strict";

  const PARAM_STRIDE = 16;
  const GRAD_STRIDE = 16;
  const TILE_SIZE = 16;
  // Four overlapping Gaussian Splats keep a curved stroke continuous while
  // leaving enough parent strokes in the physical Max splats budget. The
  // previous 12-Splat chain was fast per iteration but too sparse visually.
  const CURVE_SAMPLES = 4;
  const BRISTLE_BUNDLE_SAMPLES = 4;
  // Brush dabs spend the same final physical-Splat budget on fewer, richer
  // parent strokes: two edge-guided thin bristles and three middle marks sit
  // in front of three broad body Splats. Standard keeps the accepted four-
  // Splat Baseline.
  const BRUSH_DAB_SAMPLES = 8;
  const VARIABLE_BRUSH_DAB_SAMPLES = 9;
  // Always keep the three curve-body marks. Additional slots add middle
  // paint before fine accents. Tile lists contain only active physical dabs.
  const BRUSH_DAB_SAMPLE_PRIORITY = [5, 6, 7, 4, 1, 3, 8, 0, 2];

  function brushDabSampleMask(count) {
    const bounded = Math.max(3, Math.min(9, Math.round(Number(count) || 6)));
    return BRUSH_DAB_SAMPLE_PRIORITY.slice(0, bounded)
      .reduce((mask, sample) => mask | (1 << sample), 0);
  }

  function allocateBrushDabCounts(strokePlan, physicalBudget, fillBudget = false) {
    const budget = Math.max(0, Math.floor(Number(physicalBudget) || 0));
    if (strokePlan.length * 3 > budget) throw new Error("Brush chain budget needs at least three dabs per curve.");
    const plan = strokePlan.map((stroke) => ({
      ...stroke,
      brush_dab_count: Math.max(3, Math.min(9, Math.round(Number(stroke.brush_dab_count)
        || (3 + Math.min(6, Math.floor(clamp01(stroke.random) * 7)))))),
    }));
    let total = plan.reduce((sum, stroke) => sum + stroke.brush_dab_count, 0);
    // Seed-stable ordering, no per-update randomness. Existing counts survive
    // growth except when the final physical budget requires reconciliation.
    const order = plan.map((stroke, index) => ({ index, random: Number(stroke.random) || 0 }))
      .sort((a, b) => a.random - b.random || a.index - b.index);
    const target = fillBudget ? Math.min(budget, plan.length * 9) : Math.min(total, budget);
    while (total !== target) {
      for (const { index } of order) {
        if (total === target) break;
        const delta = total < target ? 1 : -1;
        const next = plan[index].brush_dab_count + delta;
        if (next < 3 || next > 9) continue;
        plan[index].brush_dab_count = next;
        total += delta;
      }
    }
    return { plan, physicalSplatCount: total };
  }
  const STROKE_TEXTURE_STANDARD = 0;
  const STROKE_TEXTURE_FINE_BRISTLES = 1;
  const STROKE_TEXTURE_BRUSH_DABS = 2;
  const TEXTURE_GUIDE_SPACING_PACK = 256;
  // Texture and edge guides share the existing spacing parameter. Six texture
  // bits plus two edge bits keep the bucket in the old 8-bit range, preserving
  // the accepted sub-pixel spacing precision and physical-Splat budget.
  const GUIDE_LEVEL_BASE = 4;
  const TEXTURE_GUIDE_LEVELS = 63;
  const EDGE_GUIDE_LEVELS = GUIDE_LEVEL_BASE - 1;
  const CONFIG_BYTES = 160;
  const CANVAS_LINEAR = Object.freeze([0.91, 0.885, 0.82]);

  function resolveStrokeTextureMode(options = {}) {
    if (options.brushDabs === true) return STROKE_TEXTURE_BRUSH_DABS;
    if (options.bristleBundle === true) return STROKE_TEXTURE_FINE_BRISTLES;
    return STROKE_TEXTURE_STANDARD;
  }

  function brushDabLoadedSide(random, layer) {
    return Math.sin((Number(random) || 0) * 137.3 + (Number(layer) || 0) * 11.7) >= 0
      ? 1
      : -1;
  }

  function smoothstepRange(edge0, edge1, value) {
    const t = clamp01((Number(value) - edge0) / Math.max(1e-6, edge1 - edge0));
    return t * t * (3 - 2 * t);
  }

  function physicalSplatScaleVariation(random, layer, sample, amount, mode) {
    const variation = clamp01(amount);
    if (variation <= 0) return { width: 1, length: 1 };
    if (mode === STROKE_TEXTURE_BRUSH_DABS) {
      // Stroke width is a parent-level ceiling. Preserve painterly breadth
      // without multiplying the middle marks down into a second hairline
      // family. Keep two edge accents, three mid-width marks and three bodies.
      const widthTargets = [0.40, 0.80, 0.60, 0.95, 0.88, 0.85, 1.15, 1.00, 0.95];
      const lengthTargets = [2.05, 1.34, 1.88, 1.24, 1.58, 0.84, 0.72, 0.94, 0.90];
      const index = Math.max(0, Math.min(VARIABLE_BRUSH_DAB_SAMPLES - 1, Number(sample) || 0));
      const widthJitter = 1 + 0.08 * Math.sin(
        (Number(random) || 0) * 211.1 + index * 47.3 + Number(layer) * 13.7,
      );
      const lengthJitter = 1 + 0.06 * Math.sin(
        (Number(random) || 0) * 173.9 + index * 29.7 + Number(layer) * 7.9 + 1.7,
      );
      return {
        width: Math.max(
          0.05,
          1 + (widthTargets[index] * widthJitter - 1) * variation,
        ),
        length: Math.max(
          0.35,
          1 + (lengthTargets[index] * lengthJitter - 1) * variation,
        ),
      };
    }
    const widthNoise = Math.sin(
      (Number(random) || 0) * 211.1 + Number(sample) * 47.3 + Number(layer) * 13.7,
    );
    const lengthNoise = Math.sin(
      (Number(random) || 0) * 173.9 + Number(sample) * 29.7 + Number(layer) * 7.9 + 1.7,
    );
    return {
      width: Math.max(0.15, 1 + widthNoise * 0.85 * variation),
      length: Math.max(0.35, 1 + lengthNoise * 0.65 * variation),
    };
  }

  function chainSampleProfile(
    mode,
    layer,
    sample,
    random,
    sampleCount,
    textureScore = 1,
    edgeScore = 1,
    splatSizeVariation = 0,
  ) {
    const vary = (profile) => {
      const thinLongBristle = mode === STROKE_TEXTURE_BRUSH_DABS
        && (sample === 0 || sample === 2);
      const variationGuide = thinLongBristle
        ? smoothstepRange(0.30, 0.95, edgeScore)
        : 1;
      const variation = physicalSplatScaleVariation(
        random,
        layer,
        sample,
        splatSizeVariation * variationGuide,
        mode,
      );
      return {
        ...profile,
        widthFactor: profile.widthFactor * variation.width,
        lengthFactor: profile.lengthFactor * variation.length,
      };
    };
    const fineBristle = mode === STROKE_TEXTURE_FINE_BRISTLES && layer >= 1.5;
    if (fineBristle) {
      return vary({
        t: (Math.floor(sample / 2) + 0.5) / Math.max(1, sampleCount / 2),
        normalOffsetFactor: (sample % 2 === 0 ? -1 : 1) * 0.18,
        widthFactor: 0.38,
        lengthFactor: 1,
      });
    }
    if (mode === STROKE_TEXTURE_BRUSH_DABS) {
      const side = brushDabLoadedSide(random, layer);
      const load = 0.5 + 0.5 * Math.sin((Number(random) || 0) * 97 + sample * 19);
      let t = [
        0.20 + 0.05 * load,
        0.66 + 0.08 * load,
        0.24 + 0.05 * load,
        0.70 + 0.07 * load,
        0.47 + 0.06 * load,
        0.12 + 0.03 * load,
        0.48 + 0.04 * load,
        0.84 + 0.05 * load,
        0.32 + 0.06 * load,
      ][sample] ?? 0.5;
      const layerSpread = 0.04 * Math.min(2, layer);
      const normalOffsets = [
        0.48 + layerSpread,
        0.44 + layerSpread,
        -0.46 - layerSpread,
        -0.42 - layerSpread,
        0.03,
        -0.06,
        0.04,
        -0.02,
        -0.28,
      ];
      const texture = smoothstepRange(0.12, 0.65, textureScore);
      const edge = smoothstepRange(0.30, 0.95, edgeScore);
      const body = sample >= 5;
      const thinLongBristle = sample === 0 || sample === 2;
      const shapeGuide = thinLongBristle ? edge : texture;
      if (body) {
        t = 0.5 + (t - 0.5) * (0.5 + texture * 0.5);
      } else {
        // Flat paint needs continuous coverage, not five nearly invisible
        // bristle tips. Keep their ordering but draw them into the stroke body.
        t = 0.5 + (t - 0.5) * (0.72 + shapeGuide * 0.28);
      }
      const baseWidthFactors = [0.18, 0.34, 0.16, 0.44, 0.52, 0.78, 0.92, 0.74, 0.66];
      const baseLengthFactors = [1.05, 0.95, 1.02, 0.92, 1.18, 0.72, 0.82, 0.70, 0.80];
      const flatNormalOffsets = [0.16, 0.08, -0.16, -0.08, 0];
      const widthFactor = body
        ? 0.88 + (baseWidthFactors[sample] - 0.88) * texture
        : 0.62 + (baseWidthFactors[sample] - 0.62) * shapeGuide;
      const lengthFactor = body
        ? 1.45 + (baseLengthFactors[sample] - 1.45) * texture
        : 1.15 + (baseLengthFactors[sample] - 1.15) * shapeGuide;
      const normalOffset = body
        ? normalOffsets[sample]
        : flatNormalOffsets[sample]
          + (normalOffsets[sample] - flatNormalOffsets[sample]) * shapeGuide;
      return vary({
        t,
        normalOffsetFactor: normalOffset * side,
        widthFactor,
        lengthFactor,
      });
    }
    return vary({
      t: sample / Math.max(1, sampleCount - 1),
      normalOffsetFactor: 0,
      widthFactor: 0.46,
      lengthFactor: 1,
    });
  }

  function clamp01(value) {
    return Math.max(0, Math.min(1, Number(value) || 0));
  }

  function packChainSpacing(spacing, textureScore, edgeScore, enabled) {
    const boundedSpacing = Math.max(0.5, Math.min(
      TEXTURE_GUIDE_SPACING_PACK - 0.5,
      Number(spacing) || 0.5,
    ));
    if (!enabled) return boundedSpacing;
    const level = Math.max(0, Math.min(
      TEXTURE_GUIDE_LEVELS,
      Math.round(clamp01(textureScore) * TEXTURE_GUIDE_LEVELS),
    ));
    const edgeLevel = Math.max(0, Math.min(
      EDGE_GUIDE_LEVELS,
      Math.round(clamp01(edgeScore) * EDGE_GUIDE_LEVELS),
    ));
    return (
      level * GUIDE_LEVEL_BASE + edgeLevel
    ) * TEXTURE_GUIDE_SPACING_PACK + boundedSpacing;
  }

  function unpackChainSpacing(value, enabled) {
    const packed = Math.max(0.5, Number(value) || 0.5);
    if (!enabled) return packed;
    const guideBucket = Math.floor(packed / TEXTURE_GUIDE_SPACING_PACK);
    return Math.max(0.5, packed - guideBucket * TEXTURE_GUIDE_SPACING_PACK);
  }

  function unpackChainTextureScore(value, enabled) {
    if (!enabled) return 1;
    const packed = Math.max(0.5, Number(value) || 0.5);
    const guideBucket = Math.floor(packed / TEXTURE_GUIDE_SPACING_PACK);
    return clamp01(
      Math.floor(guideBucket / GUIDE_LEVEL_BASE) / TEXTURE_GUIDE_LEVELS,
    );
  }

  function unpackChainEdgeScore(value, enabled) {
    if (!enabled) return 1;
    const packed = Math.max(0.5, Number(value) || 0.5);
    const guideBucket = Math.floor(packed / TEXTURE_GUIDE_SPACING_PACK);
    return clamp01((guideBucket % GUIDE_LEVEL_BASE) / EDGE_GUIDE_LEVELS);
  }

  function signalSrgbToLinear(value) {
    const channel = clamp01(value);
    return channel <= 0.04045
      ? channel / 12.92
      : ((channel + 0.055) / 1.055) ** 2.4;
  }

  function linearToSignalSrgb(value) {
    const channel = clamp01(value);
    return channel <= 0.0031308
      ? channel * 12.92
      : 1.055 * channel ** (1 / 2.4) - 0.055;
  }

  function logit(value) {
    const bounded = Math.max(1e-4, Math.min(1 - 1e-4, Number(value) || 0.5));
    return Math.log(bounded / (1 - bounded));
  }

  function sigmoid(value) {
    return 1 / (1 + Math.exp(-Number(value)));
  }

  function validateInput(image, strokePlan) {
    const width = Math.max(1, Math.round(Number(image?.width) || 0));
    const height = Math.max(1, Math.round(Number(image?.height) || 0));
    if (!(image?.rgb instanceof Float32Array) || image.rgb.length !== width * height * 3) {
      throw new Error("Flow ribbon trainer requires width*height*3 Float32 RGB input.");
    }
    if (!Array.isArray(strokePlan) || strokePlan.length < 1) {
      throw new Error("Flow ribbon trainer requires a non-empty connected-ribbon stroke plan.");
    }
    return { width, height };
  }

  function estimateCanvasLinear(target, width, height) {
    const border = Math.max(1, Math.round(Math.min(width, height) * 0.06));
    const stride = Math.max(1, Math.floor(Math.min(width, height) / 256));
    const channels = [[], [], []];
    const collect = (borderOnly) => {
      for (let y = 0; y < height; y += stride) {
        for (let x = 0; x < width; x += stride) {
          if (borderOnly && x >= border && x < width - border && y >= border && y < height - border) {
            continue;
          }
          const base = (y * width + x) * 4;
          if (target[base + 3] < 0.05) continue;
          for (let channel = 0; channel < 3; channel += 1) {
            channels[channel].push(target[base + channel]);
          }
        }
      }
    };
    collect(true);
    if (channels[0].length < 8) collect(false);
    if (channels[0].length < 1) return CANVAS_LINEAR.slice();
    return channels.map((values, channel) => {
      values.sort((a, b) => a - b);
      const middle = Math.floor(values.length / 2);
      const median = values.length % 2
        ? values[middle]
        : (values[middle - 1] + values[middle]) * 0.5;
      return Number.isFinite(median) ? clamp01(median) : CANVAS_LINEAR[channel];
    });
  }

  function cubicPointFromParams(params, base, t) {
    const s = 1 - t;
    const weights = [s * s * s, 3 * s * s * t, 3 * s * t * t, t * t * t];
    let x = 0;
    let y = 0;
    for (let control = 0; control < 4; control += 1) {
      x += params[base + control * 2] * weights[control];
      y += params[base + control * 2 + 1] * weights[control];
    }
    return [x, y];
  }

  function cubicTangentFromParams(params, base, t) {
    const s = 1 - t;
    const x =
      3 * s * s * (params[base + 2] - params[base]) +
      6 * s * t * (params[base + 4] - params[base + 2]) +
      3 * t * t * (params[base + 6] - params[base + 4]);
    const y =
      3 * s * s * (params[base + 3] - params[base + 1]) +
      6 * s * t * (params[base + 5] - params[base + 3]) +
      3 * t * t * (params[base + 7] - params[base + 5]);
    const inverseLength = 1 / Math.max(1e-7, Math.hypot(x, y));
    return [x * inverseLength, y * inverseLength];
  }

  function summarizeScaleValues(values) {
    if (values.length === 0) {
      return { minimum: 0, maximum: 0, mean: 0, standard_deviation: 0, ratio: 0 };
    }
    let minimum = Infinity;
    let maximum = -Infinity;
    let total = 0;
    for (const value of values) {
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
      total += value;
    }
    const mean = total / values.length;
    let varianceTotal = 0;
    for (const value of values) varianceTotal += (value - mean) ** 2;
    return {
      minimum,
      maximum,
      mean,
      standard_deviation: Math.sqrt(varianceTotal / values.length),
      ratio: maximum / Math.max(1e-6, minimum),
    };
  }

  function physicalSplatScaleStats(params, data) {
    if (!(params instanceof Float32Array) || data?.sampleCount < 1) return null;
    const halfShort = [];
    const halfLong = [];
    const aspect = [];
    for (let index = 0; index < data.strokeCount; index += 1) {
      const base = index * PARAM_STRIDE;
      const layer = params[base + 14];
      if (layer > 2.5) continue;
      const widthPx = Math.max(0.55, params[base + 8]);
      const spacing = unpackChainSpacing(params[base + 15], data.textureGuidedDabs);
      const textureScore = unpackChainTextureScore(
        params[base + 15],
        data.textureGuidedDabs,
      );
      const edgeScore = unpackChainEdgeScore(
        params[base + 15],
        data.textureGuidedDabs,
      );
      for (let sample = 0; sample < data.sampleCount; sample += 1) {
        if (data.sampleMasks && !(data.sampleMasks[index] & (1 << sample))) continue;
        const profile = chainSampleProfile(
          data.strokeTextureMode,
          layer,
          sample,
          params[base + 13],
          data.sampleCount,
          textureScore,
          edgeScore,
          data.splatSizeVariation,
        );
        const sigmaLong = Math.max(
          0.55,
          Math.max(spacing * 0.72, widthPx * 0.85) * profile.lengthFactor,
        );
        const sigmaShort = Math.max(0.38, widthPx * profile.widthFactor);
        halfLong.push(sigmaLong);
        halfShort.push(sigmaShort);
        aspect.push(Math.max(sigmaLong, sigmaShort) / Math.max(
          1e-6,
          Math.min(sigmaLong, sigmaShort),
        ));
      }
    }
    return {
      physical_splat_count: halfShort.length,
      variation_percent: data.splatSizeVariation * 100,
      // Nominal full short-axis widths, before occlusion; excludes backcoat.
      full_short_width_bins_px: {
        below_2: halfShort.filter((value) => value * 2 < 2).length,
        from_2_to_4: halfShort.filter((value) => value * 2 >= 2 && value * 2 < 4).length,
        from_4_to_8: halfShort.filter((value) => value * 2 >= 4 && value * 2 < 8).length,
        from_8_to_16: halfShort.filter((value) => value * 2 >= 8 && value * 2 < 16).length,
        at_least_16: halfShort.filter((value) => value * 2 >= 16).length,
        at_minimum: halfShort.filter((value) => value <= 0.380001).length,
      },
      half_short_px: summarizeScaleValues(halfShort),
      half_long_px: summarizeScaleValues(halfLong),
      aspect_ratio: summarizeScaleValues(aspect),
    };
  }

  function sampledCubicArcLengthFromParams(params, base, segments = 24) {
    let previous = cubicPointFromParams(params, base, 0);
    let length = 0;
    for (let index = 1; index <= segments; index += 1) {
      const point = cubicPointFromParams(params, base, index / segments);
      length += Math.hypot(point[0] - previous[0], point[1] - previous[1]);
      previous = point;
    }
    return length;
  }

  function fnv1aFloat32(values) {
    const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
    let hash = 0x811c9dc5;
    for (let index = 0; index < bytes.length; index += 1) {
      hash = Math.imul(hash ^ bytes[index], 0x01000193) >>> 0;
    }
    return hash.toString(16).padStart(8, "0");
  }

  function buildResidualPriorityTileMasks({
    width,
    height,
    target,
    residualRender,
    tileLists,
    tileCols,
    tileRows,
    tileSampleStride,
  }) {
    const tileCount = tileCols * tileRows;
    const masks = new Uint32Array(tileCount);
    const stride = Math.max(1, Math.min(16, Math.round(tileSampleStride)));
    if (!(residualRender instanceof Float32Array)
      || residualRender.length !== width * height * 4
      || stride <= 1) {
      return {
        masks,
        enabled: false,
        highTileCount: 0,
        omittedTileCount: 0,
        activationCount: 0,
      };
    }
    const ranked = [];
    for (let tileY = 0; tileY < tileRows; tileY += 1) {
      for (let tileX = 0; tileX < tileCols; tileX += 1) {
        const tile = tileY * tileCols + tileX;
        const x0 = tileX * TILE_SIZE;
        const y0 = tileY * TILE_SIZE;
        const x1 = Math.min(width, x0 + TILE_SIZE);
        const y1 = Math.min(height, y0 + TILE_SIZE);
        let residual = 0;
        let samples = 0;
        for (let y = y0; y < y1; y += 2) {
          for (let x = x0; x < x1; x += 2) {
            const pixel = y * width + x;
            const targetBase = pixel * 4;
            const renderBase = pixel * 4;
            residual += (
              Math.abs(target[targetBase] - residualRender[renderBase])
              + Math.abs(target[targetBase + 1] - residualRender[renderBase + 1])
              + Math.abs(target[targetBase + 2] - residualRender[renderBase + 2])
            ) / 3 + clamp01(residualRender[renderBase + 3]) * 0.18;
            samples += 1;
          }
        }
        const candidateCost = Math.max(1, tileLists[tile].length);
        ranked.push({
          tile,
          candidateCost,
          score: residual / Math.max(1, samples) / Math.sqrt(candidateCost),
        });
      }
    }
    ranked.sort((a, b) => b.score - a.score || a.tile - b.tile);
    const highTileCount = Math.floor(tileCount / 4);
    const middleEnd = Math.floor(tileCount * 3 / 4);
    const phaseLoads = new Float64Array(stride);
    let activationCount = 0;
    for (let rank = 0; rank < ranked.length; rank += 1) {
      const item = ranked[rank];
      const activations = rank < highTileCount ? 2 : rank < middleEnd ? 1 : 0;
      for (let activation = 0; activation < activations; activation += 1) {
        let selectedPhase = -1;
        let selectedLoad = Infinity;
        for (let phase = 0; phase < stride; phase += 1) {
          if ((masks[item.tile] & (1 << phase)) !== 0) continue;
          if (phaseLoads[phase] < selectedLoad) {
            selectedLoad = phaseLoads[phase];
            selectedPhase = phase;
          }
        }
        if (selectedPhase < 0) break;
        masks[item.tile] |= 1 << selectedPhase;
        phaseLoads[selectedPhase] += item.candidateCost;
        activationCount += 1;
      }
    }
    return {
      masks,
      enabled: true,
      highTileCount,
      omittedTileCount: tileCount - middleEnd,
      activationCount,
    };
  }

  function prepareTrainingData(image, strokePlan, options = {}) {
    const { width, height } = validateInput(image, strokePlan);
    const strokeCount = Math.min(
      strokePlan.length,
      Math.max(1, Math.round(Number(options.maxStrokes) || strokePlan.length)),
    );
    const strokeTextureMode = resolveStrokeTextureMode(options);
    const bristleBundle = strokeTextureMode === STROKE_TEXTURE_FINE_BRISTLES;
    const brushDabs = strokeTextureMode === STROKE_TEXTURE_BRUSH_DABS;
    const textureGuidedDabs = brushDabs && options.textureGuidedDabs === true;
    const splatSizeVariation = clamp01(options.splatSizeVariation);
    const fixedOpacity = Number.isFinite(Number(options.fixedOpacity))
      ? Math.max(0.05, Math.min(0.995, Number(options.fixedOpacity)))
      : null;
    const variableBrushDabs = brushDabs && options.variableBrushDabs === true;
    const sampleCount = brushDabs
      ? variableBrushDabs ? VARIABLE_BRUSH_DAB_SAMPLES : BRUSH_DAB_SAMPLES
      : bristleBundle ? BRISTLE_BUNDLE_SAMPLES : CURVE_SAMPLES;
    const sampleMasks = new Uint16Array(strokeCount);
    const params = new Float32Array(strokeCount * PARAM_STRIDE);
    const flowUnderpaintFlags = new Uint8Array(strokeCount);
    const chainUnderpaintFlags = new Uint8Array(strokeCount);
    for (let index = 0; index < strokeCount; index += 1) {
      const stroke = strokePlan[index];
      const base = index * PARAM_STRIDE;
      flowUnderpaintFlags[index] = stroke.underpaint_splat === true ? 1 : 0;
      chainUnderpaintFlags[index] = stroke.underpaint_chain === true ? 1 : 0;
      sampleMasks[index] = flowUnderpaintFlags[index] || chainUnderpaintFlags[index] ? 1
        : variableBrushDabs ? brushDabSampleMask(stroke.brush_dab_count) : (1 << sampleCount) - 1;
      params[base] = Number(stroke.start_x);
      params[base + 1] = Number(stroke.start_y);
      params[base + 2] = Number(stroke.control_1_x);
      params[base + 3] = Number(stroke.control_1_y);
      params[base + 4] = Number(stroke.control_2_x);
      params[base + 5] = Number(stroke.control_2_y);
      params[base + 6] = Number(stroke.end_x);
      params[base + 7] = Number(stroke.end_y);
      params[base + 8] = Math.max(0.65, Number(stroke.half_width_px) || 0.65);
      params[base + 9] = logit(fixedOpacity ?? stroke.opacity ?? 0.75);
      params[base + 10] = clamp01(
        Number.isFinite(stroke.paint_linear_r)
          ? stroke.paint_linear_r
          : signalSrgbToLinear(stroke.color_r),
      );
      params[base + 11] = clamp01(
        Number.isFinite(stroke.paint_linear_g)
          ? stroke.paint_linear_g
          : signalSrgbToLinear(stroke.color_g),
      );
      params[base + 12] = clamp01(
        Number.isFinite(stroke.paint_linear_b)
          ? stroke.paint_linear_b
          : signalSrgbToLinear(stroke.color_b),
      );
      params[base + 13] = Number(stroke.random) || 0;
      params[base + 14] = Math.max(0, Math.min(3, Math.round(Number(stroke.layer) || 0)));
      const controlPolylineLength =
        Math.hypot(params[base + 2] - params[base], params[base + 3] - params[base + 1]) +
        Math.hypot(params[base + 4] - params[base + 2], params[base + 5] - params[base + 3]) +
        Math.hypot(params[base + 6] - params[base + 4], params[base + 7] - params[base + 5]);
      const strokeBristleBundle = bristleBundle && params[base + 14] >= 1.5;
      const longitudinalSampleCount = brushDabs
        ? 3
        : strokeBristleBundle ? sampleCount / 2 : sampleCount;
      const chainSpacing = Math.max(
        0.5,
        controlPolylineLength / Math.max(
          1,
          strokeBristleBundle ? longitudinalSampleCount : longitudinalSampleCount - 1,
        ),
      );
      params[base + 15] = packChainSpacing(
        chainSpacing,
        stroke.dab_visibility_score ?? stroke.texture_score ?? 1,
        stroke.edge_score ?? 1,
        textureGuidedDabs && stroke.underpaint_splat !== true,
      );
    }
    const initialDetailParams = options.initialDetailParams instanceof Float32Array
      ? options.initialDetailParams
      : null;
    const initialDetailOffset = Math.max(0, Math.round(Number(options.initialDetailOffset) || 0));
    if (initialDetailParams && initialDetailOffset < strokeCount) {
      const detailCapacity = strokeCount - initialDetailOffset;
      const initialStrokeCount = Math.floor(initialDetailParams.length / PARAM_STRIDE);
      const copyStrokeCount = Math.min(detailCapacity, initialStrokeCount);
      params.set(
        initialDetailParams.subarray(0, copyStrokeCount * PARAM_STRIDE),
        initialDetailOffset * PARAM_STRIDE,
      );
    }
    const anchors = params.slice();
    const detailGeometryAnchorParams = options.detailGeometryAnchorParams instanceof Float32Array
      ? options.detailGeometryAnchorParams
      : null;
    if (detailGeometryAnchorParams && initialDetailOffset < strokeCount) {
      const detailCapacity = strokeCount - initialDetailOffset;
      const anchorStrokeCount = Math.floor(detailGeometryAnchorParams.length / PARAM_STRIDE);
      const copyStrokeCount = Math.min(detailCapacity, anchorStrokeCount);
      for (let stroke = 0; stroke < copyStrokeCount; stroke += 1) {
        const targetBase = (initialDetailOffset + stroke) * PARAM_STRIDE;
        const sourceBase = stroke * PARAM_STRIDE;
        for (let component = 0; component < 8; component += 1) {
          anchors[targetBase + component] = detailGeometryAnchorParams[sourceBase + component];
        }
      }
    }

    // Keep unscaled width anchors. Phase broadening is temporary, not a new
    // topology target that would be multiplied again at the next growth event.
    const widthPhaseLifts = new Float32Array(strokeCount);
    if (options.widthTrainingPhases === true) {
      for (let index = 0; index < strokeCount; index += 1) {
        const base = index * PARAM_STRIDE;
        if (params[base + 14] > 2.5) continue;
        const bounds = widthTrainingBounds(anchors[base + 8], width, height, 0, options);
        const before = params[base + 8];
        params[base + 8] = Math.max(bounds.minimum, Math.min(bounds.maximum, before));
        widthPhaseLifts[index] = params[base + 8] - before;
      }
    }

    const target = new Float32Array(width * height * 4);
    for (let pixel = 0; pixel < width * height; pixel += 1) {
      target[pixel * 4] = signalSrgbToLinear(image.rgb[pixel * 3]);
      target[pixel * 4 + 1] = signalSrgbToLinear(image.rgb[pixel * 3 + 1]);
      target[pixel * 4 + 2] = signalSrgbToLinear(image.rgb[pixel * 3 + 2]);
      target[pixel * 4 + 3] = image.alpha?.[pixel] ?? 1;
    }
    const canvasLinear = estimateCanvasLinear(target, width, height);

    const tileCols = Math.ceil(width / TILE_SIZE);
    const tileRows = Math.ceil(height / TILE_SIZE);
    const continuousAdamCarry = options.continuousState !== false
      && options.continuousState?.previousTrainingState != null;
    // Fresh rows start with zero moments. Survivor rows may begin with a
    // read-back Adam state. For beta1=.9, beta2=.999 the Cauchy-Schwarz
    // normalized-step bound is < 7.28 at every age; 8 also covers float error.
    // Position support already includes maxPositionDelta below.
    let widthTravel = 0;
    if (options.widthTrainingPhases === true) {
      for (let step = 0; step < Math.max(1, Number(options.iterations) || 1); step += 1) {
        const t = step + 1;
        const ratio = 0.9 ** 2 / 0.999;
        const momentBound = Math.sqrt(
          (1 - 0.9) ** 2 / (1 - 0.999)
          * (1 - 0.999 ** t) / (1 - 0.9 ** t) ** 2
          * (1 - ratio ** t) / (1 - ratio),
        );
        widthTravel += Math.abs(widthTrainingSettings(step, options).widthLearningRate)
          * (continuousAdamCarry ? 8 : momentBound);
      }
    }
    const tileLists = Array.from({ length: tileCols * tileRows }, () => []);
    const curveSplatChain = options.representation === "curve-splat-chain";
    const addCandidateToTiles = (candidate, minimumX, minimumY, maximumX, maximumY) => {
      const tileX0 = Math.max(0, Math.floor(minimumX / TILE_SIZE));
      const tileY0 = Math.max(0, Math.floor(minimumY / TILE_SIZE));
      const tileX1 = Math.min(tileCols - 1, Math.floor(maximumX / TILE_SIZE));
      const tileY1 = Math.min(tileRows - 1, Math.floor(maximumY / TILE_SIZE));
      for (let tileY = tileY0; tileY <= tileY1; tileY += 1) {
        for (let tileX = tileX0; tileX <= tileX1; tileX += 1) {
          tileLists[tileY * tileCols + tileX].push(candidate);
        }
      }
    };
    // The flow plan is coarse -> medium -> fine. Standard front-to-back alpha
    // needs fine strokes first so the final visual order matches the CPU
    // render-only gate where later fine strokes are painted over coarse ones.
    for (let index = strokeCount - 1; index >= 0; index -= 1) {
      const base = index * PARAM_STRIDE;
      if (flowUnderpaintFlags[index]) {
        const centerX = params[base];
        const centerY = params[base + 1];
        const axisX = params[base + 2] - centerX;
        const axisY = params[base + 3] - centerY;
        const sigmaLong = Math.max(0.85, Math.hypot(axisX, axisY));
        const sigmaShort = Math.max(0.65, params[base + 8]);
        const directionX = axisX / Math.max(1e-6, sigmaLong);
        const directionY = axisY / Math.max(1e-6, sigmaLong);
        const normalX = -directionY;
        const normalY = directionX;
        // Compact quartic support ends at q=1.16. Keep the CPU tile bounds in
        // parity with that finite support instead of the retired 4-sigma rear
        // Gaussian bounds.
        const supportScale = 1.16 ** 0.25;
        const extentX = supportScale
          * Math.hypot(sigmaLong * directionX, sigmaShort * normalX) + 2;
        const extentY = supportScale
          * Math.hypot(sigmaLong * directionY, sigmaShort * normalY) + 2;
        addCandidateToTiles(
          curveSplatChain ? index * sampleCount : index,
          centerX - extentX,
          centerY - extentY,
          centerX + extentX,
          centerY + extentY,
        );
        continue;
      }
      if (curveSplatChain) {
        const widthPx = Math.max(0.55, params[base + 8]);
        const spacing = unpackChainSpacing(params[base + 15], textureGuidedDabs);
        // Static candidates must cover any width allowed during this stage,
        // including a phase boundary in a short/fixed-count run. The control
        // hull plus movement limit also covers tangent rotation after Adam.
        const phasedBounds = options.widthTrainingPhases === true;
        const stageEnd = Math.max(0, (Number(options.iterations) || 1) - 1);
        const permittedWidth = phasedBounds ? Math.max(widthPx,
          widthTrainingBounds(anchors[base + 8], width, height, 0, options).maximum,
          widthTrainingBounds(anchors[base + 8], width, height, stageEnd, options).maximum,
        ) : widthPx;
        const widthLearningScale = params[base + 14] > 1.5 && params[base + 14] < 2.5
          ? Math.max(1, Math.min(8, Number(options.frontWidthLearningScale) || 1)) : 1;
        const widthCeiling = Math.min(permittedWidth, widthPx + widthTravel * widthLearningScale + 1e-3);
        for (let sample = 0; sample < sampleCount; sample += 1) {
          if (!(sampleMasks[index] & (1 << sample))) continue;
          const profile = chainSampleProfile(
            strokeTextureMode,
            params[base + 14],
            sample,
            params[base + 13],
            sampleCount,
            unpackChainTextureScore(params[base + 15], textureGuidedDabs),
            unpackChainEdgeScore(params[base + 15], textureGuidedDabs),
            splatSizeVariation,
          );
          const t = profile.t;
          if (phasedBounds) {
            const sigmaLong = Math.max(0.55,
              Math.max(spacing * 0.72, widthCeiling * 0.85) * profile.lengthFactor);
            const sigmaShort = Math.max(0.38, widthCeiling * profile.widthFactor);
            const support = strokeTextureMode === STROKE_TEXTURE_BRUSH_DABS
              ? 1.16 ** 0.25 : 4;
            const radius = support * Math.hypot(sigmaLong, sigmaShort)
              + Math.abs(profile.normalOffsetFactor) * widthCeiling
              + Math.max(0, Number(options.maxPositionDelta ?? 6)) + 2;
            const xs = [0, 2, 4, 6].map((c) => anchors[base + c]);
            const ys = [1, 3, 5, 7].map((c) => anchors[base + c]);
            addCandidateToTiles(index * sampleCount + sample,
              Math.min(...xs) - radius, Math.min(...ys) - radius,
              Math.max(...xs) + radius, Math.max(...ys) + radius);
            continue;
          }
          let [pointX, pointY] = cubicPointFromParams(params, base, t);
          const [tangentX, tangentY] = cubicTangentFromParams(params, base, t);
          const normalX = -tangentY;
          const normalY = tangentX;
          pointX += normalX * profile.normalOffsetFactor * widthPx;
          pointY += normalY * profile.normalOffsetFactor * widthPx;
          const sigmaLong = Math.max(
            0.55,
            Math.max(spacing * 0.72, widthPx * 0.85) * profile.lengthFactor,
          );
          const sigmaShort = Math.max(0.38, widthPx * profile.widthFactor);
          const extentX = 4 * Math.hypot(sigmaLong * tangentX, sigmaShort * normalX) + 6;
          const extentY = 4 * Math.hypot(sigmaLong * tangentY, sigmaShort * normalY) + 6;
          addCandidateToTiles(
            index * sampleCount + sample,
            pointX - extentX,
            pointY - extentY,
            pointX + extentX,
            pointY + extentY,
          );
        }
        continue;
      }
      let minimumX = Infinity;
      let minimumY = Infinity;
      let maximumX = -Infinity;
      let maximumY = -Infinity;
      for (let control = 0; control < 4; control += 1) {
        minimumX = Math.min(minimumX, params[base + control * 2]);
        minimumY = Math.min(minimumY, params[base + control * 2 + 1]);
        maximumX = Math.max(maximumX, params[base + control * 2]);
        maximumY = Math.max(maximumY, params[base + control * 2 + 1]);
      }
      const padding = options.widthTrainingPhases === true
        ? (params[base + 8] + widthTravel * Math.max(1, Math.min(8,
          Number(options.frontWidthLearningScale) || 1))) * 2.6
          + Math.max(0, Number(options.maxPositionDelta ?? 6)) + 2
        : params[base + 8] * 2.6 + 2;
      addCandidateToTiles(
        index,
        minimumX - padding,
        minimumY - padding,
        maximumX + padding,
        maximumY + padding,
      );
    }
    const tileOffsets = new Uint32Array(tileLists.length + 1);
    let candidateCount = 0;
    for (let tile = 0; tile < tileLists.length; tile += 1) {
      tileOffsets[tile] = candidateCount;
      candidateCount += tileLists[tile].length;
    }
    tileOffsets[tileLists.length] = candidateCount;
    const tileIndices = new Uint32Array(candidateCount);
    let cursor = 0;
    for (const tile of tileLists) {
      tileIndices.set(tile, cursor);
      cursor += tile.length;
    }
    const tileSampleStride = Math.max(1, Math.min(
      16,
      Math.round(Number(options.tileSampleStride) || 8),
    ));
    const residualPriorityTiles = options.residualPriorityTileSampling === true
      ? buildResidualPriorityTileMasks({
          width,
          height,
          target,
          residualRender: options.residualRender,
          tileLists,
          tileCols,
          tileRows,
          tileSampleStride,
        })
      : {
          masks: new Uint32Array(tileLists.length),
          enabled: false,
          highTileCount: 0,
          omittedTileCount: 0,
          activationCount: 0,
        };
    let pixelCandidatePairs = 0;
    for (let tileY = 0; tileY < tileRows; tileY += 1) {
      for (let tileX = 0; tileX < tileCols; tileX += 1) {
        const tile = tileY * tileCols + tileX;
        const tileWidth = Math.min(TILE_SIZE, width - tileX * TILE_SIZE);
        const tileHeight = Math.min(TILE_SIZE, height - tileY * TILE_SIZE);
        pixelCandidatePairs += tileLists[tile].length * tileWidth * tileHeight;
      }
    }
    const underpaintParentCount = flowUnderpaintFlags.reduce((sum, value) => sum + value, 0)
      + chainUnderpaintFlags.reduce((sum, value) => sum + value, 0);
    const detailParentCount = strokeCount - underpaintParentCount;
    const underpaintSplatCount = underpaintParentCount;
    const physicalSplatCount = curveSplatChain
      ? variableBrushDabs
        ? sampleMasks.reduce((sum, mask) => {
            for (let active = mask; active; active &= active - 1) sum += 1;
            return sum;
          }, 0)
        : underpaintSplatCount + detailParentCount * sampleCount
      : strokeCount;
    const initialPhysicalSplatScaleStats = curveSplatChain
      ? physicalSplatScaleStats(params, {
          strokeCount,
          sampleCount,
          sampleMasks,
          strokeTextureMode,
          textureGuidedDabs,
          splatSizeVariation,
        })
      : null;
    return {
      width,
      height,
      strokeCount,
      params,
      anchors,
      widthPhaseLifts,
      target,
      canvasLinear,
      tileCols,
      tileRows,
      tileOffsets,
      tileIndices,
      tileSamplingMasks: residualPriorityTiles.masks,
      tileSampleStride,
      residualPriorityTileSampling: residualPriorityTiles.enabled,
      residualPriorityHighTileCount: residualPriorityTiles.highTileCount,
      residualPriorityOmittedTileCount: residualPriorityTiles.omittedTileCount,
      residualPriorityActivationCount: residualPriorityTiles.activationCount,
      maxTileCandidates: tileLists.reduce((maximum, tile) => Math.max(maximum, tile.length), 0),
      meanTileCandidates: tileLists.length ? candidateCount / tileLists.length : 0,
      tileCandidateEntries: candidateCount,
      pixelCandidatePairs,
      physicalSplatCount,
      sampleCount,
      sampleMasks,
      variableBrushDabs,
      bristleBundle,
      brushDabs,
      textureGuidedDabs,
      splatSizeVariation,
      strokeTextureMode,
      underpaintParentCount,
      underpaintSplatCount,
      initialPhysicalSplatScaleStats,
    };
  }

  const COMMON_WGSL = String.raw`
struct Config {
  width: u32,
  height: u32,
  stroke_count: u32,
  tile_cols: u32,
  tile_rows: u32,
  param_stride: u32,
  sample_count: u32,
  iteration: u32,
  color_anchor: f32,
  geometry_anchor: f32,
  width_anchor: f32,
  opacity_anchor: f32,
  lr_position: f32,
  lr_width: f32,
  lr_opacity: f32,
  lr_color: f32,
  beta1: f32,
  beta2: f32,
  epsilon: f32,
  loss_scale: f32,
  max_width: f32,
  max_position_delta: f32,
  representation: f32,
  tile_sample_stride: u32,
  max_curve_arc: f32,
  canvas_linear_r: f32,
  canvas_linear_g: f32,
  canvas_linear_b: f32,
  stroke_motion_coherence: f32,
  bristle_bundle: f32,
  residual_priority_sampling: f32,
  fixed_opacity_logit: f32,
  front_width_learning_scale: f32,
  texture_guided_dabs: f32,
  splat_size_variation: f32,
  width_minimum_factor: f32,
  width_maximum_factor: f32,
}

struct StrokeEval {
  alpha: f32,
  pigment: vec3<f32>,
  pigment_scale: f32,
  coverage: f32,
  opacity: f32,
  modulation: f32,
  soft_d2: f32,
  width_effective: f32,
  edge2: f32,
  along: f32,
  side: f32,
}

struct ChainKernelSample {
  kernel: f32,
  delta_gradient: vec2<f32>,
  width_gradient: f32,
}

fn param_at(stroke: u32, component: u32) -> f32 {
  return params[stroke * config.param_stride + component];
}

fn chain_texture_score(stroke: u32) -> f32 {
  if (config.texture_guided_dabs < 0.5 || !chain_uses_brush_dabs()) { return 1.0; }
  let guide_bucket = floor(param_at(stroke, 15u) / 256.0);
  return clamp(
    floor(guide_bucket / 4.0) / 63.0,
    0.0,
    1.0
  );
}

fn chain_edge_score(stroke: u32) -> f32 {
  if (config.texture_guided_dabs < 0.5 || !chain_uses_brush_dabs()) { return 1.0; }
  let guide_bucket = floor(param_at(stroke, 15u) / 256.0);
  return clamp((guide_bucket - floor(guide_bucket / 4.0) * 4.0) / 3.0, 0.0, 1.0);
}

fn chain_spacing(stroke: u32) -> f32 {
  let packed = max(0.5, param_at(stroke, 15u));
  if (config.texture_guided_dabs < 0.5 || !chain_uses_brush_dabs()) { return packed; }
  let guide_bucket = floor(packed / 256.0);
  return max(0.5, packed - guide_bucket * 256.0);
}

fn cubic_point(stroke: u32, t: f32) -> vec2<f32> {
  let s = 1.0 - t;
  let p0 = vec2<f32>(param_at(stroke, 0u), param_at(stroke, 1u));
  let p1 = vec2<f32>(param_at(stroke, 2u), param_at(stroke, 3u));
  let p2 = vec2<f32>(param_at(stroke, 4u), param_at(stroke, 5u));
  let p3 = vec2<f32>(param_at(stroke, 6u), param_at(stroke, 7u));
  return s*s*s*p0 + 3.0*s*s*t*p1 + 3.0*s*t*t*p2 + t*t*t*p3;
}

fn cubic_tangent(stroke: u32, t: f32) -> vec2<f32> {
  let s = 1.0 - t;
  let p0 = vec2<f32>(param_at(stroke, 0u), param_at(stroke, 1u));
  let p1 = vec2<f32>(param_at(stroke, 2u), param_at(stroke, 3u));
  let p2 = vec2<f32>(param_at(stroke, 4u), param_at(stroke, 5u));
  let p3 = vec2<f32>(param_at(stroke, 6u), param_at(stroke, 7u));
  return 3.0*s*s*(p1-p0) + 6.0*s*t*(p2-p1) + 3.0*t*t*(p3-p2);
}

fn chain_uses_bristle(stroke: u32) -> bool {
  return config.bristle_bundle > 0.5
    && config.bristle_bundle < 1.5
    && param_at(stroke, 14u) >= 1.5;
}

fn chain_uses_brush_dabs() -> bool {
  return config.bristle_bundle > 1.5;
}

fn chain_brush_loaded_side(stroke: u32) -> f32 {
  let random = param_at(stroke, 13u);
  let layer = param_at(stroke, 14u);
  return select(-1.0, 1.0, sin(random * 137.3 + layer * 11.7) >= 0.0);
}

fn chain_splat_width_variation(stroke: u32, sample: u32) -> f32 {
  var amount = clamp(config.splat_size_variation, 0.0, 1.0);
  let random = param_at(stroke, 13u);
  let layer = param_at(stroke, 14u);
  let noise = sin(random * 211.1 + f32(sample) * 47.3 + layer * 13.7);
  if (chain_uses_brush_dabs()) {
    if (config.texture_guided_dabs > 0.5 && (sample == 0u || sample == 2u)) {
      amount *= smoothstep(0.30, 0.95, chain_edge_score(stroke));
    }
    var family_scale = 1.00;
    if (sample == 0u) { family_scale = 0.40; }
    else if (sample == 1u) { family_scale = 0.80; }
    else if (sample == 2u) { family_scale = 0.60; }
    else if (sample == 3u) { family_scale = 0.95; }
    else if (sample == 4u) { family_scale = 0.88; }
    else if (sample == 5u) { family_scale = 0.85; }
    else if (sample == 6u) { family_scale = 1.15; }
    else if (sample == 8u) { family_scale = 0.95; }
    let jitter = 1.0 + 0.08 * noise;
    return max(0.05, mix(1.0, family_scale * jitter, amount));
  }
  return max(0.15, 1.0 + noise * 0.85 * amount);
}

fn chain_splat_length_variation(stroke: u32, sample: u32) -> f32 {
  var amount = clamp(config.splat_size_variation, 0.0, 1.0);
  let random = param_at(stroke, 13u);
  let layer = param_at(stroke, 14u);
  let noise = sin(random * 173.9 + f32(sample) * 29.7 + layer * 7.9 + 1.7);
  if (chain_uses_brush_dabs()) {
    if (config.texture_guided_dabs > 0.5 && (sample == 0u || sample == 2u)) {
      amount *= smoothstep(0.30, 0.95, chain_edge_score(stroke));
    }
    var family_scale = 0.94;
    if (sample == 0u) { family_scale = 2.05; }
    else if (sample == 1u) { family_scale = 1.34; }
    else if (sample == 2u) { family_scale = 1.88; }
    else if (sample == 3u) { family_scale = 1.24; }
    else if (sample == 4u) { family_scale = 1.58; }
    else if (sample == 5u) { family_scale = 0.84; }
    else if (sample == 6u) { family_scale = 0.72; }
    else if (sample == 8u) { family_scale = 0.90; }
    let jitter = 1.0 + 0.06 * noise;
    return max(0.35, mix(1.0, family_scale * jitter, amount));
  }
  return max(0.35, 1.0 + noise * 0.65 * amount);
}

fn chain_sample_t(stroke: u32, sample: u32) -> f32 {
  if (chain_uses_bristle(stroke)) {
    let longitudinal_count = max(1u, config.sample_count / 2u);
    return (f32(sample / 2u) + 0.5) / f32(longitudinal_count);
  }
  if (chain_uses_brush_dabs()) {
    let random = param_at(stroke, 13u);
    let load = 0.5 + 0.5 * sin(random * 97.0 + f32(sample) * 19.0);
    var dab_t = 0.84 + 0.05 * load;
    if (sample == 0u) { dab_t = 0.20 + 0.05 * load; }
    else if (sample == 1u) { dab_t = 0.66 + 0.08 * load; }
    else if (sample == 2u) { dab_t = 0.24 + 0.05 * load; }
    else if (sample == 3u) { dab_t = 0.70 + 0.07 * load; }
    else if (sample == 4u) { dab_t = 0.47 + 0.06 * load; }
    else if (sample == 5u) { dab_t = 0.12 + 0.03 * load; }
    else if (sample == 6u) { dab_t = 0.48 + 0.04 * load; }
    else if (sample == 8u) { dab_t = 0.32 + 0.06 * load; }
    if (config.texture_guided_dabs > 0.5) {
      let texture = smoothstep(0.12, 0.65, chain_texture_score(stroke));
      let edge = smoothstep(0.30, 0.95, chain_edge_score(stroke));
      let shape_guide = select(texture, edge, sample == 0u || sample == 2u);
      if (sample < 5u) {
        dab_t = 0.5 + (dab_t - 0.5) * (0.72 + shape_guide * 0.28);
      } else {
        dab_t = 0.5 + (dab_t - 0.5) * (0.5 + texture * 0.5);
      }
    }
    return dab_t;
  }
  return f32(sample) / max(1.0, f32(config.sample_count - 1u));
}

fn chain_normal_offset_factor(stroke: u32, sample: u32) -> f32 {
  if (chain_uses_bristle(stroke)) {
    return select(-0.18, 0.18, sample % 2u == 1u);
  }
  if (!chain_uses_brush_dabs()) { return 0.0; }
  let side = chain_brush_loaded_side(stroke);
  let layer = clamp(param_at(stroke, 14u), 0.0, 2.0);
  let layer_spread = 0.04 * layer;
  var factor = -0.02;
  if (sample == 0u) { factor = 0.48 + layer_spread; }
  else if (sample == 1u) { factor = 0.44 + layer_spread; }
  else if (sample == 2u) { factor = -0.46 - layer_spread; }
  else if (sample == 3u) { factor = -0.42 - layer_spread; }
  else if (sample == 4u) { factor = 0.03; }
  else if (sample == 5u) { factor = -0.06; }
  else if (sample == 6u) { factor = 0.04; }
  else if (sample == 8u) { factor = -0.28; }
  if (sample < 5u && config.texture_guided_dabs > 0.5) {
    let texture = smoothstep(0.12, 0.65, chain_texture_score(stroke));
    let edge = smoothstep(0.30, 0.95, chain_edge_score(stroke));
    let shape_guide = select(texture, edge, sample == 0u || sample == 2u);
    var flat_factor = 0.0;
    if (sample == 0u) { flat_factor = 0.16; }
    else if (sample == 1u) { flat_factor = 0.08; }
    else if (sample == 2u) { flat_factor = -0.16; }
    else if (sample == 3u) { flat_factor = -0.08; }
    factor = mix(flat_factor, factor, shape_guide);
  }
  return factor * side;
}

fn chain_width_factor(stroke: u32, sample: u32) -> f32 {
  var factor = 0.46;
  if (chain_uses_bristle(stroke)) {
    factor = 0.38;
  } else if (chain_uses_brush_dabs()) {
    factor = 0.74;
    if (sample == 0u) { factor = 0.18; }
    else if (sample == 1u) { factor = 0.34; }
    else if (sample == 2u) { factor = 0.16; }
    else if (sample == 3u) { factor = 0.44; }
    else if (sample == 4u) { factor = 0.52; }
    else if (sample == 5u) { factor = 0.78; }
    else if (sample == 6u) { factor = 0.92; }
    else if (sample == 8u) { factor = 0.66; }
  }
  if (chain_uses_brush_dabs() && config.texture_guided_dabs > 0.5) {
    let texture = smoothstep(0.12, 0.65, chain_texture_score(stroke));
    let edge = smoothstep(0.30, 0.95, chain_edge_score(stroke));
    let shape_guide = select(texture, edge, sample == 0u || sample == 2u);
    if (sample < 5u) {
      factor = mix(0.62, factor, shape_guide);
    } else {
      factor = mix(0.88, factor, texture);
    }
  }
  return factor * chain_splat_width_variation(stroke, sample);
}

fn chain_length_factor(stroke: u32, sample: u32) -> f32 {
  var factor = 1.0;
  if (chain_uses_brush_dabs()) {
    factor = 0.70;
    if (sample == 0u) { factor = 1.05; }
    else if (sample == 1u) { factor = 0.95; }
    else if (sample == 2u) { factor = 1.02; }
    else if (sample == 3u) { factor = 0.92; }
    else if (sample == 4u) { factor = 1.18; }
    else if (sample == 5u) { factor = 0.72; }
    else if (sample == 6u) { factor = 0.82; }
    else if (sample == 8u) { factor = 0.80; }
  }
  if (chain_uses_brush_dabs() && config.texture_guided_dabs > 0.5) {
    let texture = smoothstep(0.12, 0.65, chain_texture_score(stroke));
    let edge = smoothstep(0.30, 0.95, chain_edge_score(stroke));
    let shape_guide = select(texture, edge, sample == 0u || sample == 2u);
    if (sample < 5u) {
      factor = mix(1.15, factor, shape_guide);
    } else {
      factor = mix(1.45, factor, texture);
    }
  }
  return factor * chain_splat_length_variation(stroke, sample);
}

fn chain_opacity_factor(stroke: u32, sample: u32) -> f32 {
  if (chain_uses_bristle(stroke)) { return 0.90; }
  return 1.0;
}

fn chain_micro_modulation(stroke: u32, sample: u32, t: f32, random: f32) -> f32 {
  // Brush dabs carry texture in their contour and pigment. Their connected
  // interior follows Fixed stroke opacity without a second alpha modulation.
  if (chain_uses_brush_dabs()) { return 1.0; }
  return (
    0.92 + 0.08 * (0.5 + 0.5 * sin(t * 17.0 + random * 83.0))
  ) * chain_opacity_factor(stroke, sample);
}

fn chain_kernel_sample(
  stroke: u32,
  sample: u32,
  along_distance: f32,
  side_distance: f32,
  sigma_long: f32,
  sigma_short: f32,
  width: f32,
  spacing: f32
) -> ChainKernelSample {
  if (!chain_uses_brush_dabs()) {
    let q = along_distance * along_distance / (sigma_long * sigma_long)
      + side_distance * side_distance / (sigma_short * sigma_short);
    let gaussian = exp(-0.5 * min(24.0, q));
    return ChainKernelSample(
      gaussian,
      -gaussian * vec2<f32>(
        along_distance / (sigma_long * sigma_long),
        side_distance / (sigma_short * sigma_short)
      ),
      gaussian * side_distance * side_distance
        / max(1e-5, sigma_short * sigma_short * sigma_short)
        * chain_width_factor(stroke, sample)
    );
  }

  // The existing Brush Algorithm uses a connected quartic contour with a
  // flat interior and a differentiable feather. Flow reuses that footprint
  // contract for each curve-aligned Splat instead of Gaussian alpha tails.
  let u = along_distance / max(0.0001, sigma_long);
  let v = side_distance / max(0.0001, sigma_short);
  let u2 = u * u;
  let v2 = v * v;
  let u4 = u2 * u2;
  let v4 = v2 * v2;
  let q = u4 + v4;
  let feather = 0.16;
  let raw_t = (q - (1.0 - feather)) / (2.0 * feather);
  let edge_t = clamp(raw_t, 0.0, 1.0);
  let kernel = 1.0 - edge_t * edge_t * (3.0 - 2.0 * edge_t);
  let d_kernel_d_q = select(
    0.0,
    -6.0 * edge_t * (1.0 - edge_t) / (2.0 * feather),
    raw_t > 0.0 && raw_t < 1.0
  );
  let delta_gradient = d_kernel_d_q * vec2<f32>(
    4.0 * u * u2 / max(0.0001, sigma_long),
    4.0 * v * v2 / max(0.0001, sigma_short)
  );
  let width_factor = chain_width_factor(stroke, sample);
  let length_factor = chain_length_factor(stroke, sample);
  var d_q_d_width = 0.0;
  if (width * width_factor > 0.38) {
    d_q_d_width -= 4.0 * v4 * width_factor / max(0.0001, sigma_short);
  }
  if (width * 0.85 >= spacing * 0.72 && width * 0.85 * length_factor > 0.55) {
    d_q_d_width -= 4.0 * u4 * 0.85 * length_factor / max(0.0001, sigma_long);
  }
  return ChainKernelSample(kernel, delta_gradient, d_kernel_d_q * d_q_d_width);
}

// Pull the local kernel derivative through normalized tangent rotation,
// including the normal-offset movement of the dab center. Spacing is a
// fixed per-stage parameter in forward and is deliberately held constant.
fn chain_tangent_pullback(
  normal: vec2<f32>, tangent_length: f32,
  along: f32, side: f32, offset: f32, kernel_gradient: vec2<f32>
) -> vec2<f32> {
  return normal * (kernel_gradient.x * (side + offset) - kernel_gradient.y * along)
    / max(1e-7, tangent_length);
}

fn coverage_backcoat_kernel(
  along_distance: f32,
  side_distance: f32,
  half_long: f32,
  half_short: f32
) -> f32 {
  let u = along_distance / max(0.0001, half_long);
  let v = side_distance / max(0.0001, half_short);
  let u2 = u * u;
  let v2 = v * v;
  let q = u2 * u2 + v2 * v2;
  let feather = 0.16;
  let edge_t = clamp((q - (1.0 - feather)) / (2.0 * feather), 0.0, 1.0);
  return 1.0 - edge_t * edge_t * (3.0 - 2.0 * edge_t);
}

fn chain_pigment_scale(stroke: u32, sample: u32, random: f32) -> f32 {
  if (chain_uses_bristle(stroke)) {
    let lane = select(-1.0, 1.0, sample % 2u == 1u);
    let variation = 0.055 + 0.025 * (0.5 + 0.5 * sin(random * 97.0));
    return 1.0 + lane * variation;
  }
  if (!chain_uses_brush_dabs()) { return 1.0; }
  let load = 0.5 + 0.5 * sin(random * 97.0 + f32(sample) * 19.0);
  var pigment = 0.96 + 0.03 * load;
  if (sample == 0u) { pigment = 0.70 + 0.12 * load; }
  else if (sample == 1u) { pigment = 0.76 + 0.12 * load; }
  else if (sample == 2u) { pigment = 0.74 + 0.14 * load; }
  else if (sample == 3u) { pigment = 0.80 + 0.12 * load; }
  else if (sample == 4u) { pigment = 0.86 + 0.08 * load; }
  else if (sample == 5u) { pigment = 0.98; }
  else if (sample == 6u) { pigment = 1.00; }
  if (config.texture_guided_dabs > 0.5) {
    let texture = smoothstep(0.08, 0.72, chain_texture_score(stroke));
    let edge = smoothstep(0.12, 0.68, chain_edge_score(stroke));
    let pigment_guide = select(texture, edge, sample == 0u || sample == 2u);
    pigment = mix(1.0, pigment, pigment_guide);
  }
  return pigment;
}

fn evaluate_stroke(stroke: u32, pixel: vec2<f32>) -> StrokeEval {
  if (config.representation > 0.5) {
    let width = max(0.55, param_at(stroke, 8u));
    let spacing = chain_spacing(stroke);
    let sigma_long = max(0.55, max(spacing * 0.72, width * 0.85));
    let sigma_short = max(0.38, width * 0.46);
    let opacity = 1.0 / (1.0 + exp(-param_at(stroke, 9u)));
    let random = param_at(stroke, 13u);
    let layer = param_at(stroke, 14u);
    var transmittance = 1.0;
    var center = vec2<f32>(0.0);
    var tangent_sum = vec2<f32>(0.0);
    var along_sum = 0.0;
    var support_sum = 0.0;
    for (var sample = 0u; sample < config.sample_count; sample += 1u) {
      let t = f32(sample) / max(1.0, f32(config.sample_count - 1u));
      let point = cubic_point(stroke, t);
      let tangent = normalize(cubic_tangent(stroke, t) + vec2<f32>(1e-7, 0.0));
      let normal = vec2<f32>(-tangent.y, tangent.x);
      let delta = pixel - point;
      let along_distance = dot(delta, tangent);
      let side_distance = dot(delta, normal);
      let q = along_distance * along_distance / (sigma_long * sigma_long)
        + side_distance * side_distance / (sigma_short * sigma_short);
      let gaussian = exp(-0.5 * min(24.0, q));
      let micro_modulation = 0.92 + 0.08 * (0.5 + 0.5 * sin(t * 17.0 + random * 83.0));
      let micro_alpha = min(0.99, opacity * gaussian * micro_modulation);
      let visible_support = transmittance * micro_alpha;
      center += point * visible_support;
      tangent_sum += tangent * visible_support;
      along_sum += t * visible_support;
      support_sum += visible_support;
      transmittance *= 1.0 - micro_alpha;
    }
    let inverse_support = 1.0 / max(1e-8, support_sum);
    center *= inverse_support;
    let tangent = normalize(tangent_sum * inverse_support + vec2<f32>(1e-7, 0.0));
    let along = along_sum * inverse_support;
    let side = dot(pixel - center, vec2<f32>(-tangent.y, tangent.x)) / max(0.5, width);
    let alpha = min(0.999, 1.0 - transmittance);
    let pigment_scale = 1.0;
    let pigment = vec3<f32>(param_at(stroke, 10u), param_at(stroke, 11u), param_at(stroke, 12u));
    return StrokeEval(
      alpha,
      pigment,
      pigment_scale,
      min(1.0, alpha / max(1e-5, opacity)),
      opacity,
      1.0,
      0.0,
      width,
      sigma_short * sigma_short,
      along,
      side
    );
  }
  let width = max(0.55, param_at(stroke, 8u));
  let tau = max(0.36, width * width * 0.22);
  var minimum_d2 = 1e30;
  for (var sample = 0u; sample < config.sample_count; sample += 1u) {
    let t = f32(sample) / max(1.0, f32(config.sample_count - 1u));
    let delta = cubic_point(stroke, t) - pixel;
    minimum_d2 = min(minimum_d2, dot(delta, delta));
  }
  var weight_sum = 0.0;
  var center = vec2<f32>(0.0);
  var tangent = vec2<f32>(0.0);
  var along = 0.0;
  for (var sample = 0u; sample < config.sample_count; sample += 1u) {
    let t = f32(sample) / max(1.0, f32(config.sample_count - 1u));
    let point = cubic_point(stroke, t);
    let delta = point - pixel;
    let weight = exp(-(dot(delta, delta) - minimum_d2) / tau);
    weight_sum += weight;
    center += point * weight;
    tangent += cubic_tangent(stroke, t) * weight;
    along += t * weight;
  }
  let inverse_weight = 1.0 / max(1e-8, weight_sum);
  center *= inverse_weight;
  tangent = normalize(tangent * inverse_weight + vec2<f32>(1e-7, 0.0));
  along *= inverse_weight;
  let soft_d2 = minimum_d2 - tau * log(max(1e-8, weight_sum / f32(config.sample_count)));
  let taper = 0.34 + 0.66 * pow(max(0.0, sin(3.14159265 * clamp(along, 0.0, 1.0))), 0.58);
  let width_effective = max(0.4, width * taper);
  let edge = max(0.5, width_effective * 0.42);
  let edge2 = edge * edge;
  let coverage = 1.0 / (1.0 + exp(clamp((soft_d2 - width_effective * width_effective) / edge2, -12.0, 12.0)));
  let side = dot(pixel - center, vec2<f32>(-tangent.y, tangent.x)) / max(0.5, width_effective);
  let random = param_at(stroke, 13u);
  let layer = param_at(stroke, 14u);
  let lane = 0.5 + 0.5 * sin(side * 3.14159265 * (2.5 + layer * 1.5) + along * 11.0 + random * 37.0);
  let grain = 0.5 + 0.5 * sin(side * 31.0 - along * 19.0 + random * 83.0);
  let modulation = 0.52 + 0.38 * lane + 0.10 * grain;
  let opacity = 1.0 / (1.0 + exp(-param_at(stroke, 9u)));
  let alpha = min(0.98, opacity * coverage * modulation);
  let edge_pool = pow(max(0.0, 1.0 - min(1.0, abs(side))), 0.52);
  let pigment_scale = (0.90 + 0.12 * edge_pool) * (0.94 + 0.08 * lane + 0.02 * grain);
  let pigment = vec3<f32>(param_at(stroke, 10u), param_at(stroke, 11u), param_at(stroke, 12u)) * pigment_scale;
  return StrokeEval(alpha, pigment, pigment_scale, coverage, opacity, modulation, soft_d2, width_effective, edge2, along, side);
}

fn evaluate_chain_micro_splat(candidate: u32, pixel: vec2<f32>) -> StrokeEval {
  let stroke = candidate / config.sample_count;
  let sample = candidate % config.sample_count;
  let t = chain_sample_t(stroke, sample);
  let width = max(0.55, param_at(stroke, 8u));
  let spacing = chain_spacing(stroke);
  let sigma_long = max(0.55, max(spacing * 0.72, width * 0.85)
    * chain_length_factor(stroke, sample));
  let sigma_short = max(0.38, width * chain_width_factor(stroke, sample));
  let tangent = normalize(cubic_tangent(stroke, t) + vec2<f32>(1e-7, 0.0));
  let normal = vec2<f32>(-tangent.y, tangent.x);
  let point = cubic_point(stroke, t)
    + normal * chain_normal_offset_factor(stroke, sample) * width;
  let delta = pixel - point;
  let along_distance = dot(delta, tangent);
  let side_distance = dot(delta, normal);
  let kernel_sample = chain_kernel_sample(
    stroke,
    sample,
    along_distance,
    side_distance,
    sigma_long,
    sigma_short,
    width,
    spacing
  );
  let opacity = 1.0 / (1.0 + exp(-param_at(stroke, 9u)));
  let random = param_at(stroke, 13u);
  let micro_modulation = chain_micro_modulation(stroke, sample, t, random);
  let alpha = min(0.99, opacity * kernel_sample.kernel * micro_modulation);
  let pigment_scale = chain_pigment_scale(stroke, sample, random);
  let pigment = vec3<f32>(param_at(stroke, 10u), param_at(stroke, 11u), param_at(stroke, 12u))
    * pigment_scale;
  return StrokeEval(
    alpha,
    pigment,
    pigment_scale,
    kernel_sample.kernel,
    opacity,
    micro_modulation,
    0.0,
    width,
    sigma_short * sigma_short,
    t,
    side_distance / max(0.38, sigma_short)
  );
}

fn evaluate_flow_underpaint_splat(stroke: u32, pixel: vec2<f32>) -> StrokeEval {
  let center = vec2<f32>(param_at(stroke, 0u), param_at(stroke, 1u));
  let axis = vec2<f32>(param_at(stroke, 2u), param_at(stroke, 3u)) - center;
  let sigma_long = max(0.85, length(axis));
  let sigma_short = max(0.65, param_at(stroke, 8u));
  let tangent = axis / max(1e-6, sigma_long);
  let normal = vec2<f32>(-tangent.y, tangent.x);
  let delta = pixel - center;
  let along_distance = dot(delta, tangent);
  let side_distance = dot(delta, normal);
  let coverage = coverage_backcoat_kernel(
    along_distance,
    side_distance,
    sigma_long,
    sigma_short
  );
  let opacity = 1.0 / (1.0 + exp(-param_at(stroke, 9u)));
  let alpha = min(0.99, opacity * coverage);
  let pigment = vec3<f32>(param_at(stroke, 10u), param_at(stroke, 11u), param_at(stroke, 12u));
  return StrokeEval(
    alpha,
    pigment,
    1.0,
    coverage,
    opacity,
    1.0,
    0.0,
    sigma_short,
    sigma_short * sigma_short,
    0.5,
    side_distance / max(0.65, sigma_short)
  );
}

fn evaluate_candidate(candidate: u32, pixel: vec2<f32>) -> StrokeEval {
  let stroke = select(candidate, candidate / config.sample_count, config.representation > 0.5);
  if (param_at(stroke, 14u) > 2.5) {
    return evaluate_flow_underpaint_splat(stroke, pixel);
  }
  if (config.representation > 0.5) {
    return evaluate_chain_micro_splat(candidate, pixel);
  }
  return evaluate_stroke(candidate, pixel);
}
`;

  const TILE_SAMPLING_WGSL = String.raw`
@group(0) @binding(7) var<storage, read> tile_sampling_masks: array<u32>;

fn tile_is_active(tile: u32) -> bool {
  if (config.representation <= 0.5 || config.tile_sample_stride <= 1u) { return true; }
  let phase = config.iteration % config.tile_sample_stride;
  if (config.residual_priority_sampling > 0.5
      && (config.iteration / config.tile_sample_stride) % 2u == 0u) {
    return (tile_sampling_masks[tile] & (1u << phase)) != 0u;
  }
  let hash = tile * 1664525u + 1013904223u;
  return hash % config.tile_sample_stride == phase;
}
`;

  const FORWARD_WGSL = String.raw`
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> tile_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> rendered: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> log_transmittance: array<f32>;
${COMMON_WGSL}
${TILE_SAMPLING_WGSL}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= config.width || id.y >= config.height) { return; }
  let pixel_index = id.y * config.width + id.x;
  let tile_x = id.x / ${TILE_SIZE}u;
  let tile_y = id.y / ${TILE_SIZE}u;
  let tile = tile_y * config.tile_cols + tile_x;
  if (!tile_is_active(tile)) { return; }
  let start = tile_offsets[tile];
  let end = tile_offsets[tile + 1u];
  let pixel = vec2<f32>(f32(id.x) + 0.5, f32(id.y) + 0.5);
  var color = vec3<f32>(0.0);
  var transmittance = 1.0;
  var log_t = 0.0;
  for (var cursor = start; cursor < end; cursor += 1u) {
    let candidate = tile_indices[cursor];
    let evaluated = evaluate_candidate(candidate, pixel);
    color += transmittance * evaluated.alpha * evaluated.pigment;
    transmittance *= 1.0 - evaluated.alpha;
    log_t += log(max(1e-4, 1.0 - evaluated.alpha));
  }
  let canvas = vec3<f32>(config.canvas_linear_r, config.canvas_linear_g, config.canvas_linear_b);
  rendered[pixel_index] = vec4<f32>(color + transmittance * canvas, transmittance);
  log_transmittance[pixel_index] = log_t;
}
`;

  const BACKWARD_WGSL = String.raw`
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> tile_indices: array<u32>;
@group(0) @binding(4) var<storage, read> teacher_pixels: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> rendered: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> gradients: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> log_transmittance: array<f32>;
${COMMON_WGSL}
${TILE_SAMPLING_WGSL}

fn atomic_add_f32(index: u32, value: f32) {
  if (value != value || abs(value) > 3.0e38 || abs(value) < 1e-20) { return; }
  var old_bits = atomicLoad(&gradients[index]);
  loop {
    let old_value = bitcast<f32>(old_bits);
    let new_bits = bitcast<u32>(old_value + value);
    let result = atomicCompareExchangeWeak(&gradients[index], old_bits, new_bits);
    if (result.exchanged) { break; }
    old_bits = result.old_value;
  }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= config.width || id.y >= config.height) { return; }
  let pixel_index = id.y * config.width + id.x;
  let tile_x = id.x / ${TILE_SIZE}u;
  let tile_y = id.y / ${TILE_SIZE}u;
  let tile = tile_y * config.tile_cols + tile_x;
  if (!tile_is_active(tile)) { return; }
  let start = tile_offsets[tile];
  let end = tile_offsets[tile + 1u];
  let pixel = vec2<f32>(f32(id.x) + 0.5, f32(id.y) + 0.5);
  let result = rendered[pixel_index];
  let teacher = teacher_pixels[pixel_index];
  let normalizer = f32(max(1u, config.tile_sample_stride))
    / max(1.0, f32(config.width * config.height) * 3.0);
  let difference = result.rgb - teacher.rgb;
  let d_color = vec3<f32>(
    select(-normalizer, normalizer, difference.x >= 0.0),
    select(-normalizer, normalizer, difference.y >= 0.0),
    select(-normalizer, normalizer, difference.z >= 0.0)
  );
  let canvas = vec3<f32>(config.canvas_linear_r, config.canvas_linear_g, config.canvas_linear_b);
  var d_transmittance_after = dot(d_color, canvas);
  // The final linear transmittance can underflow to zero with opaque dabs.
  // Recover each prefix in log space so visible front marks retain gradients.
  // The forward RGB/alpha and export surface remain unchanged.
  var log_t_after = log_transmittance[pixel_index];
  var cursor = end;
  loop {
    if (cursor <= start) { break; }
    cursor -= 1u;
    let candidate = tile_indices[cursor];
    let stroke = select(candidate, candidate / config.sample_count, config.representation > 0.5);
    let evaluated = evaluate_candidate(candidate, pixel);
    let one_minus_alpha = max(1e-4, 1.0 - evaluated.alpha);
    log_t_after -= log(one_minus_alpha);
    let transmittance_before = exp(min(0.0, log_t_after));
    let d_alpha = transmittance_before * (dot(d_color, evaluated.pigment) - d_transmittance_after);
    let base = stroke * config.param_stride;
    let d_pigment = d_color * (transmittance_before * evaluated.alpha * evaluated.pigment_scale);
    let opacity_logit = param_at(stroke, 9u);
    let opacity_derivative = evaluated.opacity * (1.0 - evaluated.opacity);
    let flow_underpaint = param_at(stroke, 14u) > 2.5;
    if (flow_underpaint) {
      // Backcoat geometry, opacity, and source-cell pigment are coverage
      // invariants. Detail strokes remain the only optimized paint marks.
    } else if (config.representation > 0.5) {
      atomic_add_f32(base + 10u, d_pigment.x);
      atomic_add_f32(base + 11u, d_pigment.y);
      atomic_add_f32(base + 12u, d_pigment.z);
      let sample = candidate % config.sample_count;
      let t = chain_sample_t(stroke, sample);
      let s = 1.0 - t;
      let bernstein = vec4<f32>(s*s*s, 3.0*s*s*t, 3.0*s*t*t, t*t*t);
      let width = max(0.55, param_at(stroke, 8u));
      let spacing = chain_spacing(stroke);
      let sigma_long = max(0.55, max(spacing * 0.72, width * 0.85)
        * chain_length_factor(stroke, sample));
      let width_factor = chain_width_factor(stroke, sample);
      let sigma_short = max(0.38, width * width_factor);
      let random = param_at(stroke, 13u);
      let raw_tangent = cubic_tangent(stroke, t) + vec2<f32>(1e-7, 0.0);
      let tangent = normalize(raw_tangent);
      let normal = vec2<f32>(-tangent.y, tangent.x);
      let normal_offset_factor = chain_normal_offset_factor(stroke, sample);
      let point = cubic_point(stroke, t)
        + normal * normal_offset_factor * width;
      let delta = pixel - point;
      let along_distance = dot(delta, tangent);
      let side_distance = dot(delta, normal);
      let kernel_sample = chain_kernel_sample(
        stroke,
        sample,
        along_distance,
        side_distance,
        sigma_long,
        sigma_short,
        width,
        spacing
      );
      let micro_modulation = chain_micro_modulation(stroke, sample, t, random);
      let raw_micro_alpha = evaluated.opacity * kernel_sample.kernel * micro_modulation;
      if (raw_micro_alpha < 0.99) {
        let d_kernel = d_alpha * evaluated.opacity * micro_modulation;
        let d_point = -d_kernel * (
          tangent * kernel_sample.delta_gradient.x
          + normal * kernel_sample.delta_gradient.y
        );
        let d_tangent = d_kernel * chain_tangent_pullback(normal, length(raw_tangent),
          along_distance, side_distance, normal_offset_factor * width, kernel_sample.delta_gradient);
        let bernstein_derivative = vec4<f32>(-3.0*s*s, 3.0*s*s-6.0*s*t, 6.0*s*t-3.0*t*t, 3.0*t*t);
        for (var control = 0u; control < 4u; control += 1u) {
          let gradient = d_point * bernstein[control] + d_tangent * bernstein_derivative[control];
          atomic_add_f32(base + control * 2u, gradient.x);
          atomic_add_f32(base + control * 2u + 1u, gradient.y);
        }
        atomic_add_f32(
          base + 9u,
          d_alpha * kernel_sample.kernel * micro_modulation * opacity_derivative
        );
        atomic_add_f32(
          base + 8u,
          select(0.0, d_kernel * kernel_sample.width_gradient
            + dot(d_point, normal) * normal_offset_factor, param_at(stroke, 8u) > 0.55)
        );
      }
    } else {
      atomic_add_f32(base + 10u, d_pigment.x);
      atomic_add_f32(base + 11u, d_pigment.y);
      atomic_add_f32(base + 12u, d_pigment.z);
      atomic_add_f32(base + 9u, d_alpha * evaluated.coverage * evaluated.modulation * opacity_derivative);
      let d_coverage = d_alpha * evaluated.opacity * evaluated.modulation;
      let d_z = d_coverage * evaluated.coverage * (1.0 - evaluated.coverage);
      let d_soft_d2 = -d_z / max(1e-5, evaluated.edge2);
      let width = max(0.55, param_at(stroke, 8u));
      let tau = max(0.36, width * width * 0.22);
      var minimum_d2 = 1e30;
      for (var sample = 0u; sample < config.sample_count; sample += 1u) {
        let t = f32(sample) / max(1.0, f32(config.sample_count - 1u));
        let delta = cubic_point(stroke, t) - pixel;
        minimum_d2 = min(minimum_d2, dot(delta, delta));
      }
      var weight_sum = 0.0;
      for (var sample = 0u; sample < config.sample_count; sample += 1u) {
        let t = f32(sample) / max(1.0, f32(config.sample_count - 1u));
        let delta = cubic_point(stroke, t) - pixel;
        weight_sum += exp(-(dot(delta, delta) - minimum_d2) / tau);
      }
      for (var sample = 0u; sample < config.sample_count; sample += 1u) {
        let t = f32(sample) / max(1.0, f32(config.sample_count - 1u));
        let s = 1.0 - t;
        let bernstein = vec4<f32>(s*s*s, 3.0*s*s*t, 3.0*s*t*t, t*t*t);
        let point = cubic_point(stroke, t);
        let delta = point - pixel;
        let weight = exp(-(dot(delta, delta) - minimum_d2) / tau) / max(1e-8, weight_sum);
        let d_point = d_soft_d2 * 2.0 * weight * delta;
        for (var control = 0u; control < 4u; control += 1u) {
          atomic_add_f32(base + control * 2u, d_point.x * bernstein[control]);
          atomic_add_f32(base + control * 2u + 1u, d_point.y * bernstein[control]);
        }
      }
      let taper = evaluated.width_effective / max(0.55, width);
      let d_width = d_z * 2.0 * evaluated.width_effective * taper / max(1e-5, evaluated.edge2);
      atomic_add_f32(base + 8u, d_width);
    }

    let d_transmittance_before = evaluated.alpha * dot(d_color, evaluated.pigment)
      + (1.0 - evaluated.alpha) * d_transmittance_after;
    d_transmittance_after = d_transmittance_before;
  }
}
`;

  // Curve Splat Chain keeps the same forward pass and per-micro-splat
  // derivatives as BACKWARD_WGSL. The only difference is accumulation: one
  // 8x8 workgroup owns a 16x16 tile, each lane processes a 2x2 pixel quad,
  // and gradients are reduced once per parent curve before global atomics.
  // This removes the dominant many-pixels-to-one-curve CAS contention without
  // changing the standard-alpha order, sampled tiles, or Adam update.
  const CHAIN_QUAD_BACKWARD_WGSL = String.raw`
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> tile_indices: array<u32>;
@group(0) @binding(4) var<storage, read> teacher_pixels: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> rendered: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> gradients: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> log_transmittance: array<f32>;
${COMMON_WGSL}
${TILE_SAMPLING_WGSL}

var<workgroup> reduce_0_3: array<vec4<f32>, 64>;
var<workgroup> reduce_4_7: array<vec4<f32>, 64>;
var<workgroup> reduce_8_11: array<vec4<f32>, 64>;
var<workgroup> reduce_12_15: array<vec4<f32>, 64>;

fn atomic_add_f32(index: u32, value: f32) {
  if (value != value || abs(value) > 3.0e38 || abs(value) < 1e-20) { return; }
  var old_bits = atomicLoad(&gradients[index]);
  loop {
    let old_value = bitcast<f32>(old_bits);
    let new_bits = bitcast<u32>(old_value + value);
    let result = atomicCompareExchangeWeak(&gradients[index], old_bits, new_bits);
    if (result.exchanged) { break; }
    old_bits = result.old_value;
  }
}

fn reduce_curve_gradient(
  local_index: u32,
  stroke: u32,
  gradient_0_3: vec4<f32>,
  gradient_4_7: vec4<f32>,
  gradient_8_11: vec4<f32>,
  gradient_12_15: vec4<f32>
) {
  reduce_0_3[local_index] = gradient_0_3;
  reduce_4_7[local_index] = gradient_4_7;
  reduce_8_11[local_index] = gradient_8_11;
  reduce_12_15[local_index] = gradient_12_15;
  workgroupBarrier();
  var stride = 32u;
  loop {
    if (local_index < stride) {
      reduce_0_3[local_index] += reduce_0_3[local_index + stride];
      reduce_4_7[local_index] += reduce_4_7[local_index + stride];
      reduce_8_11[local_index] += reduce_8_11[local_index + stride];
      reduce_12_15[local_index] += reduce_12_15[local_index + stride];
    }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride /= 2u;
  }
  if (local_index == 0u) {
    let base = stroke * config.param_stride;
    let g0 = reduce_0_3[0];
    let g1 = reduce_4_7[0];
    let g2 = reduce_8_11[0];
    let g3 = reduce_12_15[0];
    atomic_add_f32(base + 0u, g0.x);
    atomic_add_f32(base + 1u, g0.y);
    atomic_add_f32(base + 2u, g0.z);
    atomic_add_f32(base + 3u, g0.w);
    atomic_add_f32(base + 4u, g1.x);
    atomic_add_f32(base + 5u, g1.y);
    atomic_add_f32(base + 6u, g1.z);
    atomic_add_f32(base + 7u, g1.w);
    atomic_add_f32(base + 8u, g2.x);
    atomic_add_f32(base + 9u, g2.y);
    atomic_add_f32(base + 10u, g2.z);
    atomic_add_f32(base + 11u, g2.w);
    atomic_add_f32(base + 12u, g3.x);
  }
  workgroupBarrier();
}

@compute @workgroup_size(8, 8)
fn main(
  @builtin(local_invocation_index) local_index: u32,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let tile = workgroup_id.y * config.tile_cols + workgroup_id.x;
  if (tile >= config.tile_cols * config.tile_rows || !tile_is_active(tile)) { return; }
  let start = tile_offsets[tile];
  let end = tile_offsets[tile + 1u];
  let local_x = local_index % 8u;
  let local_y = local_index / 8u;
  let base_x = workgroup_id.x * ${TILE_SIZE}u + local_x * 2u;
  let base_y = workgroup_id.y * ${TILE_SIZE}u + local_y * 2u;
  let offsets = array<vec2<u32>, 4>(
    vec2<u32>(0u, 0u),
    vec2<u32>(1u, 0u),
    vec2<u32>(0u, 1u),
    vec2<u32>(1u, 1u)
  );
  let normalizer = f32(max(1u, config.tile_sample_stride))
    / max(1.0, f32(config.width * config.height) * 3.0);
  let canvas = vec3<f32>(config.canvas_linear_r, config.canvas_linear_g, config.canvas_linear_b);
  var valid_pixels: array<u32, 4>;
  var pixels: array<vec2<f32>, 4>;
  var d_colors: array<vec3<f32>, 4>;
  var d_transmittances: array<f32, 4>;
  var log_t_after: array<f32, 4>;
  for (var pixel_slot = 0u; pixel_slot < 4u; pixel_slot += 1u) {
    let x = base_x + offsets[pixel_slot].x;
    let y = base_y + offsets[pixel_slot].y;
    let valid = x < config.width && y < config.height;
    valid_pixels[pixel_slot] = select(0u, 1u, valid);
    pixels[pixel_slot] = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);
    d_colors[pixel_slot] = vec3<f32>(0.0);
    d_transmittances[pixel_slot] = 0.0;
    log_t_after[pixel_slot] = 0.0;
    if (valid) {
      let pixel_index = y * config.width + x;
      let result = rendered[pixel_index];
      let teacher = teacher_pixels[pixel_index];
      let difference = result.rgb - teacher.rgb;
      d_colors[pixel_slot] = vec3<f32>(
        select(-normalizer, normalizer, difference.x >= 0.0),
        select(-normalizer, normalizer, difference.y >= 0.0),
        select(-normalizer, normalizer, difference.z >= 0.0)
      );
      d_transmittances[pixel_slot] = dot(d_colors[pixel_slot], canvas);
      log_t_after[pixel_slot] = log_transmittance[pixel_index];
    }
  }

  var current_stroke = 0xffffffffu;
  var gradient_0_3 = vec4<f32>(0.0);
  var gradient_4_7 = vec4<f32>(0.0);
  var gradient_8_11 = vec4<f32>(0.0);
  var gradient_12_15 = vec4<f32>(0.0);
  var cursor = end;
  loop {
    if (cursor <= start) { break; }
    cursor -= 1u;
    let candidate = tile_indices[cursor];
    let stroke = candidate / config.sample_count;
    if (current_stroke != 0xffffffffu && stroke != current_stroke) {
      reduce_curve_gradient(
        local_index,
        current_stroke,
        gradient_0_3,
        gradient_4_7,
        gradient_8_11,
        gradient_12_15
      );
      gradient_0_3 = vec4<f32>(0.0);
      gradient_4_7 = vec4<f32>(0.0);
      gradient_8_11 = vec4<f32>(0.0);
      gradient_12_15 = vec4<f32>(0.0);
    }
    current_stroke = stroke;
    let underpaint = param_at(stroke, 14u) > 2.5;
    let sample = candidate % config.sample_count;
    let t = chain_sample_t(stroke, sample);
    let s = 1.0 - t;
    let bernstein = vec4<f32>(s*s*s, 3.0*s*s*t, 3.0*s*t*t, t*t*t);
    let width = max(0.55, param_at(stroke, 8u));
    let spacing = chain_spacing(stroke);
    let sigma_long = max(0.55, max(spacing * 0.72, width * 0.85)
      * chain_length_factor(stroke, sample));
    let width_factor = chain_width_factor(stroke, sample);
    let sigma_short = max(0.38, width * width_factor);
    let random = param_at(stroke, 13u);
    let raw_tangent = cubic_tangent(stroke, t) + vec2<f32>(1e-7, 0.0);
    let tangent = normalize(raw_tangent);
    let normal = vec2<f32>(-tangent.y, tangent.x);
    let normal_offset_factor = chain_normal_offset_factor(stroke, sample);
    let point = cubic_point(stroke, t)
      + normal * normal_offset_factor * width;
    for (var pixel_slot = 0u; pixel_slot < 4u; pixel_slot += 1u) {
      if (valid_pixels[pixel_slot] == 0u) { continue; }
      let pixel = pixels[pixel_slot];
      let evaluated = evaluate_chain_micro_splat(candidate, pixel);
      let one_minus_alpha = max(1e-4, 1.0 - evaluated.alpha);
      log_t_after[pixel_slot] -= log(one_minus_alpha);
      let transmittance_before = exp(min(0.0, log_t_after[pixel_slot]));
      let d_alpha = transmittance_before * (
        dot(d_colors[pixel_slot], evaluated.pigment) - d_transmittances[pixel_slot]
      );
      let d_pigment = d_colors[pixel_slot] * (
        transmittance_before * evaluated.alpha * evaluated.pigment_scale
      );
      if (underpaint) {
        // Complete fixed source-cell backcoat; no parameter gradient.
      } else {
        gradient_8_11.z += d_pigment.x;
        gradient_8_11.w += d_pigment.y;
        gradient_12_15.x += d_pigment.z;
        let delta = pixel - point;
        let along_distance = dot(delta, tangent);
        let side_distance = dot(delta, normal);
        let kernel_sample = chain_kernel_sample(
          stroke,
          sample,
          along_distance,
          side_distance,
          sigma_long,
          sigma_short,
          width,
          spacing
        );
        let micro_modulation = chain_micro_modulation(stroke, sample, t, random);
        let raw_micro_alpha = evaluated.opacity * kernel_sample.kernel * micro_modulation;
        if (raw_micro_alpha < 0.99) {
          let d_kernel = d_alpha * evaluated.opacity * micro_modulation;
          let d_point = -d_kernel * (
            tangent * kernel_sample.delta_gradient.x
            + normal * kernel_sample.delta_gradient.y
          );
          let d_tangent = d_kernel * chain_tangent_pullback(normal, length(raw_tangent),
            along_distance, side_distance, normal_offset_factor * width, kernel_sample.delta_gradient);
          let db = vec4<f32>(-3.0*s*s, 3.0*s*s-6.0*s*t, 6.0*s*t-3.0*t*t, 3.0*t*t);
          gradient_0_3 += vec4<f32>(
            d_point.x * bernstein.x,
            d_point.y * bernstein.x,
            d_point.x * bernstein.y,
            d_point.y * bernstein.y
          ) + vec4<f32>(d_tangent * db.x, d_tangent * db.y);
          gradient_4_7 += vec4<f32>(
            d_point.x * bernstein.z,
            d_point.y * bernstein.z,
            d_point.x * bernstein.w,
            d_point.y * bernstein.w
          ) + vec4<f32>(d_tangent * db.z, d_tangent * db.w);
          let opacity_derivative = evaluated.opacity * (1.0 - evaluated.opacity);
          gradient_8_11.y += d_alpha * kernel_sample.kernel * micro_modulation
            * opacity_derivative;
          gradient_8_11.x += select(0.0, d_kernel * kernel_sample.width_gradient
            + dot(d_point, normal) * normal_offset_factor, param_at(stroke, 8u) > 0.55);
        }
      }
      d_transmittances[pixel_slot] = evaluated.alpha * dot(d_colors[pixel_slot], evaluated.pigment)
        + (1.0 - evaluated.alpha) * d_transmittances[pixel_slot];
    }
  }
  if (current_stroke != 0xffffffffu) {
    reduce_curve_gradient(
      local_index,
      current_stroke,
      gradient_0_3,
      gradient_4_7,
      gradient_8_11,
      gradient_12_15
    );
  }
}
`;

  const UPDATE_WGSL = String.raw`
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read_write> params: array<f32>;
@group(0) @binding(2) var<storage, read> anchors: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradients: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> first_moment: array<f32>;
@group(0) @binding(5) var<storage, read_write> second_moment: array<f32>;
@group(0) @binding(6) var<storage, read_write> adam_steps: array<u32>;

struct Config {
  width: u32,
  height: u32,
  stroke_count: u32,
  tile_cols: u32,
  tile_rows: u32,
  param_stride: u32,
  sample_count: u32,
  iteration: u32,
  color_anchor: f32,
  geometry_anchor: f32,
  width_anchor: f32,
  opacity_anchor: f32,
  lr_position: f32,
  lr_width: f32,
  lr_opacity: f32,
  lr_color: f32,
  beta1: f32,
  beta2: f32,
  epsilon: f32,
  loss_scale: f32,
  max_width: f32,
  max_position_delta: f32,
  representation: f32,
  tile_sample_stride: u32,
  max_curve_arc: f32,
  canvas_linear_r: f32,
  canvas_linear_g: f32,
  canvas_linear_b: f32,
  stroke_motion_coherence: f32,
  bristle_bundle: f32,
  residual_priority_sampling: f32,
  fixed_opacity_logit: f32,
  front_width_learning_scale: f32,
  texture_guided_dabs: f32,
  splat_size_variation: f32,
  width_minimum_factor: f32,
  width_maximum_factor: f32,
}

fn update_cubic_point(base: u32, t: f32) -> vec2<f32> {
  let s = 1.0 - t;
  let p0 = vec2<f32>(params[base], params[base + 1u]);
  let p1 = vec2<f32>(params[base + 2u], params[base + 3u]);
  let p2 = vec2<f32>(params[base + 4u], params[base + 5u]);
  let p3 = vec2<f32>(params[base + 6u], params[base + 7u]);
  return s*s*s*p0 + 3.0*s*s*t*p1 + 3.0*s*t*t*p2 + t*t*t*p3;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let stroke = id.x;
  if (stroke >= config.stroke_count) { return; }
  let base = stroke * config.param_stride;
  let prior_step = min(0xfffffffeu, adam_steps[stroke]);
  let step = f32(prior_step + 1u);
  let bias1 = max(1e-8, 1.0 - pow(config.beta1, step));
  let bias2 = max(1e-8, 1.0 - pow(config.beta2, step));
  let coverage_backcoat = params[base + 14u] > 2.5;
  for (var component = 0u; component <= 12u; component += 1u) {
    let index = base + component;
    if (coverage_backcoat && component <= 12u) {
      first_moment[index] = 0.0;
      second_moment[index] = 0.0;
      params[index] = anchors[index];
      continue;
    }
    var gradient = bitcast<f32>(atomicLoad(&gradients[index])) * config.loss_scale;
    let value = params[index];
    let anchor = anchors[index];
    var learning_rate = config.lr_position;
    if (component <= 7u) {
      if (config.representation > 0.5 && params[base + 14u] < 2.5) {
        let axis = component % 2u;
        let translation_gradient = 0.25 * (
          bitcast<f32>(atomicLoad(&gradients[base + axis]))
          + bitcast<f32>(atomicLoad(&gradients[base + axis + 2u]))
          + bitcast<f32>(atomicLoad(&gradients[base + axis + 4u]))
          + bitcast<f32>(atomicLoad(&gradients[base + axis + 6u]))
        ) * config.loss_scale;
        // Decompose the four Bezier-control gradients into coherent stroke
        // translation plus residual deformation. Keeping some residual lets
        // the curve bend while preventing four unrelated point updates from
        // dissolving the visible brush stroke.
        gradient = mix(
          gradient,
          translation_gradient,
          clamp(config.stroke_motion_coherence, 0.0, 1.0)
        );
      }
      gradient += config.geometry_anchor * (value - anchor);
    } else if (component == 8u) {
      let fine_front = params[base + 14u] > 1.5 && params[base + 14u] < 2.5;
      let width_learning_scale = select(
        1.0,
        max(1.0, config.front_width_learning_scale),
        fine_front
      );
      gradient += config.width_anchor / width_learning_scale * (value - anchor);
      learning_rate = config.lr_width * width_learning_scale;
    } else if (component == 9u) {
      if (config.fixed_opacity_logit > 0.0) {
        first_moment[index] = 0.0;
        second_moment[index] = 0.0;
        params[index] = config.fixed_opacity_logit;
        continue;
      }
      gradient += config.opacity_anchor * (value - anchor);
      learning_rate = config.lr_opacity;
    } else if (component <= 12u) {
      gradient += config.color_anchor * (value - anchor);
      learning_rate = config.lr_color;
    }
    let moment1 = config.beta1 * first_moment[index] + (1.0 - config.beta1) * gradient;
    let moment2 = config.beta2 * second_moment[index] + (1.0 - config.beta2) * gradient * gradient;
    first_moment[index] = moment1;
    second_moment[index] = moment2;
    var next = value - learning_rate * (moment1 / bias1) / (sqrt(moment2 / bias2) + config.epsilon);
    if (component <= 7u) {
      next = clamp(
        next,
        anchor - config.max_position_delta,
        anchor + config.max_position_delta
      );
    } else if (component == 8u) {
      let underpaint = params[base + 14u] > 2.5;
      var maximum = config.max_width * select(1.0, 4.0, underpaint);
      if (config.width_maximum_factor > 0.0 && !underpaint) {
        maximum = min(maximum, anchor * config.width_maximum_factor);
      }
      let minimum = min(maximum, max(0.55, anchor * config.width_minimum_factor));
      next = clamp(next, minimum, maximum);
    } else if (component == 9u) {
      next = clamp(next, -5.5, 5.5);
    } else if (component <= 12u) {
      next = clamp(next, 0.0, 1.0);
    }
    params[index] = next;
  }
  if (coverage_backcoat) {
    adam_steps[stroke] = 0u;
  } else {
    adam_steps[stroke] = min(0xfffffffeu, prior_step + 1u);
  }
  if (config.max_curve_arc > 0.0 && params[base + 14u] < 2.5) {
    var previous = update_cubic_point(base, 0.0);
    var curve_arc = 0.0;
    for (var sample = 1u; sample <= 24u; sample += 1u) {
      let point = update_cubic_point(base, f32(sample) / 24.0);
      curve_arc += length(point - previous);
      previous = point;
    }
    if (curve_arc > config.max_curve_arc) {
      let center = update_cubic_point(base, 0.5);
      let scale = config.max_curve_arc / max(1e-6, curve_arc);
      for (var control = 0u; control < 4u; control += 1u) {
        let index = base + control * 2u;
        let point = vec2<f32>(params[index], params[index + 1u]);
        let capped = center + (point - center) * scale;
        params[index] = capped.x;
        params[index + 1u] = capped.y;
      }
    }
  }
  // Arc-length correction also moves control points, so enforce the complete-run
  // movement contract last. A limit of zero must keep generated geometry exact.
  for (var component = 0u; component <= 7u; component += 1u) {
    let index = base + component;
    params[index] = clamp(
      params[index],
      anchors[index] - config.max_position_delta,
      anchors[index] + config.max_position_delta
    );
  }
}
`;

  // Training-time phases, not the three paint-depth layers. Use the complete
  // run's real update index so growth-stage Adam restarts do not restart P1.
  function widthTrainingSettings(iteration, options = {}) {
    const total = Math.max(1, Math.floor(Number(options.globalIterations ?? options.iterations) || 1));
    const step = Math.max(0, (Number(options.globalIterationOffset) || 0) + (Number(iteration) || 0));
    const progress = total === 1 ? 1 : clamp01(step / (total - 1));
    const enabled = options.widthTrainingPhases === true;
    const phase = !enabled || progress >= 2 / 3 ? 3 : progress < 1 / 3 ? 1 : 2;
    const freedom = enabled ? 1 - smoothstepRange(1 / 3, 2 / 3, progress) : 0;
    return {
      phase,
      progress,
      widthAnchor: (options.widthAnchor ?? 0.0008) * (1 - 0.75 * freedom),
      widthLearningRate: (options.widthLearningRate ?? 0.010) * (1 + 0.5 * freedom),
      minimumWidthFactor: phase === 1 ? 1.5 : phase === 2 ? 1 : 0,
      maximumWidthFactor: phase === 1 ? 3 : phase === 2 ? 2 : 0,
      maximumWidthMultiplier: phase === 1 ? 3 : phase === 2 ? 2 : 1,
    };
  }

  function widthTrainingBounds(nominalHalfWidth, width, height, iteration, options) {
    const settings = widthTrainingSettings(iteration, options);
    const cap = Math.max(2, Math.min(width, height) * 0.09) * settings.maximumWidthMultiplier;
    const maximum = settings.maximumWidthFactor > 0
      ? Math.min(cap, nominalHalfWidth * settings.maximumWidthFactor) : cap;
    return {
      minimum: Math.min(maximum, Math.max(0.55, nominalHalfWidth * settings.minimumWidthFactor)),
      maximum,
    };
  }

  function paramsWithoutPhaseLift(params, lifts) {
    const next = params.slice();
    for (let index = 0; index < lifts.length; index += 1) {
      next[index * PARAM_STRIDE + 8] = Math.max(0.55, next[index * PARAM_STRIDE + 8] - lifts[index]);
    }
    return next;
  }

  function finiteOptimizerRow(values, base) {
    if (!(values instanceof Float32Array) || base < 0 || base + PARAM_STRIDE > values.length) {
      return false;
    }
    for (let component = 0; component < PARAM_STRIDE; component += 1) {
      if (!Number.isFinite(values[base + component])) return false;
    }
    return true;
  }

  // Rebuild GPU state only for rows whose topology identity survived the
  // stage boundary. Backcoat, growth, split, merge, and invalid readback rows
  // stay zeroed so their Adam bias correction begins at their actual birth.
  function remapContinuousOptimizerState(data, continuousState) {
    const rowCount = data.strokeCount;
    const firstMoment = new Float32Array(rowCount * PARAM_STRIDE);
    const secondMoment = new Float32Array(rowCount * PARAM_STRIDE);
    const adamSteps = new Uint32Array(rowCount);
    const detailOffset = Math.max(0, Math.min(
      rowCount,
      Math.round(Number(data.underpaintParentCount) || 0),
    ));
    const detailCount = rowCount - detailOffset;
    const enabled = continuousState !== false;
    const previous = enabled && continuousState && typeof continuousState === "object"
      ? continuousState.previousTrainingState
      : null;
    const survivorRows = enabled && continuousState && typeof continuousState === "object"
      ? continuousState.survivorRows
      : null;
    const previousFirstMoment = previous?.firstMoment;
    const previousSecondMoment = previous?.secondMoment;
    const previousAdamSteps = previous?.adamSteps;
    const previousDetailOffset = Math.max(0, Math.round(
      Number(continuousState?.previousDetailOffset) || 0,
    ));
    const sourceAvailable = survivorRows != null
      && previousFirstMoment instanceof Float32Array
      && previousSecondMoment instanceof Float32Array
      && previousAdamSteps instanceof Uint32Array;
    let retainedDetailRows = 0;
    for (let detailRow = 0; detailRow < detailCount; detailRow += 1) {
      const previousDetailRow = Number(survivorRows?.[detailRow]);
      if (!sourceAvailable || !Number.isInteger(previousDetailRow) || previousDetailRow < 0) continue;
      const sourceStroke = previousDetailOffset + previousDetailRow;
      const sourceBase = sourceStroke * PARAM_STRIDE;
      const targetBase = (detailOffset + detailRow) * PARAM_STRIDE;
      if (!finiteOptimizerRow(previousFirstMoment, sourceBase)
        || !finiteOptimizerRow(previousSecondMoment, sourceBase)
        || sourceStroke < 0
        || sourceStroke >= previousAdamSteps.length) continue;
      firstMoment.set(previousFirstMoment.subarray(sourceBase, sourceBase + PARAM_STRIDE), targetBase);
      secondMoment.set(previousSecondMoment.subarray(sourceBase, sourceBase + PARAM_STRIDE), targetBase);
      adamSteps[detailOffset + detailRow] = previousAdamSteps[sourceStroke];
      retainedDetailRows += 1;
    }
    return {
      firstMoment,
      secondMoment,
      adamSteps,
      diagnostics: {
        mode: enabled ? "survivor-row-remap" : "disabled",
        scope: "this-stage-detail-row-initialization",
        source_available: sourceAvailable,
        retained_detail_rows: retainedDetailRows,
        reset_detail_rows: detailCount - retainedDetailRows,
        frozen_backcoat_rows: detailOffset,
      },
    };
  }

  function configBytes(data, iteration, options, forceFullFrame = false) {
    const buffer = new ArrayBuffer(CONFIG_BYTES);
    const view = new DataView(buffer);
    const u32 = (offset, value) => view.setUint32(offset, value >>> 0, true);
    const f32 = (offset, value) => view.setFloat32(offset, Number(value), true);
    u32(0, data.width);
    u32(4, data.height);
    u32(8, data.strokeCount);
    u32(12, data.tileCols);
    u32(16, data.tileRows);
    u32(20, PARAM_STRIDE);
    u32(24, data.sampleCount);
    u32(28, iteration);
    f32(32, options.colorAnchor ?? 0.0035);
    f32(36, options.geometryAnchor ?? 0.00035);
    const widthTraining = widthTrainingSettings(iteration, options);
    f32(40, widthTraining.widthAnchor);
    f32(44, options.opacityAnchor ?? 0.0008);
    // Both moments continue observing the same loss. Only parameter writes
    // alternate, so per-stroke Adam age/bias correction stays well defined.
    const colorStep = ((Number(options.globalIterationOffset) || 0) + iteration) % 5 === 4;
    const shapeRate = options.alternateShapeColor === true && colorStep ? 0 : 1;
    const colorRate = options.alternateShapeColor === true && !colorStep ? 0 : 1;
    f32(48, (options.positionLearningRate ?? 0.015) * shapeRate);
    f32(52, widthTraining.widthLearningRate * shapeRate);
    f32(56, options.opacityLearningRate ?? 0.006);
    f32(60, (options.colorLearningRate ?? 0.020) * colorRate);
    f32(64, 0.9);
    f32(68, 0.999);
    f32(72, 1e-8);
    f32(76, 1);
    f32(80, Math.max(2, Math.min(data.width, data.height) * 0.09) * widthTraining.maximumWidthMultiplier);
    f32(84, options.maxPositionDelta ?? 6);
    f32(88, options.representation === "curve-splat-chain" ? 1 : 0);
    u32(
      92,
      forceFullFrame || options.representation !== "curve-splat-chain"
        ? 1
        : data.tileSampleStride,
    );
    f32(96, Math.max(0, Number(options.maximumCurveArcPx) || 0));
    f32(100, data.canvasLinear[0]);
    f32(104, data.canvasLinear[1]);
    f32(108, data.canvasLinear[2]);
    f32(112, Math.max(0, Math.min(1, Number(options.strokeMotionCoherence) || 0)));
    f32(116, data.strokeTextureMode);
    f32(120, data.residualPriorityTileSampling ? 1 : 0);
    const fixedOpacity = Number(options.fixedOpacity);
    f32(124, Number.isFinite(fixedOpacity) && fixedOpacity > 0
      ? logit(Math.max(0.05, Math.min(0.995, fixedOpacity)))
      : 0);
    f32(128, Math.max(1, Math.min(8, Number(options.frontWidthLearningScale) || 1)));
    f32(132, data.textureGuidedDabs ? 1 : 0);
    f32(136, Math.max(0, Math.min(1, Number(options.splatSizeVariation) || 0)));
    f32(140, widthTraining.minimumWidthFactor);
    f32(144, widthTraining.maximumWidthFactor);
    return new Uint8Array(buffer);
  }

  function createBuffer(device, label, dataOrSize, usage) {
    const byteLength = typeof dataOrSize === "number" ? dataOrSize : dataOrSize.byteLength;
    const buffer = device.createBuffer({
      label,
      size: Math.max(4, Math.ceil(byteLength / 4) * 4),
      usage,
      mappedAtCreation: typeof dataOrSize !== "number",
    });
    if (typeof dataOrSize !== "number") {
      const target = new Uint8Array(buffer.getMappedRange());
      target.set(new Uint8Array(dataOrSize.buffer, dataOrSize.byteOffset, dataOrSize.byteLength));
      buffer.unmap();
    }
    return buffer;
  }

  async function createComputePipeline(device, label, code) {
    const module = device.createShaderModule({ label: `${label} shader`, code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) {
      throw new Error(`${label} WGSL: ${errors.map((message) => message.message).join(" | ")}`);
    }
    return device.createComputePipeline({
      label,
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  function makeBindGroups(device, pipelines, buffers) {
    return {
      forward: device.createBindGroup({
        layout: pipelines.forward.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: buffers.config } },
          { binding: 1, resource: { buffer: buffers.params } },
          { binding: 2, resource: { buffer: buffers.tileOffsets } },
          { binding: 3, resource: { buffer: buffers.tileIndices } },
          { binding: 4, resource: { buffer: buffers.rendered } },
          { binding: 7, resource: { buffer: buffers.tileSamplingMasks } },
          { binding: 8, resource: { buffer: buffers.logTransmittance } },
        ],
      }),
      backward: device.createBindGroup({
        layout: pipelines.backward.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: buffers.config } },
          { binding: 1, resource: { buffer: buffers.params } },
          { binding: 2, resource: { buffer: buffers.tileOffsets } },
          { binding: 3, resource: { buffer: buffers.tileIndices } },
          { binding: 4, resource: { buffer: buffers.target } },
          { binding: 5, resource: { buffer: buffers.rendered } },
          { binding: 6, resource: { buffer: buffers.gradients } },
          { binding: 7, resource: { buffer: buffers.tileSamplingMasks } },
          { binding: 8, resource: { buffer: buffers.logTransmittance } },
        ],
      }),
      update: device.createBindGroup({
        layout: pipelines.update.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: buffers.config } },
          { binding: 1, resource: { buffer: buffers.params } },
          { binding: 2, resource: { buffer: buffers.anchors } },
          { binding: 3, resource: { buffer: buffers.gradients } },
          { binding: 4, resource: { buffer: buffers.firstMoment } },
          { binding: 5, resource: { buffer: buffers.secondMoment } },
          { binding: 6, resource: { buffer: buffers.adamSteps } },
        ],
      }),
    };
  }

  function encodeForward(encoder, pipeline, bindGroup, data) {
    const pass = encoder.beginComputePass({ label: "Flow ribbon forward" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(data.width / 8), Math.ceil(data.height / 8));
    pass.end();
  }

  function encodeBackward(encoder, pipeline, bindGroup, data, chainQuadBackward = false) {
    const pass = encoder.beginComputePass({ label: "Flow ribbon backward" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      chainQuadBackward ? data.tileCols : Math.ceil(data.width / 8),
      chainQuadBackward ? data.tileRows : Math.ceil(data.height / 8),
    );
    pass.end();
  }

  function encodeUpdate(encoder, pipeline, bindGroup, data) {
    const pass = encoder.beginComputePass({ label: "Flow ribbon Adam" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(data.strokeCount / 64));
    pass.end();
  }

  function renderedImageSummary(data, rendered) {
    const rgb = new Float32Array(data.width * data.height * 3);
    let l1 = 0;
    let mse = 0;
    let sourceOpaquePixels = 0;
    let backgroundExposureCount = 0;
    let over005Count = 0;
    let over010Count = 0;
    let transmittanceTotal = 0;
    let maximumTransmittance = 0;
    let zeroTransmittanceCount = 0;
    let creamLeakTotal = 0;
    let canvasLikeSourceCount = 0;
    const luminanceBuckets = {
      dark: { alpha: 0, under: 0, count: 0 },
      mid: { alpha: 0, under: 0, count: 0 },
      light: { alpha: 0, under: 0, count: 0 },
    };
    for (let pixel = 0; pixel < data.width * data.height; pixel += 1) {
      for (let channel = 0; channel < 3; channel += 1) {
        const signal = linearToSignalSrgb(rendered[pixel * 4 + channel]);
        rgb[pixel * 3 + channel] = signal;
        const difference = signal - linearToSignalSrgb(data.target[pixel * 4 + channel]);
        l1 += Math.abs(difference);
        mse += difference * difference;
      }
      if (data.target[pixel * 4 + 3] >= 0.05) {
        const transmittance = clamp01(rendered[pixel * 4 + 3]);
        const compositeAlpha = 1 - transmittance;
        const red = data.target[pixel * 4];
        const green = data.target[pixel * 4 + 1];
        const blue = data.target[pixel * 4 + 2];
        const canvasDistance = Math.sqrt((
          (red - data.canvasLinear[0]) ** 2 +
          (green - data.canvasLinear[1]) ** 2 +
          (blue - data.canvasLinear[2]) ** 2
        ) / 3);
        const luminance = red * 0.2126 + green * 0.7152 + blue * 0.0722;
        const bucket = luminance < 0.25
          ? luminanceBuckets.dark
          : luminance < 0.70
            ? luminanceBuckets.mid
            : luminanceBuckets.light;
        sourceOpaquePixels += 1;
        transmittanceTotal += transmittance;
        maximumTransmittance = Math.max(maximumTransmittance, transmittance);
        zeroTransmittanceCount += transmittance === 0 ? 1 : 0;
        creamLeakTotal += transmittance * canvasDistance;
        canvasLikeSourceCount += canvasDistance < 0.04 ? 1 : 0;
        backgroundExposureCount += transmittance > 0.01 ? 1 : 0;
        over005Count += transmittance > 0.05 ? 1 : 0;
        over010Count += transmittance > 0.10 ? 1 : 0;
        bucket.alpha += compositeAlpha;
        bucket.under += compositeAlpha < 0.99 ? 1 : 0;
        bucket.count += 1;
      }
    }
    const sampleCount = data.width * data.height * 3;
    const coverageDenominator = Math.max(1, sourceOpaquePixels);
    l1 /= sampleCount;
    mse /= sampleCount;
    return {
      image: {
        width: data.width,
        height: data.height,
        rgb,
        // RGB is already composited over the adaptive source canvas. Keep presentation
        // opaque and expose the real composite alpha through coverage_stats.
        alpha: new Float32Array(data.width * data.height).fill(1),
      },
      rgb_l1_signal: l1,
      psnr_signal_db: mse > 0 ? -10 * Math.log10(mse) : Infinity,
      coverage_stats: {
        background_exposure_count: backgroundExposureCount,
        background_exposure_ratio: backgroundExposureCount / coverageDenominator,
        background_exposure_alpha_threshold: 0.99,
        source_opaque_pixel_count: sourceOpaquePixels,
        mean_transmittance: transmittanceTotal / coverageDenominator,
        maximum_transmittance: maximumTransmittance,
        zero_transmittance_count: zeroTransmittanceCount,
        minimum_composite_alpha: 1 - maximumTransmittance,
        transmittance_over_0_01_ratio: backgroundExposureCount / coverageDenominator,
        transmittance_over_0_05_ratio: over005Count / coverageDenominator,
        transmittance_over_0_10_ratio: over010Count / coverageDenominator,
        cream_canvas_leak_linear_mean: creamLeakTotal / coverageDenominator,
        canvas_linear: data.canvasLinear.slice(),
        canvas_signal_srgb: data.canvasLinear.map(linearToSignalSrgb),
        canvas_like_source_ratio: canvasLikeSourceCount / coverageDenominator,
        luminance_buckets: Object.fromEntries(Object.entries(luminanceBuckets).map(([name, bucket]) => [name, {
          count: bucket.count,
          mean_alpha: bucket.alpha / Math.max(1, bucket.count),
          under_0_99_count: bucket.under,
          under_0_99_ratio: bucket.under / Math.max(1, bucket.count),
        }])),
      },
    };
  }

  async function readRenderedSnapshot(device, pipeline, bindGroup, buffers, data, usage) {
    const encoder = device.createCommandEncoder({ label: "Flow ribbon progress preview" });
    encodeForward(encoder, pipeline, bindGroup, data);
    encoder.copyBufferToBuffer(
      buffers.rendered,
      0,
      buffers.previewReadback,
      0,
      data.width * data.height * 16,
    );
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await buffers.previewReadback.mapAsync(usage.READ);
    const rendered = new Float32Array(buffers.previewReadback.getMappedRange().slice(0));
    buffers.previewReadback.unmap();
    return renderedImageSummary(data, rendered);
  }

  function summarizeResult(
    data,
    rendered,
    finalParams,
    elapsedMs,
    iterations,
    runState = {},
    options = {},
    optimizerState = null,
  ) {
    const renderedSummary = renderedImageSummary(data, rendered);
    let colorDrift = 0;
    let geometryDriftSquared = 0;
    let widthDrift = 0;
    let opacityDrift = 0;
    let opacityAnchorMean = 0;
    let opacityFinalMean = 0;
    let anchorCurveArcTotal = 0;
    let finalCurveArcTotal = 0;
    let anchorCurveArcMaximum = 0;
    let finalCurveArcMaximum = 0;
    const layerParameterTotals = Array.from({ length: 4 }, () => ({
      count: 0,
      anchorWidth: 0,
      finalWidth: 0,
      anchorWidthMinimum: Infinity,
      anchorWidthMaximum: -Infinity,
      finalWidthMinimum: Infinity,
      finalWidthMaximum: -Infinity,
      widthGrowthCount: 0,
      widthShrinkCount: 0,
      anchorOpacity: 0,
      finalOpacity: 0,
    }));
    for (let stroke = 0; stroke < data.strokeCount; stroke += 1) {
      const base = stroke * PARAM_STRIDE;
      const dr = finalParams[base + 10] - data.anchors[base + 10];
      const dg = finalParams[base + 11] - data.anchors[base + 11];
      const db = finalParams[base + 12] - data.anchors[base + 12];
      colorDrift += Math.hypot(dr, dg, db);
      for (let component = 0; component < 8; component += 1) {
        const difference = finalParams[base + component] - data.anchors[base + component];
        geometryDriftSquared += difference * difference;
      }
      widthDrift += Math.abs(finalParams[base + 8] - data.anchors[base + 8]);
      const anchorOpacity = sigmoid(data.anchors[base + 9]);
      const finalOpacity = sigmoid(finalParams[base + 9]);
      opacityDrift += Math.abs(finalOpacity - anchorOpacity);
      opacityAnchorMean += anchorOpacity;
      opacityFinalMean += finalOpacity;
      const layer = Math.max(0, Math.min(3, Math.round(finalParams[base + 14])));
      const layerTotals = layerParameterTotals[layer];
      layerTotals.count += 1;
      layerTotals.anchorWidth += data.anchors[base + 8];
      layerTotals.finalWidth += finalParams[base + 8];
      layerTotals.anchorWidthMinimum = Math.min(
        layerTotals.anchorWidthMinimum,
        data.anchors[base + 8],
      );
      layerTotals.anchorWidthMaximum = Math.max(
        layerTotals.anchorWidthMaximum,
        data.anchors[base + 8],
      );
      layerTotals.finalWidthMinimum = Math.min(
        layerTotals.finalWidthMinimum,
        finalParams[base + 8],
      );
      layerTotals.finalWidthMaximum = Math.max(
        layerTotals.finalWidthMaximum,
        finalParams[base + 8],
      );
      layerTotals.widthGrowthCount += finalParams[base + 8] > data.anchors[base + 8] + 1e-4
        ? 1
        : 0;
      layerTotals.widthShrinkCount += finalParams[base + 8] < data.anchors[base + 8] - 1e-4
        ? 1
        : 0;
      layerTotals.anchorOpacity += anchorOpacity;
      layerTotals.finalOpacity += finalOpacity;
      const anchorCurveArc = sampledCubicArcLengthFromParams(data.anchors, base);
      const finalCurveArc = sampledCubicArcLengthFromParams(finalParams, base);
      anchorCurveArcTotal += anchorCurveArc;
      finalCurveArcTotal += finalCurveArc;
      anchorCurveArcMaximum = Math.max(anchorCurveArcMaximum, anchorCurveArc);
      finalCurveArcMaximum = Math.max(finalCurveArcMaximum, finalCurveArc);
    }
    const curveSplatChain = options.representation === "curve-splat-chain";
    const curveSampleEvaluationsPerCandidate = curveSplatChain ? 3 : 84;
    const tileSampleStride = curveSplatChain ? data.tileSampleStride : 1;
    const result = {
      image: renderedSummary.image,
      metadata: {
        candidate: curveSplatChain ? "BR-CAND-62" : "BR-CAND-60",
        mode: curveSplatChain
          ? "curve-controlled-standard-alpha-splat-chain-webgpu"
          : "differentiable-connected-ribbon-webgpu",
        qa_only: true,
        backend: "WebGPU compute",
        iterations,
        requested_iterations: runState.requestedIterations ?? iterations,
        stopped: runState.stopped === true,
        stroke_count: data.strokeCount,
        detail_parent_count: data.strokeCount - data.underpaintParentCount,
        underpaint_parent_count: data.underpaintParentCount,
        splat_count: data.physicalSplatCount,
        underpaint_splat_count: data.underpaintSplatCount,
        detail_splat_count: data.physicalSplatCount - data.underpaintSplatCount,
        splat_underpainting: data.underpaintSplatCount > 0,
        coverage_backcoat: data.underpaintSplatCount > 0,
        coverage_backcoat_kernel: data.underpaintSplatCount > 0
          ? "compact-quartic-flat-interior-v1"
          : "disabled",
        coverage_backcoat_geometry_trainable: false,
        coverage_backcoat_pigment_trainable: false,
        representation: curveSplatChain ? "curve-splat-chain" : "fused-ribbon",
        micro_splats_per_chain: curveSplatChain ? data.variableBrushDabs ? null : data.sampleCount : 0,
        variable_brush_dabs: data.variableBrushDabs,
        maximum_chain_slots: data.sampleCount,
        bristle_bundle: curveSplatChain && data.bristleBundle,
        brush_dabs: curveSplatChain && data.brushDabs,
        texture_guided_dabs: curveSplatChain && data.textureGuidedDabs,
        splat_size_variation_percent: curveSplatChain
          ? data.splatSizeVariation * 100
          : 0,
        initial_physical_splat_scale_stats: data.initialPhysicalSplatScaleStats,
        final_physical_splat_scale_stats: curveSplatChain
          ? physicalSplatScaleStats(finalParams, data)
          : null,
        stroke_texture_mode: data.strokeTextureMode,
        standard_alpha_micro_splats: curveSplatChain,
        input_signal_space: "sRGB",
        working_space: "linear sRGB float32",
        output_signal_space: "sRGB",
        loss: "linear-sRGB RGB L1 plus soft parameter anchors",
        rgb_l1_signal: renderedSummary.rgb_l1_signal,
        psnr_signal_db: renderedSummary.psnr_signal_db,
        coverage_stats: {
          ...renderedSummary.coverage_stats,
          step: iterations,
        },
        optimizer_state: optimizerState?.diagnostics || {
          mode: options.continuousState === false ? "disabled" : "survivor-row-remap",
          scope: "this-stage-detail-row-initialization",
          source_available: false,
          retained_detail_rows: 0,
          reset_detail_rows: data.strokeCount - data.underpaintParentCount,
          frozen_backcoat_rows: data.underpaintParentCount,
        },
        mean_color_anchor_drift_linear: colorDrift / data.strokeCount,
        control_point_rms_drift_px: Math.sqrt(geometryDriftSquared / (data.strokeCount * 8)),
        mean_width_drift_px: widthDrift / data.strokeCount,
        mean_opacity_drift: opacityDrift / data.strokeCount,
        mean_opacity_anchor: opacityAnchorMean / data.strokeCount,
        mean_opacity_final: opacityFinalMean / data.strokeCount,
        opacity_trainable: !(Number.isFinite(Number(options.fixedOpacity))
          && Number(options.fixedOpacity) > 0),
        fixed_stroke_opacity: Number.isFinite(Number(options.fixedOpacity))
          ? Math.max(0.05, Math.min(0.995, Number(options.fixedOpacity)))
          : null,
        front_width_learning_scale: Math.max(
          1,
          Math.min(8, Number(options.frontWidthLearningScale) || 1),
        ),
        layer_parameter_stats: layerParameterTotals.map((totals, layer) => ({
          layer,
          role: ["coarse-rear", "medium", "fine-front", "residual-rear"][layer],
          count: totals.count,
          mean_anchor_width_px: totals.anchorWidth / Math.max(1, totals.count),
          mean_final_width_px: totals.finalWidth / Math.max(1, totals.count),
          minimum_anchor_width_px: totals.count ? totals.anchorWidthMinimum : 0,
          maximum_anchor_width_px: totals.count ? totals.anchorWidthMaximum : 0,
          minimum_final_width_px: totals.count ? totals.finalWidthMinimum : 0,
          maximum_final_width_px: totals.count ? totals.finalWidthMaximum : 0,
          width_growth_count: totals.widthGrowthCount,
          width_growth_fraction: totals.widthGrowthCount / Math.max(1, totals.count),
          width_shrink_count: totals.widthShrinkCount,
          width_shrink_fraction: totals.widthShrinkCount / Math.max(1, totals.count),
          mean_anchor_opacity: totals.anchorOpacity / Math.max(1, totals.count),
          mean_final_opacity: totals.finalOpacity / Math.max(1, totals.count),
        })),
        mean_anchor_curve_arc_px: anchorCurveArcTotal / data.strokeCount,
        mean_final_curve_arc_px: finalCurveArcTotal / data.strokeCount,
        maximum_anchor_curve_arc_px: anchorCurveArcMaximum,
        maximum_final_curve_arc_px: finalCurveArcMaximum,
        final_parameter_hash_fnv1a32: fnv1aFloat32(finalParams),
        elapsed_ms: elapsedMs,
        iterations_per_second: iterations / Math.max(1e-6, elapsedMs / 1000),
        tile_candidates_mean: data.meanTileCandidates,
        tile_candidates_max: data.maxTileCandidates,
        tile_candidate_entries: data.tileCandidateEntries,
        pixel_candidate_pairs_per_iteration: data.pixelCandidatePairs,
        tile_sample_stride: tileSampleStride,
        tile_sampling: curveSplatChain ? "rotating-uniform-hash" : "full-frame",
        residual_priority_tile_sampling: data.residualPriorityTileSampling,
        residual_priority_tile_schedule: data.residualPriorityTileSampling
          ? "alternating-residual-weighted-and-uniform"
          : "disabled",
        residual_priority_high_tile_count: data.residualPriorityHighTileCount,
        residual_priority_omitted_tile_count: data.residualPriorityOmittedTileCount,
        residual_priority_activation_count: data.residualPriorityActivationCount,
        backward_accumulation: curveSplatChain && options.quadBackward !== false
          ? "16x16-tile-2x2-lane-quad-parent-curve-reduction"
          : "8x8-per-pixel-global-atomic",
        chain_quad_backward: curveSplatChain && options.quadBackward !== false,
        sampled_pixel_candidate_pairs_per_iteration:
          Math.ceil(data.pixelCandidatePairs / tileSampleStride),
        curve_sample_evaluations_per_candidate: curveSampleEvaluationsPerCandidate,
        curve_sample_evaluations_per_iteration:
          Math.ceil(data.pixelCandidatePairs * curveSampleEvaluationsPerCandidate / tileSampleStride),
      },
    };
    if (options.returnTrainingState) {
      result.trainingState = {
        params: finalParams.slice(),
        topologyParams: paramsWithoutPhaseLift(finalParams, data.widthPhaseLifts),
        renderedLinearRgba: rendered.slice(),
        firstMoment: optimizerState?.firstMoment?.slice() || new Float32Array(
          data.strokeCount * PARAM_STRIDE,
        ),
        secondMoment: optimizerState?.secondMoment?.slice() || new Float32Array(
          data.strokeCount * PARAM_STRIDE,
        ),
        adamSteps: optimizerState?.adamSteps?.slice() || new Uint32Array(data.strokeCount),
      };
    }
    return result;
  }

  async function train(image, strokePlan, options = {}) {
    if (!global.navigator?.gpu) throw new Error("WebGPU is unavailable for Flow Paint training.");
    const iterations = Math.max(1, Math.min(10000, Math.round(Number(options.iterations) || 300)));
    const progressInterval = Math.max(1, Math.min(
      iterations,
      Math.round(Number(options.progressInterval) || 20),
    ));
    const previewInterval = Math.max(0, Math.min(
      iterations,
      Math.round(Number(options.previewInterval) || 0),
    ));
    const data = prepareTrainingData(image, strokePlan, options);
    const continuousOptimizerState = remapContinuousOptimizerState(data, options.continuousState);
    const chainQuadBackward = options.representation === "curve-splat-chain"
      && options.quadBackward !== false;
    const adapter = await global.navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter is available for Flow Paint training.");
    const device = await adapter.requestDevice();
    let deviceLost = null;
    device.lost.then((info) => { deviceLost = info?.message || info?.reason || "unknown"; });
    const U = global.GPUBufferUsage;
    const buffers = {
      config: createBuffer(device, "Flow ribbon config", CONFIG_BYTES, U.UNIFORM | U.COPY_DST),
      params: createBuffer(device, "Flow ribbon params", data.params, U.STORAGE | U.COPY_SRC),
      anchors: createBuffer(device, "Flow ribbon anchors", data.anchors, U.STORAGE),
      tileOffsets: createBuffer(device, "Flow ribbon tile offsets", data.tileOffsets, U.STORAGE),
      tileIndices: createBuffer(device, "Flow ribbon tile indices", data.tileIndices, U.STORAGE),
      tileSamplingMasks: createBuffer(
        device,
        "Flow ribbon residual-priority tile masks",
        data.tileSamplingMasks,
        U.STORAGE,
      ),
      target: createBuffer(device, "Flow ribbon target", data.target, U.STORAGE),
      rendered: createBuffer(device, "Flow ribbon rendered", data.width * data.height * 16, U.STORAGE | U.COPY_SRC),
      logTransmittance: createBuffer(device, "Flow ribbon log transmittance", data.width * data.height * 4, U.STORAGE),
      gradients: createBuffer(device, "Flow ribbon gradients", data.strokeCount * GRAD_STRIDE * 4, U.STORAGE | U.COPY_DST),
      firstMoment: createBuffer(
        device,
        "Flow ribbon Adam m",
        continuousOptimizerState.firstMoment,
        U.STORAGE | U.COPY_DST | U.COPY_SRC,
      ),
      secondMoment: createBuffer(
        device,
        "Flow ribbon Adam v",
        continuousOptimizerState.secondMoment,
        U.STORAGE | U.COPY_DST | U.COPY_SRC,
      ),
      adamSteps: createBuffer(
        device,
        "Flow ribbon Adam steps",
        continuousOptimizerState.adamSteps,
        U.STORAGE | U.COPY_DST | U.COPY_SRC,
      ),
      previewReadback: createBuffer(
        device,
        "Flow ribbon preview readback",
        data.width * data.height * 16,
        U.COPY_DST | U.MAP_READ,
      ),
    };
    const pipelines = {
      forward: await createComputePipeline(device, "Flow ribbon forward", FORWARD_WGSL),
      backward: await createComputePipeline(
        device,
        chainQuadBackward ? "Curve chain quad backward" : "Flow ribbon backward",
        chainQuadBackward ? CHAIN_QUAD_BACKWARD_WGSL : BACKWARD_WGSL,
      ),
      update: await createComputePipeline(device, "Flow ribbon update", UPDATE_WGSL),
    };
    const bindGroups = makeBindGroups(device, pipelines, buffers);
    const startedAt = performance.now();
    let completedIterations = 0;
    let stopped = false;
    let finalReadbacks = null;
    try {
      for (let iteration = 0; iteration < iterations; iteration += 1) {
        device.queue.writeBuffer(buffers.config, 0, configBytes(data, iteration, options));
        const encoder = device.createCommandEncoder({ label: `Flow ribbon iteration ${iteration + 1}` });
        encoder.clearBuffer(buffers.gradients);
        encodeForward(encoder, pipelines.forward, bindGroups.forward, data);
        encodeBackward(
          encoder,
          pipelines.backward,
          bindGroups.backward,
          data,
          chainQuadBackward,
        );
        encodeUpdate(encoder, pipelines.update, bindGroups.update, data);
        device.queue.submit([encoder.finish()]);
        if ((iteration + 1) % progressInterval === 0 || iteration + 1 === iterations) {
          await device.queue.onSubmittedWorkDone();
          if (deviceLost) throw new Error(`Flow Paint WebGPU device lost: ${deviceLost}`);
          completedIterations = iteration + 1;
          if (previewInterval > 0 && (
            (iteration + 1) % previewInterval === 0 || iteration + 1 === iterations
          )) {
            device.queue.writeBuffer(buffers.config, 0, configBytes(data, iteration, options, true));
            const preview = await readRenderedSnapshot(
              device,
              pipelines.forward,
              bindGroups.forward,
              buffers,
              data,
              global.GPUMapMode,
            );
            await options.onPreview?.({
              iteration: iteration + 1,
              iterations,
              ...preview,
            });
          }
          options.onProgress?.({
            iteration: iteration + 1,
            iterations,
            elapsed_ms: performance.now() - startedAt,
            widthTraining: widthTrainingSettings(iteration, options),
          });
          if (options.shouldStop?.()) {
            stopped = true;
            break;
          }
          await options.waitWhilePaused?.();
        }
      }
      device.queue.writeBuffer(buffers.config, 0, configBytes(data, completedIterations, options, true));
      const finalEncoder = device.createCommandEncoder({ label: "Flow ribbon final render" });
      encodeForward(finalEncoder, pipelines.forward, bindGroups.forward, data);
      finalReadbacks = {};
      finalReadbacks.rendered = device.createBuffer({
        label: "Flow ribbon render readback",
        size: data.width * data.height * 16,
        usage: U.COPY_DST | U.MAP_READ,
      });
      finalReadbacks.params = device.createBuffer({
        label: "Flow ribbon params readback",
        size: data.params.byteLength,
        usage: U.COPY_DST | U.MAP_READ,
      });
      if (options.returnTrainingState) {
        finalReadbacks.firstMoment = device.createBuffer({
          label: "Flow ribbon Adam m readback",
          size: data.strokeCount * PARAM_STRIDE * 4,
          usage: U.COPY_DST | U.MAP_READ,
        });
        finalReadbacks.secondMoment = device.createBuffer({
          label: "Flow ribbon Adam v readback",
          size: data.strokeCount * PARAM_STRIDE * 4,
          usage: U.COPY_DST | U.MAP_READ,
        });
        finalReadbacks.adamSteps = device.createBuffer({
          label: "Flow ribbon Adam step readback",
          size: data.strokeCount * 4,
          usage: U.COPY_DST | U.MAP_READ,
        });
      }
      finalEncoder.copyBufferToBuffer(
        buffers.rendered,
        0,
        finalReadbacks.rendered,
        0,
        data.width * data.height * 16,
      );
      finalEncoder.copyBufferToBuffer(buffers.params, 0, finalReadbacks.params, 0, data.params.byteLength);
      if (options.returnTrainingState) {
        finalEncoder.copyBufferToBuffer(
          buffers.firstMoment,
          0,
          finalReadbacks.firstMoment,
          0,
          data.strokeCount * PARAM_STRIDE * 4,
        );
        finalEncoder.copyBufferToBuffer(
          buffers.secondMoment,
          0,
          finalReadbacks.secondMoment,
          0,
          data.strokeCount * PARAM_STRIDE * 4,
        );
        finalEncoder.copyBufferToBuffer(
          buffers.adamSteps,
          0,
          finalReadbacks.adamSteps,
          0,
          data.strokeCount * 4,
        );
      }
      device.queue.submit([finalEncoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const mapRequests = [
        finalReadbacks.rendered.mapAsync(global.GPUMapMode.READ),
        finalReadbacks.params.mapAsync(global.GPUMapMode.READ),
      ];
      if (options.returnTrainingState) {
        mapRequests.push(
          finalReadbacks.firstMoment.mapAsync(global.GPUMapMode.READ),
          finalReadbacks.secondMoment.mapAsync(global.GPUMapMode.READ),
          finalReadbacks.adamSteps.mapAsync(global.GPUMapMode.READ),
        );
      }
      await Promise.all(mapRequests);
      const rendered = new Float32Array(finalReadbacks.rendered.getMappedRange().slice(0));
      const finalParams = new Float32Array(finalReadbacks.params.getMappedRange().slice(0));
      const finalOptimizerState = options.returnTrainingState ? {
        firstMoment: new Float32Array(finalReadbacks.firstMoment.getMappedRange().slice(0)),
        secondMoment: new Float32Array(finalReadbacks.secondMoment.getMappedRange().slice(0)),
        adamSteps: new Uint32Array(finalReadbacks.adamSteps.getMappedRange().slice(0)),
        diagnostics: continuousOptimizerState.diagnostics,
      } : continuousOptimizerState;
      for (const buffer of Object.values(finalReadbacks)) {
        buffer.unmap();
        buffer.destroy();
      }
      finalReadbacks = null;
      return summarizeResult(
        data,
        rendered,
        finalParams,
        performance.now() - startedAt,
        completedIterations,
        { requestedIterations: iterations, stopped },
        options,
        finalOptimizerState,
      );
    } finally {
      Object.values(finalReadbacks || {}).forEach((buffer) => buffer.destroy());
      Object.values(buffers).forEach((buffer) => buffer.destroy());
      device.destroy();
    }
  }

  global.Image2SplatPaintFlowRibbonTrainer = Object.freeze({
    prepareTrainingData,
    allocateBrushDabCounts,
    widthTrainingSettings,
    train,
    constants: Object.freeze({
      PARAM_STRIDE,
      TILE_SIZE,
      CURVE_SAMPLES,
      BRISTLE_BUNDLE_SAMPLES,
      BRUSH_DAB_SAMPLES,
      VARIABLE_BRUSH_DAB_SAMPLES,
      STROKE_TEXTURE_STANDARD,
      STROKE_TEXTURE_FINE_BRISTLES,
      STROKE_TEXTURE_BRUSH_DABS,
    }),
  });
})(globalThis);
