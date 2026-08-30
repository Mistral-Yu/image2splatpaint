(function installFlowPaintReference(global) {
  "use strict";

  const TAU = Math.PI * 2;
  const BROAD_FLOW_LAYERS = Object.freeze([
    Object.freeze({
      share: 0.06,
      length: 0.110,
      halfWidth: 0.0170,
      opacity: 0.90,
      colorLimit: 0.28,
      brightnessMin: 0.66,
      brightnessMax: 1.34,
      saturation: 1.16,
    }),
    Object.freeze({
      share: 0.225,
      length: 0.063,
      halfWidth: 0.0090,
      opacity: 0.82,
      colorLimit: 0.23,
      brightnessMin: 0.66,
      brightnessMax: 1.34,
      saturation: 1.16,
    }),
    Object.freeze({
      share: 0.715,
      length: 0.035,
      halfWidth: 0.00475,
      opacity: 0.74,
      colorLimit: 0.18,
      brightnessMin: 0.66,
      brightnessMax: 1.34,
      saturation: 1.16,
    }),
  ]);

  // BR-CAND-57 keeps the upstream-inspired three-level seed hierarchy, but is
  // independently implemented. It deliberately uses more, longer, thinner
  // paths than BR-CAND-56 so the final training-sized teacher still contains
  // individually readable fine marks after one downsample.
  const FINE_FLOW_LAYERS = Object.freeze([
    Object.freeze({
      share: 0.06,
      length: 0.220,
      halfWidth: 0.0260,
      opacity: 0.94,
      colorLimit: 0.13,
      brightnessMin: 0.92,
      brightnessMax: 1.08,
      saturation: 1.06,
      crispOpaqueStroke: true,
    }),
    Object.freeze({
      share: 0.225,
      length: 0.125,
      halfWidth: 0.0150,
      opacity: 0.95,
      colorLimit: 0.10,
      brightnessMin: 0.88,
      brightnessMax: 1.12,
      saturation: 1.09,
      crispOpaqueStroke: true,
    }),
    Object.freeze({
      share: 0.715,
      length: 0.066,
      halfWidth: 0.0090,
      opacity: 0.96,
      colorLimit: 0.075,
      brightnessMin: 0.82,
      brightnessMax: 1.18,
      saturation: 1.12,
      crispOpaqueStroke: true,
    }),
  ]);

  // BR-CAND-59 is the render-only gate for an independent Flow Paint
  // primitive. Unlike BR-CAND-58 it does not approximate a path with one
  // Brush Splat: one seed owns one continuous cubic ribbon and one pigment
  // anchor. Width and coverage are intentionally lower than the fine teacher
  // so individual strokes remain legible instead of becoming a smooth coat.
  const RIBBON_FLOW_LAYERS = Object.freeze([
    Object.freeze({
      share: 0.20,
      length: 0.245,
      halfWidth: 0.0160,
      opacity: 0.84,
      colorLimit: 0.15,
      brightnessMin: 0.88,
      brightnessMax: 1.12,
      saturation: 1.08,
      colorRadius: 0.0040,
      cubicRibbon: true,
    }),
    Object.freeze({
      share: 0.40,
      length: 0.145,
      halfWidth: 0.0082,
      opacity: 0.76,
      colorLimit: 0.115,
      brightnessMin: 0.84,
      brightnessMax: 1.16,
      saturation: 1.11,
      colorRadius: 0.0016,
      cubicRibbon: true,
    }),
    Object.freeze({
      share: 0.40,
      length: 0.078,
      halfWidth: 0.0040,
      opacity: 0.68,
      colorLimit: 0.085,
      brightnessMin: 0.80,
      brightnessMax: 1.20,
      saturation: 1.14,
      colorRadius: 0.00045,
      cubicRibbon: true,
    }),
  ]);

  function referenceProfile(options) {
    const ribbon = options.profile === "connected-ribbon-v1";
    const fine = options.profile === "fine-layered-v2";
    return ribbon
      ? {
          candidate: "BR-CAND-59",
          mode: "independent-connected-cubic-ribbon",
          layers: RIBBON_FLOW_LAYERS,
          underpaintScale: 0,
          canvasLinear: [0.91, 0.885, 0.82],
          fixedStrokeBudget: true,
          independentPrimitive: true,
        }
      : fine
      ? {
          candidate: "BR-CAND-57",
          mode: "oversampled-fine-flow-stroke-reference",
          layers: FINE_FLOW_LAYERS,
          underpaintScale: 0,
          canvasLinear: [0.91, 0.885, 0.82],
          fixedStrokeBudget: true,
        }
      : {
          candidate: "BR-CAND-56",
          mode: "connected-flow-stroke-reference",
          layers: BROAD_FLOW_LAYERS,
          underpaintScale: 0.70,
          fixedStrokeBudget: false,
        };
  }

  function clamp01(value) {
    return Math.max(0, Math.min(1, Number(value) || 0));
  }

  function fract(value) {
    return value - Math.floor(value);
  }

  function hash01(value) {
    let x = (Math.trunc(value) ^ 0x9e3779b9) >>> 0;
    x ^= x >>> 16;
    x = Math.imul(x, 0x21f0aaad);
    x ^= x >>> 15;
    x = Math.imul(x, 0x735a2d97);
    x ^= x >>> 15;
    return (x >>> 0) / 4294967296;
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

  function validateImage(image) {
    const width = Math.max(1, Math.round(Number(image?.width) || 0));
    const height = Math.max(1, Math.round(Number(image?.height) || 0));
    if (!(image?.rgb instanceof Float32Array) || image.rgb.length !== width * height * 3) {
      throw new Error("Flow paint reference requires width*height*3 Float32 RGB input.");
    }
    if (image.alpha && (!(image.alpha instanceof Float32Array) || image.alpha.length !== width * height)) {
      throw new Error("Flow paint reference alpha length does not match the input image.");
    }
    return { width, height };
  }

  function buildIntegral(source, width, height) {
    const stride = width + 1;
    const integral = new Float32Array(stride * (height + 1));
    for (let y = 0; y < height; y += 1) {
      let row = 0;
      for (let x = 0; x < width; x += 1) {
        row += source[y * width + x];
        integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1] + row;
      }
    }
    return integral;
  }

  function integralMean(integral, width, height, x, y, radius) {
    const stride = width + 1;
    const x0 = Math.max(0, x - radius);
    const y0 = Math.max(0, y - radius);
    const x1 = Math.min(width, x + radius + 1);
    const y1 = Math.min(height, y + radius + 1);
    const sum = integral[y1 * stride + x1]
      - integral[y0 * stride + x1]
      - integral[y1 * stride + x0]
      + integral[y0 * stride + x0];
    return sum / Math.max(1, (x1 - x0) * (y1 - y0));
  }

  function smoothstep(minimum, maximum, value) {
    const t = clamp01((Number(value) - minimum) / Math.max(1e-8, maximum - minimum));
    return t * t * (3 - 2 * t);
  }

  function sampledQuantile(values, quantile) {
    if (!(values instanceof Float32Array) || values.length === 0) return 0;
    const stride = Math.max(1, Math.floor(values.length / 8192));
    const sample = [];
    for (let index = 0; index < values.length; index += stride) {
      const value = values[index];
      if (Number.isFinite(value)) sample.push(value);
    }
    if (sample.length === 0) return 0;
    sample.sort((a, b) => a - b);
    return sample[Math.max(0, Math.min(
      sample.length - 1,
      Math.round(clamp01(quantile) * (sample.length - 1)),
    ))];
  }

  function sampledPositiveQuantile(values, quantile) {
    if (!(values instanceof Float32Array) || values.length === 0) return 0;
    const stride = Math.max(1, Math.floor(values.length / 8192));
    const sample = [];
    for (let index = 0; index < values.length; index += stride) {
      const value = values[index];
      if (Number.isFinite(value) && value > 1e-9) sample.push(value);
    }
    if (sample.length === 0) return 0;
    sample.sort((a, b) => a - b);
    return sample[Math.max(0, Math.min(
      sample.length - 1,
      Math.round(clamp01(quantile) * (sample.length - 1)),
    ))];
  }

  function scalarBilinear(values, width, height, x, y) {
    const px = Math.max(0, Math.min(width - 1, Number(x) || 0));
    const py = Math.max(0, Math.min(height - 1, Number(y) || 0));
    const x0 = Math.floor(px);
    const y0 = Math.floor(py);
    const x1 = Math.min(width - 1, x0 + 1);
    const y1 = Math.min(height - 1, y0 + 1);
    const fx = px - x0;
    const fy = py - y0;
    const top = values[y0 * width + x0] * (1 - fx) + values[y0 * width + x1] * fx;
    const bottom = values[y1 * width + x0] * (1 - fx) + values[y1 * width + x1] * fx;
    return top * (1 - fy) + bottom * fy;
  }

  function buildFlowXdogGuide(luminance, fieldX, fieldY, coherence, width, height) {
    const count = width * height;
    if (fieldX?.length !== count || fieldY?.length !== count) {
      return {
        score: new Float32Array(count),
        summary: {
          algorithm: "disabled-no-flow-field",
          mean_score: 0,
        },
      };
    }
    const shortSide = Math.max(1, Math.min(width, height));
    const narrowSigma = Math.max(0.8, Math.min(1.5, shortSide / 512));
    const wideSigma = narrowSigma * 1.6;
    const normalRadius = Math.max(2, Math.min(8, Math.ceil(wideSigma * 2.6)));
    const normalDog = new Float32Array(count);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = y * width + x;
        const normalX = -fieldY[index];
        const normalY = fieldX[index];
        let narrow = 0;
        let wide = 0;
        let narrowWeight = 0;
        let wideWeight = 0;
        for (let offset = -normalRadius; offset <= normalRadius; offset += 1) {
          const sample = scalarBilinear(
            luminance, width, height, x + normalX * offset, y + normalY * offset,
          );
          const narrowKernel = Math.exp(-(offset * offset) / (2 * narrowSigma * narrowSigma));
          const wideKernel = Math.exp(-(offset * offset) / (2 * wideSigma * wideSigma));
          narrow += sample * narrowKernel;
          wide += sample * wideKernel;
          narrowWeight += narrowKernel;
          wideWeight += wideKernel;
        }
        // Equal normalized kernels are exactly zero on flat fields. Absolute
        // response retains both dark and light source contours.
        normalDog[index] = Math.abs(
          narrow / Math.max(1e-8, narrowWeight)
          - wide / Math.max(1e-8, wideWeight),
        );
      }
    }
    const flowRadius = Math.max(2, Math.min(6, Math.round(shortSide / 128)));
    const flowSigma = Math.max(1, flowRadius * 0.62);
    const response = new Float32Array(count);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = y * width + x;
        const tangentX = fieldX[index];
        const tangentY = fieldY[index];
        let sum = normalDog[index];
        let weightTotal = 1;
        for (let offset = 1; offset <= flowRadius; offset += 1) {
          const weight = Math.exp(-(offset * offset) / (2 * flowSigma * flowSigma));
          sum += scalarBilinear(
            normalDog, width, height, x + tangentX * offset, y + tangentY * offset,
          ) * weight;
          sum += scalarBilinear(
            normalDog, width, height, x - tangentX * offset, y - tangentY * offset,
          ) * weight;
          weightTotal += weight * 2;
        }
        response[index] = sum / weightTotal;
      }
    }
    const ridge = new Float32Array(count);
    for (let y = 1; y < height - 1; y += 1) {
      for (let x = 1; x < width - 1; x += 1) {
        const index = y * width + x;
        const tangentX = fieldX[index];
        const tangentY = fieldY[index];
        const normalX = -tangentY;
        const normalY = tangentX;
        const value = response[index];
        if (
          value >= scalarBilinear(response, width, height, x + normalX, y + normalY)
          && value >= scalarBilinear(response, width, height, x - normalX, y - normalY)
        ) {
          ridge[index] = value * smoothstep(0.18, 0.55, coherence?.[index] ?? 0);
        }
      }
    }
    const low = sampledPositiveQuantile(ridge, 0.50);
    const high = sampledPositiveQuantile(ridge, 0.96);
    const range = high - low;
    const score = new Float32Array(count);
    let scoreTotal = 0;
    for (let index = 0; index < count; index += 1) {
      score[index] = range > 1e-9 ? smoothstep(low, high, ridge[index]) ** 0.90 : 0;
      scoreTotal += score[index];
    }
    return {
      score,
      summary: {
        algorithm: "linear-srgb-flow-xdog-guide",
        narrow_sigma_px: narrowSigma,
        wide_sigma_px: wideSigma,
        normal_radius_px: normalRadius,
        tangent_radius_px: flowRadius,
        ridge_p50: low,
        ridge_p96: high,
        mean_score: scoreTotal / Math.max(1, count),
      },
    };
  }

  function buildTextureGuide(
    linearRgb,
    luminance,
    edgeStrength,
    edgeCoherence,
    width,
    height,
    fieldX = null,
    fieldY = null,
  ) {
    const count = width * height;
    const luminanceSquared = new Float32Array(count);
    const colorEdge = new Float32Array(count);
    const sampleChannel = (x, y, channel) => linearRgb[
      (Math.max(0, Math.min(height - 1, y)) * width
        + Math.max(0, Math.min(width - 1, x))) * 3 + channel
    ];
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = y * width + x;
        luminanceSquared[index] = luminance[index] * luminance[index];
        let colorGradientSquared = 0;
        for (let channel = 0; channel < 3; channel += 1) {
          const gx = (sampleChannel(x + 1, y, channel) - sampleChannel(x - 1, y, channel)) * 0.5;
          const gy = (sampleChannel(x, y + 1, channel) - sampleChannel(x, y - 1, channel)) * 0.5;
          colorGradientSquared += gx * gx + gy * gy;
        }
        colorEdge[index] = Math.sqrt(colorGradientSquared / 3);
      }
    }

    const luminanceIntegral = buildIntegral(luminance, width, height);
    const luminanceSquaredIntegral = buildIntegral(luminanceSquared, width, height);
    const shortSide = Math.max(1, Math.min(width, height));
    const fineRadius = Math.max(2, Math.round(shortSide / 128));
    const broadRadius = Math.max(fineRadius + 1, Math.round(shortSide / 64));
    const raw = new Float32Array(count);
    const edgeRaw = new Float32Array(count);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = y * width + x;
        const fineMean = integralMean(luminanceIntegral, width, height, x, y, fineRadius);
        const fineMeanSquared = integralMean(
          luminanceSquaredIntegral,
          width,
          height,
          x,
          y,
          fineRadius,
        );
        const broadMean = integralMean(luminanceIntegral, width, height, x, y, broadRadius);
        const broadMeanSquared = integralMean(
          luminanceSquaredIntegral,
          width,
          height,
          x,
          y,
          broadRadius,
        );
        const fineDeviation = Math.sqrt(Math.max(0, fineMeanSquared - fineMean * fineMean));
        const broadDeviation = Math.sqrt(Math.max(0, broadMeanSquared - broadMean * broadMean));
        raw[index] = edgeStrength[index] * 0.38
          + fineDeviation * 0.32
          + broadDeviation * 0.20
          + colorEdge[index] * 0.10;
        // Keep the thin-bristle guide independent from local texture energy.
        // A structure-tensor coherence gate suppresses isotropic grain while
        // retaining coherent luminance and colour boundaries in linear sRGB.
        const coherence = clamp01(edgeCoherence?.[index] ?? 0);
        edgeRaw[index] = (
          edgeStrength[index] * 0.78 + colorEdge[index] * 0.22
        ) * (0.15 + Math.sqrt(coherence) * 0.85);
      }
    }

    const low = sampledQuantile(raw, 0.12);
    const high = sampledQuantile(raw, 0.90);
    const range = high - low;
    const edgeLow = sampledQuantile(edgeRaw, 0.60);
    const edgeHigh = sampledQuantile(edgeRaw, 0.95);
    const edgeRange = edgeHigh - edgeLow;
    const flowXdog = buildFlowXdogGuide(
      luminance, fieldX, fieldY, edgeCoherence, width, height,
    );
    const score = new Float32Array(count);
    const edgeScore = new Float32Array(count);
    const dabVisibility = new Float32Array(count);
    const darkFlat = new Float32Array(count);
    let scoreTotal = 0;
    let edgeScoreTotal = 0;
    let darkFlatPixels = 0;
    for (let index = 0; index < count; index += 1) {
      const normalized = range > 1e-7
        ? smoothstep(low, high, raw[index]) ** 0.85
        : 0;
      const darkness = 1 - smoothstep(0.035, 0.20, luminance[index]);
      const flatDarkness = (1 - normalized) * darkness;
      score[index] = normalized;
      const coherentEdgeScore = edgeRange > 1e-7
        ? smoothstep(edgeLow, edgeHigh, edgeRaw[index]) ** 1.25
        : 0;
      // Preserve every established coherent colour edge, then add connected
      // Flow-XDoG evidence only in the remaining score range. Texture energy
      // never activates the two thin bristles by itself.
      edgeScore[index] = clamp01(
        coherentEdgeScore
        + (1 - coherentEdgeScore) * flowXdog.score[index] * 0.78,
      );
      darkFlat[index] = flatDarkness;
      // Darkness alone never removes a dab. It only strengthens suppression
      // when the same source area is also locally flat.
      dabVisibility[index] = clamp01(normalized * (1 - flatDarkness * 0.65));
      scoreTotal += normalized;
      edgeScoreTotal += edgeScore[index];
      if (flatDarkness >= 0.60) darkFlatPixels += 1;
    }
    return {
      width,
      height,
      score,
      edgeScore,
      dabVisibility,
      darkFlat,
      summary: {
        fine_radius_px: fineRadius,
        broad_radius_px: broadRadius,
        raw_p12: low,
        raw_p90: high,
        mean_score: scoreTotal / Math.max(1, count),
        edge_raw_p60: edgeLow,
        edge_raw_p95: edgeHigh,
        mean_edge_score: edgeScoreTotal / Math.max(1, count),
        edge_score_mode: "coherent-colour-plus-flow-xdog-78",
        flow_xdog: flowXdog.summary,
        dark_flat_pixel_fraction: darkFlatPixels / Math.max(1, count),
      },
    };
  }

  function makeAnalysis(image, width, height, options = {}) {
    const count = width * height;
    const linearRgb = new Float32Array(count * 3);
    const luminance = new Float32Array(count);
    for (let i = 0; i < count; i += 1) {
      const offset = i * 3;
      const red = signalSrgbToLinear(image.rgb[offset]);
      const green = signalSrgbToLinear(image.rgb[offset + 1]);
      const blue = signalSrgbToLinear(image.rgb[offset + 2]);
      linearRgb[offset] = red;
      linearRgb[offset + 1] = green;
      linearRgb[offset + 2] = blue;
      luminance[i] = red * 0.2126 + green * 0.7152 + blue * 0.0722;
    }

    const tensorXX = new Float32Array(count);
    const tensorXY = new Float32Array(count);
    const tensorYY = new Float32Array(count);
    const sampleLuma = (x, y) => luminance[
      Math.max(0, Math.min(height - 1, y)) * width + Math.max(0, Math.min(width - 1, x))
    ];
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = y * width + x;
        const gx = (sampleLuma(x + 1, y) - sampleLuma(x - 1, y)) * 0.5;
        const gy = (sampleLuma(x, y + 1) - sampleLuma(x, y - 1)) * 0.5;
        tensorXX[index] = gx * gx;
        tensorXY[index] = gx * gy;
        tensorYY[index] = gy * gy;
      }
    }

    const integralXX = buildIntegral(tensorXX, width, height);
    const integralXY = buildIntegral(tensorXY, width, height);
    const integralYY = buildIntegral(tensorYY, width, height);
    const fieldX = new Float32Array(count);
    const fieldY = new Float32Array(count);
    const confidence = new Float32Array(count);
    const edgeStrength = new Float32Array(count);
    const tensorRadius = Math.max(2, Math.round(Math.min(width, height) / 120));
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = y * width + x;
        const jxx = integralMean(integralXX, width, height, x, y, tensorRadius);
        const jxy = integralMean(integralXY, width, height, x, y, tensorRadius);
        const jyy = integralMean(integralYY, width, height, x, y, tensorRadius);
        const trace = jxx + jyy;
        const separation = Math.hypot(jxx - jyy, 2 * jxy);
        const coherence = trace > 1e-9 ? Math.min(1, separation / trace) : 0;
        const normal = 0.5 * Math.atan2(2 * jxy, jxx - jyy);
        const structureAngle = normal + Math.PI * 0.5;
        const nx = x / Math.max(1, width - 1) - 0.5;
        const ny = y / Math.max(1, height - 1) - 0.5;
        const fallbackAngle = Math.atan2(ny * 1.12, nx) + Math.PI * 0.5
          + Math.sin(nx * 11 - ny * 13) * 0.16;
        const structureWeight = Math.max(0, Math.min(1, (coherence - 0.04) / 0.42));
        const doubleX = (1 - structureWeight) * Math.cos(2 * fallbackAngle)
          + structureWeight * Math.cos(2 * structureAngle);
        const doubleY = (1 - structureWeight) * Math.sin(2 * fallbackAngle)
          + structureWeight * Math.sin(2 * structureAngle);
        const angle = 0.5 * Math.atan2(doubleY, doubleX);
        fieldX[index] = Math.cos(angle);
        fieldY[index] = Math.sin(angle);
        confidence[index] = coherence;
        edgeStrength[index] = Math.sqrt(Math.max(0, trace));
      }
    }
    const textureGuide = options.textureGuide === true
      ? buildTextureGuide(
          linearRgb,
          luminance,
          edgeStrength,
          confidence,
          width,
          height,
          fieldX,
          fieldY,
        )
      : null;
    return { linearRgb, luminance, fieldX, fieldY, confidence, edgeStrength, textureGuide };
  }

  function blurredUnderpaint(analysis, width, height, scale, canvasLinear = null) {
    const count = width * height;
    const output = new Float32Array(count * 3);
    if (canvasLinear) {
      for (let i = 0; i < count; i += 1) {
        output[i * 3] = canvasLinear[0];
        output[i * 3 + 1] = canvasLinear[1];
        output[i * 3 + 2] = canvasLinear[2];
      }
      return output;
    }
    const radius = Math.max(2, Math.round(Math.min(width, height) * 0.018));
    for (let channel = 0; channel < 3; channel += 1) {
      const plane = new Float32Array(count);
      for (let i = 0; i < count; i += 1) plane[i] = analysis.linearRgb[i * 3 + channel];
      const integral = buildIntegral(plane, width, height);
      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          output[(y * width + x) * 3 + channel] = integralMean(
            integral,
            width,
            height,
            x,
            y,
            radius,
          ) * scale;
        }
      }
    }
    return output;
  }

  function fieldAt(analysis, width, height, x, y, previousX = 0, previousY = 0) {
    const ix = Math.max(0, Math.min(width - 1, Math.round(x)));
    const iy = Math.max(0, Math.min(height - 1, Math.round(y)));
    const index = iy * width + ix;
    let dx = analysis.fieldX[index];
    let dy = analysis.fieldY[index];
    if (previousX || previousY) {
      if (dx * previousX + dy * previousY < 0) {
        dx = -dx;
        dy = -dy;
      }
    }
    return [dx, dy];
  }

  function sampleLinearRgb(analysis, width, height, x, y, target) {
    const px = Math.max(0, Math.min(width - 1, x));
    const py = Math.max(0, Math.min(height - 1, y));
    const x0 = Math.floor(px);
    const y0 = Math.floor(py);
    const x1 = Math.min(width - 1, x0 + 1);
    const y1 = Math.min(height - 1, y0 + 1);
    const fx = px - x0;
    const fy = py - y0;
    for (let channel = 0; channel < 3; channel += 1) {
      const top = analysis.linearRgb[(y0 * width + x0) * 3 + channel] * (1 - fx)
        + analysis.linearRgb[(y0 * width + x1) * 3 + channel] * fx;
      const bottom = analysis.linearRgb[(y1 * width + x0) * 3 + channel] * (1 - fx)
        + analysis.linearRgb[(y1 * width + x1) * 3 + channel] * fx;
      target[channel] = top * (1 - fy) + bottom * fy;
    }
    return target;
  }

  function createSplatUnderpaintPlan(image, options = {}) {
    const { width, height } = validateImage(image);
    const requested = Math.max(0, Math.round(Number(options.count) || 0));
    const representation = options.representation === "curve-splat-chain"
      ? "curve-splat-chain"
      : "fused-ribbon";
    if (requested === 0) {
      return {
        strokePlan: [],
        metadata: {
          requested_splats: 0,
          accepted_splats: 0,
          representation,
          source_texture_used: false,
          coverage_guarantee: "disabled",
        },
      };
    }

    const analysis = makeAnalysis(image, width, height);
    const residualRender = options.residualRender instanceof Float32Array
      && options.residualRender.length === width * height * 4
      ? options.residualRender
      : null;
    const aspect = width / Math.max(1, height);
    const rows = Math.max(1, Math.min(requested, Math.round(Math.sqrt(requested / aspect))));
    const columnsFloor = Math.floor(requested / rows);
    const rowsWithExtraColumn = requested - columnsFloor * rows;
    const strokePlan = [];
    const color = [0, 0, 0];
    let generated = 0;

    for (let row = 0; row < rows; row += 1) {
      const columns = columnsFloor + (row < rowsWithExtraColumn ? 1 : 0);
      if (columns < 1) continue;
      const cellWidth = width / columns;
      const cellHeight = height / rows;
      for (let column = 0; column < columns && generated < requested; column += 1) {
        const random = hash01((generated + 1) * 65537 + Math.round(Number(options.seed) || 240825));
        const minimumX = Math.max(0, Math.floor(column * cellWidth));
        const maximumX = Math.min(width - 1, Math.ceil((column + 1) * cellWidth) - 1);
        const minimumY = Math.max(0, Math.floor(row * cellHeight));
        const maximumY = Math.min(height - 1, Math.ceil((row + 1) * cellHeight) - 1);
        // Geometry stays at the exact grid-cell center. Pigment may be sampled
        // from the highest residual inside the cell, but moving the Splat to
        // that pixel would reopen gaps between neighboring backcoat cells.
        const actualCellWidth = maximumX + 1 - minimumX;
        const actualCellHeight = maximumY + 1 - minimumY;
        const centerX = (minimumX + maximumX + 1) * 0.5;
        const centerY = (minimumY + maximumY + 1) * 0.5;
        let selectedIndex = Math.min(height - 1, Math.floor(centerY)) * width
          + Math.min(width - 1, Math.floor(centerX));
        let selectedTransmittance = residualRender ? residualRender[selectedIndex * 4 + 3] : 1;
        let selectedScore = -Infinity;
        const cellMean = [0, 0, 0];
        let cellMeanWeight = 0;
        for (let y = minimumY; y <= maximumY; y += 1) {
          for (let x = minimumX; x <= maximumX; x += 1) {
            const index = y * width + x;
            const sourceAlpha = image.alpha?.[index] ?? 1;
            const targetOffset = index * 3;
            cellMean[0] += analysis.linearRgb[targetOffset] * sourceAlpha;
            cellMean[1] += analysis.linearRgb[targetOffset + 1] * sourceAlpha;
            cellMean[2] += analysis.linearRgb[targetOffset + 2] * sourceAlpha;
            cellMeanWeight += sourceAlpha;
            const transmittance = residualRender
              ? clamp01(residualRender[index * 4 + 3])
              : 1;
            const residual = residualRender
              ? (
                  Math.abs(analysis.linearRgb[targetOffset] - residualRender[index * 4])
                  + Math.abs(analysis.linearRgb[targetOffset + 1] - residualRender[index * 4 + 1])
                  + Math.abs(analysis.linearRgb[targetOffset + 2] - residualRender[index * 4 + 2])
                ) / 3
              : 0.5;
            const edge = analysis.edgeStrength[index];
            const edgeWeight = edge / (edge + 0.025);
            const score = sourceAlpha * transmittance * transmittance
              * (0.35 + residual * 0.65)
              * (0.82 + edgeWeight * 0.18);
            if (score > selectedScore) {
              selectedScore = score;
              selectedIndex = index;
              selectedTransmittance = transmittance;
            }
          }
        }

        const inverseMeanWeight = 1 / Math.max(1e-6, cellMeanWeight);
        // The rear layer is a stable source-color closure, not another detail
        // optimizer. A complete cell mean avoids propagating one bright or
        // high-error source pixel into a whole opaque backcoat mark.
        color[0] = cellMean[0] * inverseMeanWeight;
        color[1] = cellMean[1] * inverseMeanWeight;
        color[2] = cellMean[2] * inverseMeanWeight;
        const edge = analysis.edgeStrength[selectedIndex];
        const edgeWeight = edge / (edge + 0.025);
        const [directionX, directionY] = fieldAt(
          analysis,
          width,
          height,
          centerX,
          centerY,
        );
        const cellDiagonal = Math.hypot(actualCellWidth, actualCellHeight);
        // Every cell corner lies at most 0.5*diagonal from the center. Even if
        // that vector is entirely on the short axis, q <= (0.5/0.58)^4 =
        // 0.552, safely inside the compact kernel's flat q<=0.84 interior.
        // The structure-oriented overlap also avoids exposing an axis-aligned
        // grid when the rear layer is visible between front marks.
        const sigmaLong = Math.max(0.85, cellDiagonal * 0.72);
        const sigmaShort = Math.max(0.65, cellDiagonal * 0.58);
        const axisX = centerX + directionX * sigmaLong;
        const axisY = centerY + directionY * sigmaLong;
        const opacity = 0.995;
        strokePlan.push({
          center_x: centerX,
          center_y: centerY,
          start_x: centerX,
          start_y: centerY,
          control_1_x: axisX,
          control_1_y: axisY,
          control_2_x: centerX,
          control_2_y: centerY,
          end_x: centerX,
          end_y: centerY,
          path_length_px: 0,
          half_width_px: sigmaShort,
          opacity,
          layer: 3,
          random,
          underpaint_splat: true,
          underpaint_chain: false,
          coverage_backcoat: true,
          underpaint_sigma_long_px: sigmaLong,
          underpaint_sigma_short_px: sigmaShort,
          coverage_cell_min_x: minimumX,
          coverage_cell_max_x: maximumX + 1,
          coverage_cell_min_y: minimumY,
          coverage_cell_max_y: maximumY + 1,
          residual_transmittance: selectedTransmittance,
          edge_strength: edgeWeight,
          color_r: linearToSignalSrgb(color[0]),
          color_g: linearToSignalSrgb(color[1]),
          color_b: linearToSignalSrgb(color[2]),
          paint_linear_r: color[0],
          paint_linear_g: color[1],
          paint_linear_b: color[2],
        });
        generated += 1;
      }
    }

    return {
      strokePlan,
      metadata: {
        requested_splats: requested,
        accepted_splats: strokePlan.length,
        underpaint_parent_count: strokePlan.length,
        representation,
        source_texture_used: false,
        residual_render_used: Boolean(residualRender),
        pigment_initialization: "cell-mean-linear-srgb",
        placement: "fixed-grid-source-colored-compact-brush-backcoat",
        coverage_kernel: "compact-quartic-flat-interior-v1",
        coverage_guarantee: "one-flat-interior-splat-per-grid-cell",
        geometry_trainable: false,
        pigment_trainable: false,
        opacity: 0.995,
        layer: 3,
      },
    };
  }

  function linearColorDistance(a, b) {
    const dr = a[0] - b[0];
    const dg = a[1] - b[1];
    const db = a[2] - b[2];
    return Math.sqrt(dr * dr * 0.24 + dg * dg * 0.56 + db * db * 0.20);
  }

  function placeTextureGuidedSeeds(
    image,
    analysis,
    width,
    height,
    layerIndex,
    requested,
    minimumDistance,
    seedBase,
  ) {
    const guide = analysis.textureGuide;
    const minimumCellDistance = minimumDistance * 0.68;
    const cellSize = Math.max(0.5, minimumCellDistance / Math.SQRT2);
    const cols = Math.max(1, Math.ceil(width / cellSize));
    const rows = Math.max(1, Math.ceil(height / cellSize));
    const grid = Array.from({ length: cols * rows }, () => []);
    const seeds = [];
    const maxLocalDistance = minimumDistance * 1.32;
    const scanRadius = Math.max(2, Math.ceil(maxLocalDistance / cellSize) + 1);
    const tryCandidate = (sequence, relaxed = false) => {
      const index = seedBase + sequence + 1;
      const x = fract(0.5 + index * 0.7548776662466927) * width;
      const y = fract(0.5 + index * 0.5698402909980532) * height;
      const pixel = Math.max(0, Math.min(height - 1, Math.floor(y))) * width
        + Math.max(0, Math.min(width - 1, Math.floor(x)));
      if ((image.alpha?.[pixel] ?? 1) < 0.05) return false;
      const textureScore = clamp01(guide.score[pixel]);
      const acceptance = 0.12 + 0.88 * Math.sqrt(textureScore);
      if (!relaxed && hash01(index * 7 + layerIndex * 977) > acceptance) return false;
      const localDistance = minimumDistance * (1.32 - textureScore * 0.64)
        * (relaxed ? 0.82 : 1);
      const gx = Math.max(0, Math.min(cols - 1, Math.floor(x / cellSize)));
      const gy = Math.max(0, Math.min(rows - 1, Math.floor(y / cellSize)));
      for (let oy = -scanRadius; oy <= scanRadius; oy += 1) {
        for (let ox = -scanRadius; ox <= scanRadius; ox += 1) {
          const cellX = gx + ox;
          const cellY = gy + oy;
          if (cellX < 0 || cellY < 0 || cellX >= cols || cellY >= rows) continue;
          for (const neighborIndex of grid[cellY * cols + cellX]) {
            const neighbor = seeds[neighborIndex];
            const separation = (localDistance + neighbor.localDistance) * 0.5;
            if (Math.hypot(neighbor.x - x, neighbor.y - y) < separation) return false;
          }
        }
      }
      const seed = {
        x,
        y,
        random: hash01(index * 17 + layerIndex * 131),
        textureScore,
        edgeScore: clamp01(guide.edgeScore?.[pixel] ?? textureScore),
        dabVisibility: clamp01(guide.dabVisibility[pixel]),
        darkFlatScore: clamp01(guide.darkFlat[pixel]),
        localDistance,
      };
      grid[gy * cols + gx].push(seeds.length);
      seeds.push(seed);
      return true;
    };

    const weightedAttempts = requested * 96;
    for (let attempt = 0; attempt < weightedAttempts && seeds.length < requested; attempt += 1) {
      tryCandidate(attempt, false);
    }
    // Keep the physical Splat budget exact even on nearly uniform images. The
    // fallback only relaxes separation; its quasi-random order still samples
    // high-score pixels first through the completed weighted pass above.
    const relaxedAttempts = requested * 128;
    for (let attempt = 0; attempt < relaxedAttempts && seeds.length < requested; attempt += 1) {
      tryCandidate(weightedAttempts + attempt, true);
    }
    return seeds;
  }

  function placeSeeds(image, analysis, width, height, layerIndex, requested, minimumDistance, seedBase) {
    if (analysis.textureGuide) {
      return placeTextureGuidedSeeds(
        image,
        analysis,
        width,
        height,
        layerIndex,
        requested,
        minimumDistance,
        seedBase,
      );
    }
    const cellSize = Math.max(0.75, minimumDistance / Math.SQRT2);
    const cols = Math.max(1, Math.ceil(width / cellSize));
    const rows = Math.max(1, Math.ceil(height / cellSize));
    const grid = new Int32Array(cols * rows).fill(-1);
    const seeds = [];
    const attempts = requested * 42;
    for (let attempt = 0; attempt < attempts && seeds.length < requested; attempt += 1) {
      const index = seedBase + attempt + 1;
      const x = fract(0.5 + index * 0.7548776662466927) * width;
      const y = fract(0.5 + index * 0.5698402909980532) * height;
      const pixel = Math.max(0, Math.min(height - 1, Math.floor(y))) * width
        + Math.max(0, Math.min(width - 1, Math.floor(x)));
      if ((image.alpha?.[pixel] ?? 1) < 0.05) continue;
      const density = 0.70 + analysis.confidence[pixel] * 0.30;
      if (hash01(index * 7 + layerIndex * 977) > density) continue;
      const gx = Math.max(0, Math.min(cols - 1, Math.floor(x / cellSize)));
      const gy = Math.max(0, Math.min(rows - 1, Math.floor(y / cellSize)));
      let valid = true;
      for (let oy = -2; oy <= 2 && valid; oy += 1) {
        for (let ox = -2; ox <= 2; ox += 1) {
          const cellX = gx + ox;
          const cellY = gy + oy;
          if (cellX < 0 || cellY < 0 || cellX >= cols || cellY >= rows) continue;
          const neighborIndex = grid[cellY * cols + cellX];
          if (neighborIndex < 0) continue;
          const neighbor = seeds[neighborIndex];
          if (Math.hypot(neighbor.x - x, neighbor.y - y) < minimumDistance) {
            valid = false;
            break;
          }
        }
      }
      if (!valid) continue;
      const seed = { x, y, random: hash01(index * 17 + layerIndex * 131) };
      grid[gy * cols + gx] = seeds.length;
      seeds.push(seed);
    }
    return seeds;
  }

  function traceHalf(image, analysis, width, height, seed, sign, distance, colorLimit, steps) {
    const points = [];
    const originColor = sampleLinearRgb(analysis, width, height, seed.x, seed.y, [0, 0, 0]);
    const probeColor = [0, 0, 0];
    let x = seed.x;
    let y = seed.y;
    let previousX = 0;
    let previousY = 0;
    const stepLength = distance / Math.max(1, steps);
    for (let step = 0; step < steps; step += 1) {
      let [dx, dy] = fieldAt(analysis, width, height, x, y, previousX, previousY);
      dx *= sign;
      dy *= sign;
      if (previousX || previousY) {
        if (dx * previousX + dy * previousY < 0) {
          dx = -dx;
          dy = -dy;
        }
      }
      const midpointX = x + dx * stepLength * 0.5;
      const midpointY = y + dy * stepLength * 0.5;
      let [nextX, nextY] = fieldAt(analysis, width, height, midpointX, midpointY, dx, dy);
      if (nextX * dx + nextY * dy < 0) {
        nextX = -nextX;
        nextY = -nextY;
      }
      const nx = x + nextX * stepLength;
      const ny = y + nextY * stepLength;
      if (nx < 0.5 || ny < 0.5 || nx >= width - 0.5 || ny >= height - 0.5) break;
      const pixel = Math.floor(ny) * width + Math.floor(nx);
      if ((image.alpha?.[pixel] ?? 1) < 0.05) break;
      sampleLinearRgb(analysis, width, height, nx, ny, probeColor);
      if (linearColorDistance(originColor, probeColor) > colorLimit) break;
      x = nx;
      y = ny;
      previousX = nextX;
      previousY = nextY;
      points.push([x, y]);
    }
    return points;
  }

  function makeStroke(image, analysis, width, height, seed, layer, layerIndex, shortSide) {
    const distance = Math.max(2, shortSide * layer.length * (0.82 + seed.random * 0.36));
    const steps = Math.max(4, Math.min(14, Math.round(distance / 1.6)));
    const backward = traceHalf(image, analysis, width, height, seed, -1, distance * 0.5, layer.colorLimit, steps);
    const forward = traceHalf(image, analysis, width, height, seed, 1, distance * 0.5, layer.colorLimit, steps);
    const points = backward.reverse();
    points.push([seed.x, seed.y], ...forward);
    if (points.length < 2) {
      const [dx, dy] = fieldAt(analysis, width, height, seed.x, seed.y);
      points.length = 0;
      points.push(
        [seed.x - dx, seed.y - dy],
        [seed.x + dx, seed.y + dy],
      );
    }
    const colorRadius = shortSide * (
      Number.isFinite(layer.colorRadius)
        ? layer.colorRadius
        : [0.0094, 0.0043, 0.0014][layerIndex]
    );
    const color = [0, 0, 0];
    const colorProbe = [0, 0, 0];
    const colorOffsets = [
      [0, 0],
      [-colorRadius, 0],
      [colorRadius, 0],
      [0, -colorRadius],
      [0, colorRadius],
    ];
    for (const [offsetX, offsetY] of colorOffsets) {
      sampleLinearRgb(
        analysis,
        width,
        height,
        seed.x + offsetX,
        seed.y + offsetY,
        colorProbe,
      );
      color[0] += colorProbe[0] / colorOffsets.length;
      color[1] += colorProbe[1] / colorOffsets.length;
      color[2] += colorProbe[2] / colorOffsets.length;
    }
    return {
      points,
      halfWidth: Math.max(0.7, shortSide * layer.halfWidth * (0.82 + seed.random * 0.36)),
      opacity: layer.opacity,
      random: seed.random,
      layerIndex,
      brightnessMin: layer.brightnessMin,
      brightnessMax: layer.brightnessMax,
      saturation: layer.saturation,
      crispOpaqueStroke: Boolean(layer.crispOpaqueStroke),
      color,
    };
  }

  function strokeLength(points) {
    let length = 0;
    for (let i = 1; i < points.length; i += 1) {
      length += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
    }
    return length;
  }

  function pointAtPathFraction(points, fraction) {
    if (points.length === 1) return [...points[0]];
    const total = strokeLength(points);
    if (!(total > 1e-6)) return [...points[0]];
    const target = Math.max(0, Math.min(1, fraction)) * total;
    let traversed = 0;
    for (let index = 1; index < points.length; index += 1) {
      const first = points[index - 1];
      const second = points[index];
      const segment = Math.hypot(second[0] - first[0], second[1] - first[1]);
      if (traversed + segment >= target) {
        const local = segment > 1e-6 ? (target - traversed) / segment : 0;
        return [
          first[0] + (second[0] - first[0]) * local,
          first[1] + (second[1] - first[1]) * local,
        ];
      }
      traversed += segment;
    }
    return [...points[points.length - 1]];
  }

  function cubicControlsForPath(points) {
    const p0 = [...points[0]];
    const p3 = [...points[points.length - 1]];
    const nearStart = pointAtPathFraction(points, 0.24);
    const nearEnd = pointAtPathFraction(points, 0.76);
    const chord = Math.max(1e-6, Math.hypot(p3[0] - p0[0], p3[1] - p0[1]));
    const startLength = Math.max(1e-6, Math.hypot(nearStart[0] - p0[0], nearStart[1] - p0[1]));
    const endLength = Math.max(1e-6, Math.hypot(p3[0] - nearEnd[0], p3[1] - nearEnd[1]));
    const handle = chord * 0.34;
    const p1 = [
      p0[0] + (nearStart[0] - p0[0]) / startLength * handle,
      p0[1] + (nearStart[1] - p0[1]) / startLength * handle,
    ];
    const p2 = [
      p3[0] - (p3[0] - nearEnd[0]) / endLength * handle,
      p3[1] - (p3[1] - nearEnd[1]) / endLength * handle,
    ];
    return [p0, p1, p2, p3];
  }

  function cubicPointAt(controls, t) {
    const [p0, p1, p2, p3] = controls;
    const s = 1 - t;
    return [
      s * s * s * p0[0] + 3 * s * s * t * p1[0] + 3 * s * t * t * p2[0] + t * t * t * p3[0],
      s * s * s * p0[1] + 3 * s * s * t * p1[1] + 3 * s * t * t * p2[1] + t * t * t * p3[1],
    ];
  }

  function sampledCubicArcLength(controls, segments = 24) {
    let previous = cubicPointAt(controls, 0);
    let length = 0;
    for (let index = 1; index <= segments; index += 1) {
      const point = cubicPointAt(controls, index / segments);
      length += Math.hypot(point[0] - previous[0], point[1] - previous[1]);
      previous = point;
    }
    return length;
  }

  function capCubicArcLength(controls, maximumArcPx) {
    const originalArcLength = sampledCubicArcLength(controls);
    if (!(maximumArcPx > 0) || originalArcLength <= maximumArcPx) {
      return { controls, originalArcLength, arcLength: originalArcLength, capped: false };
    }
    const center = cubicPointAt(controls, 0.5);
    const scale = maximumArcPx / Math.max(1e-6, originalArcLength);
    const cappedControls = controls.map((point) => [
      center[0] + (point[0] - center[0]) * scale,
      center[1] + (point[1] - center[1]) * scale,
    ]);
    return {
      controls: cappedControls,
      originalArcLength,
      arcLength: sampledCubicArcLength(cappedControls),
      capped: true,
    };
  }

  function sampleCubicRibbon(controls, segments = 16) {
    const points = [];
    for (let index = 0; index <= segments; index += 1) {
      points.push(cubicPointAt(controls, index / segments));
    }
    return points;
  }

  function strokePaintColor(stroke) {
    const brightness = stroke.brightnessMin
      + stroke.random * (stroke.brightnessMax - stroke.brightnessMin);
    const colorLuma = stroke.color[0] * 0.2126 + stroke.color[1] * 0.7152 + stroke.color[2] * 0.0722;
    const ribbonLumaOffset = stroke.cubicRibbon
      ? (stroke.random - 0.5) * (0.015 + (1 - clamp01(colorLuma)) * 0.075)
      : 0;
    return stroke.color.map((channel) => clamp01(
      (colorLuma + (channel - colorLuma) * stroke.saturation) * brightness + ribbonLumaOffset,
    ));
  }

  function rasterStroke(output, width, height, stroke) {
    const cubicControls = stroke.cubicRibbon
      ? stroke.cubicControls || cubicControlsForPath(stroke.points)
      : null;
    const rasterPoints = cubicControls ? sampleCubicRibbon(cubicControls) : stroke.points;
    const pathLength = Math.max(1e-6, strokeLength(rasterPoints));
    const cumulative = new Float32Array(rasterPoints.length);
    let minimumX = Infinity;
    let minimumY = Infinity;
    let maximumX = -Infinity;
    let maximumY = -Infinity;
    for (let i = 0; i < rasterPoints.length; i += 1) {
      minimumX = Math.min(minimumX, rasterPoints[i][0]);
      minimumY = Math.min(minimumY, rasterPoints[i][1]);
      maximumX = Math.max(maximumX, rasterPoints[i][0]);
      maximumY = Math.max(maximumY, rasterPoints[i][1]);
      if (i > 0) {
        cumulative[i] = cumulative[i - 1] + Math.hypot(
          rasterPoints[i][0] - rasterPoints[i - 1][0],
          rasterPoints[i][1] - rasterPoints[i - 1][1],
        );
      }
    }
    const baseColor = strokePaintColor(stroke);
    const padding = stroke.halfWidth + 1.5;
    const x0 = Math.max(0, Math.floor(minimumX - padding));
    const y0 = Math.max(0, Math.floor(minimumY - padding));
    const x1 = Math.min(width - 1, Math.ceil(maximumX + padding));
    const y1 = Math.min(height - 1, Math.ceil(maximumY + padding));
    for (let y = y0; y <= y1; y += 1) {
      for (let x = x0; x <= x1; x += 1) {
        const pixelX = x + 0.5;
        const pixelY = y + 0.5;
        let closestDistanceSquared = Infinity;
        let closestAlong = 0;
        let closestSide = 0;
        for (let segment = 1; segment < rasterPoints.length; segment += 1) {
          const a = rasterPoints[segment - 1];
          const b = rasterPoints[segment];
          const vx = b[0] - a[0];
          const vy = b[1] - a[1];
          const segmentLengthSquared = vx * vx + vy * vy;
          if (segmentLengthSquared < 1e-8) continue;
          const segmentLength = Math.sqrt(segmentLengthSquared);
          const localT = Math.max(0, Math.min(1, (
            (pixelX - a[0]) * vx + (pixelY - a[1]) * vy
          ) / segmentLengthSquared));
          const closestX = a[0] + vx * localT;
          const closestY = a[1] + vy * localT;
          const dx = pixelX - closestX;
          const dy = pixelY - closestY;
          const distanceSquared = dx * dx + dy * dy;
          if (distanceSquared < closestDistanceSquared) {
            closestDistanceSquared = distanceSquared;
            closestAlong = (cumulative[segment - 1] + localT * segmentLength) / pathLength;
            closestSide = (dx * -vy + dy * vx) / Math.max(1e-6, segmentLength);
          }
        }
        const pressure = stroke.cubicRibbon
          ? 0.34 + 0.66 * Math.sin(Math.PI * clamp01(closestAlong)) ** 0.58
          : 0.44 + 0.56 * Math.max(0, Math.sin(Math.PI * closestAlong)) ** 0.42;
        const halfWidth = stroke.halfWidth * pressure;
        const distance = Math.sqrt(closestDistanceSquared);
        if (distance > halfWidth + 0.75) continue;
        const side = closestSide / Math.max(0.5, halfWidth);
        const edgeSoftness = stroke.cubicRibbon ? 0.8 : stroke.crispOpaqueStroke ? 0.55 : 1.5;
        const edge = clamp01((halfWidth + edgeSoftness * 0.5 - distance) / edgeSoftness);
        const body = clamp01(1 - (distance / Math.max(0.75, halfWidth)) ** 2);
        const stripeCount = Math.max(1, Math.min(4, Math.round(stroke.halfWidth * 0.72)));
        const fiberPhase = stroke.crispOpaqueStroke
          ? side * Math.PI * stripeCount + closestAlong * 8 + stroke.random * TAU * 7
          : side * 23 + closestAlong * 17 + stroke.random * TAU * 7;
        const fiberWave = 0.5 + 0.5 * Math.sin(fiberPhase);
        const microWave = 0.5 + 0.5 * Math.sin(fiberPhase * 2.73 + 1.7);
        // Keep the mark itself visible even when adjacent strokes sample nearly
        // identical source colors. Wet-paint-flow gets much of its readable
        // brushwork from pigment pooling across each ribbon, not from the
        // later relief pass. This independently implemented profile applies
        // the same physical idea to the QA teacher: darker shoulders, a
        // loaded center, stable bristle bands, and sparse longitudinal pigment
        // breaks. It does not change the Brush Splat kernel or optimizer.
        const edgePool = Math.max(0, 1 - Math.min(1, Math.abs(side))) ** 0.52;
        const pigmentBreak = 0.955 + 0.045 * hash01(
          Math.floor(closestAlong * 72) + stroke.random * 8191 + stroke.layerIndex * 131,
        );
        const fibers = stroke.crispOpaqueStroke
          ? 0.76 + 0.20 * fiberWave + 0.04 * microWave
          : 0.88 + 0.09 * Math.sin(fiberPhase) + 0.035 * Math.sin(fiberPhase * 2.73 + 1.7);
        const ribbonLane = 0.5 + 0.5 * Math.sin(
          side * Math.PI * (2.5 + stroke.layerIndex * 1.5) + closestAlong * 11 + stroke.random * 37,
        );
        const ribbonGrain = 0.5 + 0.5 * Math.sin(
          side * 31 - closestAlong * 19 + stroke.random * 83,
        );
        const alpha = stroke.cubicRibbon
          ? clamp01(
              stroke.opacity * edge * (0.58 + 0.42 * body) *
              (0.52 + 0.38 * ribbonLane + 0.10 * ribbonGrain),
            )
          : stroke.crispOpaqueStroke
          ? clamp01(stroke.opacity * edge * (0.84 + 0.16 * body) * (0.88 + 0.12 * fiberWave))
          : clamp01(stroke.opacity * edge * (0.24 + 0.76 * body) * (0.86 + 0.14 * fibers));
        const target = (y * width + x) * 3;
        const pigmentScale = stroke.cubicRibbon
          ? (0.84 + 0.24 * edgePool) * (0.88 + 0.20 * ribbonLane + 0.04 * ribbonGrain)
          : stroke.crispOpaqueStroke
          ? (0.88 + 0.22 * edgePool) *
            (0.91 + 0.08 * fiberWave + 0.025 * microWave) *
            pigmentBreak
          : 0.90 + 0.16 * fibers;
        for (let channel = 0; channel < 3; channel += 1) {
          const pigment = clamp01(baseColor[channel] * pigmentScale);
          output[target + channel] += (pigment - output[target + channel]) * alpha;
        }
      }
    }
    return pathLength;
  }

  function createFlowPaintReference(image, options = {}) {
    const { width, height } = validateImage(image);
    const profile = referenceProfile(options);
    const planOnly = options.planOnly === true;
    const layers = profile.layers;
    const strength = Math.max(0, Math.min(1, Number(options.strength ?? 1)));
    const seed = Math.round(Number(options.seed ?? 240825));
    const textureGuidedAllocation = options.textureGuidedAllocation === true;
    const analysis = makeAnalysis(image, width, height, {
      textureGuide: textureGuidedAllocation,
    });
    const outputLinear = planOnly
      ? null
      : blurredUnderpaint(
          analysis,
          width,
          height,
          profile.underpaintScale,
          profile.canvasLinear,
        );
    const shortSide = Math.max(1, Math.min(width, height));
    const maximumRibbonArcFraction = Math.max(
      0,
      Math.min(0.5, Number(options.maximumRibbonArcFraction) || 0),
    );
    const maximumRibbonArcPx = shortSide * maximumRibbonArcFraction;
    const opaquePixels = image.alpha
      ? image.alpha.reduce((sum, alpha) => sum + (alpha >= 0.05 ? 1 : 0), 0)
      : width * height;
    const minimumStrokes = Math.max(1, Math.round(Number(options.minimumStrokes) || 256));
    const requestedLimit = Math.max(minimumStrokes, Math.round(Number(options.maxStrokes) || 14000));
    const requestedTotal = profile.fixedStrokeBudget
      ? requestedLimit
      : Math.min(requestedLimit, Math.round(opaquePixels * 0.097));
    const layerCounts = layers.map((layer, index) => (
      index === layers.length - 1
        ? 0
        : Math.round(requestedTotal * layer.share)
    ));
    layerCounts[layers.length - 1] = Math.max(0, requestedTotal - layerCounts[0] - layerCounts[1]);
    const summaries = [];
    const strokePlan = options.includeStrokePlan ? [] : null;
    let totalAccepted = 0;
    let totalLength = 0;
    let longCount = 0;
    let cappedRibbonCount = 0;
    let maximumOriginalRibbonArcPx = 0;
    let maximumFinalRibbonArcPx = 0;
    let textureScoreTotal = 0;
    let edgeScoreTotal = 0;
    let darkFlatParentCount = 0;
    for (let layerIndex = 0; layerIndex < layers.length; layerIndex += 1) {
      const layer = layers[layerIndex];
      const requested = layerCounts[layerIndex];
      const minimumDistance = Math.max(0.8, Math.sqrt(opaquePixels / Math.max(1, requested)) * 0.46);
      const seeds = placeSeeds(
        image,
        analysis,
        width,
        height,
        layerIndex,
        requested,
        minimumDistance,
        seed + layerIndex * 104729,
      );
      let layerLength = 0;
      let layerLong = 0;
      for (const placed of seeds) {
        const stroke = makeStroke(image, analysis, width, height, placed, layer, layerIndex, shortSide);
        stroke.cubicRibbon = Boolean(layer.cubicRibbon);
        if (stroke.cubicRibbon) {
          const capped = capCubicArcLength(
            cubicControlsForPath(stroke.points),
            maximumRibbonArcPx,
          );
          stroke.cubicControls = capped.controls;
          cappedRibbonCount += capped.capped ? 1 : 0;
          maximumOriginalRibbonArcPx = Math.max(maximumOriginalRibbonArcPx, capped.originalArcLength);
          maximumFinalRibbonArcPx = Math.max(maximumFinalRibbonArcPx, capped.arcLength);
        }
        const length = planOnly
          ? Math.max(1e-6, strokeLength(
              stroke.cubicRibbon
                ? sampleCubicRibbon(stroke.cubicControls || cubicControlsForPath(stroke.points))
                : stroke.points,
            ))
          : rasterStroke(outputLinear, width, height, stroke);
        if (strokePlan) {
          const controls = stroke.cubicControls || cubicControlsForPath(stroke.points);
          const paintColor = strokePaintColor(stroke);
          const first = controls[0];
          const last = controls[3];
          strokePlan.push({
            center_x: placed.x,
            center_y: placed.y,
            start_x: first[0],
            start_y: first[1],
            end_x: last[0],
            end_y: last[1],
            control_1_x: controls[1][0],
            control_1_y: controls[1][1],
            control_2_x: controls[2][0],
            control_2_y: controls[2][1],
            path_length_px: length,
            half_width_px: stroke.halfWidth,
            opacity: stroke.opacity,
            layer: layerIndex,
            random: stroke.random,
            texture_score: clamp01(placed.textureScore ?? 1),
            edge_score: clamp01(placed.edgeScore ?? placed.textureScore ?? 1),
            dab_visibility_score: clamp01(placed.dabVisibility ?? 1),
            dark_flat_score: clamp01(placed.darkFlatScore ?? 0),
            color_r: linearToSignalSrgb(stroke.color[0]),
            color_g: linearToSignalSrgb(stroke.color[1]),
            color_b: linearToSignalSrgb(stroke.color[2]),
            paint_linear_r: paintColor[0],
            paint_linear_g: paintColor[1],
            paint_linear_b: paintColor[2],
          });
        }
        layerLength += length;
        textureScoreTotal += clamp01(placed.textureScore ?? 1);
        edgeScoreTotal += clamp01(placed.edgeScore ?? placed.textureScore ?? 1);
        if (clamp01(placed.darkFlatScore ?? 0) >= 0.60) darkFlatParentCount += 1;
        if (length >= stroke.halfWidth * 2.25) layerLong += 1;
      }
      totalAccepted += seeds.length;
      totalLength += layerLength;
      longCount += layerLong;
      summaries.push({
        layer: layerIndex,
        requested,
        accepted: seeds.length,
        mean_path_length_px: seeds.length ? layerLength / seeds.length : 0,
        long_stroke_fraction: seeds.length ? layerLong / seeds.length : 0,
        nominal_half_width_px: shortSide * layer.halfWidth,
      });
    }

    const rgb = planOnly ? image.rgb : new Float32Array(width * height * 3);
    if (!planOnly) {
      for (let i = 0; i < width * height; i += 1) {
        for (let channel = 0; channel < 3; channel += 1) {
          const painted = outputLinear[i * 3 + channel];
          const mixed = analysis.linearRgb[i * 3 + channel]
            + (painted - analysis.linearRgb[i * 3 + channel]) * strength;
          rgb[i * 3 + channel] = linearToSignalSrgb(mixed);
        }
      }
    }
    return {
      image: {
        ...image,
        width,
        height,
        rgb,
        alpha: image.alpha || new Float32Array(width * height).fill(1),
      },
      metadata: {
        candidate: profile.candidate,
        enabled: true,
        qa_only: true,
        mode: profile.mode,
        source_copy: false,
        splat_primitive_changed: false,
        training_optimizer_changed: false,
        independent_primitive: Boolean(profile.independentPrimitive),
        primitive: profile.independentPrimitive ? "connected-cubic-ribbon" : "flow-polyline-reference",
        input_signal_space: "sRGB",
        mixing_space: "linear sRGB",
        output_signal_space: "sRGB",
        seed,
        strength,
        plan_only: planOnly,
        requested_strokes: requestedTotal,
        accepted_strokes: totalAccepted,
        mean_path_length_px: totalAccepted ? totalLength / totalAccepted : 0,
        long_stroke_fraction: totalAccepted ? longCount / totalAccepted : 0,
        maximum_ribbon_arc_fraction: maximumRibbonArcFraction,
        maximum_ribbon_arc_px: maximumRibbonArcPx,
        capped_ribbon_count: cappedRibbonCount,
        maximum_original_ribbon_arc_px: maximumOriginalRibbonArcPx,
        maximum_final_ribbon_arc_px: maximumFinalRibbonArcPx,
        source_texture_used: textureGuidedAllocation,
        texture_guided_allocation: textureGuidedAllocation,
        mean_parent_texture_score: totalAccepted
          ? textureScoreTotal / totalAccepted
          : 0,
        mean_parent_edge_score: totalAccepted
          ? edgeScoreTotal / totalAccepted
          : 0,
        dark_flat_parent_fraction: totalAccepted
          ? darkFlatParentCount / totalAccepted
          : 0,
        texture_guide: analysis.textureGuide?.summary || null,
        layers: summaries,
      },
      strokePlan,
      textureGuide: analysis.textureGuide,
    };
  }

  global.Image2SplatPaintFlowPaintReference = Object.freeze({
    createFlowPaintReference,
    createSplatUnderpaintPlan,
    buildTextureGuide,
    signalSrgbToLinear,
    linearToSignalSrgb,
  });
})(globalThis);
