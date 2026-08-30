import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

async function loadInstalledGlobal(path, name) {
  const context = vm.createContext({
    ArrayBuffer,
    DataView,
    Math,
    Number,
    Object,
    Set,
    TextEncoder,
    Uint8Array,
  });
  context.globalThis = context;
  vm.runInContext(await readFile(new URL(`../${path}`, import.meta.url), "utf8"), context, { filename: path });
  return context[name];
}

test("virtual-camera gradient balance keeps a weak front anchor at 50% sampling", async () => {
  const source = await readFile(new URL("../web/core/algorithms.js", import.meta.url), "utf8");
  const start = source.indexOf("function virtualCameraGradientBalance(");
  const end = source.indexOf("\nfunction virtualCameraShuffle(", start);
  assert.ok(start >= 0 && end > start);
  const context = vm.createContext({ Math, Number });
  vm.runInContext(`${source.slice(start, end)}\nthis.balance = virtualCameraGradientBalance;`, context);
  assert.equal(context.balance(0, 128).frontGradientAnchorWeight, 0);
  assert.equal(context.balance(32, 128).frontGradientAnchorWeight, 0.125);
  assert.equal(context.balance(64, 128).frontGradientAnchorWeight, 0.25);
  assert.equal(context.balance(96, 128).frontGradientAnchorWeight, 0.5);
  assert.equal(context.balance(128, 128).frontGradientAnchorWeight, 1);
});

test("numeric helpers preserve clamping and step normalization contracts", async () => {
  const numeric = await loadInstalledGlobal("web/core/numeric-utils.js", "Image2SplatPaintNumeric");
  assert.equal(numeric.clampNumber("12", 0, 10, 4), 10);
  assert.equal(numeric.clampNumber("invalid", 0, 10, 4), 4);
  assert.equal(numeric.normalizeStepInteger(18, { min: 1, max: 20, fallback: 5, step: 4 }), 17);
  assert.deepEqual(Array.from(numeric.hexColorToRgb("#ff8000")), [1, 128 / 255, 0]);
});

test("flow-painted Brush reference is deterministic, bounded, and keeps connected strokes", async () => {
  const context = vm.createContext({
    Float32Array,
    Int32Array,
    Math,
    Number,
    Object,
  });
  context.globalThis = context;
  vm.runInContext(
    await readFile(new URL("../web/training/flow-paint-reference.js", import.meta.url), "utf8"),
    context,
    { filename: "web/training/flow-paint-reference.js" },
  );
  const width = 48;
  const height = 32;
  const rgb = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 3;
      rgb[offset] = x / (width - 1);
      rgb[offset + 1] = 0.2 + 0.6 * y / (height - 1);
      rgb[offset + 2] = (x + y) % 9 < 4 ? 0.18 : 0.72;
    }
  }
  const input = { width, height, rgb, alpha: new Float32Array(width * height).fill(1) };
  const first = context.Image2SplatPaintFlowPaintReference.createFlowPaintReference(input, {
    seed: 77,
    maxStrokes: 512,
  });
  const second = context.Image2SplatPaintFlowPaintReference.createFlowPaintReference(input, {
    seed: 77,
    maxStrokes: 512,
  });
  assert.deepEqual(Array.from(first.image.rgb), Array.from(second.image.rgb));
  assert.ok(first.image.rgb.every((value) => Number.isFinite(value) && value >= 0 && value <= 1));
  assert.equal(first.metadata.splat_primitive_changed, false);
  assert.equal(first.metadata.training_optimizer_changed, false);
  assert.equal(first.metadata.mixing_space, "linear sRGB");
  assert.ok(first.metadata.accepted_strokes > 100);
  assert.ok(first.metadata.mean_path_length_px > 1.5);
  assert.ok(first.metadata.long_stroke_fraction > 0.5);

  const fine = context.Image2SplatPaintFlowPaintReference.createFlowPaintReference(input, {
    seed: 77,
    maxStrokes: 512,
    profile: "fine-layered-v2",
    includeStrokePlan: true,
  });
  assert.equal(fine.metadata.candidate, "BR-CAND-57");
  assert.equal(fine.metadata.mode, "oversampled-fine-flow-stroke-reference");
  assert.equal(fine.metadata.requested_strokes, 512);
  assert.ok(fine.metadata.accepted_strokes > 100);
  assert.ok(fine.metadata.layers[2].requested > fine.metadata.layers[1].requested);
  assert.ok(fine.metadata.layers[2].nominal_half_width_px < fine.metadata.layers[1].nominal_half_width_px);
  assert.ok(fine.metadata.layers[2].mean_path_length_px > fine.metadata.layers[2].nominal_half_width_px * 2.25);
  assert.equal(fine.strokePlan.length, fine.metadata.accepted_strokes);
  assert.ok(fine.strokePlan.every((stroke) => (
    Number.isFinite(
      stroke.center_x + stroke.center_y + stroke.path_length_px + stroke.half_width_px +
      stroke.color_r + stroke.color_g + stroke.color_b
    ) &&
    stroke.path_length_px > 0 && stroke.half_width_px > 0 && stroke.layer >= 0 && stroke.layer <= 2
  )));
  assert.ok(fine.strokePlan.every((stroke) => (
    stroke.color_r >= 0 && stroke.color_r <= 1 &&
    stroke.color_g >= 0 && stroke.color_g <= 1 &&
    stroke.color_b >= 0 && stroke.color_b <= 1
  )));
  assert.ok(fine.image.rgb.every((value) => Number.isFinite(value) && value >= 0 && value <= 1));

  const ribbon = context.Image2SplatPaintFlowPaintReference.createFlowPaintReference(input, {
    seed: 77,
    maxStrokes: 512,
    profile: "connected-ribbon-v1",
    includeStrokePlan: true,
  });
  assert.equal(ribbon.metadata.candidate, "BR-CAND-59");
  assert.equal(ribbon.metadata.independent_primitive, true);
  assert.equal(ribbon.metadata.primitive, "connected-cubic-ribbon");
  assert.equal(ribbon.metadata.splat_primitive_changed, false);
  assert.equal(ribbon.metadata.training_optimizer_changed, false);
  assert.equal(
    ribbon.metadata.layers.map((layer) => layer.requested).join(","),
    "102,205,205",
  );
  assert.equal(ribbon.strokePlan.length, ribbon.metadata.accepted_strokes);
  assert.ok(ribbon.strokePlan.every((stroke) => Number.isFinite(
    stroke.start_x + stroke.start_y + stroke.control_1_x + stroke.control_1_y +
    stroke.control_2_x + stroke.control_2_y + stroke.end_x + stroke.end_y,
  )));
  assert.ok(ribbon.image.rgb.every((value) => Number.isFinite(value) && value >= 0 && value <= 1));

  const ribbonPlanOnly = context.Image2SplatPaintFlowPaintReference.createFlowPaintReference(input, {
    seed: 77,
    maxStrokes: 512,
    profile: "connected-ribbon-v1",
    includeStrokePlan: true,
    planOnly: true,
  });
  assert.equal(ribbonPlanOnly.metadata.plan_only, true);
  assert.deepEqual(
    JSON.parse(JSON.stringify(ribbonPlanOnly.strokePlan)),
    JSON.parse(JSON.stringify(ribbon.strokePlan)),
  );

  const cappedRibbon = context.Image2SplatPaintFlowPaintReference.createFlowPaintReference(input, {
    seed: 77,
    maxStrokes: 512,
    profile: "connected-ribbon-v1",
    includeStrokePlan: true,
    maximumRibbonArcFraction: 0.10,
  });
  const maximumRibbonArcPx = Math.min(width, height) * 0.10;
  assert.equal(cappedRibbon.metadata.maximum_ribbon_arc_fraction, 0.10);
  assert.equal(cappedRibbon.metadata.maximum_ribbon_arc_px, maximumRibbonArcPx);
  assert.ok(cappedRibbon.metadata.capped_ribbon_count > 0);
  assert.ok(cappedRibbon.metadata.maximum_original_ribbon_arc_px > maximumRibbonArcPx);
  assert.ok(cappedRibbon.metadata.maximum_final_ribbon_arc_px <= maximumRibbonArcPx + 1e-5);
  assert.equal(cappedRibbon.strokePlan.length, ribbon.strokePlan.length);
  assert.ok(cappedRibbon.strokePlan.every((stroke) => stroke.path_length_px <= maximumRibbonArcPx + 0.03));
  assert.ok(cappedRibbon.image.rgb.every((value) => Number.isFinite(value) && value >= 0 && value <= 1));

  const compactChainPlan = context.Image2SplatPaintFlowPaintReference.createFlowPaintReference(input, {
    seed: 77,
    maxStrokes: 21,
    minimumStrokes: 1,
    profile: "connected-ribbon-v1",
    includeStrokePlan: true,
  });
  assert.equal(compactChainPlan.metadata.requested_strokes, 21);
  assert.ok(compactChainPlan.strokePlan.length > 0 && compactChainPlan.strokePlan.length <= 21);

  const allocationWidth = 64;
  const allocationHeight = 48;
  const allocationRgb = new Float32Array(allocationWidth * allocationHeight * 3);
  for (let y = 0; y < allocationHeight; y += 1) {
    for (let x = 0; x < allocationWidth; x += 1) {
      const offset = (y * allocationWidth + x) * 3;
      if (x < allocationWidth / 2) {
        allocationRgb[offset] = 0.035;
        allocationRgb[offset + 1] = 0.035;
        allocationRgb[offset + 2] = 0.035;
      } else {
        const value = ((x >> 2) + (y >> 2)) % 2 ? 0.16 : 0.84;
        allocationRgb[offset] = value;
        allocationRgb[offset + 1] = 0.22 + value * 0.68;
        allocationRgb[offset + 2] = 1 - value * 0.48;
      }
    }
  }
  const allocationInput = {
    width: allocationWidth,
    height: allocationHeight,
    rgb: allocationRgb,
    alpha: new Float32Array(allocationWidth * allocationHeight).fill(1),
  };
  const uniformAllocation = context.Image2SplatPaintFlowPaintReference.createFlowPaintReference(
    allocationInput,
    {
      seed: 77,
      maxStrokes: 96,
      minimumStrokes: 1,
      profile: "connected-ribbon-v1",
      includeStrokePlan: true,
      planOnly: true,
    },
  );
  const textureAllocation = context.Image2SplatPaintFlowPaintReference.createFlowPaintReference(
    allocationInput,
    {
      seed: 77,
      maxStrokes: 96,
      minimumStrokes: 1,
      profile: "connected-ribbon-v1",
      includeStrokePlan: true,
      planOnly: true,
      textureGuidedAllocation: true,
    },
  );
  const uniformDarkParents = uniformAllocation.strokePlan
    .filter((stroke) => stroke.center_x < allocationWidth / 2).length;
  const guidedDarkParents = textureAllocation.strokePlan
    .filter((stroke) => stroke.center_x < allocationWidth / 2).length;
  assert.equal(textureAllocation.strokePlan.length, uniformAllocation.strokePlan.length);
  assert.ok(guidedDarkParents < uniformDarkParents * 0.55);
  assert.equal(textureAllocation.metadata.source_texture_used, true);
  assert.ok(textureAllocation.metadata.texture_guide.dark_flat_pixel_fraction > 0.35);
  assert.ok(textureAllocation.strokePlan.every((stroke) => (
    stroke.texture_score >= 0 && stroke.texture_score <= 1
    && stroke.edge_score >= 0 && stroke.edge_score <= 1
    && stroke.dab_visibility_score >= 0 && stroke.dab_visibility_score <= 1
  )));
  assert.ok(textureAllocation.metadata.mean_parent_edge_score >= 0);

  const edgeWidth = 96;
  const edgeHeight = 64;
  const edgeRgb = new Float32Array(edgeWidth * edgeHeight * 3);
  for (let y = 0; y < edgeHeight; y += 1) {
    for (let x = 0; x < edgeWidth; x += 1) {
      let value = x < 48 ? 0.15 : 0.75;
      if (x >= 68 && x < 92 && y >= 12 && y < 52) {
        value = (x + y) % 2 ? 0.35 : 0.65;
      }
      const offset = (y * edgeWidth + x) * 3;
      edgeRgb[offset] = value;
      edgeRgb[offset + 1] = value;
      edgeRgb[offset + 2] = value;
    }
  }
  const edgeReference = context.Image2SplatPaintFlowPaintReference.createFlowPaintReference(
    {
      width: edgeWidth,
      height: edgeHeight,
      rgb: edgeRgb,
      alpha: new Float32Array(edgeWidth * edgeHeight).fill(1),
    },
    {
      seed: 77,
      maxStrokes: 128,
      minimumStrokes: 1,
      profile: "connected-ribbon-v1",
      includeStrokePlan: true,
      planOnly: true,
      textureGuidedAllocation: true,
    },
  );
  const meanGuideRegion = (field, x0, x1, y0, y1) => {
    let total = 0;
    let count = 0;
    for (let y = y0; y < y1; y += 1) {
      for (let x = x0; x < x1; x += 1) {
        total += field[y * edgeWidth + x];
        count += 1;
      }
    }
    return total / count;
  };
  const coherentBoundaryEdge = meanGuideRegion(
    edgeReference.textureGuide.edgeScore, 45, 51, 4, 60,
  );
  const checkerTextureEdge = meanGuideRegion(
    edgeReference.textureGuide.edgeScore, 70, 90, 14, 50,
  );
  const checkerTextureScore = meanGuideRegion(
    edgeReference.textureGuide.score, 70, 90, 14, 50,
  );
  assert.ok(coherentBoundaryEdge > checkerTextureEdge * 5);
  assert.ok(checkerTextureScore > 0.5);
  assert.equal(
    edgeReference.textureGuide.summary.edge_score_mode,
    "coherent-colour-plus-flow-xdog-78",
  );
  assert.equal(
    edgeReference.textureGuide.summary.flow_xdog.algorithm,
    "linear-srgb-flow-xdog-guide",
  );
  assert.ok(edgeReference.textureGuide.summary.flow_xdog.ridge_p96 > 0);
  const legacyGuide = context.Image2SplatPaintFlowPaintReference.buildTextureGuide(
    edgeRgb,
    new Float32Array(edgeWidth * edgeHeight).fill(0.5),
    new Float32Array(edgeWidth * edgeHeight),
    new Float32Array(edgeWidth * edgeHeight),
    edgeWidth,
    edgeHeight,
  );
  assert.equal(legacyGuide.width, edgeWidth);
  assert.equal(legacyGuide.height, edgeHeight);
  assert.equal(legacyGuide.summary.flow_xdog.algorithm, "disabled-no-flow-field");

  const flowUnderpaint = context.Image2SplatPaintFlowPaintReference.createSplatUnderpaintPlan(input, {
    count: 17,
    representation: "fused-ribbon",
    seed: 77,
  });
  assert.equal(flowUnderpaint.strokePlan.length, 17);
  assert.equal(flowUnderpaint.metadata.accepted_splats, 17);
  assert.equal(flowUnderpaint.metadata.source_texture_used, false);
  assert.equal(
    flowUnderpaint.metadata.coverage_guarantee,
    "one-flat-interior-splat-per-grid-cell",
  );
  assert.ok(flowUnderpaint.strokePlan.every((splat) => (
    splat.underpaint_splat === true && splat.underpaint_chain === false &&
    splat.coverage_backcoat === true && splat.layer === 3 &&
    splat.path_length_px === 0 && splat.half_width_px > 0 && splat.opacity === 0.995 &&
    [splat.paint_linear_r, splat.paint_linear_g, splat.paint_linear_b]
      .every((value) => Number.isFinite(value) && value >= 0 && value <= 1)
  )));
  assert.ok(flowUnderpaint.strokePlan.every((splat) => {
    const centerX = splat.start_x;
    const centerY = splat.start_y;
    const axisX = splat.control_1_x - centerX;
    const axisY = splat.control_1_y - centerY;
    const halfLong = Math.hypot(axisX, axisY);
    const tangentX = axisX / halfLong;
    const tangentY = axisY / halfLong;
    const normalX = -tangentY;
    const normalY = tangentX;
    const corners = [
      [splat.coverage_cell_min_x, splat.coverage_cell_min_y],
      [splat.coverage_cell_max_x, splat.coverage_cell_min_y],
      [splat.coverage_cell_min_x, splat.coverage_cell_max_y],
      [splat.coverage_cell_max_x, splat.coverage_cell_max_y],
    ];
    return corners.every(([x, y]) => {
      const dx = x - centerX;
      const dy = y - centerY;
      const along = (dx * tangentX + dy * tangentY) / halfLong;
      const side = (dx * normalX + dy * normalY) / splat.half_width_px;
      return along ** 4 + side ** 4 <= 0.84 + 1e-6;
    });
  }), "every grid-cell corner must remain inside the compact kernel's flat interior");

  const chainUnderpaint = context.Image2SplatPaintFlowPaintReference.createSplatUnderpaintPlan(input, {
    count: 5,
    representation: "curve-splat-chain",
    seed: 77,
  });
  assert.equal(chainUnderpaint.strokePlan.length, 5);
  assert.equal(chainUnderpaint.metadata.accepted_splats, 5);
  assert.ok(chainUnderpaint.strokePlan.every((splat) => (
    splat.underpaint_splat === true && splat.underpaint_chain === false &&
    splat.layer === 3 && splat.path_length_px === 0 &&
    splat.underpaint_sigma_long_px >= splat.underpaint_sigma_short_px
  )));

  const residualRender = new Float32Array(width * height * 4);
  residualRender.fill(0.25);
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    residualRender[pixel * 4 + 3] = pixel % width < width / 2 ? 0.95 : 0.05;
  }
  const residualUnderpaint = context.Image2SplatPaintFlowPaintReference.createSplatUnderpaintPlan(
    input,
    { count: 8, representation: "fused-ribbon", seed: 77, residualRender },
  );
  assert.equal(residualUnderpaint.metadata.residual_render_used, true);
  assert.equal(
    residualUnderpaint.metadata.placement,
    "fixed-grid-source-colored-compact-brush-backcoat",
  );
  assert.ok(residualUnderpaint.strokePlan.filter((splat) => splat.center_x < width / 2).length >= 4);
});

test("Flow Paint WebGPU training data keeps cubic parameters, linear target, and fine-first tiles", async () => {
  const context = vm.createContext({
    Array,
    ArrayBuffer,
    DataView,
    Float32Array,
    Math,
    Number,
    Object,
    Uint8Array,
    Uint32Array,
  });
  context.globalThis = context;
  vm.runInContext(
    await readFile(new URL("../web/training/flow-ribbon-trainer.js", import.meta.url), "utf8"),
    context,
    { filename: "web/training/flow-ribbon-trainer.js" },
  );
  const image = {
    width: 32,
    height: 16,
    rgb: new Float32Array(32 * 16 * 3).fill(0.5),
    alpha: new Float32Array(32 * 16).fill(1),
  };
  const stroke = (layer, offset) => ({
    start_x: 2 + offset,
    start_y: 4,
    control_1_x: 8 + offset,
    control_1_y: 2,
    control_2_x: 16 + offset,
    control_2_y: 8,
    end_x: 28 + offset,
    end_y: 6,
    half_width_px: 2,
    opacity: 0.75,
    random: 0.25 + layer * 0.1,
    layer,
    paint_linear_r: 0.2 + layer * 0.1,
    paint_linear_g: 0.3,
    paint_linear_b: 0.4,
  });
  const prepared = context.Image2SplatPaintFlowRibbonTrainer.prepareTrainingData(
    image,
    [stroke(0, 0), stroke(1, 0), stroke(2, 0)],
  );
  assert.equal(prepared.strokeCount, 3);
  assert.equal(prepared.params.length, 3 * 16);
  assert.ok(prepared.params.every(Number.isFinite));
  assert.ok(prepared.params[15] >= 0.5, "curve sample spacing must be stored for standard Splat chains");
  assert.ok(prepared.target[0] > 0.21 && prepared.target[0] < 0.22);
  assert.equal(prepared.canvasLinear.length, 3);
  assert.ok(prepared.canvasLinear.every((value) => value > 0.21 && value < 0.22));
  assert.equal(prepared.tileOffsets.length, prepared.tileCols * prepared.tileRows + 1);
  assert.ok(prepared.tileIndices.length > 0);
  const firstNonEmptyTile = Array.from(prepared.tileOffsets).findIndex((offset, index, offsets) => (
    index < offsets.length - 1 && offsets[index + 1] > offset
  ));
  assert.ok(firstNonEmptyTile >= 0);
  assert.equal(prepared.tileIndices[prepared.tileOffsets[firstNonEmptyTile]], 2);
  assert.deepEqual(Array.from(prepared.params), Array.from(prepared.anchors));

  const preparedOpaque = context.Image2SplatPaintFlowRibbonTrainer.prepareTrainingData(
    image,
    [stroke(0, 0), stroke(1, 0), stroke(2, 0)],
    { fixedOpacity: 0.995 },
  );
  for (let index = 0; index < preparedOpaque.strokeCount; index += 1) {
    const opacity = 1 / (1 + Math.exp(-preparedOpaque.params[index * 16 + 9]));
    assert.ok(Math.abs(opacity - 0.995) < 1e-6);
  }

  const preparedChain = context.Image2SplatPaintFlowRibbonTrainer.prepareTrainingData(
    image,
    [stroke(0, 0), stroke(1, 0), stroke(2, 0)],
    { representation: "curve-splat-chain" },
  );
  assert.ok(preparedChain.tileIndices.length > prepared.tileIndices.length);
  const splatsPerChain = context.Image2SplatPaintFlowRibbonTrainer.constants.CURVE_SAMPLES;
  assert.equal(splatsPerChain, 4);
  assert.equal(preparedChain.sampleCount, 4);
  assert.ok(preparedChain.tileIndices.every((candidate) => candidate < 3 * splatsPerChain));
  const firstChainTile = Array.from(preparedChain.tileOffsets).findIndex((offset, index, offsets) => (
    index < offsets.length - 1 && offsets[index + 1] > offset
  ));
  const firstChainStart = preparedChain.tileOffsets[firstChainTile];
  const firstChainEnd = preparedChain.tileOffsets[firstChainTile + 1];
  const firstChainParents = Array.from(
    preparedChain.tileIndices.slice(firstChainStart, firstChainEnd),
    (candidate) => Math.floor(candidate / splatsPerChain),
  );
  assert.equal(firstChainParents[0], Math.max(...firstChainParents));

  const preparedBrushDabs = context.Image2SplatPaintFlowRibbonTrainer.prepareTrainingData(
    image,
    [stroke(0, 0), stroke(1, 0), stroke(2, 0)],
    { representation: "curve-splat-chain", brushDabs: true, fixedOpacity: 0.995 },
  );
  assert.equal(
    preparedBrushDabs.sampleCount,
    context.Image2SplatPaintFlowRibbonTrainer.constants.BRUSH_DAB_SAMPLES,
  );
  assert.equal(preparedBrushDabs.sampleCount, 8);
  assert.equal(preparedBrushDabs.physicalSplatCount, preparedChain.physicalSplatCount * 2);
  assert.equal(preparedBrushDabs.bristleBundle, false);
  assert.equal(preparedBrushDabs.brushDabs, true);
  assert.equal(
    preparedBrushDabs.strokeTextureMode,
    context.Image2SplatPaintFlowRibbonTrainer.constants.STROKE_TEXTURE_BRUSH_DABS,
  );
  assert.ok(
    preparedBrushDabs.params[15] > preparedChain.params[15],
    "two-segment brush bundles need wider longitudinal support than four line samples",
  );
  for (let index = 0; index < preparedBrushDabs.strokeCount; index += 1) {
    const opacity = 1 / (1 + Math.exp(-preparedBrushDabs.params[index * 16 + 9]));
    assert.ok(Math.abs(opacity - 0.995) < 1e-6);
  }
  const preparedVariedBrushDabs = context.Image2SplatPaintFlowRibbonTrainer.prepareTrainingData(
    image,
    [stroke(0, 0), stroke(1, 0), stroke(2, 0)],
    {
      representation: "curve-splat-chain",
      brushDabs: true,
      textureGuidedDabs: true,
      fixedOpacity: 0.995,
      splatSizeVariation: 0.4,
    },
  );
  assert.equal(preparedVariedBrushDabs.splatSizeVariation, 0.4);
  assert.equal(
    preparedVariedBrushDabs.initialPhysicalSplatScaleStats.variation_percent,
    40,
  );
  assert.equal(
    preparedVariedBrushDabs.initialPhysicalSplatScaleStats.physical_splat_count,
    preparedBrushDabs.physicalSplatCount,
  );
  assert.ok(
    preparedVariedBrushDabs.initialPhysicalSplatScaleStats.half_long_px.standard_deviation
      > preparedBrushDabs.initialPhysicalSplatScaleStats.half_long_px.standard_deviation * 2,
    "physical Splat length variation must materially widen the scale distribution",
  );
  assert.ok(
    preparedVariedBrushDabs.initialPhysicalSplatScaleStats.aspect_ratio.maximum
      > preparedBrushDabs.initialPhysicalSplatScaleStats.aspect_ratio.maximum * 1.35,
    "thin-long physical Splats must remain visible at the 300% parent-width default",
  );
  assert.ok(
    Math.abs(
      preparedVariedBrushDabs.initialPhysicalSplatScaleStats.half_short_px.mean
        - preparedBrushDabs.initialPhysicalSplatScaleStats.half_short_px.mean,
    ) < preparedBrushDabs.initialPhysicalSplatScaleStats.half_short_px.mean * 0.02,
    "size variation must not make the complete physical-Splat population broader",
  );

  const guidedDabStrokes = [
    {
      ...stroke(1, 0), texture_score: 0.02, edge_score: 0.03, dab_visibility_score: 0.01,
    },
    {
      ...stroke(2, 0), texture_score: 0.98, edge_score: 0.97, dab_visibility_score: 0.96,
    },
  ];
  const preparedGuidedBrushDabs = context.Image2SplatPaintFlowRibbonTrainer.prepareTrainingData(
    image,
    guidedDabStrokes,
    {
      representation: "curve-splat-chain",
      brushDabs: true,
      textureGuidedDabs: true,
      fixedOpacity: 0.995,
    },
  );
  assert.equal(preparedGuidedBrushDabs.textureGuidedDabs, true);
  assert.equal(preparedGuidedBrushDabs.physicalSplatCount, 16);
  const firstGuideBucket = Math.floor(preparedGuidedBrushDabs.params[15] / 256);
  const secondGuideBucket = Math.floor(preparedGuidedBrushDabs.params[16 + 15] / 256);
  assert.ok(Math.floor(firstGuideBucket / 4) <= 1);
  assert.equal(firstGuideBucket % 4, 0);
  assert.ok(Math.floor(secondGuideBucket / 4) >= 60);
  assert.equal(secondGuideBucket % 4, 3);
  assert.ok(Math.abs(
    preparedGuidedBrushDabs.params[15] % 256 - preparedBrushDabs.params[15]
  ) < 1e-4);
  assert.ok(preparedGuidedBrushDabs.tileIndices.length > 0);
  assert.ok(preparedGuidedBrushDabs.tileIndices.length <= preparedBrushDabs.tileIndices.length);

  const edgeSeparatedOptions = {
    representation: "curve-splat-chain",
    brushDabs: true,
    textureGuidedDabs: true,
    fixedOpacity: 0.995,
    splatSizeVariation: 0.4,
  };
  const preparedTextureOnlyDabs = context.Image2SplatPaintFlowRibbonTrainer.prepareTrainingData(
    image,
    [{ ...stroke(2, 0), texture_score: 1, edge_score: 0, dab_visibility_score: 1 }],
    edgeSeparatedOptions,
  );
  const preparedEdgeDabs = context.Image2SplatPaintFlowRibbonTrainer.prepareTrainingData(
    image,
    [{ ...stroke(2, 0), texture_score: 1, edge_score: 1, dab_visibility_score: 1 }],
    edgeSeparatedOptions,
  );
  assert.ok(
    preparedEdgeDabs.initialPhysicalSplatScaleStats.aspect_ratio.mean
      > preparedTextureOnlyDabs.initialPhysicalSplatScaleStats.aspect_ratio.mean * 1.3,
    "thin-long physical Splats must respond to source edges rather than texture alone",
  );

  const priorityImage = {
    width: 64,
    height: 64,
    rgb: new Float32Array(64 * 64 * 3).fill(0.5),
    alpha: new Float32Array(64 * 64).fill(1),
  };
  const residualRender = new Float32Array(priorityImage.width * priorityImage.height * 4);
  for (let pixel = 0; pixel < priorityImage.width * priorityImage.height; pixel += 1) {
    residualRender[pixel * 4] = pixel % priorityImage.width < priorityImage.width / 2
      ? 0
      : 0.214;
    residualRender[pixel * 4 + 3] = pixel % priorityImage.width < priorityImage.width / 2
      ? 0.8
      : 0.1;
  }
  const preparedResidualPriority = context.Image2SplatPaintFlowRibbonTrainer.prepareTrainingData(
    priorityImage,
    [stroke(0, 0), stroke(1, 0), stroke(2, 0)],
    {
      representation: "curve-splat-chain",
      residualPriorityTileSampling: true,
      residualRender,
      tileSampleStride: 8,
    },
  );
  assert.equal(preparedResidualPriority.residualPriorityTileSampling, true);
  assert.equal(
    preparedResidualPriority.tileSamplingMasks.length,
    preparedResidualPriority.tileCols * preparedResidualPriority.tileRows,
  );
  assert.equal(
    preparedResidualPriority.residualPriorityActivationCount,
    preparedResidualPriority.tileSamplingMasks.length,
  );
  assert.ok(preparedResidualPriority.tileSamplingMasks.some((mask) => mask === 0));
  assert.ok(preparedResidualPriority.tileSamplingMasks.some((mask) => (mask & (mask - 1)) !== 0));

  const preparedFineBristles = context.Image2SplatPaintFlowRibbonTrainer.prepareTrainingData(
    image,
    [stroke(0, 0), stroke(1, 0), stroke(2, 0)],
    { representation: "curve-splat-chain", bristleBundle: true },
  );
  assert.equal(
    context.Image2SplatPaintFlowRibbonTrainer.constants.BRISTLE_BUNDLE_SAMPLES,
    4,
  );
  assert.equal(preparedFineBristles.sampleCount, preparedChain.sampleCount);
  assert.equal(preparedFineBristles.physicalSplatCount, preparedChain.physicalSplatCount);
  assert.equal(preparedFineBristles.bristleBundle, true);
  assert.ok(
    preparedFineBristles.params[2 * 16 + 15] > preparedChain.params[2 * 16 + 15],
    "fine-layer bristle pairs use two longitudinal positions while keeping four physical Splats",
  );

  const baseSplat = {
    ...stroke(0, 0),
    start_x: 16,
    start_y: 8,
    control_1_x: 16,
    control_1_y: 8,
    control_2_x: 16,
    control_2_y: 8,
    end_x: 16,
    end_y: 8,
    half_width_px: 8,
    layer: 3,
    underpaint_splat: true,
  };
  const preparedHybrid = context.Image2SplatPaintFlowRibbonTrainer.prepareTrainingData(
    image,
    [baseSplat, stroke(0, 0), stroke(1, 0), stroke(2, 0)],
  );
  assert.equal(preparedHybrid.physicalSplatCount, 4);
  assert.equal(preparedHybrid.underpaintParentCount, 1);
  assert.equal(preparedHybrid.underpaintSplatCount, 1);
  assert.equal(preparedHybrid.params[14], 3);

  const preparedChainUnderpaint = context.Image2SplatPaintFlowRibbonTrainer.prepareTrainingData(
    image,
    [baseSplat, stroke(0, 0), stroke(1, 0)],
    { representation: "curve-splat-chain" },
  );
  assert.equal(preparedChainUnderpaint.physicalSplatCount, 9);
  assert.equal(preparedChainUnderpaint.underpaintParentCount, 1);
  assert.equal(preparedChainUnderpaint.underpaintSplatCount, 1);
  assert.ok(preparedChainUnderpaint.tileIndices.some((candidate) => candidate === 0));

  const warmParams = prepared.params.slice();
  warmParams[0] += 1.25;
  const resumed = context.Image2SplatPaintFlowRibbonTrainer.prepareTrainingData(
    image,
    [baseSplat, stroke(0, 0), stroke(1, 0), stroke(2, 0)],
    { initialDetailParams: warmParams, initialDetailOffset: 1 },
  );
  assert.equal(resumed.params[16], warmParams[0]);
  assert.deepEqual(Array.from(resumed.params), Array.from(resumed.anchors));

  const borderRgb = new Float32Array(12 * 12 * 3).fill(1);
  for (let y = 0; y < 12; y += 1) {
    for (let x = 0; x < 12; x += 1) {
      if (x > 0 && x < 11 && y > 0 && y < 11) continue;
      borderRgb.fill(0.1, (y * 12 + x) * 3, (y * 12 + x + 1) * 3);
    }
  }
  const borderPrepared = context.Image2SplatPaintFlowRibbonTrainer.prepareTrainingData(
    { width: 12, height: 12, rgb: borderRgb, alpha: new Float32Array(144).fill(1) },
    [stroke(0, 0)],
  );
  assert.ok(
    borderPrepared.canvasLinear.every((value) => value < 0.02),
    "adaptive canvas must follow the source border rather than a bright image center",
  );
});

test("adaptive Flow stroke topology varies initialization and performs bounded split / merge events", async () => {
  const context = vm.createContext({
    Float32Array,
    Map,
    Math,
    Number,
    Object,
    Set,
  });
  context.globalThis = context;
  vm.runInContext(
    await readFile(new URL("../web/training/flow-stroke-topology.js", import.meta.url), "utf8"),
    context,
    { filename: "web/training/flow-stroke-topology.js" },
  );
  const width = 64;
  const height = 32;
  const rgb = new Float32Array(width * height * 3).fill(0.5);
  const image = { width, height, rgb, alpha: new Float32Array(width * height).fill(1) };
  const stroke = (index) => ({
    start_x: 2 + (index % 8) * 7,
    start_y: 5 + Math.floor(index / 8) * 7,
    control_1_x: 4 + (index % 8) * 7,
    control_1_y: 5 + Math.floor(index / 8) * 7,
    control_2_x: 7 + (index % 8) * 7,
    control_2_y: 5 + Math.floor(index / 8) * 7,
    end_x: 9 + (index % 8) * 7,
    end_y: 5 + Math.floor(index / 8) * 7,
    half_width_px: 2,
    opacity: 0.75,
    random: (index + 1) / 37,
    layer: 1,
    paint_linear_r: 0.3,
    paint_linear_g: 0.4,
    paint_linear_b: 0.5,
  });
  const referencePlan = Array.from({ length: 32 }, (_, index) => stroke(index));
  const topology = context.Image2SplatPaintFlowStrokeTopology;
  assert.equal(topology.constants.splitFraction, 0.04);
  assert.equal(topology.constants.mergeFraction, 0.01);
  assert.equal(topology.constants.maximumSplitsPerEvent, 24);
  assert.equal(topology.constants.splitApplyUntil, 0.75);
  const initialized = topology.initialize(referencePlan, 8, {
    minimumWidthFactor: 0.55,
    maximumWidthFactor: 1.65,
    frontWidthMaximumFactor: 3,
    fixedOpacity: 0.995,
  });
  assert.equal(initialized.plan.length, 8);
  assert.ok(initialized.initialDistribution.width_px.minimum < initialized.initialDistribution.width_px.maximum);
  assert.equal(initialized.initialDistribution.opacity.minimum, 0.995);
  assert.equal(initialized.initialDistribution.opacity.maximum, 0.995);

  const layeredReferencePlan = Array.from({ length: 30 }, (_, index) => ({
    ...stroke(index),
    layer: Math.floor(index / 10),
  }));
  const opaqueLayered = topology.initialize(layeredReferencePlan, 30, {
    minimumWidthFactor: 0.55,
    maximumWidthFactor: 1.65,
    frontWidthMaximumFactor: 3,
    fixedOpacity: 0.995,
    paintCurriculumEnabled: false,
  });
  for (const layer of [0, 1, 2]) {
    assert.equal(opaqueLayered.initialDistribution.opacity_by_layer[layer].minimum, 0.995);
    assert.equal(opaqueLayered.initialDistribution.opacity_by_layer[layer].maximum, 0.995);
  }
  const ordinaryWidthMaximum = Math.max(...opaqueLayered.plan
    .filter((item) => item.layer === 1)
    .map((item) => item.half_width_px));
  const frontWidthMaximum = Math.max(...opaqueLayered.plan
    .filter((item) => item.layer === 2)
    .map((item) => item.half_width_px));
  assert.ok(frontWidthMaximum > ordinaryWidthMaximum);

  const coarseCurriculum = topology.initialize(referencePlan, 8, {
    imageLongSide: width,
    paintCurriculumEnabled: true,
    startingWidthDivisor: 8,
    startingLengthPercent: 160,
    curriculumProgress: 0,
    maximumCurveArcPx: 64,
  });
  const settledCurriculum = topology.initialize(referencePlan, 8, {
    imageLongSide: width,
    paintCurriculumEnabled: true,
    startingWidthDivisor: 8,
    startingLengthPercent: 160,
    curriculumProgress: 1,
    maximumCurveArcPx: 64,
  });
  assert.ok(coarseCurriculum.initialDistribution.width_px.mean
    > settledCurriculum.initialDistribution.width_px.mean);
  assert.ok(coarseCurriculum.initialDistribution.arc_px.mean
    > settledCurriculum.initialDistribution.arc_px.mean);

  const params = new Float32Array(initialized.plan.length * 16);
  for (let index = 0; index < initialized.plan.length; index += 1) {
    const item = initialized.plan[index];
    const base = index * 16;
    params.set([
      item.start_x, item.start_y,
      item.control_1_x, item.control_1_y,
      item.control_2_x, item.control_2_y,
      item.end_x, item.end_y,
      item.half_width_px,
      Math.log(item.opacity / (1 - item.opacity)),
      item.paint_linear_r, item.paint_linear_g, item.paint_linear_b,
      item.random, item.layer, 2,
    ], base);
  }
  const rendered = new Float32Array(width * height * 4);
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    rendered[pixel * 4 + 3] = 0.35;
  }
  const evolved = topology.evolve(
    initialized,
    params,
    image,
    rendered,
    16,
    referencePlan,
    {
      splitFraction: 0.25,
      mergeFraction: 0.25,
      maximumSplitsPerEvent: 4,
      maximumMergesPerEvent: 2,
      curriculumProgress: 0.5,
      maximumCurveArcPx: 64,
      maximumWidthPx: 16,
    },
  );
  const explicitBaseline = topology.evolve(
    initialized,
    params,
    image,
    rendered,
    16,
    referencePlan,
    {
      splitFraction: 0.25,
      mergeFraction: 0.25,
      maximumSplitsPerEvent: 4,
      maximumMergesPerEvent: 2,
      curriculumProgress: 0.5,
      maximumCurveArcPx: 64,
      maximumWidthPx: 16,
      scaleMatchedResidualRepaint: false,
    },
  );
  assert.equal(
    JSON.stringify(explicitBaseline),
    JSON.stringify(evolved),
    "the default-OFF candidate must preserve the previous topology result exactly",
  );
  assert.equal(evolved.plan.length, 16);
  assert.ok(evolved.totals.splits >= 1);
  assert.ok(evolved.totals.merges >= 1);
  assert.ok(evolved.totals.sourceAdded > initialized.totals.sourceAdded);
  assert.equal(evolved.events.length, 1);
  assert.equal(evolved.events[0].count_after, 16);
  assert.ok(Number.isFinite(evolved.events[0].residual_move_count));
  assert.ok(Number.isFinite(evolved.events[0].curriculum_mean_arc_px));
  assert.ok(Number.isFinite(evolved.events[0].distribution.center_displacement_px.mean));
  assert.ok(evolved.plan.filter((item) => item.topology_kind === "split-fork").length >= 2);
  assert.ok(evolved.plan.every((item) => (
    Number.isFinite(item.start_x + item.end_y + item.half_width_px + item.opacity)
    && item.half_width_px >= 0.55
    && item.opacity === 0.995
  )));

  const scaleMatched = topology.evolve(
    initialized,
    params,
    image,
    rendered,
    16,
    referencePlan,
    {
      splitFraction: 0.25,
      mergeFraction: 0.25,
      maximumSplitsPerEvent: 4,
      maximumMergesPerEvent: 2,
      maximumCurveArcPx: 64,
      maximumWidthPx: 16,
      curriculumProgress: 0.25,
      residualMovePerEventPx: 1.5,
      maximumTotalMovementPx: 12,
      scaleMatchedResidualRepaint: true,
    },
  );
  assert.equal(scaleMatched.events[0].scale_matched_residual_repaint, true);
  assert.ok(scaleMatched.events[0].scale_matched_residual_radius_px > 0);
  assert.equal(scaleMatched.plan.length, evolved.plan.length);

  const afterSplitWindow = topology.evolve(
    initialized,
    params,
    image,
    rendered,
    16,
    referencePlan,
    {
      splitFraction: 0.25,
      mergeFraction: 0,
      maximumSplitsPerEvent: 4,
      splitApplyUntil: 0.75,
      curriculumProgress: 0.80,
      maximumCurveArcPx: 64,
      maximumWidthPx: 16,
    },
  );
  assert.equal(afterSplitWindow.events[0].split_enabled, false);
  assert.equal(afterSplitWindow.events[0].split_count, 0);
  assert.equal(afterSplitWindow.plan.length, 16);
});

test("image metadata preserves dimensions and EXIF axis semantics", async () => {
  const metadata = await loadInstalledGlobal("web/input/image-metadata.js", "Image2SplatPaintImageMetadata");
  assert.equal(metadata.orientationSwapsImageAxes(6), true);
  assert.equal(metadata.orientationSwapsImageAxes(3), false);
  assert.deepEqual(Array.from(metadata.displayOrientedImageSize(4032, 3024, 6)), [3024, 4032]);

  const png = new Uint8Array(24);
  png.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const view = new DataView(png.buffer);
  view.setUint32(16, 2048);
  view.setUint32(20, 1024);
  assert.deepEqual(
    { ...metadata.parseImageDimensions(png, "image/png") },
    { width: 2048, height: 1024, format: "png" },
  );
});

test("PLY serializer preserves frame, standard-alpha header, and 17-float rows", async () => {
  const serializer = await loadInstalledGlobal("web/export/ply-serializer.js", "Image2SplatPaintPlySerializer");
  assert.deepEqual(
    { ...serializer.plyFrameScale({ width: 800, height: 400 }) },
    { x: 1, y: 0.5, width: 800, height: 400, aspect: 2 },
  );
  const header = serializer.createPlyHeader({
    count: 1,
    image: { width: 800, height: 400 },
    boundarySigma: 3,
    layerOrderEnabled: true,
    layerDepthSpan: 0.01,
  });
  assert.match(header, /comment image2gaussianpaint_blend standard_alpha/);
  assert.equal((header.match(/^property float /gm) || []).length, 17);

  const rowBytes = 17 * 4;
  const result = serializer.serializeBinaryPly({
    header,
    count: 1,
    rowBytes,
    shC0: 0.28209479177387814,
    geometryAt: () => ({ x: 0, y: 0, sx: 0.1, sy: 0.05, theta: 0 }),
    depthAt: () => 0,
    rgbAt: () => [0.5, 0.5, 0.5],
    opacityAt: () => 0.5,
    logit: (value) => Math.log(value / (1 - value)),
  });
  assert.equal(result.byteLength, new TextEncoder().encode(header).byteLength + rowBytes);
});

test("GPU profile reducer preserves histogram percentile bins", async () => {
  const metrics = await loadInstalledGlobal("web/gpu/metrics.js", "Image2SplatPaintGpuMetrics");
  const summary = metrics.profileDistributionSummary([1, 1, 2, 0, 0, 0, 0, 0], 4, 8);
  assert.equal(summary.mean, 2);
  assert.equal(summary.p50_bin, "1");
  assert.equal(summary.p90_bin, "2-3");
  assert.equal(summary.histogram["2-3"], 2);
});

test("WGSL factories preserve byte-stable tile and optimizer reset sources", async () => {
  const context = vm.createContext({
    ILLUSTRATIVE_OIL_WGSL: "IO",
    MIP_PIXEL_SIGMA: 0.5,
    Object,
    RECTANGLE_TRAPEZOID_WGSL: "RT",
    RENDER_SIGMA: 4,
    TILE_SIZE: 16,
    VIRTUAL_TILT_WGSL: "VT",
  });
  context.globalThis = context;
  for (const path of ["web/gpu/shaders/tile-pipeline.js", "web/gpu/shaders/optimizer-reset.js"]) {
    vm.runInContext(await readFile(new URL(`../${path}`, import.meta.url), "utf8"), context, { filename: path });
  }
  const tile = context.Image2SplatPaintTilePipelineShader.create({
    opacitySupportDeclaration: "OD",
    opacitySupportFunction: "OF",
    paintSupportFunctions: "PF",
    supportSigmaExpression: "SS",
    exactTileIntersectionFunction: "EI",
    opacitySupportEarlyExit: "OE",
    exactTileIntersectionGuard: "EG",
    sortTilesFunction: "ST",
  });
  const optimizerReset = context.Image2SplatPaintOptimizerResetShader.create();
  assert.equal(tile.length, 6317);
  assert.equal(createHash("sha256").update(tile).digest("hex"), "d3c6136fb57c6f0f005de5c2e110bdbbcfe8ce5a26f084c878db86d3d92e29ba");
  assert.equal(optimizerReset.length, 1866);
  assert.equal(createHash("sha256").update(optimizerReset).digest("hex"), "946e15a4b94356a399ddae7ea2b48da581472575ec70484f897d31d0672ca2d1");
});

test("preview WGSL factories preserve generated source fingerprints", async () => {
  const context = vm.createContext({
    ILLUSTRATIVE_OIL_WGSL: "IO",
    LAYERED_OPAQUE_BRUSH_EDGE_SOFTNESS: 2,
    LAYERED_OPAQUE_BRUSH_KERNEL_EXTENT: 1,
    LAYER_CODE_RANGE: 6,
    MAX_DISCRETE_LAYER_COUNT: 8,
    MIN_DISCRETE_LAYER_COUNT: 7,
    Object,
    RECTANGLE_EDGE_SOFTNESS: 4,
    RECTANGLE_FLAG_ASYMMETRIC_SOFTNESS: 5,
    RECTANGLE_KERNEL_EXTENT: 3,
    RECTANGLE_TRAPEZOID_WGSL: "RT",
    RENDER_SIGMA: 4,
    TILE_SIZE: 16,
  });
  context.globalThis = context;
  vm.runInContext(
    await readFile(new URL("../web/gpu/shaders/preview-pipelines.js", import.meta.url), "utf8"),
    context,
  );
  const shaders = Object.keys(context.Image2SplatPaintPreviewShaders)
    .map((name) => context.Image2SplatPaintPreviewShaders[name]());
  assert.deepEqual(shaders.map((shader) => shader.length), [9565, 3557, 7465, 375, 809]);
  assert.equal(
    createHash("sha256").update(shaders.join("\u0000")).digest("hex"),
    "cb624d452f9cb10e95b80910f00dc6f22c973a25f20b8c6bd64b363a09cb54e1",
  );
});

test("metric WGSL factories preserve generated source fingerprints", async () => {
  const context = vm.createContext({
    BACKGROUND_EXPOSURE_EPSILON: 0.01,
    DEFAULT_ALPHA_TARGET: 1,
    METRIC_TILE_STRIDE: 2,
    MIP_PIXEL_SIGMA: 0.5,
    Object,
    OVERLAP_METRIC_STRIDE: 4,
    RENDER_SIGMA: 5,
    TILE_SIZE: 16,
    VIRTUAL_CAMERA_METRIC_TILE_STRIDE: 3,
    VIRTUAL_TILT_WGSL: "VT",
  });
  context.globalThis = context;
  vm.runInContext(await readFile(new URL("../web/gpu/shaders/metric-pipelines.js", import.meta.url), "utf8"), context);
  const shaders = Object.keys(context.Image2SplatPaintMetricShaders).map((name) =>
    name === "overlapMetrics"
      ? context.Image2SplatPaintMetricShaders[name]({ hiddenRgbBinding: "HB", hiddenRgbShader: "HS" })
      : context.Image2SplatPaintMetricShaders[name](),
  );
  assert.deepEqual(shaders.map((shader) => shader.length), [1862, 8933, 7701, 10818, 4788]);
  assert.equal(
    createHash("sha256").update(shaders.join("\u0000")).digest("hex"),
    "96f3b98c1226ef0eab07d6ade90cffa57be4d8217f664f81e72b4a81747618ed",
  );
});

test("compaction WGSL factories preserve generated source fingerprints", async () => {
  const context = vm.createContext({ Object });
  context.globalThis = context;
  vm.runInContext(await readFile(new URL("../web/gpu/shaders/compaction-pipelines.js", import.meta.url), "utf8"), context);
  const shaders = Object.keys(context.Image2SplatPaintCompactionShaders)
    .map((name) => context.Image2SplatPaintCompactionShaders[name]());
  assert.deepEqual(shaders.map((shader) => shader.length), [1162, 1093]);
  assert.equal(
    createHash("sha256").update(shaders.join("\u0000")).digest("hex"),
    "a88c9daca1327bb375968105080619c0fbe177270f218ebeda52639c431e6e3b",
  );
});
