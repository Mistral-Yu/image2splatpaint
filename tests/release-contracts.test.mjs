import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";
import { createHash } from "node:crypto";
import { releaseArtifactFiles, sourceJavaScriptFiles } from "../release-manifest.mjs";

test("release and source manifests contain unique existing files", async () => {
  assert.equal(new Set(releaseArtifactFiles).size, releaseArtifactFiles.length);
  assert.equal(new Set(sourceJavaScriptFiles).size, sourceJavaScriptFiles.length);
  await Promise.all([...new Set([...releaseArtifactFiles, ...sourceJavaScriptFiles])].map((path) => access(new URL(`../${path}`, import.meta.url))));
});

test("every local app script is shipped and critical classic-script order is stable", async () => {
  const html = await readFile(new URL("../web/index.html", import.meta.url), "utf8");
  const scripts = [...html.matchAll(/<script\b[^>]*\bsrc=["']([^"']+)["'][^>]*>/gi)]
    .map(([, source]) => source.split(/[?#]/, 1)[0])
    .filter((source) => !/^(?:https?:)?\/\//i.test(source))
    .map((source) => `web/${source.replace(/^\.\//, "")}`);
  for (const source of scripts) assert.ok(releaseArtifactFiles.includes(source), `${source} must be in the release manifest`);

  const indexOf = (suffix) => scripts.findIndex((path) => path.endsWith(suffix));
  assert.ok(indexOf("gpu/shaders/tile-pipeline.js") < indexOf("gpu/tile-pipelines.js"));
  assert.ok(indexOf("gpu/shaders/optimizer-reset.js") < indexOf("gpu/optimizer-runtime.js"));
  assert.ok(indexOf("gpu/shaders/training-pipelines.js") < indexOf("gpu/renderer.js"));
  assert.ok(indexOf("gpu/shaders/preview-pipelines.js") < indexOf("gpu/renderer.js"));
  assert.ok(indexOf("gpu/shaders/metric-pipelines.js") < indexOf("gpu/renderer.js"));
  assert.ok(indexOf("gpu/shaders/density-pipelines.js") < indexOf("gpu/renderer.js"));
  assert.ok(indexOf("gpu/shaders/compaction-pipelines.js") < indexOf("gpu/renderer.js"));
  assert.ok(indexOf("gpu/renderer.js") < indexOf("gpu/tile-pipelines.js"));
  assert.ok(indexOf("gpu/renderer.js") < indexOf("gpu/optimizer-runtime.js"));
  assert.ok(indexOf("app.js") < indexOf("ui/bootstrap.js"));
  assert.ok(indexOf("app.js") < indexOf("ui/controls.js"));
});

async function loadShaderFactories() {
  const context = vm.createContext({ ILLUSTRATIVE_OIL_WGSL: "IO", VIRTUAL_TILT_WGSL: "VT",
    RECTANGLE_TRAPEZOID_WGSL: "RT", MONOCHROME_LAB_L_WGSL: "ML",
    residualDestinationOracleRequested: () => false, residualTileCdfEnabled: () => true });
  for (const file of ["core/config.js", "gpu/shaders/training-pipelines.js", "gpu/shaders/density-pipelines.js"]) {
    vm.runInContext(await readFile(new URL(`../web/${file}`, import.meta.url), "utf8"), context);
  }
  return context;
}
const shaderHash = source => createHash("sha256").update(source).digest("hex");

test("generated default training WGSL stays byte-identical to the pre-Flow Baseline", async () => {
  const context = await loadShaderFactories();
  const options = { fixedPointExactGradientEnabled: false, inverseScaleOptimizationEnabled: false,
    optimizerStatsDeclaration: "OS", segmentedExactBackwardEnabled: false };
  const create = extra => context.Image2SplatPaintTrainingPipelineShaders.create.call(
    { subgroupExactBackwardEnabled: false, quadExactBackwardEnabled: false }, { ...options, ...extra });
  const baseline = create({});
  // Captured by executing the archived pre-change factories with these exact
  // constants. Test generated WGSL, not raw JS interpolation source.
  const expected = {
    renderShader: "f00d4bf9558b7d826a5aee7f856cb69e68887fba53b4fd47465aefd3b826e50b",
    ssimShader: "1156d4732a843250e6fbbfc1efcf7a0d83c42cd7a6ce25e2cb37e99df814eeaf",
    lossGradientShader: "4f9db93b1436b37beb1cce5ce3947f66c1ffbdda71a96af1eecaf6cc0873b5da",
    exactBackwardShader: "e06a9574868fb9cf02d41eae6addf21ebe3a262c0d9c79738f9e900ec9a66fd9",
    segmentedReferenceShader: "f5ff80932f5b9c0cc37b0900a0a5b5145003cd5e37ef66becd087df346039b89",
    segmentedGradientReduceShader: "4c6c41cea082babc5a83970c04afae4aca45517b280e977fa64318be7443930b",
    exactBackwardTelemetryShader: "b18c6c6a6ebd00f6cb51411354d92f87f1c6e1e680aabbfbb756e8104ee3620e",
    virtualOrderPenaltyShader: "05e12ec4c403471df52106811dd8dcd168cceaac389554848e84b6d1ea53cd4b",
    brushLocalColorFlowShader: "139cd2b701583b5ae2fc522eea24b34ef9ad43299507fea1eae9242257b99d73",
    optimizerShader: "7eb63ede3b949204a1a1101df4af165f91c7fbe246b16636a3da73fd2b2a2f6b",
  };
  for (const [key, hash] of Object.entries(expected)) assert.equal(shaderHash(baseline[key]), hash, key);
  const protectedFlow = create({ protectedPrefix: 26 });
  assert.deepEqual(Object.keys(baseline).filter(key => baseline[key] !== protectedFlow[key]), ["optimizerShader"]);
  assert.match(protectedFlow.optimizerShader, /if \(g < 26u\) \{ return; \}/);
});

test("generated default density WGSL stays byte-identical to the pre-Flow Baseline", async () => {
  const context = await loadShaderFactories();
  assert.equal(shaderHash(context.Image2SplatPaintDensityShader.create()),
    "8aba275932268b28cc6a56352e5e2b8cd8d8c61f339ecdb8c37df92cfdb87dd7");
});

test("Pages workflow uses tracked source, contract, build, and release gates", async () => {
  const workflow = await readFile(new URL("../.github/workflows/pages.yml", import.meta.url), "utf8");
  const sourceGate = workflow.indexOf("node verify-sources.mjs");
  const contractGate = workflow.indexOf("node --test tests/*.test.mjs");
  const build = workflow.indexOf("node build-release.mjs _site");
  const releaseGate = workflow.indexOf("node verify-release.mjs _site");
  assert.ok(sourceGate >= 0 && contractGate > sourceGate && build > contractGate && releaseGate > build);
  assert.doesNotMatch(workflow, /scripts\//);
});

test("GPU feature composition rejects collisions without copying implementations", async () => {
  const renderer = await readFile(new URL("../web/gpu/renderer.js", import.meta.url), "utf8");
  const tilePipelines = await readFile(new URL("../web/gpu/tile-pipelines.js", import.meta.url), "utf8");
  const tileRuntime = await readFile(new URL("../web/gpu/tile-runtime.js", import.meta.url), "utf8");
  const optimizer = await readFile(new URL("../web/gpu/optimizer-runtime.js", import.meta.url), "utf8");
  assert.match(renderer, /hasOwnProperty\.call\(WebGpuPreview\.prototype, name\)/);
  assert.match(renderer, /WebGpuPreview method collision/);
  assert.match(renderer, /class WebGpuPreviewFeatureAdapter/);
  assert.match(renderer, /descriptor\.value\.apply\(this\.owner, args\)/);
  assert.match(renderer, /this\.webGpuFeatureAdapters = new Map/);
  assert.match(tilePipelines, /registerWebGpuPreviewFeature\(WebGpuTilePipelines\.prototype/);
  assert.match(tileRuntime, /registerWebGpuPreviewFeature\(WebGpuTileRuntime\.prototype/);
  assert.match(optimizer, /registerWebGpuPreviewFeature\(WebGpuOptimizerRuntime\.prototype/);
  assert.doesNotMatch(`${tilePipelines}\n${tileRuntime}\n${optimizer}`, /Object\.defineProperty\(\s*WebGpuPreview\.prototype/);
  assert.doesNotMatch(optimizer, /\bels\./, "GPU optimizer must consume run state rather than live DOM controls");
  assert.match(optimizer, /this\.trainState\.tileCullingEnabled/);
  assert.match(optimizer, /params\.layerOrderEnabled/);
});

test("Training actions stay on one four-column row", async () => {
  const styles = await readFile(new URL("../web/styles.css", import.meta.url), "utf8");
  assert.match(
    styles,
    /\.actions\s*\{[^}]*grid-template-columns:\s*repeat\(4,\s*minmax\(0,\s*1fr\)\)/s,
    "Train, Pause, Stop, and Reset must stay on one four-column row",
  );
});

test("Curved Brush Splats is the third public Algorithm and legacy Brush is a Rectangle shape", async () => {
  const html = await readFile(new URL("../web/index.html", import.meta.url), "utf8");
  const styles = await readFile(new URL("../web/styles.css", import.meta.url), "utf8");
  const app = await readFile(new URL("../web/app.js", import.meta.url), "utf8");
  const exportRuntime = await readFile(new URL("../web/export/runtime.js", import.meta.url), "utf8");
  const bootstrap = await readFile(new URL("../web/ui/bootstrap.js", import.meta.url), "utf8");
  const controls = await readFile(new URL("../web/ui/training-controls.js", import.meta.url), "utf8");
  const algorithms = await readFile(new URL("../web/core/algorithms.js", import.meta.url), "utf8");
  const initialization = await readFile(new URL("../web/training/initialization-runtime.js", import.meta.url), "utf8");
  const integration = await readFile(new URL("../web/training/flow-splat-fusion.js", import.meta.url), "utf8");
  const topology = await readFile(new URL("../web/training/flow-stroke-topology.js", import.meta.url), "utf8");
  const flowReference = await readFile(new URL("../web/training/flow-paint-reference.js", import.meta.url), "utf8");
  const ribbonTrainer = await readFile(new URL("../web/training/flow-ribbon-trainer.js", import.meta.url), "utf8");
  assert.match(
    html,
    /value="planar-gaussian" selected>Planar Gaussian<\/option>[^]*value="rectangle-splats">Rectangle Splats<\/option>[^]*value="flow-splat-fusion">Curved Brush Splats<\/option>[^]*value="gs-virtual-camera-sampling">GS Virtual Camera Sampling<\/option>/,
  );
  assert.doesNotMatch(html, /<option value="layered-opaque-brush">/);
  assert.match(html, /id="rectanglePaintShape"[^]*value="rectangle" selected>Rectangle<\/option>[^]*value="opaque-brush">Illustrative Brush<\/option>/);
  assert.match(html, /id="rectangleShapeSettings"/);
  assert.match(html, /Illustrative Brush shape settings/);
  const rectangleSettingsMarkup = html.slice(
    html.indexOf('<section class="rectangle-paint-panel'),
    html.indexOf('<section class="virtual-camera-panel'),
  );
  assert.equal((rectangleSettingsMarkup.match(/<details\b/g) || []).length, 1);
  assert.match(rectangleSettingsMarkup, /data-testid="opaque-paint-settings"[^]*data-testid="layered-opaque-brush-settings"/);
  assert.match(controls, /rectangleShapeSettings\.hidden = !rectangleSelected/);
  assert.match(
    controls,
    /function setInputControlsDisabled\(disabled\)[^]*els\.algorithmSelect,[^]*els\.rectanglePaintShape,[^]*els\.trainSize,/,
    "Rectangle / Brush shape selection must be locked with the other training inputs",
  );
  assert.match(algorithms, /function selectedRectanglePaintShape\(\)/);
  assert.match(algorithms, /algorithm\?\.id === RECTANGLE_SPLATS_ALGORITHM_ID[^]*selectedRectanglePaintShape\(\) === "opaque-brush"/);
  assert.match(initialization, /initOpaqueLayeredPaint\(image, count, selectedRectanglePaintShape\(\)\)/);
  assert.doesNotMatch(html, /id="flowSplatFusionPanel"[^>]*\bhidden\b/);
  assert.doesNotMatch(controls, /flowSplatFusionPanel\.hidden/);
  assert.match(html, /data-settings-scope="shared" data-settings-order="1"/);
  assert.match(html, /data-testid="rectangle-paint-panel" data-settings-order="2"/);
  assert.doesNotMatch(html, /data-testid="opaque-paint-panel"[^>]*data-settings-order/);
  assert.match(html, /data-testid="flow-splat-fusion-panel" data-settings-order="3"/);
  assert.match(html, /data-testid="virtual-camera-panel" data-settings-order="4"/);
  assert.match(html, /data-testid="budget-panel" data-settings-order="5"/);
  assert.match(
    styles,
    /\[data-settings-order="1"\]\s*\{\s*order:\s*1;\s*\}[^]*\[data-settings-order="5"\]\s*\{\s*order:\s*5;\s*\}/,
  );
  assert.match(html, /\.\/training\/flow-paint-reference\.js/);
  assert.match(html, /\.\/training\/flow-ribbon-trainer\.js/);
  assert.doesNotMatch(html, /\.\/qa\/flow-(?:paint-reference|ribbon-trainer)\.js/);
  assert.match(
    app,
    /\[FLOW_SPLAT_FUSION_ALGORITHM_ID\][^]*exports: Object\.freeze\(\["png"\]\)[^]*png: true/,
  );
  assert.match(exportRuntime, /function currentFlowPngResult\(\)/);
  assert.match(exportRuntime, /async function makeFlowPreviewPngBlob\(/);
  assert.match(exportRuntime, /image2splatpaint-flow-brush-fusion\.png/);
  assert.match(html, /id="flowSplatFusionMaxArcPercent"[^>]*value="10"/);
  assert.match(html, /id="flowSplatFusionStrokeOptimization"/);
  assert.match(html, /id="flowSplatFusionStrokeTexture"/);
  assert.doesNotMatch(html, /id="flowSplatFusionTopology"/);
  assert.match(html, /id="flowSplatFusionPaintCurriculum"[^>]*\bchecked\b/);
  assert.match(html, /id="flowSplatFusionFixedOpacity"[^>]*value="0\.995"/);
  assert.doesNotMatch(html, /id="flowSplatFusionDepthLayerOpacity"/);
  assert.match(html, /id="flowSplatFusionStartingWidthDivisor"[^>]*value="32"/);
  assert.match(html, /id="flowSplatFusionStartingLengthPercent"[^>]*value="160"/);
  assert.match(html, /id="flowSplatFusionResidualMovePx"[^>]*value="6"/);
  assert.match(html, /id="flowSplatFusionScaleMatchedResidualRepaint"/);
  assert.doesNotMatch(html, /id="flowSplatFusionScaleMatchedResidualRepaint"[^>]*\bchecked\b/);
  assert.match(html, /id="flowSplatFusionInitialWidthMin"[^>]*value="55"/);
  assert.match(html, /id="flowSplatFusionInitialWidthMax"[^>]*value="165"/);
  assert.match(html, /id="flowSplatFusionFrontWidthMax"[^>]*value="300"/);
  assert.match(html, /id="flowSplatFusionFrontWidthLearning"[^>]*value="400"/);
  assert.doesNotMatch(html, /flowSplatFusionInitialOpacity(?:Min|Max)/);
  assert.match(html, /value="brush-dabs" selected>Flow Brush<\/option>/);
  assert.match(html, /value="baseline">Classic Gaussian \(4-Splat\)/);
  assert.match(html, /value="fine-bristles">Fine bristles \(Gaussian\)/);
  assert.match(controls, /forms linked strokes within the selected Min\/Max group size/);
  assert.match(html, /Paint with curved Brush Splats using linked strokes or learned internal bends/);
  assert.match(html, /id="flowLinkedSplatMin"[^>]*value="4"/);
  assert.match(html, /id="flowLinkedSplatMax"[^>]*value="9"/);
  assert.match(html, /id="flowInternalBendControlPointCount"[^>]*value="1"/);
  assert.match(html, /id="flowInternalBendControlPointPositions"[^>]*value="50"/);
  assert.match(html, /value="balanced" selected>Balanced motion/);
  assert.match(html, /id="flowSplatFusionWidthPercent"[^>]*value="300"/);
  assert.match(html, /id="flowSplatFusionSplatSizeVariation"[^>]*value="40"/);
  assert.match(html, /id="flowSplatFusionMovementLimit"[^>]*value="12"/);
  assert.match(html, /id="flowSplatUnderpainting"[^>]*\bchecked\b/);
  assert.match(html, /Curved Brush underpainting/);
  assert.match(html, /id="flowSplatUnderpaintPercent"[^>]*value="10"/);
  assert.doesNotMatch(html, /flowPaintPreviewButton|localFlowPaintTrain/);
  assert.doesNotMatch(bootstrap, /FLOW_SPLAT_FUSION_ALGORITHM_ID/);
  assert.doesNotMatch(bootstrap, /Curve Splat Chain \(Local\)/);
  assert.doesNotMatch(controls, /syncAlgorithmTrainingPreset|>Ribbons</);
  assert.match(controls, /finalSplatCountLabel\.textContent = "Max splats"/);
  assert.match(controls, /Physical Splat budget\. Detail uses complete stroke chains/);
  assert.match(controls, /The protected backcoat starts in P1|forms linked strokes within the selected Min\/Max group size/);
  assert.match(controls, /trainSize\.max = flowSelected \? "512"/);
  assert.match(integration, /trainFlowSplatFusion/);
  assert.match(integration, /: "brush-dabs";/);
  assert.match(integration, /const representation = "curve-splat-chain"/);
  assert.match(integration, /progressiveParentCounts/);
  assert.match(integration, /strokeOptimizationProfiles/);
  assert.match(integration, /flowStrokeTexture/);
  assert.match(integration, /bristleBundle: flowBristleBundle/);
  assert.match(integration, /brushDabs: flowBrushDabs/);
  assert.match(integration, /flow_brush_dabs/);
  assert.match(integration, /flow_brush_kernel: flowBrushDabs \? "compact-quartic-opaque-interior-v1" : "gaussian"/);
  assert.match(integration, /const flowTextureGuidedDabs = flowBrushDabs/);
  assert.match(integration, /flow-texture-guided-dabs/);
  assert.match(integration, /textureGuidedAllocation: flowTextureGuidedDabs/);
  assert.match(integration, /textureGuidedDabs: flowTextureGuidedDabs/);
  assert.match(integration, /flow_texture_guided_dabs/);
  assert.match(integration, /Image2SplatPaintFlowStrokeTopology\.evolve/);
  assert.match(integration, /flow_topology_split_count/);
  assert.match(integration, /flow_topology_residual_move_count/);
  assert.match(integration, /flow_scale_matched_residual_repaint/);
  assert.match(integration, /flow_fixed_stroke_opacity/);
  assert.match(integration, /fixedOpacity: fixedStrokeOpacity/);
  assert.match(integration, /flow_front_width_maximum_percent/);
  assert.match(integration, /const strokeWidthMaximumFactor = strokeWidthPercent \/ 100/);
  assert.match(integration, /frontWidthMaximumFactor: Math\.min\(/);
  assert.match(integration, /flow_stroke_width_mode: "global-parent-width-ceiling"/);
  assert.match(integration, /balanced-physical-brush-width-families/);
  assert.doesNotMatch(integration, /half_width_px: Math\.max\(0\.25, Number\(stroke\.half_width_px\) \* strokeWidthScale\)/);
  assert.match(integration, /frontWidthLearningScale,/);
  assert.match(integration, /"residual-split-clone-visible-prune-texture-guided"/);
  assert.match(integration, /: "residual-split-clone-visible-prune"/);
  assert.match(integration, /flow_topology_clone_count/);
  assert.match(integration, /flow_topology_pruned_count/);
  assert.doesNotMatch(integration, /flow-adaptive-topology/);
  assert.match(integration, /flow-residual-priority-tiles/);
  assert.match(integration, /flow_tile_list_update: "growth-boundary-only"/);
  assert.match(topology, /function splitStroke/);
  assert.match(topology, /function mergePair/);
  assert.match(topology, /sampleResidual/);
  assert.match(topology, /optimizeResidualPlacement/);
  assert.match(topology, /applyPaintCurriculum/);
  assert.match(topology, /function fixedStrokeOpacity/);
  assert.match(topology, /function widthFactorRangeForLayer/);
  assert.match(topology, /function annotateTextureGuide/);
  assert.match(topology, /edge_score: stats\.edge/);
  assert.match(topology, /splitCellCapacity/);
  assert.match(topology, /const learnedWidthDelta = Number\(target\.half_width_px\) - scheduledWidth/);
  assert.match(ribbonTrainer, /fixed_opacity_logit: f32/);
  assert.match(ribbonTrainer, /front_width_learning_scale: f32/);
  assert.match(ribbonTrainer, /texture_guided_dabs: f32/);
  assert.match(ribbonTrainer, /fn chain_texture_score/);
  assert.match(ribbonTrainer, /fn chain_edge_score/);
  assert.match(ribbonTrainer, /struct ChainKernelSample/);
  assert.match(ribbonTrainer, /fn chain_kernel_sample/);
  assert.match(ribbonTrainer, /let q = u4 \+ v4/);
  assert.match(ribbonTrainer, /let feather = 0\.16/);
  assert.match(ribbonTrainer, /if \(chain_uses_brush_dabs\(\)\) \{ return 1\.0; \}/);
  assert.match(ribbonTrainer, /smoothstepRange\(0\.12, 0\.65, textureScore\)/);
  assert.match(ribbonTrainer, /smoothstepRange\(0\.30, 0\.95, edgeScore\)/);
  assert.match(ribbonTrainer, /0\.72 \+ shapeGuide \* 0\.28/);
  assert.match(ribbonTrainer, /flatNormalOffsets = \[0\.16, 0\.08, -0\.16, -0\.08, 0\]/);
  assert.match(ribbonTrainer, /mix\(0\.62, factor, shape_guide\)/);
  assert.match(ribbonTrainer, /mix\(1\.15, factor, shape_guide\)/);
  assert.match(ribbonTrainer, /dab_t = 0\.5 \+ \(dab_t - 0\.5\) \* \(0\.5 \+ texture \* 0\.5\)/);
  assert.match(ribbonTrainer, /mix\(0\.88, factor, texture\)/);
  assert.match(ribbonTrainer, /mix\(1\.45, factor, texture\)/);
  assert.match(ribbonTrainer, /fn chain_uses_brush_dabs/);
  assert.match(ribbonTrainer, /fn chain_normal_offset_factor/);
  assert.match(ribbonTrainer, /fn chain_width_factor/);
  assert.match(flowReference, /function buildFlowXdogGuide/);
  assert.match(flowReference, /algorithm: "linear-srgb-flow-xdog-guide"/);
  assert.match(flowReference, /edge_score_mode: edgeGuidedAccents \? "coherent-colour-plus-flow-xdog-78" : "disabled"/);
  assert.match(flowReference, /normalDog\[index\] = Math\.abs/);
  assert.match(flowReference, /coherentEdgeScore[^]*\+ \(1 - coherentEdgeScore\) \* flowXdog\.score\[index\] \* 0\.78/);
  assert.match(ribbonTrainer, /splat_size_variation: f32/);
  assert.match(ribbonTrainer, /fn chain_splat_width_variation/);
  assert.match(ribbonTrainer, /fn chain_splat_length_variation/);
  assert.match(ribbonTrainer, /sample == 0u \|\| sample == 2u/);
  assert.match(ribbonTrainer, /sample == 0u\) \{ family_scale = 0\.40/);
  assert.match(ribbonTrainer, /sample == 2u\) \{ family_scale = 1\.88/);
  assert.match(ribbonTrainer, /f32\(136, Math\.max\(0, Math\.min\(1, Number\(options\.splatSizeVariation\)/);
  assert.match(ribbonTrainer, /fn chain_pigment_scale/);
  assert.match(ribbonTrainer, /const BRUSH_DAB_SAMPLES = 8/);
  assert.match(ribbonTrainer, /params\[index\] = config\.fixed_opacity_logit/);
  assert.match(integration, /createFlowGeometryAnchorParams/);
  assert.match(integration, /detailGeometryAnchorParams/);
  assert.match(integration, /FLOW_PROGRESSIVE_GROWTH_INTERVAL = 100/);
  assert.match(integration, /FLOW_ITERATION_STRIDE = 10/);
  assert.match(integration, /FLOW_PROGRESSIVE_GROWTH_APPLY_UNTIL = 0\.90/);
  assert.match(integration, /FLOW_SPLIT_APPLY_UNTIL = 0\.75/);
  assert.match(integration, /splitFraction: 0\.04/);
  assert.match(integration, /maximumSplitsPerEvent: 24/);
  assert.match(integration, /buildFlowProgressiveGrowthSchedule/);
  assert.match(integration, /progressive_growth_stage_count/);
  assert.match(integration, /progressive_settle_iterations/);
  assert.match(integration, /progressive_growth_parent_counts/);
  assert.match(integration, /createSplatUnderpaintPlan/);
  assert.match(integration, /trainingStrokePlan = \[\.\.\.rearPlan, \.\.\.detailPlan\]/);
  assert.match(integration, /flow-coverage-backcoat/);
  assert.match(integration, /fixed-grid-source-colored-compact-brush-backcoat/);
  assert.match(integration, /coverage_backcoat_geometry_trainable: false/);
  assert.match(integration, /residualRender: previousStage\?\.trainingState\.renderedLinearRgba/);
  assert.match(integration, /initialDetailParams: undefined/);
  assert.match(integration, /maximumCurveArcPx: reference\.metadata\.maximum_ribbon_arc_px/);
  assert.match(integration, /previewInterval/);
  assert.doesNotMatch(integration, /\bels\./, "Algorithm training must use the UI adapter rather than direct DOM controls");
  assert.match(ribbonTrainer, /16x16-tile-2x2-lane-quad-parent-curve-reduction/);
  assert.match(ribbonTrainer, /estimateCanvasLinear/);
  assert.match(ribbonTrainer, /config\.canvas_linear_r/);
  assert.match(ribbonTrainer, /stroke_motion_coherence/);
  assert.match(ribbonTrainer, /BRISTLE_BUNDLE_SAMPLES = 4/);
  assert.match(ribbonTrainer, /chain_uses_bristle/);
  assert.match(ribbonTrainer, /translation_gradient/);
  assert.match(ribbonTrainer, /alternating-residual-weighted-and-uniform/);
  assert.match(ribbonTrainer, /tile_sampling_masks/);
  assert.match(ribbonTrainer, /if \(config\.max_curve_arc > 0\.0 && params\[base \+ 14u\] < 2\.5\)/);
  assert.match(ribbonTrainer, /chainQuadBackward \? CHAIN_QUAD_BACKWARD_WGSL : BACKWARD_WGSL/);
  assert.match(ribbonTrainer, /evaluate_flow_underpaint_splat/);
  assert.match(ribbonTrainer, /fn coverage_backcoat_kernel/);
  assert.match(ribbonTrainer, /let coverage_backcoat = params\[base \+ 14u\] > 2\.5/);
  assert.match(ribbonTrainer, /params\[index\] = anchors\[index\]/);
  assert.match(ribbonTrainer, /cream_canvas_leak_linear_mean/);
  assert.match(ribbonTrainer, /maximum_transmittance: maximumTransmittance/);
  assert.match(integration, /dataset\.flowMaximumTransmittance/);
});
