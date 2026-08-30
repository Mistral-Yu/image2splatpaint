import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

async function orientationHelpers() {
  const source = await read("web/core/algorithms.js");
  const context = vm.createContext({
    Math, Number,
    DEFAULT_RECTANGLE_ORIENTATION: "free",
    DEFAULT_RECTANGLE_ORIENTATION_TOLERANCE: 10,
    clampNumber: (value, min, max, fallback) => Number.isFinite(Number(value))
      ? Math.min(max, Math.max(min, Number(value))) : fallback,
    document: { querySelector: () => null },
  });
  vm.runInContext(source.slice(
    source.indexOf("function normalizedRectangleOrientation("),
    source.indexOf("function selectedRectangleShapeSettings("),
  ), context);
  return context;
}

test("Rectangle orientation allowance preserves exact lock, free rotation and tapered half turns", async () => {
  const api = await orientationHelpers();
  assert.equal(api.selectedRectangleOrientationTolerance(), 10);
  assert.equal(api.selectedRectangleOrientationToleranceDegrees(), 9);
  assert.equal(api.normalizedRectangleOrientationTolerance(undefined), 0, "legacy params retain exact lock");
  assert.equal(api.normalizedRectangleOrientationTolerance(-1), 0);
  assert.equal(api.normalizedRectangleOrientationTolerance(101), 100);
  for (const orientation of ["vertical", "horizontal"]) {
    for (const [sx, sy] of [[2, 1], [1, 2], [1, 1]]) {
      const base = orientation === "vertical" ? (sx >= sy ? Math.PI / 2 : 0) : (sx >= sy ? 0 : Math.PI / 2);
      for (let index = -100; index <= 100; index += 1) {
        const theta = index * 0.173;
        assert.equal(api.constrainedRectangleTheta(theta, sx, sy, orientation, 0), base);
        assert.equal(api.constrainedRectangleTheta(theta, sx, sy, orientation, 100), theta);
        assert.equal(api.constrainedRectangleTheta(theta, sx, sy, "free", 10), theta);
        for (const percent of [0.1, 10, 50, 99.9]) {
          const next = api.constrainedRectangleTheta(theta, sx, sy, orientation, percent);
          const error = Math.asin(Math.min(1, Math.abs(Math.sin(next - base))));
          assert.ok(error <= percent * Math.PI / 200 + 1e-12);
          const again = api.constrainedRectangleTheta(next, sx, sy, orientation, percent);
          assert.ok(Math.abs(next - again) < 1e-12, "constraint is idempotent");
        }
      }
      for (const turn of [-2, -1, 0, 1, 2]) {
        const theta = base + turn * Math.PI + 0.05;
        assert.ok(Math.abs(api.constrainedRectangleTheta(theta, sx, sy, orientation, 10) - theta) < 1e-12);
      }
    }
  }
  const theta = api.constrainedRectangleTheta(0.8, 2, 1, "horizontal", 10);
  const probe = api.rectangleConstraintProbe({count: 1, scale: [2, 1], theta: [theta], rectangleOrientation: "horizontal", rectangleOrientationTolerance: 10});
  assert.ok(Math.abs(probe.max_deviation_degrees - 9) < 1e-10);
  assert.equal(probe.violation_count, 0);
});

test("Rectangle tolerance is captured once and preserved through GPU config, cloning and density", async () => {
  const algorithms = await read("web/core/algorithms.js");
  assert.match(algorithms, /config\[127\] = shape === "rectangle"[^]*normalizedRectangleOrientationTolerance\(params\?\.rectangleOrientationTolerance\) \* Math\.PI \/ 200/);
  for (const path of ["web/training/initialization.js", "web/training/densification.js"]) {
    assert.match(await read(path), /rectangleOrientationTolerance: normalizedRectangleOrientationTolerance\(params\.rectangleOrientationTolerance\)/);
  }
  const trainer = await read("web/training/trainer.js");
  assert.match(trainer, /const runRectangleOrientationToleranceDegrees = selectedRectangleOrientationToleranceDegrees\(\)/);
  assert.match(trainer, /const runRectangleOrientationTolerance = runRectangleOrientationToleranceDegrees \/ 90 \* 100/);
  assert.match(trainer, /rectangleOrientationTolerance\.value = String\(runRectangleOrientationToleranceDegrees\)/);
  assert.match(trainer, /state\.params\.rectangleOrientationTolerance = algorithmUsesRectangleKernel\(algorithm\)[^]*\? runRectangleOrientationTolerance[^]*: 0/);
  const initial = await read("web/training/initialization-runtime.js");
  assert.match(initial, /constrainedRectangleTheta\([^]*params\.rectangleOrientation,[^]*params\.rectangleOrientationTolerance,/);
  const training = await read("web/gpu/shaders/training-pipelines.js");
  assert.equal((training.match(/let tolerance = clamp\(cfg\(127u\), 0\.0, 1\.57079632679\)/g) || []).length, 2);
  assert.equal((training.match(/nextTheta \+= clamp\(delta, -tolerance, tolerance\) - delta/g) || []).length, 2);
  const density = await read("web/gpu/shaders/density-pipelines.js");
  assert.match(density, /let tolerance = clamp\(config\[127\], 0\.0, 1\.57079632679\)/);
  assert.match(density, /return theta \+ clamp\(delta, -tolerance, tolerance\) - delta/);
  assert.equal((density.match(/nextTheta = constrain_rectangle_orientation\(/g) || []).length, 3);
});

test("Rectangle tolerance UI uses 0 to 90 degrees and locks with training and Free mode", async () => {
  const html = await read("web/index.html");
  assert.match(html, /Orientation tolerance \(°\)/);
  assert.match(html, /id="rectangleOrientationTolerance"[^>]*min="0"[^>]*max="90"[^>]*value="9"/);
  assert.doesNotMatch(html, /Orientation tolerance \(%\)/);
  const controls = await read("web/ui/training-controls.js");
  assert.match(controls, /els\.rectangleOrientation,[\s]*els\.rectangleOrientationTolerance,/);
  assert.match(controls, /els\.rectangleOrientationTolerance\.disabled = state\.running \|\| !rectangleSelected \|\| selectedRectangleOrientation\(\) === "free"/);
});

test("Degree UI conversion preserves the old run contract and clamps only the input angle", async () => {
  const api = await orientationHelpers();
  for (const [input, degrees, percent] of [[-5, 0, 0], [0, 0, 0], [9, 9, 10], [45, 45, 50], [90, 90, 100], [120, 90, 100], ["bad", 9, 10]]) {
    api.document.querySelector = () => ({value: String(input)});
    assert.equal(api.selectedRectangleOrientationToleranceDegrees(), degrees);
    assert.equal(api.selectedRectangleOrientationTolerance(), percent);
    for (const orientation of ["vertical", "horizontal", "free"]) {
      for (const theta of [-3.1, -1.7, -0.3, 0, 0.4, 1.5, 2.8]) {
        const selected = api.constrainedRectangleTheta(theta, 2, 1, orientation, api.selectedRectangleOrientationTolerance());
        assert.equal(selected, api.constrainedRectangleTheta(theta, 2, 1, orientation, percent));
      }
    }
  }
  const controls = await read("web/ui/controls.js");
  assert.match(controls, /rectangleOrientationTolerance\.value = String\(selectedRectangleOrientationToleranceDegrees\(\)\)/);
});

test("GPU tolerance occupies only the reserved slot and does not affect Gaussian or Brush configs", async () => {
  const context = await orientationHelpers();
  // Config declares the real production constants; use a fresh scope to avoid
  // the helper fixtures shadowing those lexical bindings.
  const scope = vm.createContext({
    Math, Number, Float32Array,
    clampNumber: context.clampNumber,
    document: { querySelector: () => null },
  });
  vm.runInContext(await read("web/core/config.js"), scope);
  const app = await read("web/app.js");
  const limitsStart = app.indexOf("const LIMITS = {");
  vm.runInContext(app.slice(limitsStart, app.indexOf("\n};", limitsStart) + 3), scope);
  vm.runInContext(await read("web/core/algorithms.js"), scope);
  for (const kernelShape of ["gaussian", "rectangle", "opaque-brush"]) {
    const base = scope.configurePaintKernel(new Float32Array(128), {kernelShape, rectangleOrientation: "vertical"});
    const loose = scope.configurePaintKernel(new Float32Array(128), {kernelShape, rectangleOrientation: "vertical", rectangleOrientationTolerance: 10});
    assert.deepEqual(Array.from(base.slice(0, 127)), Array.from(loose.slice(0, 127)));
    assert.equal(base[127], 0);
    assert.ok(Math.abs(loose[127] - (kernelShape === "rectangle" ? Math.PI / 20 : 0)) < 1e-7);
  }
});
