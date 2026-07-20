import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const [app, index] = await Promise.all([
  readFile(new URL("../web/app.js", import.meta.url), "utf8"),
  readFile(new URL("../web/index.html", import.meta.url), "utf8"),
]);

const contracts = {
  relativeDefault: /DEFAULT_STAGE_MIN_SCALE_RATIO = 0\.05/.test(app) &&
    /stageMinScaleRatio: Math\.max\(0, Math\.min\(0\.25, finite\("stageMinScaleRatio", DEFAULT_STAGE_MIN_SCALE_RATIO\)\)\)/.test(app),
  initialGridReference: /function stageMinimumScale\(image, initialCount, trainingStage, ratio\)/.test(app) &&
    /splatGridLayout\(image, Math\.max\(1, initialCount\)\)/.test(app) &&
    /Math\.min\(referenceLayout\.baseScaleX, referenceLayout\.baseScaleY\) \* ratio/.test(app),
  p1p2Only: /if \(trainingStage === "full" \|\| ratio <= 0\) return MIN_SPLAT_SCALE/.test(app),
  p1p2BaseFloorInputs: /p1BaseScaleFloorRatio: Math\.max\(0, Math\.min\(1, finite\(/.test(app) &&
    /p2BaseScaleFloorRatio: Math\.max\(0, Math\.min\(1, finite\(/.test(app) &&
    /function stageBaseScaleFloorRatio\(trainingStage, variants\)/.test(app),
  densityAndRelocationFloor: /let baseScaleFloor = baseScale \* clamp\(config\[61\], 0\.0, 1\.0\);/.test(app) &&
    /max\(max\(splitScale, baseScaleFloor\), stageMinScale\)/.test(app),
  optimizerClamp: (app.match(/let minScale = max\(\$\{MIN_SPLAT_SCALE\}, cfg\(80u\)\)/g) || []).length === 2,
  densityClamp: /let stageMinScale = vec2<f32>\(max\(\$\{MIN_SPLAT_SCALE\}, config\[60\]\)\)/.test(app) &&
    /max\(max\(splitScale, baseScaleFloor\), stageMinScale\)/.test(app),
  publicInputs:
    /id="p1BaseScaleFloorRatio"[^>]*value="0\.50"/.test(index) &&
    /id="p2BaseScaleFloorRatio"[^>]*value="0\.35"/.test(index) &&
    /inputNumber\("#p1BaseScaleFloorRatio", DEFAULT_P1_BASE_SCALE_FLOOR_RATIO\)/.test(app) &&
    /inputNumber\("#p2BaseScaleFloorRatio", DEFAULT_P2_BASE_SCALE_FLOOR_RATIO\)/.test(app),
  observable: /stage_min_scale_ratio/.test(app) && /stage_base_scale_floor_ratio/.test(app) && /stage_min_scale_normalized/.test(app),
};

for (const [name, passed] of Object.entries(contracts)) {
  assert.equal(passed, true, `stage minimum scale contract failed: ${name}`);
}

console.log(JSON.stringify({ ok: true, contracts }, null, 2));
