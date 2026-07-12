import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";

const root = new URL("../", import.meta.url);
const [app, index, readme, ignore, license] = await Promise.all([
  readFile(new URL("web/app.js", root), "utf8"),
  readFile(new URL("web/index.html", root), "utf8"),
  readFile(new URL("README.md", root), "utf8"),
  readFile(new URL(".gitignore", root), "utf8"),
  readFile(new URL("LICENSE", root), "utf8"),
]);

const retired = [
  "Reduce oversized splats", "Confirmed tangent chains", "Symmetric long-axis split",
  "Late subpixel detail", "Reset density stats per batch", "ADC protect fine detail",
  "ADC prioritize fine detail", "areaRegularization", "confirmed_chain_at",
  "apply_symmetric_parents", "adcProtectDetail", "adcPrioritizeDetail",
];

const legacyPaths = ["pyproject.toml", "src", "tests", "viewer", "build", "web/export-formats.mjs", "web/vendor"];
const absent = async (path) => access(new URL(path, root)).then(() => false, () => true);

function functionSource(source, name) {
  const start = source.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `${name} missing`);
  const brace = source.indexOf("{", start);
  let depth = 0;
  for (let i = brace; i < source.length; i += 1) {
    if (source[i] === "{") depth += 1;
    if (source[i] === "}") depth -= 1;
    if (depth === 0) return source.slice(start, i + 1);
  }
  throw new Error(`${name} body is incomplete`);
}

const transformPlanarSplatForPly = new Function(
  `${functionSource(app, "plyFrameScale")}\n${functionSource(app, "transformPlanarSplatForPly")}\nreturn transformPlanarSplatForPly;`,
)();
const landscapeProbe = transformPlanarSplatForPly(0.5, 0.5, 0.2, 0.1, 0, { width: 800, height: 400 });
assert.ok(Math.abs(landscapeProbe.x - 0.5) < 1e-12);
assert.ok(Math.abs(landscapeProbe.y + 0.25) < 1e-12);
assert.ok(Math.abs(landscapeProbe.sx - 0.2) < 1e-12);
assert.ok(Math.abs(landscapeProbe.sy - 0.05) < 1e-12);

const theta = Math.PI / 4;
const rotatedProbe = transformPlanarSplatForPly(0, 0, 0.2, 0.1, theta, { width: 800, height: 400 });
const reconstructed = {
  xx: Math.cos(rotatedProbe.theta) ** 2 * rotatedProbe.sx ** 2 + Math.sin(rotatedProbe.theta) ** 2 * rotatedProbe.sy ** 2,
  yy: Math.sin(rotatedProbe.theta) ** 2 * rotatedProbe.sx ** 2 + Math.cos(rotatedProbe.theta) ** 2 * rotatedProbe.sy ** 2,
  xy: Math.cos(rotatedProbe.theta) * Math.sin(rotatedProbe.theta) * (rotatedProbe.sx ** 2 - rotatedProbe.sy ** 2),
};
const expected = {
  xx: 0.5 * (0.2 ** 2 + 0.1 ** 2),
  yy: 0.25 * 0.5 * (0.2 ** 2 + 0.1 ** 2),
  xy: -0.5 * 0.5 * (0.2 ** 2 - 0.1 ** 2),
};
assert.ok(Math.max(
  Math.abs(reconstructed.xx - expected.xx),
  Math.abs(reconstructed.yy - expected.yy),
  Math.abs(reconstructed.xy - expected.xy),
) < 1e-12);

const contracts = {
  retiredProductCodeAbsent: retired.every((term) => !app.includes(term) && !index.includes(term) && !readme.includes(term)),
  retainedControls: index.includes("Tile/subtile culling"),
  adaptiveDetailAlwaysOn:
    !index.includes("adaptiveDetailToggle") &&
    !index.includes("Adaptive thin details") &&
    !app.includes("els.adaptiveDetailToggle") &&
    app.includes("adaptiveDetail: true"),
  publicLicenses:
    license.includes("MIT License") &&
    license.includes("Copyright (c) 2026 Mistral-Yu") &&
    readme.includes("[MIT License](LICENSE)") &&
    readme.includes("PlayCanvas Engine") &&
    readme.includes("no PlayCanvas code or runtime") &&
    index.includes('href="../LICENSE"'),
  localEvidenceIgnored: ["checkpoints/", "doc/", "runbooks/", "outputs/", "references/", "AGENTS.md"]
    .every((entry) => ignore.split(/\r?\n/).includes(entry)),
  experimentalScriptsIgnored: [
    "scripts/*",
    "!scripts/build-pages.mjs",
    "!scripts/pages_artifact_tests.mjs",
    "!scripts/public_surface_tests.mjs",
    "!scripts/static-server-lib.mjs",
    "!scripts/static-server.mjs",
  ].every((entry) => ignore.split(/\r?\n/).includes(entry)),
  legacyPathsRemoved: (await Promise.all(legacyPaths.map(absent))).every(Boolean),
  pagesPublicSurface: readme.includes("only `index.html`, `LICENSE`, `web/`, and `.nojekyll`"),
  publicQaRestricted:
    app.includes("const QA_RUNTIME_ENABLED") &&
    app.includes('qaOverrides("__flatPhotoPhase40")') &&
    app.includes("if (QA_RUNTIME_ENABLED) window.__flatPhotoTest"),
  automaticSplatFloor:
    app.includes("AUTO_INITIAL_SPLATS_MIN = 500") &&
    app.includes("AUTO_FINAL_SPLATS_MIN = 3000") &&
    app.includes("previousEstimate / 2") &&
    app.includes("previousInitial * 2"),
  finalSplatCommitBehavior:
    app.includes('els.finalSplatCount.addEventListener("input"') &&
    app.includes("reconcileSplatCounts: false") &&
    app.includes('els.finalSplatCount.addEventListener("change"'),
  allocationAlwaysValidated:
    app.includes('this.device.pushErrorScope("out-of-memory")') &&
    !app.includes('if (verifyAllocation) {\n      this.device.pushErrorScope("out-of-memory")'),
  aspectPreservingPly:
    app.includes("transformPlanarSplatForPly") &&
    app.includes("single NDC-to-isotropic-world conversion") &&
    app.includes("comment image2gaussianpaint_frame") &&
    app.includes("aspect_ratio_preserved") &&
    app.includes("opacity_error_max") &&
    app.includes("color_error_max"),
  aspectAwareTrainingGrid:
    app.includes("function aspectAwareGridEnabled") &&
    app.includes("const baseScaleX = aspectAwareGridEnabled() ? 1.6 / cols : baseScale") &&
    app.includes("const baseScaleY = aspectAwareGridEnabled() ? 1.6 / rows : baseScale") &&
    app.includes("layout.baseScaleX") &&
    app.includes("layout.baseScaleY") &&
    app.includes("size: 56 * 4"),
  simpleExports:
    !index.includes("Compressed PLY") &&
    !index.includes("PlayCanvas SOG") &&
    !app.includes("compressed-ply") &&
    !app.includes('formatKey !== "ply"') &&
    index.indexOf('id="savePlyButton"') < index.indexOf('id="savePngButton"'),
};

assert.ok(Object.values(contracts).every(Boolean), JSON.stringify(contracts));
console.log(JSON.stringify({ ok: true, contracts }, null, 2));
