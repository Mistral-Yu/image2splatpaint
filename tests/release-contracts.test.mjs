import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";
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
  assert.ok(indexOf("gpu/renderer.js") < indexOf("gpu/tile-pipelines.js"));
  assert.ok(indexOf("gpu/renderer.js") < indexOf("gpu/optimizer-runtime.js"));
  assert.ok(indexOf("app.js") < indexOf("ui/bootstrap.js"));
  assert.ok(indexOf("app.js") < indexOf("ui/controls.js"));
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

test("GPU prototype extension rejects silent method replacement", async () => {
  const renderer = await readFile(new URL("../web/gpu/renderer.js", import.meta.url), "utf8");
  const tilePipelines = await readFile(new URL("../web/gpu/tile-pipelines.js", import.meta.url), "utf8");
  const tileRuntime = await readFile(new URL("../web/gpu/tile-runtime.js", import.meta.url), "utf8");
  const optimizer = await readFile(new URL("../web/gpu/optimizer-runtime.js", import.meta.url), "utf8");
  assert.match(renderer, /hasOwnProperty\.call\(WebGpuPreview\.prototype, name\)/);
  assert.match(renderer, /WebGpuPreview method collision/);
  assert.match(tilePipelines, /installWebGpuPreviewMethods\(WebGpuTilePipelines\.prototype/);
  assert.match(tileRuntime, /installWebGpuPreviewMethods\(WebGpuTileRuntime\.prototype/);
  assert.match(optimizer, /installWebGpuPreviewMethods\(WebGpuOptimizerRuntime\.prototype/);
});
