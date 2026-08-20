import { lstat, readdir, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { relative, resolve, sep } from "node:path";

const root = resolve(fileURLToPath(new URL(".", import.meta.url)));

const artifactFiles = Object.freeze([
  ".nojekyll",
  "LICENSE",
  "THIRD_PARTY_NOTICES.md",
  "index.html",
  "assets/source-images/README.md",
  "assets/source-images/generated-geometric-sample.jpg",
  "assets/source-images/ramen-photo.jpg",
  "web/app.js",
  "web/core/numeric-utils.js",
  "web/core/config.js",
  "web/export/canvas-blob.js",
  "web/export/ply-serializer.js",
  "web/export/ply-inspector.js",
  "web/input/image-metadata.js",
  "web/input/image-loader.js",
  "web/gpu/metrics.js",
  "web/gpu/device.js",
  "web/gpu/renderer.js",
  "web/gpu/tile-pipelines.js",
  "web/gpu/tile-runtime.js",
  "web/gpu/optimizer-runtime.js",
  "web/training/trainer.js",
  "web/ui/controls.js",
  "web/index.html",
  "web/sample-image-data.js",
  "web/styles.css",
  "web/tilt-viewer.bundle.js",
  "web/vendor/PLAYCANVAS-LICENSE.txt",
]);

function check(name, condition, failures) {
  if (!condition) failures.push(name);
}

function hasEnglishDocument(source) {
  return /<html\b[^>]*\blang=["']en(?:-[A-Za-z0-9]+)?["']/i.test(source) && !/[ぁ-んァ-ヶ一-龯々]/.test(source);
}

function metaContent(source, name) {
  const tag = source.match(new RegExp(`<meta\\b(?=[^>]*\\bname=["']${name}["'])[^>]*>`, "i"))?.[0];
  return tag?.match(/\bcontent=(["'])(.*?)\1/i)?.[2]?.trim() || "";
}

function contentSecurityPolicy(source) {
  const tag = source.match(/<meta\b(?=[^>]*\bhttp-equiv=["']Content-Security-Policy["'])[^>]*>/i)?.[0];
  return tag?.match(/\bcontent=(["'])(.*?)\1/i)?.[2] || "";
}

function localAssetReferences(source) {
  const references = [];
  for (const tag of source.matchAll(/<(?:script|link)\b[^>]*>/gi)) {
    const value = tag[0].match(/\b(?:src|href)=["']([^"']+)["']/i)?.[1];
    if (value) references.push(value);
  }
  return references.every((value) => !/^(?:https?:)?\/\//i.test(value));
}

function permissionMap(job) {
  const block = job.match(/^    permissions:\n((?:      [^\n]*\n?)*)/m)?.[1] || "";
  return Object.fromEntries(
    [...block.matchAll(/^      ([A-Za-z-]+):\s*([^\s#]+)/gm)].map(([, key, value]) => [key, value]),
  );
}

function sameMap(actual, expected) {
  return JSON.stringify(Object.entries(actual).sort()) === JSON.stringify(Object.entries(expected).sort());
}

function jobBlock(workflow, name, nextName = null) {
  const start = workflow.indexOf(`  ${name}:\n`);
  if (start < 0) return "";
  const end = nextName ? workflow.indexOf(`  ${nextName}:\n`, start + 1) : -1;
  return workflow.slice(start, end < 0 ? workflow.length : end);
}

async function listedArtifactFiles(site) {
  const files = [];
  const visit = async (directory) => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const fullPath = resolve(directory, entry.name);
      if (entry.isDirectory()) {
        await visit(fullPath);
      } else if (entry.isFile()) {
        files.push(relative(site, fullPath).split(sep).join("/"));
      } else {
        files.push(`${relative(site, fullPath).split(sep).join("/")} (not a regular file)`);
      }
    }
  };
  await visit(site);
  return files.sort();
}

export async function verifyArtifact(sitePath) {
  const site = resolve(sitePath);
  const failures = [];
  const actualFiles = await listedArtifactFiles(site);
  const unexpected = actualFiles.filter((path) => !artifactFiles.includes(path));
  const missing = artifactFiles.filter((path) => !actualFiles.includes(path));

  check("artifact has no unexpected files", unexpected.length === 0, failures);
  check("artifact has every required file", missing.length === 0, failures);

  for (const path of artifactFiles) {
    const sourcePath = resolve(root, path);
    const artifactPath = resolve(site, path);
    const [sourceInfo, artifactInfo] = await Promise.all([lstat(sourcePath), lstat(artifactPath)]);
    check(`${path} is a regular source file`, sourceInfo.isFile(), failures);
    check(`${path} is a regular artifact file`, artifactInfo.isFile(), failures);
    const [sourceBytes, artifactBytes] = await Promise.all([readFile(sourcePath), readFile(artifactPath)]);
    check(`${path} has byte parity`, sourceBytes.equals(artifactBytes), failures);
  }

  if (failures.length) {
    const details = [
      ...failures,
      ...(missing.length ? [`missing: ${missing.join(", ")}`] : []),
      ...(unexpected.length ? [`unexpected: ${unexpected.join(", ")}`] : []),
    ];
    throw new Error(`Pages artifact verification failed:\n- ${details.join("\n- ")}`);
  }

  return { files: actualFiles };
}

export async function verifyRelease(sitePath) {
  const [rootHtml, webHtml, app, styles, workflow, readme, readmeScreenshot] = await Promise.all([
    readFile(resolve(root, "index.html"), "utf8"),
    readFile(resolve(root, "web/index.html"), "utf8"),
    readFile(resolve(root, "web/app.js"), "utf8"),
    readFile(resolve(root, "web/styles.css"), "utf8"),
    readFile(resolve(root, ".github/workflows/pages.yml"), "utf8"),
    readFile(resolve(root, "README.md"), "utf8"),
    readFile(resolve(root, "assets/readme-ui.png")),
  ]);
  const failures = [];
  const algorithmSelect = webHtml.match(/<select\b(?=[^>]*\bid=["']algorithmSelect["'])[^>]*>([\s\S]*?)<\/select>/i)?.[1] || "";
  const algorithmValues = [...algorithmSelect.matchAll(/<option\b[^>]*\bvalue=["']([^"']+)["'][^>]*>/gi)].map(([, value]) => value);
  const liveMetricsInput = webHtml.match(/<input\b(?=[^>]*\bid=["']liveQualityMetrics["'])[^>]*>/i)?.[0] || "";
  const fileInput = webHtml.match(/<input\b(?=[^>]*\bid=["']fileInput["'])[^>]*>/i)?.[0] || "";
  const workflowActions = [...workflow.matchAll(/^\s*uses:\s*([^\s#]+)(?:\s+#.*)?$/gm)].map(([, value]) => value);
  const build = jobBlock(workflow, "build", "deploy");
  const deploy = jobBlock(workflow, "deploy");
  const expectedAlgorithms = [
    "planar-gaussian",
    "rectangle-splats",
    "layered-opaque-brush",
    "gs-virtual-camera-sampling",
  ];
  const csp = contentSecurityPolicy(webHtml);
  const pngSignature = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const screenshotIsPng =
    readmeScreenshot.length >= 24 &&
    readmeScreenshot.length <= 1024 * 1024 &&
    readmeScreenshot.subarray(0, 8).equals(pngSignature) &&
    readmeScreenshot.readUInt32BE(16) === 1280 &&
    readmeScreenshot.readUInt32BE(20) === 720;

  check("root launcher is English", hasEnglishDocument(rootHtml), failures);
  check("app UI is English", hasEnglishDocument(webHtml), failures);
  check("first-party app source has no Japanese UI literals", !/[ぁ-んァ-ヶ一-龯々]/.test(app), failures);
  check("root launcher has a meta description", Boolean(metaContent(rootHtml, "description")), failures);
  check("app has a meta description", Boolean(metaContent(webHtml, "description")), failures);
  check("root launcher has a CSP", Boolean(contentSecurityPolicy(rootHtml)), failures);
  check("app has a self-only CSP", csp.includes("default-src 'self'") && csp.includes("script-src 'self'") && csp.includes("style-src 'self'"), failures);
  check("app exposes the custom English file chooser", fileInput.includes('type="file"') && webHtml.includes("Choose image file") && webHtml.includes("drop-zone-file-button"), failures);
  check("app has a visible local-only privacy notice", /\b(?:no uploads?|never uploads?|processed locally|kept locally|stays? (?:on|in) (?:this|your) (?:device|browser))\b/i.test(webHtml), failures);
  check("live quality metrics are default-off", liveMetricsInput.includes('type="checkbox"') && !/\bchecked(?:\s|=|>)/i.test(liveMetricsInput), failures);
  check("app exposes exactly four public algorithms", JSON.stringify(algorithmValues) === JSON.stringify(expectedAlgorithms), failures);
  check(
    "export parity accounts for one alpha quantization step in premultiplied color",
    app.includes("const premultipliedTolerance = alphaMaximum > 0 ? 2 : 1") &&
      app.includes("alphaMaximum <= 1 && premultipliedMaximum <= premultipliedTolerance"),
    failures,
  );
  check("README uses the verified public UI screenshot", readme.includes('src="assets/readme-ui.png"') && screenshotIsPng, failures);
  check("HTML has no remote script or stylesheet", localAssetReferences(`${rootHtml}\n${webHtml}`), failures);
  check("CSS has no remote stylesheet or URL", !/(?:@import|url)\s*(?:\(|)["']?(?:https?:)?\/\//i.test(styles), failures);
  check("workflow builds pull requests", /^  pull_request:\s*$/m.test(workflow), failures);
  check("workflow never uses pull_request_target", !/\bpull_request_target\b/.test(workflow), failures);
  check("workflow has no ignored local script dependency", !workflow.includes("scripts/"), failures);
  check("workflow uses full-SHA action refs", workflowActions.length > 0 && workflowActions.every((value) => /^[^@\s]+@[0-9a-f]{40}$/i.test(value)), failures);
  check("workflow has deny-by-default permissions", /^permissions:\s*\{\}\s*$/m.test(workflow), failures);
  check("build job has read-only contents permission", sameMap(permissionMap(build), { contents: "read" }), failures);
  check("deploy job has only Pages deployment permissions", sameMap(permissionMap(deploy), { pages: "write", "id-token": "write" }), failures);
  check("deploy job is excluded from pull requests", /^    if:\s*github\.event_name\s*!=\s*['"]pull_request['"]\s*$/m.test(deploy), failures);
  check("workflow explicitly prepares the artifact before the release gate", workflow.indexOf("rm -rf _site") >= 0 && workflow.indexOf("rm -rf _site") < workflow.indexOf("node verify-release.mjs _site") && workflow.includes("mkdir -p _site/web _site/web/vendor _site/assets/source-images"), failures);

  if (failures.length) throw new Error(`Release verification failed:\n- ${failures.join("\n- ")}`);

  const artifact = sitePath ? await verifyArtifact(sitePath) : null;
  return { artifactFiles: artifact?.files || null };
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : "";
if (invokedPath === fileURLToPath(import.meta.url)) {
  const result = await verifyRelease(process.argv[2]);
  console.log(JSON.stringify({ ok: true, ...result }, null, 2));
}
