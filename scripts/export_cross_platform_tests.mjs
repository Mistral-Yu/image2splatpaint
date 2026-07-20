import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const [html, app, readme] = await Promise.all([
  readFile(resolve(root, "web/index.html"), "utf8"),
  readFile(resolve(root, "web/app.js"), "utf8"),
  readFile(resolve(root, "README.md"), "utf8"),
]);

const checks = {
  separate_png_and_splat_buttons: html.includes('id="savePngButton"') && html.includes('id="savePlyButton"') && !html.includes('<option value="png">'),
  unstable_export_rows_removed: !html.includes('id="exportCoverage"') && !html.includes('id="exportLastFile"'),
  splat_preview_png_render:
    /async function makeSplatPreviewPngBlob\(\)/.test(app) &&
    app.includes("captureRenderedRgba") &&
    app.includes("currentSplatPngSpec") &&
    html.includes("Save Splat PNG"),
  gpu_texture_readback: app.includes("copyTextureToBuffer") && app.includes('this.format.startsWith("bgra")'),
  nonblack_png_guard: app.includes("PNG frame readback is unexpectedly all black") && app.includes("nonblack_pixels"),
  png_file_protocol:
    !app.includes("requiresServer") &&
    !app.includes("Compressed exports require localhost") &&
    app.includes('saveExport({ formatKey: "png" })'),
  image_decode_feature_detection: app.includes('typeof createImageBitmap === "function"') && app.includes("using Canvas image decode"),
  pointer_pan_and_pinch: app.includes('event.pointerType !== "mouse"') && app.includes("state.canvasPinch.scale * distance"),
  visibility_pause: app.includes('document.addEventListener("visibilitychange"') && app.includes("training remains paused until Resume"),
  training_starts_fitted: app.includes('state.previewMode = "splats";\n  fitCanvases(state.image.width, state.image.height);'),
  no_os_sniffing: !/navigator\.(userAgent|platform)/.test(app),
  runtime_webgpu_detection:
    app.includes('if (!("gpu" in navigator))') &&
    app.includes('reason: "navigator.gpu unavailable"') &&
    app.includes("WebGPU unavailable; training disabled") &&
    app.includes("WebGPU required; training not started") &&
    !readme.includes("still need physical-device checks"),
};

const failed = Object.entries(checks).filter(([, value]) => !value).map(([key]) => key);
if (failed.length) throw new Error(`export/cross-platform checks failed: ${failed.join(", ")}`);
console.log(JSON.stringify(checks, null, 2));
