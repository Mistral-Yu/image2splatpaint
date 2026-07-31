<div align="center">
  <h1>Image2SplatPaint</h1>
  <p><strong>Rebuild a single image from trainable splats, directly in your browser.</strong></p>
  <p>
    Compare Gaussian, geometric, and paint-like representations,
    inspect reconstruction quality, and export the result.
  </p>
  <p>
    <a href="https://mistral-yu.github.io/image2splatpaint/web/index.html"><strong>Launch the app</strong></a>
    ·
    <a href="#quick-start">Quick start</a>
    ·
    <a href="#algorithms">Algorithms</a>
    ·
    <a href="#development">Development</a>
  </p>
  <p><code>WebGPU</code> <code>Browser-only training</code> <code>MIT</code></p>
</div>

<p align="center">
  <img src="assets/readme-ui.png" width="1280" alt="Image2SplatPaint with a loaded image, training controls, and two-row status display" />
</p>

## Quick start

1. Select **Load image**, use **Sample**, or drop an image on the canvas.
2. Choose an **Algorithm**, then set **Max image side**, splat counts, and
   **Iterations**.
3. Select **Train**. Use **Pause** or **Stop** if needed.
4. Switch between **Original** and **Splats**, then review L1, SSIM, PSNR, and
   the GPU/state indicators.
5. Open **Export** to save the current rendering as PNG. Gaussian results can
   also be saved as a standard 3DGS PLY; virtual-camera results can be inspected
   in **Tilt**.

## Requirements and privacy

> [!IMPORTANT]
> Training requires a WebGPU-capable browser and GPU. The maintained release
> target is current desktop Chrome on macOS.

Larger image and splat limits can use substantial GPU memory and take longer to
finish. The page shows a visible explanation and retry action when WebGPU cannot
be initialized.

Loaded images, training state, previews, and generated results are processed
locally in the browser. The app has no image-upload or analytics endpoint.
GitHub Pages still serves the static application files, and browser or hosting
logs are outside the app's control. Training state is not persisted across a
reload, so save PNG or PLY results you want to keep.

## Algorithms

| Algorithm | Purpose | Exact export | Tilt |
| --- | --- | --- | --- |
| `Planar Gaussian` | Stable front-view approximation of one image | PNG and standard 3DGS PLY | No |
| `Rectangle Splats` | Analytic rectangle, trapezoid, or triangle paint shapes | PNG | No |
| `Layered Opaque Brush` | Layered opaque illustrative-oil brush shapes | PNG | No |
| `GS Virtual Camera Sampling` | Thin-depth 3DGS-style training from front and virtual teachers | PNG and standard 3DGS PLY | Yes |

All four choices share one custom WebGPU optimizer and standard front-to-back
alpha compositing. Each algorithm keeps its own initialization, kernel, opacity
semantics, settings, metrics, and export capability. Grid initialization and
density growth respect the source image's pixel aspect. `GS Virtual Camera
Sampling` can additionally learn a small bounded depth and is the only path that
enables virtual-camera teachers and the `Tilt` tab.

The selector configures the next run. Once a result exists, Export eligibility
and Tilt availability stay bound to that completed result until Reset, Clear,
image replacement, or a new completed run.

<details>
<summary><strong>Advanced Rectangle and Brush behavior</strong></summary>

`Rectangle Splats` exposes `Min` and `Max` values for `Edge / long edge`.
`1 / 1` keeps the current rectangular kernel, `0 / 1` makes a triangle,
equal nonzero values keep both parallel edges at the same width, and
`0 <= Min <= Max <= 1` produces a deterministic trapezoid range. Optional
Rectangle-only controls can preserve footprint area while tapering, point the
narrow edge toward stronger local structure, keep flat-region splats
rectangular, and use a harder narrow edge with a softer wide edge. `1 / 1`
retains the existing Rectangle training path.

`Rectangle Splats` uses a trainable minimum opacity floor. `Layered Opaque
Brush` instead has its own fixed paint opacity and does not share that Rectangle
setting. The former optional Brush Line layer and Brush-profile choices are no
longer part of the product UI.

</details>

## Input images

The app probes supported image headers before decoding. Large supported images
use a bounded decoder when available and are cached at no more than 4096 pixels
on the long side. `Max image side` is a separate training-time resize. The
status bar reports the current cached or training size, not the pre-cache source
dimensions. Decoder support varies by browser; the app retains a guarded
fallback for formats without bounded decode support.

## Training feedback

`Update quality metrics during training`, under `Learning rates`, optionally
refreshes L1, global/local SSIM, and PSNR at a bounded interval. It is off by
default because the read-only full-image evaluations can slow training; final
metrics are always calculated.

The two-row status area reports iteration and splat counts, L1, global and local
SSIM, RGB PSNR, image size, background/outside coverage, speed and elapsed time,
tracked GPU allocation, and the current state. Starting a new run clears the
previous run's quality values.

## Exports

- PNG: current Splats preview, including shape, effects, alpha background, and optional outside-image padding.
- PLY: available for the Gaussian algorithms. It uses standard SH0 Gaussian
  Splatting fields in aspect-preserving coordinates and stores the learned
  layer-order depth plus any enabled bounded virtual depth in `z`.

`Rectangle Splats` and `Layered Opaque Brush` use analytic non-Gaussian
kernels, so their exact export is PNG and PLY is intentionally disabled.

PLY opacity, color, scale, and rotation use standard pre-activation fields. The
training renderer and exported result use standard front-to-back alpha blending.

## Development

Open the root `index.html` directly, or serve the repository root with a static server.
Training and the self-hosted PlayCanvas `Tilt` tab both work from `file://`.
GitHub Pages is deployed by `.github/workflows/pages.yml`; the workflow publishes
only the app, the generated geometric Sample image, and the author-owned ramen
sample in `assets/source-images/`. Their provenance is recorded in
`assets/source-images/README.md`.

Run the dependency-free public release contract before publishing:

```sh
node verify-release.mjs
```

The `Tilt` tab uses a pinned, self-hosted build of
[PlayCanvas Engine](https://github.com/playcanvas/engine) 2.20.6 (MIT) to load
the generated PLY without uploading it. Training and the primary preview remain
the custom WebGPU implementation; Tilt uses the backend negotiated by PlayCanvas.
See [Third-Party Notices](THIRD_PARTY_NOTICES.md).

This repository is an experiment in AI-assisted implementation and validation
built with GPT-5.6.

## Roadmap

- Improve training methods for paint-oriented effects.
- Improve compatibility with conventional 3D Gaussian Splatting workflows.

## License

Image2SplatPaint is released under the [MIT License](LICENSE).
