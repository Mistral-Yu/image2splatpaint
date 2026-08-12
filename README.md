<div align="center">
  <h1>Image2SplatPaint</h1>
  <p><strong>Explore a new differentiable stylization medium built from trainable splats, directly in your browser.</strong></p>
  <p>
    Turn one image into Gaussian, geometric, and paint-like representations,
    inspect how they transform it, and export the result.
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

## Project direction

Image2SplatPaint explores both faithful image representation and a new form of
**differentiable image stylization inspired by Gaussian Splatting**.

Instead of treating splats only as compression primitives, the project treats
their position, scale, anisotropy, orientation, opacity, layer order, and
paint-like shape as an image-making vocabulary. `Planar Gaussian` and
`GS Virtual Camera Sampling` aim for faithful reproduction: the former targets
a stable front view, while the latter extends that goal to bounded tilted
views. `Rectangle Splats` and `Brush Splats` instead explore geometric and
illustrative stylization that may deliberately depart from the source.
Its discrete paint layers, layer-aware optimization and ordering, Brush and
Rectangle shape semantics, and paint-oriented controls are original parts of
Image2SplatPaint rather than features inherited from the referenced methods.
For the stylization algorithms, visible style, stroke structure, and the
character of the rendered image matter alongside numerical quality metrics.

## Quick start

1. Choose an **Algorithm**. Algorithm and setting changes apply to the next
   Train without reloading the page or clearing the loaded image and current
   result.
2. Select **Load image**, use **Sample**, or drop an image on the canvas.
3. Set **Max image side**, splat counts, and **Iterations**, then select
   **Train**. Use **Pause** or **Stop** if needed.
4. Switch between **Original** and **Splats**, then review the visible result.
   RGB L1, SSIM, and PSNR help evaluate fidelity and compare runs; for Rectangle
   and Brush results, also evaluate their visible style and stroke structure.
   The status area reports GPU and runtime state.
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
| `Brush Splats` | Directional illustrative brush shapes | PNG | No |
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

`Rectangle Splats` exposes `Min` and `Max` values for
`Short edge / long edge`. `1 / 1` keeps the current rectangular kernel.
Lower values taper the short parallel edge while the opposite edge stays at
full width: `0` allows triangle tips, and `0 <= Min <= Max <= 1` distributes a
deterministic mix of triangles, trapezoids, and rectangles across the paint
layers. Optional Rectangle-only controls can preserve each footprint area
while tapering, point the narrow edge toward stronger local structure, prefer
the selected Max ratio in flat regions, and use a harder narrow edge with a
softer wide edge. `1 / 1` retains the existing Rectangle training path.
`Learned opacity (before gradient)` bounds each Rectangle's trainable opacity
to `0.005...0.995`. `Opacity gradient multiplier (short / long)` is a fixed
`0...1` multiplier that changes linearly from the trapezoid's short edge to its
long edge. Final opacity is `learned opacity × gradient multiplier`. The defaults
`0.995 / 0.995` and `1 / 1` preserve the former uniform `0.995` behavior.

### Brush Splats settings

Brush settings apply at the next Train start. Experimental checkboxes remain
off by default. Equal directional endpoints are treated as a uniform/no-taper
setting, so the defaults retain the accepted Brush path.

| Setting | What it changes |
| --- | --- |
| `Learned opacity (before gradient)` | Bounds each trainable Brush opacity to Min...Max in the safe `0.005...0.995` range. The default is `0.995 / 0.995`. |
| `Opacity gradient multiplier` | A fixed `0...1` directional multiplier; it is not trained. Final opacity is learned opacity multiplied by this gradient in training, preview, standard-alpha overlap, and PNG export. The default `1 / 1` is uniform. |
| `Aspect ratio (long side / short side)` | Sets Brush-specific Min and Max anisotropy in one row. Defaults are `1 / 8`; both values override the Shared `Max anisotropy` behavior during Brush training. |
| `Train directional width taper` | Learns a separate taper amount per splat between the configured directional Min and Max widths. Equal endpoints disable the directional change; the default `1 / 1` preserves the prior untapered path. |
| `Contribution-confirmed detail promotion` | Keeps a newly split detail stroke in its inherited paint layer until contribution and footprint checks confirm that moving it forward is useful. This is experimental. |
| `Local color-flow orientation` | Softly aligns nearby directional splats when their colors and paint layers are similar. Broad patches and strong direction crossings are excluded. This is experimental. |
| `Directional stroke aspect floor` | Softly maintains a minimum long/short ratio while preserving footprint area. `Ribbon minimum` defaults to `2.2`; `Accent minimum` defaults to `2.8`; Base Patches are unchanged. These are lower bounds, while `Maximum long / short ratio` is the upper bound. This is experimental. |

The general Brush Min/Max applies to every Brush splat. When Directional stroke
aspect floor is enabled, Ribbon and Accent additionally use their stronger
family-specific minimums, capped by the Brush Max.

The former training-teacher preprocessing, saturation gradient, Brush Line
layer, Brush-profile choices, and rejected Sector-aware, optical-smoothing, and
residual-matching experiments are not part of the product UI. Their comparison
records remain in the local Brush experiment registry.

</details>

## Input images

The app probes supported image headers before decoding. Large supported images
use a bounded decoder when available and are cached at no more than 4096 pixels
on the long side. `Max image side` is a separate training-time resize. The
status bar reports the current cached or training size, not the pre-cache source
dimensions. Decoder support varies by browser; the app retains a guarded
fallback for formats without bounded decode support. Training resize and GPU
estimates use the decoded display orientation, so EXIF-rotated images keep the
same aspect ratio when Train starts.

## Training feedback

`Update quality metrics during training`, under `Shared training settings`, optionally
refreshes RGB L1, global/local SSIM, and PSNR at a bounded interval. It is off by
default because the read-only full-image evaluations can slow training and may
trigger a memory safety stop when their temporary allocation would exceed the
budget. Live values are the latest completed evaluation; final values are fixed
only by the final evaluation.

The following `Color workflow` is also shared by every algorithm. `Monochrome
underpainting` initializes and optimizes CIELAB L* lightness only until `Color
finish starts at (%)`, then returns to the standard RGB objective. It is off by
default and does not change RGB rendering or final RGB quality metrics.

The two-row status area reports iteration and splat counts, RGB L1, global and
local SSIM, RGB PSNR, image size, low-alpha (alpha below 0.99) and outside
coverage, speed and elapsed time, tracked GPU allocation, and the current state.
Starting a new run clears the previous run's quality values.

## Exports

- PNG: current Splats preview, including shape, effects, alpha background, and optional outside-image padding.
- PLY: available for the Gaussian algorithms. It uses standard SH0 Gaussian
  Splatting fields in aspect-preserving coordinates and stores the learned
  layer-order depth plus any enabled bounded virtual depth in `z`.

These PLY files are learned from one image and prioritize its front view.
Rotated views—especially from `Planar Gaussian`—will generally be less complete
than conventional 3DGS trained from calibrated multi-view photos. Virtual
Camera Sampling improves bounded tilt, but does not replace true multi-view
geometry or view-dependent color.

`Rectangle Splats` and `Brush Splats` use analytic non-Gaussian
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

The Pages workflow rebuilds `_site` from scratch and then runs
`node verify-release.mjs _site` to require byte parity with the reviewed source.

The `Tilt` tab uses a pinned, self-hosted build of
[PlayCanvas Engine](https://github.com/playcanvas/engine) 2.20.6 (MIT) to load
the generated PLY without uploading it. Training and the primary preview remain
the custom WebGPU implementation; Tilt uses the backend negotiated by PlayCanvas.
See [Third-Party Notices](THIRD_PARTY_NOTICES.md).

This repository is an experiment in AI-assisted implementation and validation
built with GPT-5.6.

## Research inspiration

The project is independently implemented, but its direction has been informed
by research that treats images as learnable sets of spatial primitives:

- [Image-GS: Content-Adaptive Image Representation via 2D Gaussians](https://arxiv.org/abs/2407.01866)
  demonstrates adaptive image representation with progressively optimized
  anisotropic 2D Gaussians.
- [Soft Anisotropic Diagrams for Differentiable Image Representation](https://luckyiyi.github.io/SAD/index.html)
  explores learnable anisotropic sites, differentiable spatial ownership, and
  content-aligned boundaries.

Image2SplatPaint takes inspiration from those broader ideas, but it is not a
reimplementation of either method. In particular, its discrete paint-layer
model, layer-aware ordering and optimization, Brush and Rectangle primitives,
opacity semantics, and paint-oriented training controls are independently
designed for this project. The implementation combines those original elements
with standard front-to-back alpha compositing in its own WebGPU optimizer,
extending the design space toward editable, paint-like, and deliberately
stylized splat representations.

## Roadmap

- Improve faithful image representation for the Planar Gaussian and Virtual
  Camera paths.
- Develop Rectangle and Brush splats as distinct stylization and image-making
  media.
- Improve training methods for paint-oriented effects, stroke structure, and
  controllable visual character.
- Improve compatibility with conventional 3D Gaussian Splatting workflows.

## License

Image2SplatPaint is released under the [MIT License](LICENSE).
