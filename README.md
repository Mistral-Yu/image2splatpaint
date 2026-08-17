<div align="center">
  <h1>Image2SplatPaint</h1>
  <p><strong>Recreate or stylize an image with trainable splats.</strong></p>
  <p>Runs directly in your WebGPU browser.</p>
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

Image2SplatPaint explores faithful image representation and differentiable
stylization with trainable splats. `Planar Gaussian` and
`GS Virtual Camera Sampling` focus on reproduction, while `Rectangle Splats`
and `Brush Splats` use geometric shapes, strokes, and independent paint layers
to create deliberately stylized results. For these stylization paths, visible
character matters alongside numerical fidelity. They can also act as trainable,
shape-aware blur filters that simplify fine detail into larger paint forms.

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
| `Local color-flow orientation` | Softly aligns nearby directional splats when their colors and paint layers are similar. Broad patches and strong direction crossings are excluded. This is experimental. |
| `Directional stroke aspect floor` | Softly maintains a minimum long/short ratio while preserving footprint area. `Ribbon minimum` defaults to `2.2`; `Accent minimum` defaults to `2.8`; Base Patches are unchanged. These are lower bounds, while `Maximum long / short ratio` is the upper bound. This is experimental. |

The general Brush Min/Max applies to every Brush splat. When Directional stroke
aspect floor is enabled, Ribbon and Accent additionally use their stronger
family-specific minimums, capped by the Brush Max.

New Brush detail children always inherit their parent's paint layer. They can
move later through the shared contribution-aware layer training; the former
birth-time one-layer promotion has been removed. When layer training moves a
Paint splat forward, stale RGB is repaired from its source-image footprint
before the new order becomes visible.

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

The status area reports progress, splat count, image quality, coverage, speed,
elapsed time, and tracked GPU use. Optional live quality updates are off by
default because full-image evaluation can slow training. The shared
`Monochrome underpainting` option begins with lightness and switches to RGB at
the selected point; final quality metrics always evaluate the RGB result.

## Exports

- PNG exports the current rendered result for every algorithm.
- Splat PNG resolution can use the training size, 2K, 4K, or a custom long side
  while preserving the trained image aspect ratio.
- Standard SH0 3DGS PLY is available for the two Gaussian algorithms.
  Rectangle and Brush use non-Gaussian kernels and therefore export PNG only.

PLY results are learned from one image and prioritize its front view. Virtual
Camera Sampling supports bounded tilt, but does not replace true multi-view
geometry or view-dependent color.

## Development

Open `index.html` directly or serve the repository as static files. Before
publishing, run:

```sh
node verify-release.mjs
```

GitHub Pages publishes the reviewed static app. Training uses the custom WebGPU
implementation; `Tilt` uses a pinned self-hosted
[PlayCanvas Engine](https://github.com/playcanvas/engine) build. See
[Third-Party Notices](THIRD_PARTY_NOTICES.md). This project is developed and
validated with AI assistance.

## Research inspiration

Image2SplatPaint was inspired by
[Image-GS](https://arxiv.org/abs/2407.01866) and
[Soft Anisotropic Diagrams](https://luckyiyi.github.io/SAD/index.html), while
its implementation and paint-oriented design are developed independently.

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
