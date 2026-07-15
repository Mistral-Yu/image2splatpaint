# Image2SplatPaint

Image2SplatPaint approximates one image with splats entirely
in the browser. Training, the primary preview, and export use JavaScript and WebGPU.
This repository was created with GPT-5.6 as an experiment in AI-assisted
implementation and validation.

## Use

[Open Image2SplatPaint on GitHub Pages](https://mistral-yu.github.io/image2splatpaint/)

WebGPU is required. The active model is a custom planar Gaussian optimizer with
sRGB colors, SH0, and planar splats with optional bounded micro-depth for layer order. Training and preview use a
custom WebGPU renderer. Grid initialization and density growth respect the
source image's pixel aspect.

The Algorithm selector currently exposes the implemented `Planar Gaussian`
backend. It is structured for future non-Gaussian splat painters without
presenting unavailable modes.

Initial splat colors come from the image. Opacity remains trainable and is
evaluated separately from RGB reconstruction quality.

## Workflow

1. Load or drop an image.
2. Set the image limit, splat counts, iterations, and optimizer controls.
3. Train while L1 and SSIM update, then compare against the original.
4. Inspect the in-memory PLY on a fixed-radius orbit up to 75 degrees, or render a 49-pose Fibonacci hemisphere in the `Tilt` tab.
5. Save the rendered frame or export the splats.

## Exports

- PNG: WebGPU render cropped to the loaded image frame.
- PLY: Graphdeco-style SH0 data in aspect-preserving planar coordinates. Learned micro-depth is on by default for stable layer ordering; disabling it exports `z = 0`.

PLY opacity, color, scale, and rotation use standard pre-activation fields. The
training renderer and exported result use standard front-to-back alpha blending.

## Compatibility

Chrome on macOS is verified. Windows, iPhone, and Android code paths are
prepared but still need physical-device checks. iOS must expose `navigator.gpu`.

## Development

```bash
node scripts/static-server.mjs 8765
node scripts/public_surface_tests.mjs
node scripts/build-pages.mjs
node scripts/pages_artifact_tests.mjs
```

Open `http://127.0.0.1:8765/`. The root `index.html` also works directly.
The training app works from `file://`; the lazy PlayCanvas `Tilt` tab requires
GitHub Pages or a local HTTP server.
GitHub Pages is deployed by `.github/workflows/pages.yml`; the artifact contains the app plus the
author-owned ramen sample in `assets/source-images/`. Its provenance is recorded
in `assets/source-images/README.md`.

The `Tilt` tab uses a pinned, self-hosted build of
[PlayCanvas Engine](https://github.com/playcanvas/engine) 2.20.6 (MIT) to load
the generated PLY without uploading it. Training and the primary preview remain
the custom WebGPU implementation; Tilt uses the backend negotiated by PlayCanvas.
See [Third-Party Notices](THIRD_PARTY_NOTICES.md).

## License

Image2SplatPaint is released under the [MIT License](LICENSE).
