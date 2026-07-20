# Image2SplatPaint

Image2SplatPaint approximates one image with splats entirely
in the browser. Training, the primary preview, and export use JavaScript and WebGPU.
This repository was created with GPT-5.6 as an experiment in AI-assisted
implementation and validation.

## Use

[Open Image2SplatPaint on GitHub Pages](https://mistral-yu.github.io/image2splatpaint/web/index.html)

WebGPU is required. The active model is a custom planar Gaussian optimizer with
SH0 and planar splats with bounded micro-depth for layer order. Virtual-camera training can optionally learn an additional thin bounded depth. Training and preview use a
custom WebGPU renderer. Grid initialization and density growth respect the
source image's pixel aspect.

The Algorithm selector separates the front-only `Planar Gaussian` optimizer
from `GS Virtual Camera Sampling`. Virtual-camera teachers and the `Tilt` tab
are only enabled for the latter, so front-only optimizer changes do not alter
the virtual-camera path.

## Workflow

1. Load or drop an image.
2. Set the image limit, splat counts, iterations, and optimizer controls.
3. Train while L1 and SSIM update, then compare against the original.
4. Save the rendered frame or export the splats.

## Exports

- PNG: current Splats preview, including shape, effects, alpha background, and optional outside-image padding.
- PLY: standard SH0 Gaussian Splatting fields in aspect-preserving coordinates. It stores the learned layer-order depth plus any enabled bounded virtual depth in `z`.

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
Training and the self-hosted PlayCanvas `Tilt` tab both work from `file://`.
GitHub Pages is deployed by `.github/workflows/pages.yml`; the artifact contains the app, the
generated geometric Sample image, and the author-owned ramen benchmark in `assets/source-images/`. Their provenance is recorded
in `assets/source-images/README.md`.

The `Tilt` tab uses a pinned, self-hosted build of
[PlayCanvas Engine](https://github.com/playcanvas/engine) 2.20.6 (MIT) to load
the generated PLY without uploading it. Training and the primary preview remain
the custom WebGPU implementation; Tilt uses the backend negotiated by PlayCanvas.
See [Third-Party Notices](THIRD_PARTY_NOTICES.md).

## License

Image2SplatPaint is released under the [MIT License](LICENSE).
