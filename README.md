# Image2GaussianPaint

Image2GaussianPaint approximates one image with planar Gaussian splats entirely
in the browser. Training, rendering, and export use JavaScript and WebGPU.
This repository was created with GPT-5.6 as an experiment in AI-assisted
implementation and validation.

## Use

[Open Image2GaussianPaint on GitHub Pages](https://mistral-yu.github.io/image2gaussianpaint/)

WebGPU is required. The active model is a custom planar Gaussian optimizer with
sRGB colors, SH0, and every splat center on `z = 0`. Training and preview use a
custom WebGPU renderer. Grid initialization and density growth respect the
source image's pixel aspect.

## Workflow

1. Load or drop an image.
2. Set the image limit, splat counts, iterations, and optimizer controls.
3. Train while L1 and SSIM update, then compare against the original.
4. Save the rendered frame or export the splats.

## Exports

- PNG: WebGPU render cropped to the loaded image frame.
- PLY: Graphdeco-style SH0 data with `z = 0` and source-image aspect preserved
  in planar world coordinates.

PLY opacity, color, scale, and rotation use standard pre-activation fields. The
training renderer uses order-independent normalized weighted blending, so
overlapping transparency can look different in depth-sorted 3DGS viewers.

## Compatibility

macOS browsers are verified. Windows, iPhone, and Android code paths are
prepared but still need physical-device checks. iOS must expose `navigator.gpu`.

## Development

```bash
node scripts/static-server.mjs 8765
node scripts/public_surface_tests.mjs
node scripts/build-pages.mjs
node scripts/pages_artifact_tests.mjs
```

Open `http://127.0.0.1:8765/`. The root `index.html` also works directly. GitHub Pages is
deployed by `.github/workflows/pages.yml`; the artifact contains only `index.html`, `LICENSE`, `web/`, and `.nojekyll`.

[PlayCanvas Engine](https://github.com/playcanvas/engine) (MIT) was consulted
for GPU resource-accounting design, but no PlayCanvas code or runtime is
included in the released app.

## License

Image2GaussianPaint is released under the [MIT License](LICENSE).
