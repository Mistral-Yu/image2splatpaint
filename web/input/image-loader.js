const RGBA8_FLOAT_CONVERSION_ROWS_PER_YIELD = 32;

async function rgba8ToFloatArrays(imageData, width, height, yieldFrame = nextFrame) {
  const rgba = imageData.data;
  const rgb = new Float32Array(width * height * 3);
  const alpha = new Float32Array(width * height);
  for (let row = 0; row < height; row += 1) {
    const rgbaStart = row * width * 4;
    const rgbStart = row * width * 3;
    const alphaStart = row * width;
    for (let x = 0, rgbaOffset = rgbaStart, rgbOffset = rgbStart; x < width; x += 1, rgbaOffset += 4, rgbOffset += 3) {
      rgb[rgbOffset] = rgba[rgbaOffset] / 255;
      rgb[rgbOffset + 1] = rgba[rgbaOffset + 1] / 255;
      rgb[rgbOffset + 2] = rgba[rgbaOffset + 2] / 255;
      alpha[alphaStart + x] = rgba[rgbaOffset + 3] / 255;
    }
    if ((row + 1) % RGBA8_FLOAT_CONVERSION_ROWS_PER_YIELD === 0 && row + 1 < height) {
      await yieldFrame();
    }
  }
  return { rgb, alpha };
}

async function resizeLoadedImageToMaxSide(maxSide) {
  if (!state.image) return false;
  const sourceImage = state.image;
  // The decoded cache is display-oriented. Header dimensions can still be in
  // encoded EXIF orientation, so using originalWidth/originalHeight here can
  // swap the aspect ratio when Train is pressed. The cache dimensions also
  // let a later larger run resize again from sourceBitmap without upscaling.
  const sourceWidth = state.image.cacheWidth || state.image.width;
  const sourceHeight = state.image.cacheHeight || state.image.height;
  const targetSide = Math.round(clampNumber(maxSide, LIMITS.trainSizeMin, LIMITS.trainSizeMax, Math.max(sourceWidth, sourceHeight)));
  const [width, height] = resizedSize(sourceWidth, sourceHeight, targetSide);
  if (width === state.image.width && height === state.image.height) return false;

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  let imageData;
  try {
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("2D canvas is unavailable for image resize.");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    if (state.image.sourceBitmap) {
      ctx.drawImage(state.image.sourceBitmap, 0, 0, width, height);
    } else {
      const source = document.createElement("canvas");
      source.width = state.image.width;
      source.height = state.image.height;
      try {
        const sourceContext = source.getContext("2d");
        if (!sourceContext) throw new Error("2D canvas is unavailable for image resize.");
        sourceContext.putImageData(
          rgbToImageData(state.image.rgb, state.image.width, state.image.height, state.image.alpha),
          0,
          0,
        );
        ctx.drawImage(source, 0, 0, width, height);
      } finally {
        releaseCanvasBackingStore(source);
      }
    }
    imageData = ctx.getImageData(0, 0, width, height);
  } finally {
    releaseCanvasBackingStore(canvas);
  }
  const { rgb, alpha } = await rgba8ToFloatArrays(imageData, width, height);
  if (state.image !== sourceImage) return false;

  cancelCompletedResultGpuRecovery();
  state.webgpu.renderer?.disposeTrainState();
  state.webgpu.renderer?.disposeResultRenderState();
  state.image = {
    ...sourceImage,
    width,
    height,
    resizeMode: "max-side",
    resizeScale: Math.max(width, height) /
      Math.max(1, state.image.originalWidth || sourceWidth, state.image.originalHeight || sourceHeight),
    rgb,
    alpha,
  };
  updateImageSizeStatus();
  state.params = null;
  state.flowSplatResult = null;
  state.metrics = null;
  state.previewPadding = previewPaddingSpec(state.image, null, false);
  state.lastDownload = "";
  els.stepText.textContent = "0 / 0";
  els.splatText.textContent = "-";
  els.lossText.textContent = "-";
  els.psnrText.textContent = "-";
  els.ssimText.textContent = "-";
  els.regionalSsimText.textContent = "-";
  els.boundaryText.textContent = "-";
  els.coverageText.textContent = "- / -";
  resetTrainingTiming();
  clearSplatAdjustmentBaseline();
  els.previewCanvas.width = width;
  els.previewCanvas.height = height;
  els.gpuCanvas.width = width;
  els.gpuCanvas.height = height;
  fitCanvases(width, height);
  setPreviewMode("original");
  updateDownloads(false);
  setStatus("image loaded");
  setTrainingMessage(`Prepared ${state.image.fileName} at ${width} x ${height}.`, "success");
  updateMemoryRecommendation();
  updateVirtualCameraCoverageEstimate();
  return true;
}

function beginImageLoad(message = "Loading image...") {
  invalidateTrainingRun("image load");
  destroyTiltViewer({ restoreCanvas: true });
  const loadGeneration = state.imageLoadGeneration + 1;
  state.imageLoadGeneration = loadGeneration;
  state.imageLoading = true;
  setStatus("loading image");
  setTrainingMessage(message);
  publishState();
  return loadGeneration;
}

function finishImageLoad(loadGeneration) {
  if (loadGeneration !== state.imageLoadGeneration) return;
  state.imageLoading = false;
  publishState();
}

async function loadFile(file, { loadGeneration: inheritedLoadGeneration = null } = {}) {
  const inheritedLoad = Number.isFinite(inheritedLoadGeneration);
  if (!inheritedLoad && trainingLifecycleInputLocked()) {
    throw new Error("Stop training before loading another image.");
  }
  if (inheritedLoad && inheritedLoadGeneration !== state.imageLoadGeneration) return false;
  if (!file?.type?.startsWith("image/")) {
    throw new Error("Choose an image file.");
  }
  if (file.size > MAX_INPUT_FILE_BYTES) {
    throw new Error(`Image file is too large (maximum ${Math.round(MAX_INPUT_FILE_BYTES / MB)} MB).`);
  }
  const loadGeneration = inheritedLoad
    ? inheritedLoadGeneration
    : beginImageLoad(`Loading ${file.name}...`);
  setStatus("loading image");
  setTrainingMessage(`Loading ${file.name}...`);
  publishState();
  state.webgpu.renderer?.disposeTrainState();
  let decoded = null;
  let cacheCanvas = null;
  let decodedReleased = false;
  const releaseDecoded = () => {
    if (decodedReleased) return;
    decodedReleased = true;
    decoded?.close?.();
  };
  try {
    decoded = await decodeImageFile(file);
    if (loadGeneration !== state.imageLoadGeneration) return false;
    const originalWidth = decoded.originalWidth || decoded.width;
    const originalHeight = decoded.originalHeight || decoded.height;
    if (originalWidth * originalHeight > MAX_INPUT_DECODED_PIXELS) {
      throw new Error(`Decoded image is too large (maximum ${MAX_INPUT_DECODED_PIXELS.toLocaleString()} pixels).`);
    }
    els.trainSize.value = String(clampNumber(els.trainSize.value, LIMITS.trainSizeMin, LIMITS.trainSizeMax, DEFAULT_MAX_SIDE));
    const [width, height] = resizedSize(
      decoded.width,
      decoded.height,
      INPUT_CACHE_MAX_SIDE,
      INPUT_CACHE_MAX_SIDE,
    );
    const loadScale = Math.max(width, height) / Math.max(1, originalWidth, originalHeight);
    const canvas = document.createElement("canvas");
    cacheCanvas = canvas;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("2D canvas is unavailable for image load.");
    ctx.drawImage(decoded.source, 0, 0, width, height);
    const imageData = ctx.getImageData(0, 0, width, height);
    releaseDecoded();
    const { rgb, alpha } = await rgba8ToFloatArrays(imageData, width, height);
    if (loadGeneration !== state.imageLoadGeneration) return false;
    cancelCompletedResultGpuRecovery();
    state.webgpu.renderer?.disposeResultRenderState();
    releaseImageSource(state.image);
    state.image = {
      fileName: file.name,
      width,
      height,
      originalWidth,
      originalHeight,
      cacheWidth: width,
      cacheHeight: height,
      inputCacheMaxSide: INPUT_CACHE_MAX_SIDE,
      inputCacheResized: width !== originalWidth || height !== originalHeight,
      inputDecodeMode: decoded.mode,
      inputOrientation: decoded.orientation || 1,
      resizeMode: "input-cache",
      resizeScale: loadScale,
      rgb,
      alpha,
      sourceBitmap: canvas,
    };
    cacheCanvas = null;
    if (loadGeneration !== state.imageLoadGeneration) return false;
    updateImageSizeStatus();
    state.params = null;
    state.flowSplatResult = null;
    state.metrics = null;
    state.previewPadding = previewPaddingSpec(state.image, null, false);
    state.lastDownload = "";
    els.stepText.textContent = "0 / 0";
    els.splatText.textContent = "-";
    els.lossText.textContent = "-";
    els.psnrText.textContent = "-";
    els.ssimText.textContent = "-";
    els.regionalSsimText.textContent = "-";
    els.boundaryText.textContent = "-";
    els.coverageText.textContent = "- / -";
    resetTrainingTiming();
    clearSplatAdjustmentBaseline();
    els.dropZone.classList.add("ready");
    els.previewCanvas.width = width;
    els.previewCanvas.height = height;
    els.gpuCanvas.width = width;
    els.gpuCanvas.height = height;
    fitCanvases(width, height);
    setPreviewMode("original");
    updateDownloads(false);
    updateMemoryRecommendation();
    updateVirtualCameraCoverageEstimate();
    syncAlgorithmRequirements();
    setStatus("image loaded");
    const cacheNote = state.image.inputCacheResized
      ? ` Cached at ${width} x ${height}; Max image side is applied separately when Train starts.`
      : "";
    setTrainingMessage(`Loaded ${file.name} (${originalWidth} x ${originalHeight}).${cacheNote}`, "success");
    log(`loaded ${file.name}; input cache=${width}x${height} source=${originalWidth}x${originalHeight} orientation=${decoded.orientation || 1} scale=${loadScale.toFixed(3)} decode=${decoded.mode} sRGB; training resize deferred`);
    return true;
  } finally {
    releaseDecoded();
    releaseCanvasBackingStore(cacheCanvas);
    finishImageLoad(loadGeneration);
  }
}

async function probeImageDimensions(file) {
  const bytes = new Uint8Array(
    await file.slice(0, Math.min(file.size, MAX_IMAGE_HEADER_PROBE_BYTES)).arrayBuffer(),
  );
  return parseImageDimensions(bytes, file.type);
}

async function decodeImageFile(file) {
  const probed = await probeImageDimensions(file).catch(() => null);
  if (!probed) {
    throw new Error(
      "Image dimensions could not be read safely before decode. Use PNG, JPEG, WebP, TIFF, AVIF, HEIC, BMP, or GIF.",
    );
  }
  if (probed.width * probed.height > MAX_INPUT_DECODED_PIXELS) {
    throw new Error(`Decoded image is too large (maximum ${MAX_INPUT_DECODED_PIXELS.toLocaleString()} pixels).`);
  }
  const encodedWidth = probed.width;
  const encodedHeight = probed.height;
  const orientation = probed.orientation || 1;
  const [originalWidth, originalHeight] = displayOrientedImageSize(
    encodedWidth,
    encodedHeight,
    orientation,
  );
  const [desiredWidth, desiredHeight] = resizedSize(
    originalWidth,
    originalHeight,
    INPUT_CACHE_MAX_SIDE,
    INPUT_CACHE_MAX_SIDE,
  );
  const needsBoundedDecode = desiredWidth !== originalWidth || desiredHeight !== originalHeight;
  const swapsAxes = orientationSwapsImageAxes(orientation);

  // WebKit can lose EXIF orientation when createImageBitmap() receives a Blob,
  // while its HTMLImageElement path is display-oriented. Dimensions alone
  // cannot reveal whether pixels were already rotated when resizeWidth and
  // resizeHeight force an exact output size, so avoid manual double rotation.
  if (swapsAxes) {
    return decodeImageFileViaHtmlElement(file, {
      expectedWidth: originalWidth,
      expectedHeight: originalHeight,
      orientation,
      mode: needsBoundedDecode ? "html-image-bounded-exif" : "html-image-exif",
    });
  }

  if (needsBoundedDecode && typeof ImageDecoder === "function") {
    let decoder = null;
    let frame = null;
    try {
      const type = file.type === "image/jpg" ? "image/jpeg" : file.type;
      if (type && await ImageDecoder.isTypeSupported(type)) {
        const data = await file.arrayBuffer();
        decoder = new ImageDecoder({
          type,
          data,
          desiredWidth,
          desiredHeight,
          preferAnimation: false,
          transfer: [data],
        });
        const result = await decoder.decode({ frameIndex: 0, completeFramesOnly: true });
        frame = result.image;
        const width = frame.displayWidth || frame.codedWidth;
        const height = frame.displayHeight || frame.codedHeight;
        if (width > 0 && height > 0) {
          return {
            source: frame,
            width,
            height,
            originalWidth,
            originalHeight,
            orientation,
            mode: width <= desiredWidth && height <= desiredHeight
              ? "webcodecs-bounded"
              : "webcodecs-best-effort",
            close() {
              frame?.close();
              decoder?.close();
              frame = null;
              decoder = null;
            },
          };
        }
      }
    } catch (error) {
      log(`ImageDecoder bounded decode failed; using ImageBitmap: ${error.message}`);
    }
    frame?.close();
    decoder?.close();
  }

  if (typeof createImageBitmap === "function") {
    try {
      const bitmap = needsBoundedDecode
        ? await createImageBitmap(file, {
          resizeWidth: desiredWidth,
          resizeHeight: desiredHeight,
          resizeQuality: "high",
        })
        : await createImageBitmap(file);
      const decodedAspect = bitmap.width / Math.max(1, bitmap.height);
      const expectedAspect = desiredWidth / Math.max(1, desiredHeight);
      if (Math.abs(decodedAspect - expectedAspect) > Math.max(1e-6, expectedAspect * 0.001)) {
        bitmap.close?.();
        throw new Error(`decoded aspect ${decodedAspect.toFixed(6)} differs from expected ${expectedAspect.toFixed(6)}`);
      }
      return {
        source: bitmap,
        width: bitmap.width,
        height: bitmap.height,
        originalWidth: originalWidth || bitmap.width,
        originalHeight: originalHeight || bitmap.height,
        orientation,
        mode: needsBoundedDecode ? "imagebitmap-bounded" : "imagebitmap",
        close: () => bitmap.close?.(),
      };
    } catch (error) {
      log(`createImageBitmap decode failed; using Canvas image decode: ${error.message}`);
    }
  }
  return decodeImageFileViaHtmlElement(file, {
    expectedWidth: originalWidth,
    expectedHeight: originalHeight,
    orientation,
    mode: "html-image-canvas",
  });
}

async function decodeImageFileViaHtmlElement(file, {
  expectedWidth,
  expectedHeight,
  orientation,
  mode,
}) {
  const url = URL.createObjectURL(file);
  let image = null;
  try {
    image = await loadImageElement(url);
    if (image.naturalWidth * image.naturalHeight > MAX_INPUT_DECODED_PIXELS) {
      throw new Error(`Decoded image is too large (maximum ${MAX_INPUT_DECODED_PIXELS.toLocaleString()} pixels).`);
    }
    const fallbackAspect = image.naturalWidth / Math.max(1, image.naturalHeight);
    const expectedAspect = expectedWidth / Math.max(1, expectedHeight);
    if (Math.abs(fallbackAspect - expectedAspect) > Math.max(1e-6, expectedAspect * 0.001)) {
      throw new Error(
        `Image orientation could not be decoded safely (expected aspect ${expectedAspect.toFixed(6)}, got ${fallbackAspect.toFixed(6)}).`,
      );
    }
    const [width, height] = resizedSize(
      image.naturalWidth,
      image.naturalHeight,
      INPUT_CACHE_MAX_SIDE,
      INPUT_CACHE_MAX_SIDE,
    );
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    try {
      const context = canvas.getContext("2d");
      if (!context) throw new Error("2D canvas is unavailable for image decode.");
      context.drawImage(image, 0, 0, width, height);
      return {
        source: canvas,
        width,
        height,
        originalWidth: image.naturalWidth,
        originalHeight: image.naturalHeight,
        orientation,
        mode,
        close() {
          releaseCanvasBackingStore(canvas);
        },
      };
    } catch (error) {
      releaseCanvasBackingStore(canvas);
      throw error;
    }
  } finally {
    image?.removeAttribute("src");
    URL.revokeObjectURL(url);
  }
}

function imagePathCandidates(value) {
  const path = value.trim().replaceAll("\\", "/");
  if (!path) throw new Error("Enter an image path.");
  if (/^file:\/\//.test(path)) {
    if (location.protocol !== "file:") throw new Error("file:// images only work when the app itself is opened with file://.");
    return [path];
  }
  if (/^[a-zA-Z]:\//.test(path)) {
    if (location.protocol === "file:") return [`file:///${path}`];
    throw new Error("Local absolute paths need file:// launch, file input/drop, or a local server path.");
  }
  if (path.startsWith("/Users/") || path.startsWith("/home/")) {
    if (location.protocol === "file:") return [`file://${path}`];
    throw new Error("Local absolute paths need file:// launch, file input/drop, or a local server path.");
  }
  if (/^https?:\/\//.test(path) || path.startsWith("/")) return [path];
  const clean = path.replace(/^\.\//, "");
  return [`../${clean}`, clean, `./${clean}`];
}

async function loadPathImage() {
  const path = els.pathInput.value;
  const name = path.split(/[\\/]/).filter(Boolean).pop() || "path-image";
  const candidates = imagePathCandidates(path);
  const loadGeneration = beginImageLoad(`Loading ${name}...`);
  try {
    const file = await loadFirstImagePath(candidates, name);
    if (loadGeneration !== state.imageLoadGeneration) return false;
    state.lastInputMode = "path";
    return await loadFile(file, { loadGeneration });
  } finally {
    finishImageLoad(loadGeneration);
  }
}

async function loadFirstImagePath(urls, name) {
  const errors = [];
  for (const url of urls) {
    try {
      return await loadImageUrlAsFile(url, name);
    } catch (error) {
      const label = url.startsWith("data:") ? "embedded sample" : url;
      errors.push(`${label}: ${error.message}`);
    }
  }
  throw new Error(`image load failed. Tried ${errors.join(" | ")}`);
}

async function loadImageUrlAsFile(url, name) {
  if (url.startsWith("data:")) {
    const match = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
    if (!match) throw new Error("invalid embedded image data");
    const encoded = match[3];
    const upperBound = match[2]
      ? Math.ceil(encoded.length * 0.75)
      : encoded.length;
    if (upperBound > MAX_INPUT_FILE_BYTES) {
      throw new Error(`Image file is too large (maximum ${Math.round(MAX_INPUT_FILE_BYTES / MB)} MB).`);
    }
    const binary = match[2] ? atob(encoded) : decodeURIComponent(encoded);
    const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    if (bytes.byteLength > MAX_INPUT_FILE_BYTES) {
      throw new Error(`Image file is too large (maximum ${Math.round(MAX_INPUT_FILE_BYTES / MB)} MB).`);
    }
    return new File([bytes], name, { type: match[1] || "image/jpeg" });
  }
  if (location.protocol !== "file:") {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`image fetch failed: ${response.status}`);
    const declaredBytes = Number(response.headers.get("content-length"));
    if (Number.isFinite(declaredBytes) && declaredBytes > MAX_INPUT_FILE_BYTES) {
      throw new Error(`Image file is too large (maximum ${Math.round(MAX_INPUT_FILE_BYTES / MB)} MB).`);
    }
    const blob = await response.blob();
    if (blob.size > MAX_INPUT_FILE_BYTES) {
      throw new Error(`Image file is too large (maximum ${Math.round(MAX_INPUT_FILE_BYTES / MB)} MB).`);
    }
    return new File([blob], name, { type: blob.type || "image/jpeg" });
  }
  const image = await loadImageElement(url);
  if (image.naturalWidth * image.naturalHeight > MAX_INPUT_DECODED_PIXELS) {
    image.src = "";
    throw new Error(`Decoded image is too large (maximum ${MAX_INPUT_DECODED_PIXELS.toLocaleString()} pixels).`);
  }
  const canvas = document.createElement("canvas");
  const [width, height] = resizedSize(
    image.naturalWidth,
    image.naturalHeight,
    INPUT_CACHE_MAX_SIDE,
    INPUT_CACHE_MAX_SIDE,
  );
  canvas.width = width;
  canvas.height = height;
  try {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2D canvas is unavailable for local image load.");
    ctx.drawImage(image, 0, 0, width, height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!blob) throw new Error("image canvas export failed");
    return new File([blob], name.replace(/\.[^.]+$/, ".png"), { type: "image/png" });
  } finally {
    canvas.width = 1;
    canvas.height = 1;
    image.src = "";
  }
}

function loadImageElement(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`image load failed: ${url}`));
    image.src = url;
  });
}

async function loadGeneratedSample() {
  const path = "assets/source-images/generated-geometric-sample.jpg";
  const urls = [...new Set([
    ...(location.protocol === "file:" && EMBEDDED_SAMPLE_IMAGE_URL ? [EMBEDDED_SAMPLE_IMAGE_URL] : []),
    SAMPLE_IMAGE_URL,
    ...imagePathCandidates(path),
  ])];
  const loadGeneration = beginImageLoad("Loading sample image...");
  try {
    const file = await loadFirstImagePath(urls, "generated-geometric-sample.jpg");
    if (loadGeneration !== state.imageLoadGeneration) return false;
    state.lastInputMode = "sample";
    return await loadFile(file, { loadGeneration });
  } finally {
    finishImageLoad(loadGeneration);
  }
}
