import {
  Application,
  Asset,
  AssetListLoader,
  Color,
  Entity,
  FILLMODE_NONE,
  GAMMA_SRGB,
  RESOLUTION_AUTO,
  TONEMAP_NONE,
  Vec3,
} from "./vendor/playcanvas-2.20.6.min.mjs";
import {
  TILT_FOV_DEGREES,
  cameraDiagnostics,
  clampOrbitAngles,
  fitOrbitRadius,
  frameWorldCorners,
  orbitCameraPose,
} from "./tilt-camera.mjs";

const ENGINE_VERSION = "2.20.6";

function loadAssets(assets, registry, signal) {
  const loader = new AssetListLoader(assets, registry);
  return new Promise((resolve, reject) => {
    const cleanup = () => signal?.removeEventListener("abort", abort);
    const abort = () => {
      loader.destroy();
      cleanup();
      reject(new DOMException("Tilt viewer loading was cancelled.", "AbortError"));
    };
    if (signal?.aborted) {
      abort();
      return;
    }
    signal?.addEventListener("abort", abort, { once: true });
    loader.load((error, failed) => {
      cleanup();
      loader.destroy();
      if (error) {
        reject(new Error(`PlayCanvas could not load the PLY (${failed?.length || 0} failed assets).`));
      } else {
        resolve();
      }
    });
  });
}

function plyVertexCount(buffer) {
  const prefix = new TextDecoder("ascii").decode(new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 4096)));
  return Number(prefix.match(/^element vertex (\d+)$/m)?.[1] || 0);
}

function plySupportHalfExtents(buffer, fallback, sigma = 3) {
  const bytes = new Uint8Array(buffer);
  const prefix = new TextDecoder("ascii").decode(bytes.subarray(0, Math.min(bytes.byteLength, 4096)));
  const marker = "end_header\n";
  const headerEnd = prefix.indexOf(marker);
  const vertices = Number(prefix.match(/^element vertex (\d+)$/m)?.[1] || 0);
  const dataOffset = headerEnd < 0 ? -1 : headerEnd + marker.length;
  if (dataOffset < 0 || vertices <= 0 || dataOffset + vertices * 68 > buffer.byteLength) return fallback;

  const view = new DataView(buffer, dataOffset);
  let x = Math.max(1e-3, Number(fallback?.x) || 1);
  let y = Math.max(1e-3, Number(fallback?.y) || 1);
  for (let index = 0; index < vertices; index += 1) {
    const row = index * 68;
    const centerX = view.getFloat32(row, true);
    const centerY = view.getFloat32(row + 4, true);
    const sx = Math.exp(view.getFloat32(row + 40, true));
    const sy = Math.exp(view.getFloat32(row + 44, true));
    const theta = 2 * Math.atan2(view.getFloat32(row + 64, true), view.getFloat32(row + 52, true));
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const extentX = sigma * Math.hypot(c * sx, s * sy);
    const extentY = sigma * Math.hypot(s * sx, c * sy);
    if (!Number.isFinite(centerX + centerY + extentX + extentY)) continue;
    x = Math.max(x, Math.abs(centerX) + extentX);
    y = Math.max(y, Math.abs(centerY) + extentY);
  }
  return { x, y };
}

async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

function waitForSplatSort(scene, signal, timeoutMs = 5000) {
  return new Promise((resolve, reject) => {
    let timer = 0;
    const cleanup = () => {
      clearTimeout(timer);
      scene.off("gsplat:sorted", sorted);
      signal?.removeEventListener("abort", abort);
    };
    const sorted = (sortTime) => {
      cleanup();
      resolve(Number(sortTime) || 0);
    };
    const abort = () => {
      cleanup();
      reject(new DOMException("Tilt viewer rendering was cancelled.", "AbortError"));
    };
    scene.once("gsplat:sorted", sorted);
    signal?.addEventListener("abort", abort, { once: true });
    timer = setTimeout(() => {
      cleanup();
      reject(new Error("PlayCanvas GSplat sort did not complete in time."));
    }, timeoutMs);
  });
}

function nextAnimationFrame() {
  return new Promise((resolve) => requestAnimationFrame(resolve));
}

export async function createTiltViewer({ canvas, plyBuffer, frame, signal, onCameraChange }) {
  if (!(canvas instanceof HTMLCanvasElement)) throw new Error("Tilt viewer canvas is unavailable.");
  if (!(plyBuffer instanceof ArrayBuffer) || plyBuffer.byteLength === 0) throw new Error("Tilt viewer PLY is empty.");
  signal?.throwIfAborted?.();

  let blobUrl = URL.createObjectURL(new Blob([plyBuffer], { type: "application/octet-stream" }));
  let app;
  try {
    app = new Application(canvas, {
      graphicsDeviceOptions: {
        antialias: false,
        alpha: false,
        preserveDrawingBuffer: true,
        powerPreference: "high-performance",
      },
    });
    app.setCanvasFillMode(FILLMODE_NONE, canvas.clientWidth, canvas.clientHeight);
    app.setCanvasResolution(RESOLUTION_AUTO);
    // Sorting is asynchronous. Render only on camera/resize changes, then request
    // one more frame when PlayCanvas reports that the new order is available.
    app.autoRender = false;
    app.start();

    const camera = new Entity("Tilt Camera");
    camera.addComponent("camera", {
      clearColor: new Color(0, 0, 0, 1),
      farClip: 100,
      fov: TILT_FOV_DEGREES,
      gammaCorrection: GAMMA_SRGB,
      nearClip: 0.01,
      toneMapping: TONEMAP_NONE,
    });
    app.root.addChild(camera);

    const asset = new Asset("Image2SplatPaint PLY", "gsplat", {
      url: blobUrl,
      filename: "image2splatpaint.ply",
    });
    const vertices = plyVertexCount(plyBuffer);
    const sortTimeoutMs = Math.min(30000, Math.max(5000, 5000 + vertices / 50000 * 1000));
    await loadAssets([asset], app.assets, signal);
    URL.revokeObjectURL(blobUrl);
    blobUrl = "";

    const firstSort = waitForSplatSort(app.scene, signal, sortTimeoutMs);
    const splat = new Entity("Planar Gaussian painting");
    splat.addComponent("gsplat", { asset });
    app.root.addChild(splat);

    const support = plySupportHalfExtents(plyBuffer, frame);
    let pitch = 0;
    let yaw = 0;
    let orbitRadius = 1;
    let lastCameraDiagnostics = null;
    let destroyed = false;

    const requestPresentedFrame = async () => {
      if (destroyed) return;
      app.renderNextFrame = true;
      await nextAnimationFrame();
      await nextAnimationFrame();
    };

    const onSorted = () => {
      if (!destroyed) app.renderNextFrame = true;
    };
    app.scene.on("gsplat:sorted", onSorted);

    const resize = () => {
      if (destroyed || !canvas.clientWidth || !canvas.clientHeight) return;
      app.resizeCanvas(canvas.clientWidth, canvas.clientHeight);
      orbitRadius = fitOrbitRadius(support, {
        width: canvas.clientWidth,
        height: canvas.clientHeight,
      });
      app.renderNextFrame = true;
    };

    const updateCamera = () => {
      if (destroyed) return;
      const viewport = {
        width: Math.max(1, canvas.clientWidth),
        height: Math.max(1, canvas.clientHeight),
      };
      const pose = orbitCameraPose(pitch, yaw, orbitRadius);
      camera.setPosition(...pose.position);
      camera.lookAt(0, 0, 0);
      const analytic = cameraDiagnostics(frame, viewport, pitch, yaw, orbitRadius, TILT_FOV_DEGREES);
      const playCanvasCorners = frameWorldCorners(frame).map((corner) => {
        const screen = camera.camera.worldToScreen(
          new Vec3(corner[0], corner[1], corner[2]),
          new Vec3(),
        );
        return { x: screen.x, y: screen.y, depth: screen.z };
      });
      const cornerErrorMaxPx = Math.max(...playCanvasCorners.map((corner, index) => Math.hypot(
        corner.x - analytic.corners[index].x,
        corner.y - analytic.corners[index].y,
      )));
      lastCameraDiagnostics = {
        ...analytic,
        playCanvasCorners,
        cornerErrorMaxPx,
      };
      onCameraChange?.(lastCameraDiagnostics);
      app.renderNextFrame = true;
    };

    const resizeObserver = new ResizeObserver(() => {
      resize();
      updateCamera();
    });
    resizeObserver.observe(canvas);
    resize();
    updateCamera();
    const firstSortMs = await firstSort;
    await requestPresentedFrame();
    const plyDigest = await sha256Hex(plyBuffer);

    return {
      engineVersion: ENGINE_VERSION,
      backend: app.graphicsDevice.isWebGPU ? "WebGPU" : "WebGL 2",
      plyDigest,
      plyByteLength: plyBuffer.byteLength,
      vertices,
      firstSortMs,
      setTilt(nextPitch, nextYaw) {
        const clamped = clampOrbitAngles(nextPitch, nextYaw);
        pitch = clamped.pitchDegrees;
        yaw = clamped.yawDegrees;
        updateCamera();
        return { pitch, yaw };
      },
      render() {
        app.renderNextFrame = true;
      },
      async setTiltAndWait(nextPitch, nextYaw) {
        const clamped = clampOrbitAngles(nextPitch, nextYaw);
        const requestedPitch = clamped.pitchDegrees;
        const requestedYaw = clamped.yawDegrees;
        if (requestedPitch === pitch && requestedYaw === yaw) {
          await requestPresentedFrame();
          return { pitch, yaw, sortMs: 0 };
        }
        const sort = waitForSplatSort(app.scene, signal, sortTimeoutMs);
        pitch = requestedPitch;
        yaw = requestedYaw;
        updateCamera();
        const sortMs = await sort;
        await requestPresentedFrame();
        return { pitch, yaw, sortMs };
      },
      async renderAndWait() {
        await requestPresentedFrame();
      },
      async captureFrameBlob(type = "image/jpeg", quality = 0.78) {
        await requestPresentedFrame();
        return new Promise((resolve, reject) => {
          canvas.toBlob(
            (blob) => blob ? resolve(blob) : reject(new Error("Tilt frame capture failed.")),
            type,
            quality,
          );
        });
      },
      async refreshCameraDiagnostics() {
        resize();
        updateCamera();
        await requestPresentedFrame();
        updateCamera();
        return lastCameraDiagnostics;
      },
      diagnostics() {
        return {
          engineVersion: ENGINE_VERSION,
          backend: app.graphicsDevice.isWebGPU ? "WebGPU" : "WebGL 2",
          plyDigest,
          plyByteLength: plyBuffer.byteLength,
          vertices,
          firstSortMs,
          pitch,
          yaw,
          orbitRadius,
          fovDegrees: TILT_FOV_DEGREES,
          camera: lastCameraDiagnostics,
        };
      },
      destroy() {
        if (destroyed) return;
        destroyed = true;
        resizeObserver.disconnect();
        app.scene.off("gsplat:sorted", onSorted);
        app.destroy();
        if (blobUrl) URL.revokeObjectURL(blobUrl);
      },
    };
  } catch (error) {
    app?.destroy?.();
    if (blobUrl) URL.revokeObjectURL(blobUrl);
    throw error;
  }
}
