import {
  ASPECT_MANUAL,
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
  canonicalOrbitRadius,
  clampOrbitAngles,
  fitOrbitRadius,
  frameWorldCorners,
  orbitCameraPose,
  trainingCameraMarkerGeometry,
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

export async function createTiltViewer({
  canvas,
  plyBuffer,
  frame,
  supportFrame = frame,
  signal,
  onCameraChange,
  cameraPool = null,
}) {
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

    const sharedFovDegrees = Math.max(25, Math.min(55, Number(cameraPool?.fov_degrees) || TILT_FOV_DEGREES));
    const camera = new Entity("Tilt Camera");
    camera.addComponent("camera", {
      aspectRatioMode: ASPECT_MANUAL,
      clearColor: new Color(0, 0, 0, 1),
      farClip: 100,
      fov: sharedFovDegrees,
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

    let pitch = 0;
    let yaw = 0;
    const orbitRadius = Number.isFinite(Number(cameraPool?.orbit_radius))
      ? Math.max(0.01, Number(cameraPool.orbit_radius))
      : canonicalOrbitRadius(frame, {
        fovDegrees: sharedFovDegrees,
        maxAngleDegrees: 75,
      });
    let radiusMode = "training";
    let fitAllOrbitRadius = orbitRadius;
    let viewerOrbitRadius = orbitRadius;
    let lastCameraDiagnostics = null;
    let destroyed = false;
    let cameraMarkersVisible = true;
    let cameraPoolSnapshot = cameraPool ? structuredClone(cameraPool) : null;
    let cameraMarkers = trainingCameraMarkerGeometry(cameraPoolSnapshot);
    const markerOverviewPitch = 20;
    const markerOverviewYaw = -42;
    let markerOverviewRadius = orbitRadius * 2.65;
    const frontCameraColor = new Color(0.12, 0.72, 0.95, 1);
    const virtualCameraColor = new Color(1, 0.48, 0.12, 1);
    const activeCameraColor = new Color(1, 0.9, 0.18, 1);

    const drawMarkerSegment = (start, end, color) => {
      app.drawLine(new Vec3(...start), new Vec3(...end), color, false);
    };

    const drawCameraMarkers = () => {
      if (!cameraMarkersVisible) return;
      for (const marker of cameraMarkers) {
        const color = marker.active
          ? activeCameraColor
          : marker.kind === "front" ? frontCameraColor : virtualCameraColor;
        const scale = orbitRadius * (marker.active ? 0.025 : 0.016);
        for (let axis = 0; axis < 3; axis += 1) {
          const low = [...marker.position];
          const high = [...marker.position];
          low[axis] -= scale;
          high[axis] += scale;
          drawMarkerSegment(low, high, color);
        }
        const corners = marker.frustumCorners;
        for (let index = 0; index < corners.length; index += 1) {
          drawMarkerSegment(marker.position, corners[index], color);
          drawMarkerSegment(corners[index], corners[(index + 1) % corners.length], color);
        }
      }
    };
    app.on("prerender", drawCameraMarkers);

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
      camera.camera.aspectRatioMode = ASPECT_MANUAL;
      camera.camera.aspectRatio = canvas.clientWidth / canvas.clientHeight;
      app.renderNextFrame = true;
    };

    const updateCamera = () => {
      if (destroyed) return;
      const viewport = {
        width: Math.max(1, canvas.clientWidth),
        height: Math.max(1, canvas.clientHeight),
      };
      const viewPitch = cameraMarkersVisible ? markerOverviewPitch : pitch;
      const viewYaw = cameraMarkersVisible ? markerOverviewYaw : yaw;
      const viewRadius = cameraMarkersVisible ? markerOverviewRadius : viewerOrbitRadius;
      const pose = orbitCameraPose(viewPitch, viewYaw, viewRadius);
      camera.setPosition(...pose.position);
      camera.lookAt(0, 0, 0);
      const analytic = cameraDiagnostics(frame, viewport, viewPitch, viewYaw, viewRadius, sharedFovDegrees);
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
        viewMode: cameraMarkersVisible ? "camera-pool-overview" : "center-orbit",
        trainingOrbitRadius: orbitRadius,
        fitAllOrbitRadius,
        radiusMode,
        viewerOrbitRadius,
        supportFrame: { ...supportFrame },
        playCanvasCorners,
        cornerErrorMaxPx,
      };
      onCameraChange?.(lastCameraDiagnostics);
      app.renderNextFrame = true;
    };

    const resizeObserver = new ResizeObserver(() => {
      resize();
      fitAllOrbitRadius = fitOrbitRadius(supportFrame, {
        width: Math.max(1, canvas.clientWidth),
        height: Math.max(1, canvas.clientHeight),
      }, {
        fovDegrees: sharedFovDegrees,
        maxAngleDegrees: 75,
      });
      viewerOrbitRadius = radiusMode === "fit" ? fitAllOrbitRadius : orbitRadius;
      markerOverviewRadius = Math.max(orbitRadius, viewerOrbitRadius) * 2.65;
      updateCamera();
    });
    resizeObserver.observe(canvas);
    resize();
    fitAllOrbitRadius = fitOrbitRadius(supportFrame, {
      width: Math.max(1, canvas.clientWidth),
      height: Math.max(1, canvas.clientHeight),
    }, {
      fovDegrees: sharedFovDegrees,
      maxAngleDegrees: 75,
    });
    viewerOrbitRadius = orbitRadius;
    markerOverviewRadius = Math.max(orbitRadius, viewerOrbitRadius) * 2.65;
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
          viewerOrbitRadius,
          fitAllOrbitRadius,
          radiusMode,
          fovDegrees: sharedFovDegrees,
          camera: lastCameraDiagnostics,
          cameraMarkers: {
            visible: cameraMarkersVisible,
            count: cameraMarkers.length,
            sampleCount: cameraMarkers.reduce((total, marker) => total + marker.multiplicity, 0),
            frontCount: cameraMarkers.filter((marker) => marker.kind === "front").length,
            virtualCount: cameraMarkers.filter((marker) => marker.kind === "virtual").length,
            frontSampleCount: cameraMarkers.reduce(
              (total, marker) => total + (marker.kind === "front" ? marker.multiplicity : 0),
              0,
            ),
            virtualSampleCount: cameraMarkers.reduce(
              (total, marker) => total + (marker.kind === "virtual" ? marker.multiplicity : 0),
              0,
            ),
            samplingEnabled: Boolean(cameraPoolSnapshot?.sampling_enabled),
            activeCameraId: cameraPoolSnapshot?.active_camera_id || "",
            trainingOrbitRadius: orbitRadius,
          },
        };
      },
      setCameraMarkersVisible(visible) {
        cameraMarkersVisible = Boolean(visible);
        updateCamera();
        return cameraMarkersVisible;
      },
      setRadiusMode(mode) {
        radiusMode = mode === "fit" ? "fit" : "training";
        viewerOrbitRadius = radiusMode === "fit" ? fitAllOrbitRadius : orbitRadius;
        markerOverviewRadius = Math.max(orbitRadius, viewerOrbitRadius) * 2.65;
        updateCamera();
        return radiusMode;
      },
      setActiveCamera(cameraId) {
        if (!cameraPoolSnapshot) return false;
        cameraPoolSnapshot.active_camera_id = String(cameraId || "");
        cameraMarkers = trainingCameraMarkerGeometry(cameraPoolSnapshot);
        app.renderNextFrame = true;
        return cameraMarkers.some((marker) => marker.active);
      },
      destroy() {
        if (destroyed) return;
        destroyed = true;
        resizeObserver.disconnect();
        app.scene.off("gsplat:sorted", onSorted);
        app.off("prerender", drawCameraMarkers);
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
