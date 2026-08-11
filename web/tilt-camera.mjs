// Stay inside the open front hemisphere. A camera at exactly 90 degrees makes
// the planar inverse projection singular, but 75 degrees is not a hard limit.
export const TILT_MAX_ANGLE_DEGREES = 89;
export const TILT_FOV_DEGREES = 50;
export const FIBONACCI_HEMISPHERE_POSE_COUNT = 49;
export const FIBONACCI_GOLDEN_ANGLE_RADIANS = Math.PI * (3 - Math.sqrt(5));

const EPSILON = 1e-9;

function finite(value, fallback = 0) {
  return Number.isFinite(Number(value)) ? Number(value) : fallback;
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalize(value) {
  const length = Math.hypot(value[0], value[1], value[2]);
  if (!Number.isFinite(length) || length < EPSILON) return [0, 0, 0];
  return value.map((component) => component / length);
}

export function clampTiltAngle(value) {
  return Math.max(-TILT_MAX_ANGLE_DEGREES, Math.min(TILT_MAX_ANGLE_DEGREES, finite(value)));
}

export function clampOrbitAngles(pitchDegrees, yawDegrees, maxPolarDegrees = TILT_MAX_ANGLE_DEGREES) {
  const pitch = clampTiltAngle(pitchDegrees) * Math.PI / 180;
  const yaw = clampTiltAngle(yawDegrees) * Math.PI / 180;
  const cosPitch = Math.cos(pitch);
  const direction = [
    Math.sin(yaw) * cosPitch,
    Math.sin(pitch),
    Math.cos(yaw) * cosPitch,
  ];
  const maxPolar = Math.max(0, Math.min(TILT_MAX_ANGLE_DEGREES, finite(maxPolarDegrees, TILT_MAX_ANGLE_DEGREES))) * Math.PI / 180;
  const polar = Math.acos(Math.max(-1, Math.min(1, direction[2])));
  if (polar <= maxPolar + EPSILON) {
    return { pitchDegrees: pitch * 180 / Math.PI, yawDegrees: yaw * 180 / Math.PI, polarDegrees: polar * 180 / Math.PI, direction };
  }
  const azimuth = Math.atan2(direction[1], direction[0]);
  const sine = Math.sin(maxPolar);
  const capped = [sine * Math.cos(azimuth), sine * Math.sin(azimuth), Math.cos(maxPolar)];
  const cappedPitch = Math.asin(capped[1]);
  const cappedYaw = Math.atan2(capped[0], capped[2]);
  return {
    pitchDegrees: cappedPitch * 180 / Math.PI,
    yawDegrees: cappedYaw * 180 / Math.PI,
    polarDegrees: maxPolar * 180 / Math.PI,
    direction: capped,
  };
}

export function orbitCameraPose(pitchDegrees, yawDegrees, radius) {
  const clamped = clampOrbitAngles(pitchDegrees, yawDegrees);
  const pitch = clamped.pitchDegrees * Math.PI / 180;
  const yaw = clamped.yawDegrees * Math.PI / 180;
  const safeRadius = Math.max(0.01, finite(radius, 1));
  const cosPitch = Math.cos(pitch);
  const position = [
    safeRadius * Math.sin(yaw) * cosPitch,
    safeRadius * Math.sin(pitch),
    safeRadius * Math.cos(yaw) * cosPitch,
  ];
  const target = [0, 0, 0];
  const forward = normalize(position.map((component) => -component));
  const right = normalize(cross(forward, [0, 1, 0]));
  const up = normalize(cross(right, forward));
  return {
    pitchDegrees: clamped.pitchDegrees,
    yawDegrees: clamped.yawDegrees,
    polarDegrees: clamped.polarDegrees,
    radius: safeRadius,
    position,
    target,
    forward,
    right,
    up,
  };
}

export function orbitProjectionContract(frame, viewport, pitchDegrees, yawDegrees, radius, fovDegrees = TILT_FOV_DEGREES) {
  const pose = orbitCameraPose(pitchDegrees, yawDegrees, radius);
  const width = Math.max(1, finite(viewport?.width, 1));
  const height = Math.max(1, finite(viewport?.height, 1));
  const longSide = Math.max(1, finite(frame?.width ?? frame?.x, 1), finite(frame?.height ?? frame?.y, 1));
  const frameX = Math.max(EPSILON, finite(frame?.width ?? frame?.x, 1) / longSide);
  const frameY = Math.max(EPSILON, finite(frame?.height ?? frame?.y, 1) / longSide);
  return {
    ...pose,
    fovDegrees: Math.max(1, finite(fovDegrees, TILT_FOV_DEGREES)),
    tanHalfFov: Math.tan(Math.max(1, finite(fovDegrees, TILT_FOV_DEGREES)) * Math.PI / 360),
    aspect: width / height,
    frameX,
    frameY,
    viewport: { width, height },
  };
}

export function projectPlanarSplatPoint(point, contract, z = 0) {
  const world = [
    contract.frameX * finite(point?.[0]),
    -contract.frameY * finite(point?.[1]),
    finite(z),
  ];
  const cameraX = dot(world, contract.right);
  const cameraY = dot(world, contract.up);
  const depth = contract.radius + dot(world, contract.forward);
  const ndcX = cameraX / Math.max(EPSILON, depth * contract.tanHalfFov * contract.aspect);
  const ndcY = -cameraY / Math.max(EPSILON, depth * contract.tanHalfFov);
  return {
    point: [ndcX, ndcY],
    depth,
    valid: Number.isFinite(ndcX + ndcY + depth) && depth > EPSILON,
  };
}

export function inverseProjectPlanarSplatPoint(point, contract, z = 0) {
  const u = finite(point?.[0]);
  const v = finite(point?.[1]);
  const depthBase = contract.radius + contract.forward[2] * finite(z);
  const ax = contract.frameX * contract.right[0];
  const ay = -contract.frameY * contract.right[1];
  const bx = contract.frameX * contract.up[0];
  const by = -contract.frameY * contract.up[1];
  const cx = contract.frameX * contract.forward[0];
  const cy = -contract.frameY * contract.forward[1];
  const focalX = 1 / Math.max(EPSILON, contract.tanHalfFov * contract.aspect);
  const focalY = 1 / Math.max(EPSILON, contract.tanHalfFov);
  const a00 = u * cx - focalX * ax;
  const a01 = u * cy - focalX * ay;
  const a10 = v * cx + focalY * bx;
  const a11 = v * cy + focalY * by;
  const rhsX = focalX * contract.right[2] * z - u * depthBase;
  const rhsY = -focalY * contract.up[2] * z - v * depthBase;
  const determinant = a00 * a11 - a01 * a10;
  if (!Number.isFinite(determinant) || Math.abs(determinant) < EPSILON) return { point: [0, 0], valid: false };
  const x = (rhsX * a11 - a01 * rhsY) / determinant;
  const y = (a00 * rhsY - rhsX * a10) / determinant;
  return { point: [x, y], valid: Number.isFinite(x + y) };
}

export function fibonacciHemispherePoses(
  count = FIBONACCI_HEMISPHERE_POSE_COUNT,
  maxPolarDegrees = TILT_MAX_ANGLE_DEGREES,
) {
  const safeCount = Math.max(2, Math.round(finite(count, FIBONACCI_HEMISPHERE_POSE_COUNT)));
  const safeMaxPolar = Math.max(0, Math.min(TILT_MAX_ANGLE_DEGREES, finite(maxPolarDegrees, TILT_MAX_ANGLE_DEGREES)));
  const maxPolar = safeMaxPolar * Math.PI / 180;
  const minCosine = Math.cos(maxPolar);
  return Array.from({ length: safeCount }, (_, index) => {
    const fraction = index / (safeCount - 1);
    const cosine = 1 - fraction * (1 - minCosine);
    const polar = Math.acos(Math.max(-1, Math.min(1, cosine)));
    const azimuth = index === 0 ? 0 : (index * FIBONACCI_GOLDEN_ANGLE_RADIANS) % (Math.PI * 2);
    const sine = Math.sin(polar);
    const direction = [
      sine * Math.cos(azimuth),
      sine * Math.sin(azimuth),
      cosine,
    ];
    const pitch = Math.asin(Math.max(-1, Math.min(1, direction[1])));
    const yaw = Math.atan2(direction[0], direction[2]);
    return {
      index,
      id: `fib-${String(index).padStart(2, "0")}`,
      fraction,
      polarDegrees: polar * 180 / Math.PI,
      azimuthDegrees: azimuth * 180 / Math.PI,
      pitchDegrees: pitch * 180 / Math.PI,
      yawDegrees: yaw * 180 / Math.PI,
      direction,
    };
  });
}

export function projectWorldPoint(point, pose, viewport, fovDegrees = TILT_FOV_DEGREES) {
  const width = Math.max(1, finite(viewport?.width, 1));
  const height = Math.max(1, finite(viewport?.height, 1));
  const relative = [
    finite(point?.[0]) - pose.position[0],
    finite(point?.[1]) - pose.position[1],
    finite(point?.[2]) - pose.position[2],
  ];
  const depth = dot(relative, pose.forward);
  const halfHeight = depth * Math.tan(Math.max(1, finite(fovDegrees, TILT_FOV_DEGREES)) * Math.PI / 360);
  const halfWidth = halfHeight * width / height;
  const ndcX = dot(relative, pose.right) / Math.max(EPSILON, halfWidth);
  const ndcY = dot(relative, pose.up) / Math.max(EPSILON, halfHeight);
  return {
    x: (ndcX + 1) * width * 0.5,
    y: (1 - ndcY) * height * 0.5,
    ndcX,
    ndcY,
    depth,
    valid: Number.isFinite(ndcX + ndcY + depth) && depth > EPSILON,
  };
}

export function frameWorldCorners(frame) {
  const x = Math.max(EPSILON, finite(frame?.x, 1));
  const y = Math.max(EPSILON, finite(frame?.y, 1));
  return [
    [-x, y, 0],
    [x, y, 0],
    [x, -y, 0],
    [-x, -y, 0],
  ];
}

export function projectFrameCorners(frame, pose, viewport, fovDegrees = TILT_FOV_DEGREES) {
  return frameWorldCorners(frame).map((corner) => projectWorldPoint(corner, pose, viewport, fovDegrees));
}

function projectedInside(corners, margin) {
  return corners.every((corner) => corner.valid && Math.abs(corner.ndcX) <= margin && Math.abs(corner.ndcY) <= margin);
}

export function fitOrbitRadius(frame, viewport, options = {}) {
  const fovDegrees = Math.max(1, finite(options.fovDegrees, TILT_FOV_DEGREES));
  const maxAngle = Math.max(0, Math.min(TILT_MAX_ANGLE_DEGREES, finite(options.maxAngleDegrees, TILT_MAX_ANGLE_DEGREES)));
  const margin = Math.max(0.5, Math.min(0.98, finite(options.ndcMargin, 0.9)));
  const support = Math.max(EPSILON, Math.hypot(finite(frame?.x, 1), finite(frame?.y, 1)));
  let low = support * 1.001;
  let high = support * 32;
  const angles = Array.isArray(options.poses) && options.poses.length
    ? options.poses.map((pose) => [finite(pose.pitchDegrees), finite(pose.yawDegrees)])
    : fibonacciHemispherePoses(FIBONACCI_HEMISPHERE_POSE_COUNT, maxAngle)
      .map((pose) => [pose.pitchDegrees, pose.yawDegrees]);
  for (let iteration = 0; iteration < 40; iteration += 1) {
    const radius = (low + high) * 0.5;
    const fits = angles.every(([pitch, yaw]) => projectedInside(
      projectFrameCorners(frame, orbitCameraPose(pitch, yaw, radius), viewport, fovDegrees),
      margin,
    ));
    if (fits) high = radius;
    else low = radius;
  }
  return high;
}

export function canonicalOrbitRadius(frame, options = {}) {
  const x = Math.max(EPSILON, finite(frame?.x, 1));
  const y = Math.max(EPSILON, finite(frame?.y, 1));
  const scale = 1024 / Math.max(x, y);
  return fitOrbitRadius(frame, { width: x * scale, height: y * scale }, options);
}

export function cameraDiagnostics(frame, viewport, pitchDegrees, yawDegrees, radius, fovDegrees = TILT_FOV_DEGREES) {
  const pose = orbitCameraPose(pitchDegrees, yawDegrees, radius);
  const corners = projectFrameCorners(frame, pose, viewport, fovDegrees);
  return {
    ...pose,
    fovDegrees,
    viewport: { width: Math.max(1, finite(viewport?.width, 1)), height: Math.max(1, finite(viewport?.height, 1)) },
    center: projectWorldPoint([0, 0, 0], pose, viewport, fovDegrees),
    corners,
  };
}

export function trainingCameraMarkerGeometry(snapshot) {
  const cameras = Array.isArray(snapshot?.cameras) ? snapshot.cameras : [];
  const radius = Math.max(0.01, finite(snapshot?.orbit_radius, 1));
  const fovDegrees = Math.max(1, finite(snapshot?.fov_degrees, TILT_FOV_DEGREES));
  const aspect = Math.max(EPSILON, finite(snapshot?.image_aspect, 1));
  const target = Array.isArray(snapshot?.target) && snapshot.target.length === 3
    ? snapshot.target.map((value) => finite(value))
    : [0, 0, 0];
  const activeId = String(snapshot?.active_camera_id || "");
  const unique = new Map();
  for (const camera of cameras) {
    const pitchDegrees = finite(camera?.pitch_degrees);
    const yawDegrees = finite(camera?.yaw_degrees);
    const key = `${pitchDegrees.toFixed(6)}:${yawDegrees.toFixed(6)}`;
    const existing = unique.get(key);
    if (existing) {
      existing.ids.push(String(camera?.id || key));
      existing.multiplicity += Math.max(1, Math.round(finite(camera?.multiplicity, 1)));
      existing.active ||= String(camera?.id || "") === activeId;
      continue;
    }
    const cameraFovDegrees = Math.max(
      1,
      finite(camera?.intrinsics?.fov_degrees, fovDegrees),
    );
    const pose = orbitCameraPose(pitchDegrees, yawDegrees, radius);
    const forward = normalize(target.map((value, index) => value - pose.position[index]));
    const right = normalize(cross(forward, [0, 1, 0]));
    const up = normalize(cross(right, forward));
    const frustumDepth = radius * 0.055;
    const halfHeight = frustumDepth * Math.tan(cameraFovDegrees * Math.PI / 360);
    const halfWidth = halfHeight * aspect;
    const center = pose.position.map((value, index) => value + forward[index] * frustumDepth);
    const frustumCorners = [
      [-1, 1], [1, 1], [1, -1], [-1, -1],
    ].map(([x, y]) => center.map((value, index) => value + right[index] * halfWidth * x + up[index] * halfHeight * y));
    unique.set(key, {
      ids: [String(camera?.id || key)],
      kind: camera?.kind === "front" ? "front" : "virtual",
      pitchDegrees: pose.pitchDegrees,
      yawDegrees: pose.yawDegrees,
      position: pose.position,
      target,
      forward,
      fovDegrees: cameraFovDegrees,
      intrinsics: camera?.intrinsics ? { ...camera.intrinsics } : null,
      frustumCorners,
      multiplicity: Math.max(1, Math.round(finite(camera?.multiplicity, 1))),
      active: String(camera?.id || "") === activeId,
    });
  }
  return [...unique.values()];
}
