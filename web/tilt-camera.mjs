export const TILT_MAX_ANGLE_DEGREES = 75;
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
