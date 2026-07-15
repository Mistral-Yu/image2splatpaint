const REQUIRED_PROPERTIES = [
  "x", "y", "z", "nx", "ny", "nz",
  "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
  "scale_0", "scale_1", "scale_2",
  "rot_0", "rot_1", "rot_2", "rot_3",
];

export function inspectIndependentPly(input) {
  const bytes = input instanceof Uint8Array
    ? input
    : new Uint8Array(input.buffer, input.byteOffset || 0, input.byteLength);
  const marker = new TextEncoder().encode("end_header\n");
  let markerOffset = -1;
  outer: for (let index = 0; index <= bytes.length - marker.length; index += 1) {
    for (let offset = 0; offset < marker.length; offset += 1) {
      if (bytes[index + offset] !== marker[offset]) continue outer;
    }
    markerOffset = index;
    break;
  }
  if (markerOffset < 0) throw new Error("independent PLY parser: end_header missing");
  const dataOffset = markerOffset + marker.length;
  const header = new TextDecoder("ascii").decode(bytes.subarray(0, dataOffset));
  if (!header.includes("format binary_little_endian 1.0")) throw new Error("independent PLY parser: binary little-endian format required");
  const vertices = Number(header.match(/^element vertex (\d+)$/m)?.[1] || 0);
  const properties = [...header.matchAll(/^property float (\S+)$/gm)].map((match) => match[1]);
  if (JSON.stringify(properties) !== JSON.stringify(REQUIRED_PROPERTIES)) {
    throw new Error(`independent PLY parser: property mismatch ${properties.join(",")}`);
  }
  const rowBytes = properties.length * 4;
  const payloadBytes = bytes.length - dataOffset;
  if (payloadBytes !== vertices * rowBytes) {
    throw new Error(`independent PLY parser: payload ${payloadBytes}/${vertices * rowBytes}`);
  }
  const frameMatch = header.match(/^comment image2gaussianpaint_frame (\d+) (\d+)$/m);
  const layerMatch = header.match(/^comment image2gaussianpaint_layer_order (micro_z|flat_z0) ([0-9.eE+-]+)$/m);
  const frameWidth = Number(frameMatch?.[1] || 0);
  const frameHeight = Number(frameMatch?.[2] || 0);
  const layerMode = layerMatch?.[1] || "";
  const layerSpan = Number(layerMatch?.[2] || 0);
  const view = new DataView(bytes.buffer, bytes.byteOffset + dataOffset, payloadBytes);
  let finite = true;
  let planarQuaternion = true;
  let validOpacity = true;
  let validScale = true;
  let zAbsMax = 0;
  for (let vertex = 0; vertex < vertices; vertex += 1) {
    const row = vertex * rowBytes;
    for (let property = 0; property < properties.length; property += 1) {
      if (!Number.isFinite(view.getFloat32(row + property * 4, true))) finite = false;
    }
    const z = view.getFloat32(row + 2 * 4, true);
    const opacity = 1 / (1 + Math.exp(-view.getFloat32(row + 9 * 4, true)));
    const sx = Math.exp(view.getFloat32(row + 10 * 4, true));
    const sy = Math.exp(view.getFloat32(row + 11 * 4, true));
    const sz = Math.exp(view.getFloat32(row + 12 * 4, true));
    const qx = view.getFloat32(row + 14 * 4, true);
    const qy = view.getFloat32(row + 15 * 4, true);
    zAbsMax = Math.max(zAbsMax, Math.abs(z));
    if (qx !== 0 || qy !== 0) planarQuaternion = false;
    if (!(opacity > 0 && opacity < 1)) validOpacity = false;
    if (!(sx > 0 && sy > 0 && sz > 0)) validScale = false;
  }
  const zContract = layerMode === "flat_z0"
    ? zAbsMax === 0
    : layerMode === "micro_z" && zAbsMax <= layerSpan * 0.501;
  return {
    valid:
      vertices > 0 &&
      frameWidth > 0 &&
      frameHeight > 0 &&
      header.includes("comment image2gaussianpaint_blend standard_alpha") &&
      finite &&
      planarQuaternion &&
      validOpacity &&
      validScale &&
      zContract,
    vertices,
    properties,
    rowBytes,
    payloadBytes,
    frameWidth,
    frameHeight,
    frameAspect: frameWidth / frameHeight,
    standardAlpha: header.includes("comment image2gaussianpaint_blend standard_alpha"),
    layerMode,
    layerSpan,
    zAbsMax,
    finite,
    planarQuaternion,
    validOpacity,
    validScale,
    zContract,
  };
}
