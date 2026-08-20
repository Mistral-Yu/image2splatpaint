(function installPlySerializer(global) {
  function serializeBinaryPly({
    header,
    count,
    rowBytes,
    shC0,
    geometryAt,
    depthAt,
    rgbAt,
    opacityAt,
    logit,
  }) {
    const headerBytes = new TextEncoder().encode(header);
    const buffer = new ArrayBuffer(headerBytes.byteLength + count * rowBytes);
    const bytes = new Uint8Array(buffer);
    bytes.set(headerBytes, 0);
    const view = new DataView(buffer, headerBytes.byteLength);
    let offset = 0;
    const writeFloat = (value) => {
      view.setFloat32(offset, value, true);
      offset += 4;
    };

    for (let index = 0; index < count; index += 1) {
      const geometry = geometryAt(index);
      const rgb = rgbAt(index);
      const halfTheta = geometry.theta * 0.5;
      writeFloat(geometry.x);
      writeFloat(geometry.y);
      writeFloat(depthAt(index));
      writeFloat(0);
      writeFloat(0);
      writeFloat(0);
      writeFloat((rgb[0] - 0.5) / shC0);
      writeFloat((rgb[1] - 0.5) / shC0);
      writeFloat((rgb[2] - 0.5) / shC0);
      writeFloat(logit(opacityAt(index)));
      writeFloat(Math.log(Math.max(geometry.sx, 1e-6)));
      writeFloat(Math.log(Math.max(geometry.sy, 1e-6)));
      writeFloat(Math.log(1e-4));
      writeFloat(Math.cos(halfTheta));
      writeFloat(0);
      writeFloat(0);
      writeFloat(Math.sin(halfTheta));
    }
    return buffer;
  }

  global.Image2SplatPaintPlySerializer = Object.freeze({ serializeBinaryPly });
})(globalThis);
