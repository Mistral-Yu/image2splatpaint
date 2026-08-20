(function installPlySerializer(global) {
  function plyFrameScale(image) {
    const width = Math.max(1, Number(image?.width) || 1);
    const height = Math.max(1, Number(image?.height) || 1);
    const longSide = Math.max(width, height);
    return {
      x: width / longSide,
      y: height / longSide,
      width,
      height,
      aspect: width / height,
    };
  }

  function transformPlanarSplat(x, y, sx, sy, theta, image) {
    const frame = plyFrameScale(image);
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const sx2 = sx * sx;
    const sy2 = sy * sy;
    const covarianceX = frame.x * frame.x * (c * c * sx2 + s * s * sy2);
    const covarianceY = frame.y * frame.y * (s * s * sx2 + c * c * sy2);
    const covarianceXY = -frame.x * frame.y * c * s * (sx2 - sy2);
    const trace = covarianceX + covarianceY;
    const delta = Math.hypot(covarianceX - covarianceY, 2 * covarianceXY);
    const lambda0 = Math.max(1e-12, 0.5 * (trace + delta));
    const lambda1 = Math.max(1e-12, 0.5 * (trace - delta));
    return {
      x: x * frame.x,
      y: -y * frame.y,
      sx: Math.sqrt(lambda0),
      sy: Math.sqrt(lambda1),
      theta: 0.5 * Math.atan2(2 * covarianceXY, covarianceX - covarianceY),
      frame,
    };
  }

  function createPlyHeader({ count, image, boundarySigma, layerOrderEnabled, layerDepthSpan }) {
    const properties = [
      "property float x",
      "property float y",
      "property float z",
      "property float nx",
      "property float ny",
      "property float nz",
      "property float f_dc_0",
      "property float f_dc_1",
      "property float f_dc_2",
      "property float opacity",
      "property float scale_0",
      "property float scale_1",
      "property float scale_2",
      "property float rot_0",
      "property float rot_1",
      "property float rot_2",
      "property float rot_3",
    ];
    const frame = plyFrameScale(image);
    return `ply\nformat binary_little_endian 1.0\ncomment image2gaussianpaint_frame ${frame.width} ${frame.height}\ncomment image2gaussianpaint_blend standard_alpha\ncomment image2gaussianpaint_edge_containment ${boundarySigma}\ncomment image2gaussianpaint_layer_order ${layerOrderEnabled ? "micro_z" : "flat_z0"} ${layerDepthSpan}\nelement vertex ${count}\n${properties.join("\n")}\nend_header\n`;
  }

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

  global.Image2SplatPaintPlySerializer = Object.freeze({
    createPlyHeader,
    plyFrameScale,
    serializeBinaryPly,
    transformPlanarSplat,
  });
})(globalThis);
