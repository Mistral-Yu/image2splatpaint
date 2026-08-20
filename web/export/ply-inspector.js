(function installPlyInspector(global) {
  function inspectPlyContract(buffer, sourceParams, sourceImage, dependencies) {
    const {
      plyFrameScale,
      selectedBoundarySigma,
      anisotropyLimitsForParams,
      plyLayerDepth,
      transformPlanarSplatForPly,
      rotatedSplatExtent,
      shC0,
      minSplatScale,
      renderSigma,
      plyLayerDepthSpan,
      defaultVirtualDepthThickness,
    } = dependencies;
    const bytes = new Uint8Array(buffer);
    const marker = new TextEncoder().encode("end_header\n");
    let markerOffset = -1;
    for (let i = 0; i <= bytes.length - marker.length; i += 1) {
      if (marker.every((value, index) => bytes[i + index] === value)) {
        markerOffset = i;
        break;
      }
    }
    if (markerOffset < 0) throw new Error("PLY end_header marker missing");
    const dataOffset = markerOffset + marker.length;
    const header = new TextDecoder("ascii").decode(bytes.subarray(0, dataOffset));
    const vertexMatch = header.match(/element vertex (\d+)/);
    const frameMatch = header.match(/comment image2gaussianpaint_frame (\d+) (\d+)/);
    const boundaryMatch = header.match(/comment image2gaussianpaint_edge_containment ([0-9.eE+-]+)/);
    const layerMatch = header.match(/comment image2gaussianpaint_layer_order (micro_z|flat_z0) ([0-9.eE+-]+)/);
    const properties = [...header.matchAll(/^property float (\S+)$/gm)].map((match) => match[1]);
    const vertices = Number(vertexMatch?.[1] || 0);
    const frameWidth = Number(frameMatch?.[1] || 0);
    const frameHeight = Number(frameMatch?.[2] || 0);
    const expectedFrame = plyFrameScale(sourceImage);
    const boundarySigma = Math.max(0, Number(boundaryMatch?.[1] ?? sourceParams?.boundarySigma ?? selectedBoundarySigma()));
    const frameAspect = frameHeight > 0 ? frameWidth / frameHeight : Number.NaN;
    const aspectRatioError = Number.isFinite(frameAspect) ? Math.abs(frameAspect - expectedFrame.aspect) : Number.POSITIVE_INFINITY;
    const rowBytes = properties.length * 4;
    const view = new DataView(buffer, dataOffset);
    let allZZero = true;
    let zAbsMax = 0;
    let layerDepthErrorMax = 0;
    let allFinite = true;
    let planarSh0 = !properties.some((name) => name.startsWith("f_rest_"));
    let planarRotation = true;
    let geometryMatchErrorMax = 0;
    let opacityErrorMax = 0;
    let colorErrorMax = 0;
    let boundaryLeakCount = 0;
    let boundaryMaxLeak = 0;
    let renderOutsideCount = 0;
    let renderOutsideMaxExtent = 0;
    const anisotropyLimits = anisotropyLimitsForParams(sourceParams);
    let surfaceAnisotropyMax = 1;
    let detailAnisotropyMax = 1;
    let anisotropyLimitViolations = 0;
    for (let i = 0; i < vertices; i += 1) {
      const row = i * rowBytes;
      for (let p = 0; p < properties.length; p += 1) {
        if (!Number.isFinite(view.getFloat32(row + p * 4, true))) allFinite = false;
      }
      const z = view.getFloat32(row + 8, true);
      if (z !== 0) allZZero = false;
      zAbsMax = Math.max(zAbsMax, Math.abs(z));
      if (view.getFloat32(row + 14 * 4, true) !== 0 || view.getFloat32(row + 15 * 4, true) !== 0) planarRotation = false;
      const x = view.getFloat32(row, true);
      const y = view.getFloat32(row + 4, true);
      const sx = Math.exp(view.getFloat32(row + 10 * 4, true));
      const sy = Math.exp(view.getFloat32(row + 11 * 4, true));
      const theta = 2 * Math.atan2(view.getFloat32(row + 16 * 4, true), view.getFloat32(row + 13 * 4, true));
      if (sourceParams && i < sourceParams.count) {
        const expectedZ = plyLayerDepth(i, sourceParams, layerMatch?.[1] === "micro_z");
        layerDepthErrorMax = Math.max(layerDepthErrorMax, Math.abs(z - expectedZ));
        const expected = transformPlanarSplatForPly(
          sourceParams.xy[i * 2],
          sourceParams.xy[i * 2 + 1],
          sourceParams.scale[i * 2],
          sourceParams.scale[i * 2 + 1],
          sourceParams.theta?.[i] || 0,
          sourceImage,
        );
        const angleError = 0.5 * Math.abs(Math.atan2(Math.sin(2 * (theta - expected.theta)), Math.cos(2 * (theta - expected.theta))));
        geometryMatchErrorMax = Math.max(
          geometryMatchErrorMax,
          Math.abs(x - expected.x),
          Math.abs(y - expected.y),
          Math.abs(sx - expected.sx),
          Math.abs(sy - expected.sy),
          angleError,
        );
        const exportedOpacity = 1 / (1 + Math.exp(-view.getFloat32(row + 9 * 4, true)));
        opacityErrorMax = Math.max(opacityErrorMax, Math.abs(exportedOpacity - sourceParams.opacity[i]));
        for (let channel = 0; channel < 3; channel += 1) {
          const exportedColor = view.getFloat32(row + (6 + channel) * 4, true) * shC0 + 0.5;
          colorErrorMax = Math.max(colorErrorMax, Math.abs(exportedColor - sourceParams.rgb[i * 3 + channel]));
        }
        const sourceMinor = Math.max(
          minSplatScale,
          Math.min(sourceParams.scale[i * 2], sourceParams.scale[i * 2 + 1]),
        );
        const sourceRatio = Math.max(sourceParams.scale[i * 2], sourceParams.scale[i * 2 + 1]) / sourceMinor;
        const detail = Math.floor(Number(sourceParams.detailTags?.[i]) || 1) >= 2;
        const limit = detail ? anisotropyLimits.detail : anisotropyLimits.surface;
        if (detail) detailAnisotropyMax = Math.max(detailAnisotropyMax, sourceRatio);
        else surfaceAnisotropyMax = Math.max(surfaceAnisotropyMax, sourceRatio);
        if (sourceRatio > limit + 1e-5) anisotropyLimitViolations += 1;
      }
      const extent = rotatedSplatExtent(sx, sy, theta, boundarySigma);
      const leak = Math.max(
        0,
        Math.abs(x) + extent.x - expectedFrame.x,
        Math.abs(y) + extent.y - expectedFrame.y,
      );
      if (leak > 1e-6) boundaryLeakCount += 1;
      boundaryMaxLeak = Math.max(boundaryMaxLeak, leak);
      const renderExtent = rotatedSplatExtent(sx, sy, theta, renderSigma);
      const renderLeak = Math.max(
        0,
        Math.abs(x) + renderExtent.x - expectedFrame.x,
        Math.abs(y) + renderExtent.y - expectedFrame.y,
      );
      if (renderLeak > 1e-6) renderOutsideCount += 1;
      renderOutsideMaxExtent = Math.max(renderOutsideMaxExtent, renderLeak);
    }
    return {
      format: header.includes("format binary_little_endian 1.0") ? "binary_little_endian" : "unknown",
      vertices,
      properties,
      row_bytes: rowBytes,
      payload_bytes: buffer.byteLength - dataOffset,
      all_z_zero: allZZero,
      z_abs_max: zAbsMax,
      layer_order_enabled: layerMatch?.[1] === "micro_z",
      layer_depth_span: Number(layerMatch?.[2] || 0),
      layer_depth_error_max: layerDepthErrorMax,
      layer_depth_match: layerDepthErrorMax <= 1e-8 && zAbsMax <= plyLayerDepthSpan * 0.501 + (sourceParams?.virtualDepthEnabled ? Number(sourceParams.virtualDepthThickness) || defaultVirtualDepthThickness : 0) + 1e-8,
      all_finite: allFinite,
      sh_degree_0: planarSh0,
      planar_rotation: planarRotation,
      frame_width: frameWidth,
      frame_height: frameHeight,
      frame_aspect: frameAspect,
      expected_frame_aspect: expectedFrame.aspect,
      aspect_ratio_error: aspectRatioError,
      aspect_ratio_preserved: aspectRatioError <= 1e-6,
      standard_alpha_blend: header.includes("comment image2gaussianpaint_blend standard_alpha"),
      geometry_match_error_max: geometryMatchErrorMax,
      y_reflection_rotation: geometryMatchErrorMax <= 1e-5,
      y_reflection_rotation_error_max: geometryMatchErrorMax,
      opacity_error_max: opacityErrorMax,
      color_error_max: colorErrorMax,
      surface_anisotropy_limit: anisotropyLimits.surface,
      surface_anisotropy_max: surfaceAnisotropyMax,
      detail_anisotropy_limit: anisotropyLimits.detail,
      detail_anisotropy_max: detailAnisotropyMax,
      anisotropy_limit_violations: anisotropyLimitViolations,
      boundary_sigma: boundarySigma,
      boundary_leak_count: boundaryLeakCount,
      boundary_max_leak: boundaryMaxLeak,
      render_footprint_outside_count: renderOutsideCount,
      render_footprint_outside_max_extent: renderOutsideMaxExtent,
    };
  }

  global.Image2SplatPaintPlyInspector = Object.freeze({ inspectPlyContract });
})(globalThis);
