async function saveExport({ download = true, formatKey = "ply" } = {}) {
  if (state.exporting) return;
  const algorithm = displayedResultAlgorithm();
  if (!algorithmSupportsExport(formatKey, algorithm)) {
    throw new Error(`${algorithm.label} does not support ${formatKey.toUpperCase()} export.`);
  }
  if (!state.exportReady || state.metrics?.safety_stop) throw new Error("Export is not ready.");
  const coverage = exportCoverageStatus();
  if (!coverage.ok) throw new Error(`Export is not ready: ${coverage.message}.`);
  const format = EXPORT_FORMATS[formatKey] || EXPORT_FORMATS.ply;
  const pngSpec = formatKey === "png" ? currentSplatPngSpec() : null;
  const filename = pngSpec?.filename || format.filename;
  if (formatKey === "ply") assertPlyExportCapacity(state.params, state.image, download);

  state.exporting = true;
  state.exportMessage = `Preparing ${format.label}...`;
  updateExportPanel();
  publishState();
  try {
    if (formatKey === "png") {
      const { blob, width, height, exportFrame, spec, padding, nonblackPixels, meanRgb, pngRgbaParity } = await makeSplatPreviewPngBlob();
      const bytes = new Uint8Array(await blob.arrayBuffer());
      if (download) downloadBlob(filename, blob);
      state.exportMessage = `${filename} ${download ? "saved" : "validated"} (${(bytes.byteLength / 1024).toFixed(1)} KiB).`;
      state.metrics.export_history ||= [];
      state.metrics.export_history.push({
        format: formatKey,
        filename,
        bytes: bytes.byteLength,
        width,
        height,
        resolution_mode: exportFrame.mode,
        requested_long_side: exportFrame.longSide,
        splat_shape: spec.shape,
        splat_parameter_effects: Boolean(els.splatParameterEffects?.checked),
        splat_small_first_order: Boolean(spec.renderOptions.splatSmallFirstOrder),
        kernel_falloff: Number(spec.renderOptions.kernelFalloff),
        alpha_background: [...spec.renderOptions.alphaBackground],
        outside_image: Boolean(spec.renderOptions.outside),
        padding: { x: padding.x, y: padding.y },
        nonblack_pixels: nonblackPixels,
        mean_rgb: meanRgb,
        rgba_roundtrip: pngRgbaParity,
        step: state.metrics.steps_done,
      });
      eventLog(`${download ? "exported" : "validated"} ${filename} ${width}x${height} bytes=${bytes.byteLength}`);
      return {
        format: formatKey,
        filename,
        bytes,
        width,
        height,
        exportFrame,
        spec,
        padding,
        nonblackPixels,
        meanRgb,
        pngRgbaParity,
      };
    }

    const exportParams = await materializeCurrentSplatAdjustmentSnapshot();
    if (!exportParams?.params) throw new Error("Splat adjustments changed while preparing the PLY.");
    const plyParams = exportParams.params;
    const plyBuffer = makePly(plyParams, state.image);
    const plyContract = inspectPlyContract(plyBuffer, plyParams, state.image);
    const plyValid =
      plyContract.vertices === plyParams.count &&
      plyContract.row_bytes === 68 &&
      (plyContract.layer_order_enabled ? plyContract.layer_depth_match : plyContract.all_z_zero) &&
      plyContract.all_finite &&
      plyContract.sh_degree_0 &&
      plyContract.planar_rotation &&
      plyContract.standard_alpha_blend &&
      plyContract.aspect_ratio_preserved &&
      plyContract.geometry_match_error_max <= 1e-5 &&
      plyContract.opacity_error_max <= 1e-5 &&
      plyContract.color_error_max <= 1e-5 &&
      plyContract.anisotropy_limit_violations === 0 &&
      plyContract.boundary_leak_count === 0;
    if (!plyValid) throw new Error(`Canonical PLY contract failed: ${JSON.stringify(plyContract)}`);

    const bytes = new Uint8Array(plyBuffer);
    const plyDigest = await sha256Hex(plyBuffer);
    if (!splatAdjustmentSnapshotIsCurrent(exportParams.snapshot)) {
      throw new Error("Splat adjustments changed while preparing the PLY.");
    }
    const currentTiltDigest = state.tilt.verifiedRevision === currentTiltRevision()
      ? state.tilt.verifiedPlyDigest
      : "";
    if (plyDigest && currentTiltDigest && plyDigest !== currentTiltDigest) {
      throw new Error("PLY bytes differ from the current Tilt view.");
    }

    if (download) downloadBlob(format.filename, new Blob([bytes], { type: "application/octet-stream" }));
    state.exportMessage = `${format.filename} ${download ? "saved" : "validated"} (${(bytes.byteLength / 1024).toFixed(1)} KiB).`;
    state.metrics.export_history ||= [];
    state.metrics.export_history.push({
      format: formatKey,
      filename: format.filename,
      bytes: bytes.byteLength,
      sha256: plyDigest,
      step: state.metrics.steps_done,
      contract: plyContract,
    });
    document.documentElement.dataset.lastPlySha256 = plyDigest;
    document.documentElement.dataset.lastPlyBytes = String(bytes.byteLength);
    eventLog(
      `${download ? "exported" : "validated"} ${format.filename} bytes=${bytes.byteLength}` +
      ` sha256=${plyDigest ? plyDigest.slice(0, 12) : "unavailable"}` +
      ` aspect=${plyContract.frame_aspect.toFixed(6)}` +
      ` geometry_error=${plyContract.geometry_match_error_max.toExponential(2)}` +
      ` opacity_error=${plyContract.opacity_error_max.toExponential(2)}` +
      ` color_error=${plyContract.color_error_max.toExponential(2)}`,
    );
    return {
      format: formatKey,
      filename: format.filename,
      bytes,
      sha256: plyDigest,
      plyContract,
    };
  } catch (error) {
    state.exportMessage = `Export failed: ${error.message}`;
    eventLog(state.exportMessage);
    throw error;
  } finally {
    state.exporting = false;
    updateExportPanel();
    publishState();
  }
}

function inspectPlyContract(buffer = makePly(), sourceParams = state.params, sourceImage = state.image) {
  return inspectSerializedPlyContract(buffer, sourceParams, sourceImage, {
    plyFrameScale,
    selectedBoundarySigma,
    anisotropyLimitsForParams,
    plyLayerDepth,
    transformPlanarSplatForPly,
    rotatedSplatExtent,
    shC0: SH_C0,
    minSplatScale: MIN_SPLAT_SCALE,
    renderSigma: RENDER_SIGMA,
    plyLayerDepthSpan: PLY_LAYER_DEPTH_SPAN,
    defaultVirtualDepthThickness: DEFAULT_VIRTUAL_DEPTH_THICKNESS,
  });
}

