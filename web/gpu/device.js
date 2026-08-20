(function installGpuDevice(global) {
  async function requestWebGpuDevice(gpu, {
    profileRequested,
    subgroupExactBackward,
    requiredStorageBuffersPerShaderStage,
    preferredStorageBuffersPerShaderStage,
  }) {
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("adapter unavailable");
    const adapterInfo = adapter.info || null;
    const adapterFeatures = Array.from(adapter.features || []).sort();
    const timestampQuery = profileRequested && adapter.features.has("timestamp-query");
    const subgroups = adapter.features.has("subgroups");
    const adapterStorageBuffers = Number(adapter.limits?.maxStorageBuffersPerShaderStage || 8);
    if (adapterStorageBuffers < requiredStorageBuffersPerShaderStage) {
      throw new Error(
        `adapter supports ${adapterStorageBuffers} storage buffers per shader stage; ${requiredStorageBuffersPerShaderStage} required`,
      );
    }
    const requiredLimits = {
      maxStorageBuffersPerShaderStage: Math.min(
        adapterStorageBuffers,
        preferredStorageBuffersPerShaderStage,
      ),
    };
    let device;
    try {
      device = await adapter.requestDevice({
        requiredFeatures: [
          ...(timestampQuery ? ["timestamp-query"] : []),
          ...(subgroups ? ["subgroups"] : []),
        ],
        requiredLimits,
      });
    } catch (error) {
      device = await adapter.requestDevice({
        requiredFeatures: timestampQuery ? ["timestamp-query"] : [],
        requiredLimits,
      });
    }
    return {
      device,
      adapterInfo,
      adapterFeatures,
      limits: device.limits || adapter.limits || null,
      profile: {
        requested: profileRequested,
        timing_backend: timestampQuery && device.features.has("timestamp-query")
          ? "timestamp-query"
          : profileRequested ? "unavailable" : "off",
      },
      timestampQuery: timestampQuery && device.features.has("timestamp-query"),
      subgroups: device.features.has("subgroups"),
      subgroupExactBackward: subgroupExactBackward && device.features.has("subgroups"),
    };
  }

  global.Image2SplatPaintGpuDevice = Object.freeze({ requestWebGpuDevice });
})(globalThis);
