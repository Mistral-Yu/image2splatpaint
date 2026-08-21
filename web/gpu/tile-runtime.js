class WebGpuTileRuntime {
  async readTileCounters({ includeDistribution = false } = {}) {
    if (!this.trainState?.tileReady) return null;
    const tileCount = includeDistribution ? this.trainState.tileCount : 0;
    const controlBytes = 16;
    const distributionBytes = tileCount * 4;
    const readBuffer = this.device.createBuffer({
      size: Math.max(16, controlBytes + distributionBytes),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.trainState.tileControlBuffer, 0, readBuffer, 0, controlBytes);
      if (distributionBytes > 0) {
        encoder.copyBufferToBuffer(this.trainState.tileCountsBuffer, 0, readBuffer, controlBytes, distributionBytes);
      }
      this.device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const mapped = readBuffer.getMappedRange();
      const values = new Uint32Array(mapped, 0, 4).slice();
      const distribution = distributionBytes > 0
        ? Array.from(new Uint32Array(mapped, controlBytes, tileCount)).sort((a, b) => a - b)
        : null;
      readBuffer.unmap();
      const total = values[0];
      const overflow = values[1];
      const firstNonemptyTile = distribution?.findIndex((value) => value > 0) ?? -1;
      const candidatePercentiles = distribution ? {
        p50: percentileSorted(distribution, 0.5),
        p90: percentileSorted(distribution, 0.9),
        p99: percentileSorted(distribution, 0.99),
        maximum: distribution[distribution.length - 1] ?? 0,
        nonempty_tiles: firstNonemptyTile < 0 ? 0 : distribution.length - firstNonemptyTile,
      } : null;
      return {
        total,
        overflow,
        noop_steps: values[2],
        capacity: this.trainState.tileIndexCapacity,
        reserve_ratio: total / Math.max(1, this.trainState.tileIndexCapacity),
        tile_count: this.trainState.tileCount,
        average_candidates: total / Math.max(1, this.trainState.tileCount),
        candidate_percentiles: candidatePercentiles,
        qa_readback_count: includeDistribution ? 1 : 0,
        qa_readback_bytes: includeDistribution ? controlBytes + distributionBytes : 0,
        active_count: this.trainState.count,
        free_count: Math.max(0, this.trainState.capacity - this.trainState.count),
      };
    } finally {
      readBuffer.destroy();
    }
  }

  async readLayerTileDiagnostics(params, layerCounts = [5, 10]) {
    if (!this.trainState?.tileReady || !params?.depthOrder || params.count <= 0) return null;
    const counters = await this.readTileCounters();
    if (!counters) return null;
    const tileCount = this.trainState.tileCount;
    const total = Math.min(counters.total, this.trainState.tileIndexCapacity);
    const offsetBytes = (tileCount + 1) * 4;
    const indexBytes = total * 4;
    const readBuffer = this.device.createBuffer({
      size: Math.max(4, offsetBytes + indexBytes),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.trainState.tileOffsetsBuffer, 0, readBuffer, 0, offsetBytes);
      if (indexBytes > 0) {
        encoder.copyBufferToBuffer(this.trainState.tileIndicesBuffer, 0, readBuffer, offsetBytes, indexBytes);
      }
      this.device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const mapped = readBuffer.getMappedRange();
      const offsets = new Uint32Array(mapped, 0, tileCount + 1).slice();
      const indices = indexBytes > 0
        ? new Uint32Array(mapped, offsetBytes, total).slice()
        : new Uint32Array();
      readBuffer.unmap();
      const referenceCounts = new Uint32Array(params.count);
      for (const index of indices) {
        if (index < params.count) referenceCounts[index] += 1;
      }
      const comparisonCost = (count) => count > 1 ? count * Math.log2(count) : 0;
      const quantization = {};
      for (const rawLayerCount of layerCounts) {
        const layerCount = Math.max(2, Math.floor(rawLayerCount));
        let fullSortCost = 0;
        let bucketSortCost = 0;
        for (let tile = 0; tile < tileCount; tile += 1) {
          const begin = Math.min(total, offsets[tile]);
          const end = Math.min(total, offsets[tile + 1]);
          const bins = new Uint32Array(layerCount);
          for (let cursor = begin; cursor < end; cursor += 1) {
            const index = indices[cursor];
            if (index >= params.count) continue;
            const layer = Math.max(0, Math.min(1, params.depthOrder[index]));
            bins[Math.min(layerCount - 1, Math.floor(layer * layerCount))] += 1;
          }
          fullSortCost += comparisonCost(end - begin);
          for (const count of bins) bucketSortCost += comparisonCost(count);
        }
        quantization[layerCount] = {
          layer_count: layerCount,
          estimated_comparison_reduction_ratio: fullSortCost > 0
            ? 1 - bucketSortCost / fullSortCost
            : 0,
          full_sort_cost: fullSortCost,
          bucket_sort_cost: bucketSortCost,
        };
      }
      return {
        total_tile_references: total,
        tile_count: tileCount,
        per_splat_reference_count: referenceCounts,
        quantization,
      };
    } finally {
      readBuffer.destroy();
    }
  }

  async clearTileNoopCounter() {
    if (!this.trainState?.tileControlBuffer) return;
    const encoder = this.device.createCommandEncoder();
    encoder.clearBuffer(this.trainState.tileControlBuffer, 8, 8);
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }

  async growTileIndexCapacity(requiredTotal, { proactive = false } = {}) {
    if (!this.trainState) return false;
    if (!proactive && requiredTotal <= this.trainState.tileIndexCapacity) return true;
    const limit = tileIndexCapacityLimit(this.device);
    const requested = Math.ceil(
      Math.max(
        requiredTotal * TILE_INDEX_GROWTH_HEADROOM,
        this.trainState.tileIndexCapacity * TILE_INDEX_INITIAL_HEADROOM,
      ) / 256,
    ) * 256;
    const nextCapacity = Math.min(requested, limit);
    if (nextCapacity < requiredTotal || nextCapacity <= this.trainState.tileIndexCapacity) return false;
    const previous = this.trainState.tileIndicesBuffer;
    const growthPlan = tileGrowthMemoryPlan({
      currentReservedBytes: this.trainingMemorySnapshot().reservedBytes,
      currentTileBytes: previous?.size,
      nextTileBytes: nextCapacity * 4,
    });
    if (!growthPlan.withinBudget) {
      if (state.metrics) state.metrics.tile_growth_budget_rejection = growthPlan;
      return false;
    }
    let nextBuffer = null;
    this.device.pushErrorScope("out-of-memory");
    this.device.pushErrorScope("validation");
    try {
      nextBuffer = this.device.createBuffer({
        size: nextCapacity * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const encoder = this.device.createCommandEncoder();
      encoder.clearBuffer(nextBuffer);
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      const validationError = await this.device.popErrorScope();
      const oomError = await this.device.popErrorScope();
      if (validationError || oomError) {
        nextBuffer.destroy();
        return false;
      }
      this.trainState.tileIndicesBuffer = nextBuffer;
      this.trainState.tileIndexCapacity = nextCapacity;
      this.trainState.tileReady = false;
      previous.destroy();
      updateGpuMemoryStatus();
      return true;
    } catch (error) {
      nextBuffer?.destroy();
      await this.device.popErrorScope().catch(() => null);
      await this.device.popErrorScope().catch(() => null);
      return false;
    }
  }

}

registerWebGpuPreviewFeature(WebGpuTileRuntime.prototype, "WebGpuTileRuntime");
