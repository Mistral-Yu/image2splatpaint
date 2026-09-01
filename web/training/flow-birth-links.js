(function installFlowBirthLinks(global) {
  function classicDependencies() {
    const math = global.Image2SplatPaintBrushSupport;
    const Graph = global.Image2SplatPaintFlowBirthGraph?.BirthGraph;
    if (!math || !Graph) throw new Error("Flow birth-link dependencies were not loaded.");
    return [math, Graph];
  }

  function selectedPath() {
    const value = document.querySelector("#flowTrainingPath")?.value;
    return value === "internal-bend" ? value : "birth-linked";
  }

  function initialize(image, requestedInitialCount) {
    const maxCount = Math.max(32, Math.round(Number(els.finalSplatCount.value) || 8192));
    const enabled = Boolean(els.flowSplatUnderpainting.checked);
    const share = clampNumber(els.flowSplatUnderpaintPercent.value, 0, 50, 10) / 100;
    const fixedCount = enabled ? Math.min(maxCount - 3, Math.round(maxCount * share)) : 0;
    const count = Math.min(maxCount, fixedCount + Math.max(3, requestedInitialCount));
    // Initialize the trainable cohort across the whole image, then prepend the
    // coverage layer. Overwriting the first BSP rows would keep only one edge
    // of its spatially ordered initial placement.
    const params = initLayeredOpaqueBrush(image, count - fixedCount);
    if (fixedCount) {
      for (const [key, stride] of [["xy", 2], ["scale", 2], ["theta", 1], ["rgb", 3],
        ["opacity", 1], ["depthOrder", 1], ["detailTags", 1], ["virtualDepth", 1], ["brushTaper", 1]]) {
        if (!params[key]) continue;
        const data = new Float32Array(count * stride);
        data.set(params[key], fixedCount * stride);
        params[key] = data;
      }
      params.count = count;
    }
    params.flowBirthLinksEnabled = true;
    params.flowBirthLinkStrength = 0.01;
    params.flowBackcoatCount = fixedCount;
    params.flowTrainingSize = [image.width, image.height];
    if (fixedCount) {
      const plan = Image2SplatPaintFlowPaintReference.createSplatUnderpaintPlan(image, {
        count: fixedCount, seed: 240825,
        sizeVariation: clampNumber(els.flowSplatBackcoatSizeVariation.value, 0, 75, 40) / 100,
      });
      for (let i = 0; i < fixedCount; i++) {
        const mark = plan.strokePlan[i];
        const diagonal = Math.hypot(mark.coverage_cell_max_x - mark.coverage_cell_min_x,
          mark.coverage_cell_max_y - mark.coverage_cell_min_y);
        // Conservative base-Brush core, not a source-image background texture.
        // The narrowest family keeps all corners of this mean-color cell opaque.
        const extentPx = diagonal * 1.15;
        params.xy[i * 2] = 2 * (mark.center_x - 0.5) / Math.max(1, image.width - 1) - 1;
        params.xy[i * 2 + 1] = 2 * (mark.center_y - 0.5) / Math.max(1, image.height - 1) - 1;
        params.scale[i * 2] = extentPx / (1.5 * Math.max(1, image.width - 1) / 2);
        params.scale[i * 2 + 1] = extentPx / (1.5 * Math.max(1, image.height - 1) / 2);
        params.theta[i] = 0;
        params.depthOrder[i] = 0;
        params.detailTags[i] = 1;
        params.rgb.set([mark.color_r, mark.color_g, mark.color_b], i * 3);
      }
      const layers = Math.max(2, Number(els.discreteLayerCount.value) || 8);
      for (let i = fixedCount; i < count; i++) {
        params.depthOrder[i] = 1 / layers + (1 - 1 / layers) * (count - i) / (count - fixedCount);
      }
    }
    return params;
  }

  function configure(params) {
    if (!params.flowBirthLinksEnabled) return;
    const opacity = clampNumber(els.flowSplatFusionFixedOpacity.value, 0.05, 0.995, 0.995);
    params.minimumOpacity = opacity;
    params.maximumOpacity = opacity;
    params.opacity.fill(opacity);
  }

  const restoreShader = `
struct Frozen { p:vec4<f32>, t:vec4<f32>, c:vec4<f32> };
@group(0) @binding(0) var<storage,read> fixed:array<Frozen>;
@group(0) @binding(1) var<storage,read_write> xy:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read_write> transform:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read_write> color:array<vec4<f32>>;
@compute @workgroup_size(64) fn restore(@builtin(global_invocation_id) id:vec3<u32>){
  let i=id.x;if(i>=arrayLength(&fixed)){return;}
  xy[i]=fixed[i].p;transform[i]=fixed[i].t;color[i]=fixed[i].c;
}`;

  class Runtime {
    constructor(renderer, params, math, Graph) {
      this.renderer = renderer;
      this.math = math;
      this.graph = new Graph(params.count);
      this.fixedCount = params.flowBackcoatCount || 0;
      this.strength = params.flowBirthLinkStrength;
      this.dirty = true;
      this.passes = 0;
      this.events = [];
      this.readbackBytes = 0;
      this.uniform = null;
      this.neighbors = null;
      this.frozen = null;
      if (this.fixedCount) {
        const p = packPositions(params), t = packTransforms(params), c = packColors(params);
        const data = new Float32Array(this.fixedCount * 12);
        for (let i = 0; i < this.fixedCount; i++) {
          data.set(p.subarray(i * 4, i * 4 + 4), i * 12);
          data.set(t.subarray(i * 4, i * 4 + 4), i * 12 + 4);
          data.set(c.subarray(i * 4, i * 4 + 4), i * 12 + 8);
        }
        this.frozen = makeBuffer(renderer.device, data, GPUBufferUsage.STORAGE);
      }
    }

    buffers() { return [this.uniform, this.neighbors, this.frozen].filter(Boolean); }

    async compile(code, entryPoint) {
      const device = this.renderer.device;
      const module = device.createShaderModule({ code });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter(message => message.type === "error");
      if (errors.length) throw new Error(errors.map(message => message.message).join(" | "));
      return device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint } });
    }

    async readWords(offset, count) {
      const r = this.renderer, bytes = count * 4;
      const buffer = r.device.createBuffer({ size: Math.max(4, bytes), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      try {
        const encoder = r.device.createCommandEncoder();
        encoder.copyBufferToBuffer(r.trainState.densityControlBuffer, offset, buffer, 0, bytes);
        r.device.queue.submit([encoder.finish()]);
        await buffer.mapAsync(GPUMapMode.READ);
        const words = new Uint32Array(buffer.getMappedRange()).slice();
        buffer.unmap();this.readbackBytes += bytes;return words;
      } finally { buffer.destroy(); }
    }

    async grow(oldCount, newCount) {
      const words = await this.readWords(this.renderer.trainState.capacity * 4, newCount - oldCount);
      // GPU source selection must exclude the immutable coverage prefix.
      for (let i = 0; i < words.length; i++) {
        const mode = words[i] >>> 30, parent = (words[i] & 0x3fffffff) - 1;
        if ((mode === 1 || mode === 2) && parent < this.fixedCount) throw new Error("Flow growth selected a fixed backcoat parent.");
      }
      const events = this.graph.grow(oldCount, words, state.metrics.steps_done || 0);
      this.events.push({ kind: "grow", step: state.metrics.steps_done, births: events.length,
        linked: events.filter(event => event.linked).length, count: newCount });
      this.dirty = true;this.restoreNow();
    }

    async relocate(count) {
      const words = await this.readWords(this.renderer.trainState.capacity * 4, count);
      const roles = await this.readWords(0, count);
      for (let row = 0; row < count; row++) {
        if (row < this.fixedCount && (words[row] || roles[row])) throw new Error("Flow relocation selected a fixed backcoat row.");
        if (words[row] && (words[row] & 0x3fffffff) - 1 < this.fixedCount) throw new Error("Flow relocation selected a fixed backcoat source.");
      }
      const changed = this.graph.relocate(words, roles);
      this.events.push({ kind: "relocate", step: state.metrics.steps_done, replaced: changed.destinations.length });
      this.dirty = true;this.restoreNow();
    }

    compact(keep) {
      this.graph.compact(keep);this.dirty = true;
      this.events.push({ kind: "compact", step: state.metrics.steps_done, count: keep.length });
    }

    async prepare(image, params) {
      const r = this.renderer;
      if (this.graph.rows.length !== params.count) throw new Error("Flow lineage count is stale.");
      const fixed = Boolean(r.trainState.fixedPointExactGradient?.enabled);
      if (!this.pipeline || this.fixed !== fixed) {
        this.pipeline = await this.compile(this.math.supportShader(fixed, FIXED_POINT_EXACT_GRADIENT_SCALE), "birth_links");
        this.fixed = fixed;
      }
      if (this.frozen && !this.restorePipeline) this.restorePipeline = await this.compile(restoreShader, "restore");
      if (!this.uniform) this.uniform = r.device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const settings = this.math.settingsFromParams(params);
      if (this.dirty) {
        const p = { ...params };
        for (const key of ["xy", "scale", "theta", "rgb", "opacity", "depthOrder", "detailTags", "brushTaper", "virtualDepth"]) if (p[key]) p[key] = p[key].slice();
        await r.readTrainedColors(p);
        for (const edge of this.graph.pack(() => true, { includeDormant: true }).edges) {
          const key = `${edge.nodeA}:${edge.nodeB}`;
          if (this.graph.caps.has(key)) continue;
          const a = this.math.rowFromParams(p, edge.a), b = this.math.rowFromParams(p, edge.b);
          const delta = [(b.x - a.x) * (image.width - 1) / 2, (b.y - a.y) * (image.height - 1) / 2];
          const distance = Math.hypot(...delta);
          if (distance < 1e-5) continue;
          const n = delta.map(value => value / distance);
          const ra = this.math.brushRadius(a, n, image.width, image.height, settings);
          const rb = this.math.brushRadius(b, n.map(value => -value), image.width, image.height, settings);
          if (ra && rb) this.graph.caps.set(key, {
            capPx: Math.max(0.5, 0.65 * (ra.radius + rb.radius)),
            normalization: Math.max(1, ra.radius + rb.radius),
          });
        }
        this.packed = this.graph.pack();
        const data = new Float32Array(params.count * 8), slots = new Uint8Array(params.count);
        for (let i = 0; i < params.count * 2; i++) data[i * 4] = -1;
        for (const edge of this.packed.edges) {
          const cap = this.graph.caps.get(`${edge.nodeA}:${edge.nodeB}`);
          if (!cap) continue;
          data.set([edge.b, cap.capPx, cap.normalization, 0], edge.a * 8 + slots[edge.a]++ * 4);
          data.set([edge.a, cap.capPx, cap.normalization, 0], edge.b * 8 + slots[edge.b]++ * 4);
        }
        if (!this.neighbors || this.neighbors.size < data.byteLength) {
          this.neighbors?.destroy();
          this.neighbors = r.device.createBuffer({ size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        }
        r.device.queue.writeBuffer(this.neighbors, 0, data);this.dirty = false;
      }
      r.device.queue.writeBuffer(this.uniform, 0, new Float32Array([
        (image.width - 1) / 2, (image.height - 1) / 2, params.discreteLayerCount || 8, 0,
        params.count, this.strength, 0, 0, settings.widthStart, settings.widthEnd, settings.widthTaperEnabled ? 1 : 0, 1 - settings.feather,
      ]));
    }

    encode(encoder, params) {
      if (!this.strength || !this.packed?.edges.length) return;
      if (this.packed.count !== params.count) throw new Error("Flow neighbor buffer is stale.");
      const r = this.renderer, train = r.trainState, front = train.front;
      const buffers = [this.uniform, train.xyBuffers[front], train.transformBuffers[front], train.exactGradientBuffer, this.neighbors];
      if (this.fixed) buffers.push(train.fixedPointGradientControlBuffer);
      this.dispatch(encoder, this.pipeline, buffers, Math.ceil(params.count / 64), "flow-birth-links");this.passes++;
    }

    dispatch(encoder, pipeline, buffers, count, label) {
      if (buffers.some(buffer => !buffer)) throw new Error(`${label}: missing GPU binding ${buffers.findIndex(buffer => !buffer)}`);
      const bindGroup = this.renderer.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline);pass.setBindGroup(0, bindGroup);pass.dispatchWorkgroups(count);pass.end();
    }

    restore(encoder, front) {
      if (!this.frozen || !this.restorePipeline) return;
      const train = this.renderer.trainState;
      this.dispatch(encoder, this.restorePipeline, [this.frozen, train.xyBuffers[front], train.transformBuffers[front], train.colorBuffers[front]],
        Math.ceil(this.fixedCount / 64), "flow-fixed-backcoat");
    }

    restoreNow() {
      if (!this.frozen || !this.restorePipeline || !this.renderer.trainState) return;
      const encoder = this.renderer.device.createCommandEncoder();
      for (let front = 0; front < this.renderer.trainState.xyBuffers.length; front++) this.restore(encoder, front);
      this.renderer.device.queue.submit([encoder.finish()]);
    }

    summary() {
      const packed = this.graph.pack();
      return { path: "shared-dab-birth-links", actual_optimizer: "shared-exact-backward-adam", strength: this.strength,
        linked_dabs: packed.linkedCount, edges: packed.edges.length, groups: packed.groups.map(group => group.length),
        fixed_backcoat_count: this.fixedCount, extra_gradient_passes: this.passes, lineage_readback_bytes: this.readbackBytes, events: this.events };
    }
  }

  global.Image2SplatPaintFlowBirthLinks = Object.freeze({
    selectedPath, initialize, configure,
    async create(renderer, params) {
      const [math, Graph] = classicDependencies();
      return new Runtime(renderer, params, math, Graph);
    },
  });
})(globalThis);
