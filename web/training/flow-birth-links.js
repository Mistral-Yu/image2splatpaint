(function installFlowBirthLinks(global) {
  const PAINTERLY_LINK_STRENGTH = 0.03;
  const PAINTERLY_PIGMENT_WEIGHT = 10;
  const PAINTERLY_TANGENT_WEIGHT = 6;
  const PAINTERLY_WIDTH_WEIGHT = 2;
  const FIXED_UNDERPAINT_SAFETY_SHARE = 0.2;
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

  function selectedLinkedSplatRange() {
    const minimum = Math.max(2, Math.min(9, Math.round(
      Number(els.flowLinkedSplatMin?.value) || 4,
    )));
    const maximum = Math.max(minimum, Math.min(9, Math.round(
      Number(els.flowLinkedSplatMax?.value) || 9,
    )));
    return { min: minimum, max: maximum };
  }

  function selectedStrokeCoherence() {
    return clampNumber(els.flowStrokeCoherence?.value, 0, 100, 50) / 100;
  }

  function coherenceWeights(value = 0) {
    const amount = Math.max(0, Math.min(1, Number(value) || 0));
    return {
      link: PAINTERLY_LINK_STRENGTH * (1 + 2 * amount),
      pigment: PAINTERLY_PIGMENT_WEIGHT * (1 + 0.5 * amount),
      tangent: PAINTERLY_TANGENT_WEIGHT * (1 + amount),
      width: PAINTERLY_WIDTH_WEIGHT * (1 + 1.5 * amount),
    };
  }

  function splitUnderpaintBudget(enabled, maxCount, share, minimumDetailCount) {
    const total = enabled
      ? Math.min(maxCount - minimumDetailCount, Math.round(maxCount * share))
      : 0;
    const fixed = total > 0
      ? Math.max(1, Math.min(total, Math.round(total * FIXED_UNDERPAINT_SAFETY_SHARE)))
      : 0;
    return Object.freeze({ total, fixed, trainable: total - fixed });
  }

  function orientedUnderpaintShape(mark, image, trainable) {
    const dx = mark.control_1_x - mark.center_x;
    const dy = mark.control_1_y - mark.center_y;
    const pixelAngle = Math.atan2(dy, dx);
    const pixelHalfWidth = Math.max(1, image.width - 1) / 2;
    const pixelHalfHeight = Math.max(1, image.height - 1) / 2;
    let theta = Math.atan2(Math.sin(pixelAngle) / pixelHalfHeight,
      Math.cos(pixelAngle) / pixelHalfWidth);
    const c = Math.cos(theta), s = Math.sin(theta);
    const majorMetric = Math.hypot(pixelHalfWidth * c, pixelHalfHeight * s);
    const minorMetric = Math.hypot(pixelHalfWidth * s, pixelHalfHeight * c);
    const diagonal = Math.hypot(mark.coverage_cell_max_x - mark.coverage_cell_min_x,
      mark.coverage_cell_max_y - mark.coverage_cell_min_y);
    // Only the small frozen safety prefix needs to close every source cell by
    // itself. The larger trainable cohort starts as elongated Brush marks and
    // is allowed to move, rotate, resize, split and re-link with the optimizer.
    const minorExtentPx = trainable
      ? Math.max(mark.underpaint_sigma_short_px || 0, diagonal * 0.58)
      : diagonal * 1.15;
    const majorExtentPx = trainable
      ? Math.max(mark.underpaint_sigma_long_px || 0, minorExtentPx * 1.85)
      : minorExtentPx * 1.35;
    let sx = majorExtentPx / (1.5 * Math.max(1, majorMetric));
    let sy = minorExtentPx / (1.5 * Math.max(1, minorMetric));
    // Brush family and tangent logic use the numerically larger local scale as
    // the long axis. Preserve the intended pixel-space direction on wide or
    // tall images by swapping local axes when necessary.
    if (sx < sy) {
      [sx, sy] = [sy, sx];
      theta -= Math.PI / 2;
    }
    return Object.freeze({ sx, sy, theta });
  }

  function writeUnderpaintPlan(params, plan, start, image, { trainable, depth }) {
    for (let local = 0; local < plan.strokePlan.length; local += 1) {
      const i = start + local;
      const mark = plan.strokePlan[local];
      const shape = orientedUnderpaintShape(mark, image, trainable);
      params.xy[i * 2] = 2 * (mark.center_x - 0.5) / Math.max(1, image.width - 1) - 1;
      params.xy[i * 2 + 1] = 2 * (mark.center_y - 0.5) / Math.max(1, image.height - 1) - 1;
      params.scale[i * 2] = shape.sx;
      params.scale[i * 2 + 1] = shape.sy;
      params.theta[i] = shape.theta;
      params.depthOrder[i] = depth;
      params.detailTags[i] = 1;
      params.rgb.set([mark.color_r, mark.color_g, mark.color_b], i * 3);
    }
  }

  function initialize(image, requestedInitialCount) {
    const maxCount = Math.max(32, Math.round(Number(els.finalSplatCount.value) || 8192));
    const linkedSplats = selectedLinkedSplatRange();
    const enabled = Boolean(els.flowSplatUnderpainting.checked);
    const share = clampNumber(els.flowSplatUnderpaintPercent.value, 0, 50, 10) / 100;
    const underpaint = splitUnderpaintBudget(enabled, maxCount, share, linkedSplats.min);
    const fixedCount = underpaint.fixed;
    const count = Math.min(maxCount, underpaint.total + Math.max(linkedSplats.min, requestedInitialCount));
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
    params.flowBirthLinkStrength = PAINTERLY_LINK_STRENGTH;
    params.flowLinkedSplatMin = linkedSplats.min;
    params.flowLinkedSplatMax = linkedSplats.max;
    params.flowStrokeCoherence = selectedStrokeCoherence();
    params.flowBackcoatCount = fixedCount;
    params.flowUnderpaintCount = underpaint.total;
    params.flowTrainableUnderpaintCount = underpaint.trainable;
    params.flowTrainingSize = [image.width, image.height];
    if (underpaint.total) {
      const sizeVariation = clampNumber(els.flowSplatBackcoatSizeVariation.value, 0, 75, 40) / 100;
      const fixedPlan = Image2SplatPaintFlowPaintReference.createSplatUnderpaintPlan(image, {
        count: fixedCount, seed: 240825,
        sizeVariation,
      });
      const layers = Math.max(2, Number(els.discreteLayerCount.value) || 8);
      writeUnderpaintPlan(params, fixedPlan, 0, image, { trainable: false, depth: 0 });
      if (underpaint.trainable) {
        const trainablePlan = Image2SplatPaintFlowPaintReference.createSplatUnderpaintPlan(image, {
          count: underpaint.trainable, seed: 240825 ^ 0x9e3779b9,
          sizeVariation,
        });
        writeUnderpaintPlan(params, trainablePlan, fixedCount, image, {
          trainable: true,
          depth: 1 / layers,
        });
      }
      for (let i = underpaint.total; i < count; i++) {
        params.depthOrder[i] = 2 / layers
          + (1 - 2 / layers) * (count - i) / Math.max(1, count - underpaint.total);
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

  function mortonKey(x, y) {
    const qx = Math.max(0, Math.min(1023, Math.round((x + 1) * 511.5)));
    const qy = Math.max(0, Math.min(1023, Math.round((y + 1) * 511.5)));
    let key = 0;
    for (let bit = 0; bit < 10; bit += 1) {
      key |= ((qx >>> bit) & 1) << (bit * 2);
      key |= ((qy >>> bit) & 1) << (bit * 2 + 1);
    }
    return key >>> 0;
  }

  function brushAxis(params, row) {
    const majorIsX = params.scale[row * 2] >= params.scale[row * 2 + 1];
    return (params.theta[row] || 0) + (majorIsX ? 0 : Math.PI / 2);
  }

  function axialMismatch(a, b) {
    return Math.abs(Math.sin(a - b));
  }

  function nextChainRow(params, chain, pending, windowSize = 24) {
    const current = chain[chain.length - 1];
    const cx = params.xy[current * 2], cy = params.xy[current * 2 + 1];
    const currentAxis = brushAxis(params, current);
    let previousDirection = null;
    if (chain.length >= 2) {
      const previous = chain[chain.length - 2];
      previousDirection = Math.atan2(cy - params.xy[previous * 2 + 1], cx - params.xy[previous * 2]);
    }
    let bestIndex = 0, bestScore = Infinity, bestDistance = Infinity;
    for (let index = 0; index < Math.min(windowSize, pending.length); index += 1) {
      const candidate = pending[index];
      const dx = params.xy[candidate * 2] - cx, dy = params.xy[candidate * 2 + 1] - cy;
      const distance = Math.max(1e-6, Math.hypot(dx, dy));
      const direction = Math.atan2(dy, dx);
      const candidateAxis = brushAxis(params, candidate);
      const turn = previousDirection === null ? 0 : Math.abs(Math.sin(direction - previousDirection));
      const colorOffset = current * 3, candidateColorOffset = candidate * 3;
      const colorDistance = Math.hypot(
        (params.rgb[candidateColorOffset] || 0) - (params.rgb[colorOffset] || 0),
        (params.rgb[candidateColorOffset + 1] || 0) - (params.rgb[colorOffset + 1] || 0),
        (params.rgb[candidateColorOffset + 2] || 0) - (params.rgb[colorOffset + 2] || 0),
      );
      const score = distance * (1 + 1.35 * axialMismatch(currentAxis, direction)
        + 0.85 * axialMismatch(candidateAxis, direction) + 0.9 * turn + 0.35 * colorDistance);
      if (score < bestScore) { bestScore = score; bestIndex = index; bestDistance = distance; }
    }
    // Layer identity alone does not make a coherent stroke. Sparse layers can
    // otherwise connect opposite sides of the image after compaction.
    if (bestDistance > .18) return null;
    return pending.splice(bestIndex, 1)[0];
  }

  function localChains(params, rows, minimum, maximum) {
    const byLayer = new Map();
    for (const row of rows) {
      const layer = Math.max(0, Math.round((params.depthOrder?.[row] || 0) * 32));
      if (!byLayer.has(layer)) byLayer.set(layer, []);
      byLayer.get(layer).push(row);
    }
    const chains = [];
    for (const layerRows of byLayer.values()) {
      const pending = layerRows.slice().sort((a, b) => mortonKey(params.xy[a * 2], params.xy[a * 2 + 1])
        - mortonKey(params.xy[b * 2], params.xy[b * 2 + 1]));
      while (pending.length >= minimum) {
        const seed = pending.shift();
        const range = Math.max(1, maximum - minimum + 1);
        const desired = minimum + mortonKey(params.xy[seed * 2], params.xy[seed * 2 + 1]) % range;
        const members = Math.min(maximum, desired, pending.length + 1);
        const chain = [seed];
        while (chain.length < members && pending.length) {
          const next = nextChainRow(params, chain, pending);
          if (next === null) break;
          chain.push(next);
        }
        if (chain.length >= minimum) chains.push(chain);
      }
    }
    return chains;
  }

  function orderedGroup(graph, group) {
    const groupSet = new Set(group);
    const endpoint = group.find((node) =>
      [...(graph.nodes.get(node) || [])].filter((other) => groupSet.has(other)).length <= 1,
    ) || group[0];
    const ordered = [];
    let previous = null, current = endpoint;
    while (current !== undefined && ordered.length < group.length) {
      ordered.push(current);
      const next = [...(graph.nodes.get(current) || [])]
        .find((node) => node !== previous && groupSet.has(node) && !ordered.includes(node));
      previous = current;current = next;
    }
    return ordered;
  }

  function chainDiagnostics(graph, packed, params, image, fixedCount) {
    const rowOfNode = new Map(graph.rows.map((node, row) => [node, row]));
    const histogram = {};
    const spans = [], arcHeights = [];
    const dx = Math.max(1, image.width - 1) / 2, dy = Math.max(1, image.height - 1) / 2;
    for (const group of packed.groups) {
      histogram[group.length] = (histogram[group.length] || 0) + 1;
      const ordered = orderedGroup(graph, group);
      const points = ordered.map((node) => {
        const row = rowOfNode.get(node);
        return [params.xy[row * 2] * dx, params.xy[row * 2 + 1] * dy];
      });
      if (points.length < 2) continue;
      const a = points[0], b = points[points.length - 1], vx = b[0] - a[0], vy = b[1] - a[1];
      const span = Math.hypot(vx, vy);spans.push(span);
      let height = 0;
      if (span > 1e-5) for (const point of points.slice(1, -1)) {
        height = Math.max(height, Math.abs(vx * (point[1] - a[1]) - vy * (point[0] - a[0])) / span);
      }
      arcHeights.push(height);
    }
    const mean = values => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
    return {
      linked_ratio: packed.linkedCount / Math.max(1, params.count - fixedCount),
      group_count: packed.groups.length,
      group_size_histogram: histogram,
      mean_group_size: mean(packed.groups.map(group => group.length)),
      mean_chain_span_px: mean(spans),
      mean_arc_height_px: mean(arcHeights),
      max_arc_height_px: arcHeights.length ? Math.max(...arcHeights) : 0,
    };
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
      this.graph = new Graph(params.count, {
        minMembers: params.flowLinkedSplatMin || 2,
        maxMembers: params.flowLinkedSplatMax || 9,
      });
      this.fixedCount = params.flowBackcoatCount || 0;
      this.underpaintCount = params.flowUnderpaintCount || this.fixedCount;
      this.strength = params.flowBirthLinkStrength;
      this.strokeCoherence = params.flowStrokeCoherence || 0;
      this.dirty = true;
      this.passes = 0;
      this.events = [];
      this.readbackBytes = 0;
      this.uniform = null;
      this.neighbors = null;
      this.frozen = null;
      this.baseMinorScaleByNode = new Map();
      // The initial trainable paint already represents visible strokes. Build
      // same-layer, direction-guided roots so curvature and pressure are present before the
      // first split, instead of leaving the complete P1 cohort as isolated dabs.
      const initialRows = Array.from({length: params.count - this.fixedCount},
        (_, index) => this.fixedCount + index);
      this.graph.seedChains(localChains(params, initialRows,
        this.graph.minMembers, this.graph.maxMembers));
      for (let row = this.fixedCount; row < params.count; row += 1) {
        this.baseMinorScaleByNode.set(this.graph.rows[row], Math.max(1e-5,
          Math.min(params.scale[row * 2], params.scale[row * 2 + 1])));
      }
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
      // A split reduces the child's physical scale. Keep the original stroke
      // pressure envelope as the width target so repeated growth adds detail
      // along a stroke instead of turning every lineage into uniformly thin
      // dots by the 8192-Splat stage.
      for (const event of events) {
        const inherited = this.baseMinorScaleByNode.get(event.parent);
        if (inherited) this.baseMinorScaleByNode.set(event.child, inherited);
      }
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
      const alive = new Set(this.graph.rows);
      for (const node of this.baseMinorScaleByNode.keys()) {
        if (!alive.has(node)) this.baseMinorScaleByNode.delete(node);
      }
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
      if (!this.uniform) this.uniform = r.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const settings = this.math.settingsFromParams(params);
      if (this.dirty) {
        const p = { ...params };
        for (const key of ["xy", "scale", "theta", "rgb", "opacity", "depthOrder", "detailTags", "brushTaper", "virtualDepth"]) if (p[key]) p[key] = p[key].slice();
        await r.readTrainedColors(p);
        // Rebuild only at structural boundaries. Split/clone children are
        // reinserted into direction- and color-compatible chains instead of
        // accumulating as isolated round dots or ancestry-only fragments.
        const trainableRows = Array.from({length: params.count - this.fixedCount},
          (_, index) => this.fixedCount + index);
        this.graph.resetEdges(trainableRows);
        this.graph.seedChains(localChains(p, trainableRows,
          this.graph.minMembers, this.graph.maxMembers));
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
        // Two-dab groups are already valid visible stroke fragments. Training
        // them immediately avoids waiting for a third child before continuity
        // starts contributing gradients.
        this.packed = this.graph.pack(() => true, {
          includeDormant: this.graph.minMembers <= 2,
        });
        const data = new Float32Array(params.count * 8), slots = new Uint8Array(params.count);
        const rowOfNode = new Map(this.graph.rows.map((node, row) => [node, row]));
        const widthTargets = new Float32Array(params.count);
        for (const group of this.packed.groups) {
          // Every dab in one stroke bends to the same side. Hashing each node
          // independently made neighboring control dabs alternate left/right,
          // cancelling the visible arc into a nearly straight or vibrating row.
          const groupRoot = Math.min(...group);
          const groupBendSign = ((Math.imul(groupRoot, 2654435761) >>> 0) & 1) ? -1 : 1;
          const ordered = orderedGroup(this.graph, group);
          ordered.forEach((node, index) => {
            const row = rowOfNode.get(node);
            if (row === undefined) return;
            if (!this.baseMinorScaleByNode.has(node)) {
              this.baseMinorScaleByNode.set(node, Math.max(1e-5,
                Math.min(p.scale[row * 2], p.scale[row * 2 + 1])));
            }
            const progress = ordered.length > 1 ? index / (ordered.length - 1) : .5;
            // Narrow ends and a broad body make each linked group read as one
            // pressure-varying stroke, while retaining the parent's scale family.
            const pressure = .75 + .85 * Math.pow(Math.sin(Math.PI * progress), .8);
            // Preserve a stable bend side across compaction by encoding the
            // stable node's sign in the otherwise-positive width target.
            widthTargets[row] = groupBendSign * this.baseMinorScaleByNode.get(node) * pressure;
          });
        }
        for (let i = 0; i < params.count * 2; i++) data[i * 4] = -1;
        for (const edge of this.packed.edges) {
          const cap = this.graph.caps.get(`${edge.nodeA}:${edge.nodeB}`);
          if (!cap) continue;
          data.set([edge.b, cap.capPx, cap.normalization, widthTargets[edge.a]], edge.a * 8 + slots[edge.a]++ * 4);
          data.set([edge.a, cap.capPx, cap.normalization, widthTargets[edge.b]], edge.b * 8 + slots[edge.b]++ * 4);
        }
        if (!this.neighbors || this.neighbors.size < data.byteLength) {
          this.neighbors?.destroy();
          this.neighbors = r.device.createBuffer({ size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        }
        r.device.queue.writeBuffer(this.neighbors, 0, data);
        this.lastDiagnostics = chainDiagnostics(this.graph, this.packed, p, image, this.fixedCount);
        this.dirty = false;
      }
      const weights = coherenceWeights(this.strokeCoherence);
      r.device.queue.writeBuffer(this.uniform, 0, new Float32Array([
        (image.width - 1) / 2, (image.height - 1) / 2, params.discreteLayerCount || 8, weights.pigment,
        params.count, weights.link, weights.tangent, weights.width,
        settings.widthStart, settings.widthEnd, settings.widthTaperEnabled ? 1 : 0, 1 - settings.feather,
        this.strokeCoherence, 0, 0, 0,
      ]));
    }

    encode(encoder, params) {
      if (!this.strength || !this.packed?.edges.length) return;
      if (this.packed.count !== params.count) throw new Error("Flow neighbor buffer is stale.");
      const r = this.renderer, train = r.trainState, front = train.front;
      const buffers = [this.uniform, train.xyBuffers[front], train.transformBuffers[front], train.exactGradientBuffer,
        this.neighbors, train.colorBuffers[front]];
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
      const packed = this.graph.pack(() => true, {
        includeDormant: this.graph.minMembers <= 2,
      });
      return { path: "shared-dab-birth-links", actual_optimizer: "shared-exact-backward-adam", strength: this.strength,
        linked_dabs: packed.linkedCount, edges: packed.edges.length, groups: packed.groups.map(group => group.length),
        ...(this.lastDiagnostics || {}),
        linked_splats_min: this.graph.minMembers, linked_splats_max: this.graph.maxMembers,
        stroke_coherence: this.strokeCoherence,
        coherence_weights: coherenceWeights(this.strokeCoherence),
        fixed_backcoat_count: this.fixedCount,
        trainable_underpaint_count: Math.max(0, this.underpaintCount - this.fixedCount),
        extra_gradient_passes: this.passes, lineage_readback_bytes: this.readbackBytes, events: this.events };
    }
  }

  global.Image2SplatPaintFlowBirthLinks = Object.freeze({
    selectedPath, selectedLinkedSplatRange, selectedStrokeCoherence, initialize, configure,
    _test: Object.freeze({ localChains, chainDiagnostics, coherenceWeights,
      splitUnderpaintBudget, orientedUnderpaintShape }),
    async create(renderer, params) {
      const [math, Graph] = classicDependencies();
      return new Runtime(renderer, params, math, Graph);
    },
  });
})(globalThis);
