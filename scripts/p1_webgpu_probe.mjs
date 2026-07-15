const resultNode = document.querySelector("#result");
const dispatchCounts = [65535, 65536, 262144, 1048576];

function dispatchLayout(workgroupCount, limit) {
  const x = Math.min(workgroupCount, limit);
  const y = Math.ceil(workgroupCount / x);
  if (y > limit) throw new Error(`dispatch ${workgroupCount} exceeds ${limit}x${limit}`);
  return { x, y };
}

async function readU32(device, source, count) {
  const readback = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, count * 4);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const values = new Uint32Array(readback.getMappedRange()).slice();
  readback.unmap();
  readback.destroy();
  return values;
}

async function run() {
  if (!navigator.gpu) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU adapter unavailable");
  const device = await adapter.requestDevice();
  const maxDispatch = Number(device.limits.maxComputeWorkgroupsPerDimension || 65535);

  const dispatchModule = device.createShaderModule({ code: `
struct Config { count: u32, _pad0: u32, _pad1: u32, _pad2: u32, };
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read_write> visits: array<atomic<u32>>;
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) workgroups: vec3<u32>) {
  let g = id.x + id.y * workgroups.x * 64u;
  if (g < config.count) { atomicAdd(&visits[g], 1u); }
}` });
  const dispatchInfo = await dispatchModule.getCompilationInfo();
  const dispatchErrors = dispatchInfo.messages.filter((message) => message.type === "error");
  if (dispatchErrors.length) throw new Error(dispatchErrors.map((message) => message.message).join(" | "));
  const dispatchPipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: dispatchModule, entryPoint: "probe" },
  });

  const twoDimModule = device.createShaderModule({ code: `
struct Config { count: u32, _pad0: u32, _pad1: u32, _pad2: u32, };
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read_write> visits: array<atomic<u32>>;
@compute @workgroup_size(1)
fn probe_2d(@builtin(workgroup_id) id: vec3<u32>, @builtin(num_workgroups) workgroups: vec3<u32>) {
  let g = id.x + id.y * workgroups.x;
  if (g < config.count) { atomicAdd(&visits[g], 1u); }
}` });
  const twoDimInfo = await twoDimModule.getCompilationInfo();
  const twoDimErrors = twoDimInfo.messages.filter((message) => message.type === "error");
  if (twoDimErrors.length) throw new Error(twoDimErrors.map((message) => message.message).join(" | "));
  const twoDimPipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: twoDimModule, entryPoint: "probe_2d" },
  });

  const dispatch = [];
  for (const count of dispatchCounts) {
    const config = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const visits = device.createBuffer({
      size: count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(config, 0, new Uint32Array([count, 0, 0, 0]));
    const bindGroup = device.createBindGroup({
      layout: dispatchPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: config } },
        { binding: 1, resource: { buffer: visits } },
      ],
    });
    const encoder = device.createCommandEncoder();
    encoder.clearBuffer(visits);
    const pass = encoder.beginComputePass();
    pass.setPipeline(dispatchPipeline);
    pass.setBindGroup(0, bindGroup);
    const workgroups = Math.ceil(count / 64);
    const layout = dispatchLayout(workgroups, maxDispatch);
    pass.dispatchWorkgroups(layout.x, layout.y);
    pass.end();
    device.queue.submit([encoder.finish()]);
    const values = await readU32(device, visits, count);
    let firstBad = -1;
    let sum = 0;
    for (let i = 0; i < values.length; i += 1) {
      sum += values[i];
      if (firstBad < 0 && values[i] !== 1) firstBad = i;
    }
    dispatch.push({ count, workgroups, ...layout, sum, first_bad: firstBad, ok: firstBad < 0 && sum === count });
    config.destroy();
    visits.destroy();
  }

  const twoDimCount = 72929;
  const twoDimConfig = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const twoDimVisits = device.createBuffer({
    size: twoDimCount * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(twoDimConfig, 0, new Uint32Array([twoDimCount, 0, 0, 0]));
  const twoDimBindGroup = device.createBindGroup({
    layout: twoDimPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: twoDimConfig } },
      { binding: 1, resource: { buffer: twoDimVisits } },
    ],
  });
  const twoDimEncoder = device.createCommandEncoder();
  twoDimEncoder.clearBuffer(twoDimVisits);
  const twoDimPass = twoDimEncoder.beginComputePass();
  twoDimPass.setPipeline(twoDimPipeline);
  twoDimPass.setBindGroup(0, twoDimBindGroup);
  const twoDimLayout = dispatchLayout(twoDimCount, maxDispatch);
  twoDimPass.dispatchWorkgroups(twoDimLayout.x, twoDimLayout.y);
  twoDimPass.end();
  device.queue.submit([twoDimEncoder.finish()]);
  const twoDimValues = await readU32(device, twoDimVisits, twoDimCount);
  let twoDimFirstBad = -1;
  let twoDimSum = 0;
  for (let i = 0; i < twoDimValues.length; i += 1) {
    twoDimSum += twoDimValues[i];
    if (twoDimFirstBad < 0 && twoDimValues[i] !== 1) twoDimFirstBad = i;
  }
  const dispatch2d = {
    count: twoDimCount,
    max_dispatch: maxDispatch,
    ...twoDimLayout,
    sum: twoDimSum,
    first_bad: twoDimFirstBad,
    exceeds_device_x_limit: twoDimCount > maxDispatch,
    ok: twoDimLayout.y > 1 && twoDimFirstBad < 0 && twoDimSum === twoDimCount,
  };
  twoDimConfig.destroy();
  twoDimVisits.destroy();

  const prefixModule = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage, read> counts: array<u32>;
@group(0) @binding(1) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> control: array<atomic<u32>>;
@compute @workgroup_size(1)
fn prefix() {
  let capacity = min(arrayLength(&indices), 0x7fffffffu);
  var acceptedTotal = 0u;
  var requiredTotal = 0u;
  for (var tile = 0u; tile < arrayLength(&counts); tile += 1u) {
    offsets[tile] = acceptedTotal;
    let count = counts[tile];
    if (count > 0xffffffffu - requiredTotal) { requiredTotal = 0xffffffffu; }
    else { requiredTotal += count; }
    acceptedTotal += min(count, capacity - acceptedTotal);
  }
  let overflow = requiredTotal > capacity;
  offsets[arrayLength(&counts)] = acceptedTotal | select(0u, 0x80000000u, overflow);
  atomicStore(&control[0], requiredTotal);
  var overflowAmount = 0u;
  if (overflow) { overflowAmount = requiredTotal - capacity; }
  atomicStore(&control[1], overflowAmount);
}` });
  const prefixInfo = await prefixModule.getCompilationInfo();
  const prefixErrors = prefixInfo.messages.filter((message) => message.type === "error");
  if (prefixErrors.length) throw new Error(prefixErrors.map((message) => message.message).join(" | "));
  const prefixPipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: prefixModule, entryPoint: "prefix" },
  });
  const counts = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const offsets = device.createBuffer({
    size: 12,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const indices = device.createBuffer({ size: 16 * 4, usage: GPUBufferUsage.STORAGE });
  const control = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(counts, 0, new Uint32Array([0xfffffff0, 0x40]));
  const prefixBindGroup = device.createBindGroup({
    layout: prefixPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: counts } },
      { binding: 1, resource: { buffer: offsets } },
      { binding: 2, resource: { buffer: indices } },
      { binding: 3, resource: { buffer: control } },
    ],
  });
  const prefixEncoder = device.createCommandEncoder();
  prefixEncoder.clearBuffer(offsets);
  prefixEncoder.clearBuffer(control);
  const prefixPass = prefixEncoder.beginComputePass();
  prefixPass.setPipeline(prefixPipeline);
  prefixPass.setBindGroup(0, prefixBindGroup);
  prefixPass.dispatchWorkgroups(1);
  prefixPass.end();
  device.queue.submit([prefixEncoder.finish()]);
  const offsetValues = await readU32(device, offsets, 3);
  const controlValues = await readU32(device, control, 4);
  const prefix = {
    offsets: [...offsetValues],
    control: [...controlValues],
    ok:
      offsetValues[0] === 0 &&
      offsetValues[1] === 16 &&
      offsetValues[2] === (0x80000000 | 16) >>> 0 &&
      controlValues[0] === 0xffffffff &&
      controlValues[1] === 0xffffffef,
  };
  for (const buffer of [counts, offsets, indices, control]) buffer.destroy();
  device.destroy();

  const result = {
    ok: dispatch.every((entry) => entry.ok) && dispatch2d.ok && prefix.ok,
    dispatch,
    dispatch_2d: dispatch2d,
    prefix,
  };
  if (!result.ok) throw new Error(`P1 WebGPU probe failed: ${JSON.stringify(result)}`);
  return result;
}

run()
  .then((result) => {
    document.documentElement.dataset.probe = "passed";
    resultNode.textContent = JSON.stringify(result, null, 2);
  })
  .catch((error) => {
    document.documentElement.dataset.probe = "failed";
    resultNode.textContent = error.stack || error.message;
  });
