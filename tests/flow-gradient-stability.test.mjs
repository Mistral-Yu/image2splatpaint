import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");
test("Flow backward preserves visible gradients behind an underflowed opaque stack", async () => {
  const f = Math.fround;
  const multiplier = f(1 - f(0.99));
  let t = 1, logT = 0;
  for (let i = 0; i < 32; i++) {
    t = f(t * multiplier);
    logT = f(logT + f(Math.log(multiplier)));
  }
  assert.equal(t, 0);
  for (let i = 31; i >= 0; i--) {
    logT = f(logT - f(Math.log(multiplier)));
    t /= multiplier;
  }
  assert.equal(t, 0, "legacy reverse division loses even the front gradient");
  assert.ok(Math.abs(Math.exp(Math.min(0, logT)) - 1) < 0.0001);
  const source = await read("web/training/flow-ribbon-trainer.js");
  assert.equal((source.match(/var<storage, read> log_transmittance/g) || []).length, 2);
  assert.equal((source.match(/let transmittance_before = exp\(min\(0\.0, log_t_after/g) || []).length, 2);
  assert.ok(!source.includes("transmittance_after / one_minus_alpha"));
  assert.ok(source.includes("color += transmittance * evaluated.alpha * evaluated.pigment;"));
  assert.ok(source.includes("transmittance *= 1.0 - evaluated.alpha;"));
});

test("Flow curve rotation pullback is used in both backward paths, not in forward", async () => {
  const source = await read("web/training/flow-ribbon-trainer.js");
  assert.equal((source.match(/let d_tangent = d_kernel \* chain_tangent_pullback/g) || []).length, 2);
  assert.ok(source.includes("kernel_gradient.x * (side + offset) - kernel_gradient.y * along"));
  assert.equal((source.match(/param_at\(stroke, 8u\) > 0.55\)/g) || []).length, 2);
  // Independent central differences through a normalized moving frame.
  function kernel(d, q, offset) {
    const length = Math.hypot(...d), e = d.map(x => x / length), n = [-e[1], e[0]];
    const delta = q.map((x, i) => x - n[i] * offset);
    const a = delta[0]*e[0] + delta[1]*e[1], b = delta[0]*n[0] + delta[1]*n[1];
    return { k: Math.exp(-a*a/8-b*b/2), a, b, n, length };
  }
  for (const d of [[3,1],[-2,4],[5,-3]]) for (const offset of [-.7,0,.8]) {
    const q=[1.8,-.2], k=kernel(d,q,offset), ka=-k.k*k.a/4, kb=-k.k*k.b;
    const analytic=k.n.map(x=>x*(ka*(k.b+offset)-kb*k.a)/k.length);
    for(let c=0;c<2;c++) {
      const plus=d.slice(),minus=d.slice();plus[c]+=1e-5;minus[c]-=1e-5;
      assert.ok(Math.abs((kernel(plus,q,offset).k-kernel(minus,q,offset).k)/2e-5-analytic[c])<1e-7);
    }
  }
});

test("Variable backcoat cells preserve count, deterministic pigment, and complete flat-interior coverage", async () => {
  const context=vm.createContext({Float32Array,Float64Array});
  vm.runInContext(await read("web/training/flow-paint-reference.js"),context);
  const create=context.Image2SplatPaintFlowPaintReference.createSplatUnderpaintPlan;
  const image={width:83,height:61,rgb:new Float32Array(83*61*3).fill(.4)};
  const base={count:47,representation:"curve-splat-chain",seed:240825};
  assert.equal(JSON.stringify(create(image,base)),JSON.stringify(create(image,{...base,sizeVariation:0})));
  for(const variation of [.4,.75]) {
    const result=create(image,{...base,sizeVariation:variation});
    assert.equal(JSON.stringify(result),JSON.stringify(create(image,{...base,sizeVariation:variation})));
    assert.equal(result.strokePlan.length,47);
    const coverage=new Uint8Array(83*61);
    for(const s of result.strokePlan) {
      assert.equal(s.opacity,.995);
      const long=s.underpaint_sigma_long_px,short=s.underpaint_sigma_short_px;
      const ex=(s.control_1_x-s.center_x)/long,ey=(s.control_1_y-s.center_y)/long;
      for(let y=s.coverage_cell_min_y;y<s.coverage_cell_max_y;y++)for(let x=s.coverage_cell_min_x;x<s.coverage_cell_max_x;x++) {
        const dx=x+.5-s.center_x,dy=y+.5-s.center_y;
        const q=((dx*ex+dy*ey)/long)**4+((-dx*ey+dy*ex)/short)**4;
        assert.ok(q<.84);coverage[y*83+x]=1;
      }
    }
    assert.ok(coverage.every(x=>x===1));
    assert.ok(new Set(result.strokePlan.map(s=>s.half_width_px.toFixed(2))).size>15);
  }
});

test("Alternating writes keep Adam age continuous and the backcoat control is wired and locked", async () => {
  const source=await read("web/training/flow-ribbon-trainer.js");
  const context=vm.createContext({Float32Array,DataView,ArrayBuffer});
  vm.runInContext(source.replace("global.Image2SplatPaintFlowRibbonTrainer =",
    "global.testConfigBytes = configBytes; global.Image2SplatPaintFlowRibbonTrainer ="),context);
  const data={width:64,height:64,strokeCount:2,tileCols:4,tileRows:4,sampleCount:8,
    tileSampleStride:8,canvasLinear:[.1,.2,.3],strokeTextureMode:2};
  for(let step=0;step<10;step++){
    const bytes=context.testConfigBytes(data,step,{alternateShapeColor:true,iterations:10});
    const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
    assert.equal(view.getFloat32(48,true)===0,step%5===4);
    assert.equal(view.getFloat32(52,true)===0,step%5===4);
    assert.equal(view.getFloat32(60,true)===0,step%5!==4);
  }
  const controls=await read("web/ui/training-controls.js");
  assert.ok(controls.includes("els.flowSplatBackcoatSizeVariation.disabled = state.running"));
  assert.ok((await read("web/index.html")).includes('id="flowSplatBackcoatSizeVariation" data-testid="flow-splat-backcoat-size-variation" type="number" min="0" max="75" step="5" value="40"'));
});
