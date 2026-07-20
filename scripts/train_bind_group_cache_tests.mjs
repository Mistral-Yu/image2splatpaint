import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const app = await readFile(new URL("../web/app.js", import.meta.url), "utf8");

const contracts = {
  qaOnlyDefaultOff:
    /bindGroupCacheQuery === "1"/.test(app) &&
    /QA_RUNTIME_ENABLED && query\.has\("bind-group-cache"\) && bindGroupCacheQuery === "1"/.test(app),
  stateScopedCache: /bindGroupCache: new Map\(\)/.test(app) && /bindGroupCacheStats: \{ hits: 0, misses: 0 \}/.test(app),
  frontStageKeyed: /front,\s*trainingStage,/.test(app) && /bindGroupKeyBase/.test(app),
  majorIterationGroupsCached: ["render", "ssim", "loss-gradient", "exact-backward", "exact-optimizer"]
    .every((name) => app.includes(`cachedBindGroup("${name}"`)),
  observable: /bind_group_cache_hits/.test(app) && /bind_group_cache_misses/.test(app),
};

for (const [name, passed] of Object.entries(contracts)) {
  assert.equal(passed, true, `train bind-group cache contract failed: ${name}`);
}

console.log(JSON.stringify({ ok: true, contracts }, null, 2));
