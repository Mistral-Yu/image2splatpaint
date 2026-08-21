import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

async function loadTrainingSession() {
  const context = vm.createContext({ Error, Object });
  context.globalThis = context;
  vm.runInContext(
    await readFile(new URL("../web/training/session.js", import.meta.url), "utf8"),
    context,
    { filename: "web/training/session.js" },
  );
  return context.Image2SplatPaintTrainingSession;
}

function currentState(session, renderer, image) {
  return {
    trainingRun: session,
    trainingGeneration: session.generation,
    webgpu: { renderer },
    image,
    params: null,
    metrics: null,
  };
}

test("TrainingSession preserves ownership and optional params/metrics checks", async () => {
  const TrainingSession = await loadTrainingSession();
  const renderer = { deviceLost: false };
  const image = {};
  const session = new TrainingSession({ generation: 4, renderer, image });
  const current = currentState(session, renderer, image);
  assert.equal(session.owns(current), true);

  const params = {};
  const metrics = {};
  session.updateOwnership({ params, metrics });
  current.params = params;
  current.metrics = metrics;
  assert.equal(session.owns(current), true);
  current.params = {};
  assert.equal(session.owns(current), false);
});

test("TrainingSession validates both sides of an awaited GPU operation", async () => {
  const TrainingSession = await loadTrainingSession();
  const renderer = { deviceLost: false };
  const image = {};
  const session = new TrainingSession({ generation: 2, renderer, image });
  const current = currentState(session, renderer, image);
  let resolveOperation;
  const operation = new Promise((resolve) => { resolveOperation = resolve; });
  const guarded = session.awaitCurrent(current, operation);
  session.cancel();
  resolveOperation("complete");
  await assert.rejects(guarded, (error) => error.trainingRunCancelled === true);
});
