import { spawnSync } from "node:child_process";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { sourceJavaScriptFiles } from "./release-manifest.mjs";

const root = resolve(fileURLToPath(new URL(".", import.meta.url)));
const failures = [];
for (const filename of sourceJavaScriptFiles) {
  const result = spawnSync(process.execPath, ["--check", resolve(root, filename)], {
    encoding: "utf8",
  });
  if (result.status !== 0) failures.push(`${filename}: ${result.stderr.trim() || result.stdout.trim()}`);
}

if (failures.length) throw new Error(`Source verification failed:\n- ${failures.join("\n- ")}`);
console.log(JSON.stringify({ ok: true, files: sourceJavaScriptFiles.length }, null, 2));
