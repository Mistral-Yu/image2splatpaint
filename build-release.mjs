import { cp, mkdir, rm } from "node:fs/promises";
import { dirname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { releaseArtifactFiles } from "./release-manifest.mjs";

const root = resolve(fileURLToPath(new URL(".", import.meta.url)));
const output = resolve(root, process.argv[2] || "_site");
if (output === root || !output.startsWith(`${root}${sep}`)) {
  throw new Error("Release output must be a child directory of the repository.");
}

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
for (const filename of releaseArtifactFiles) {
  const destination = resolve(output, filename);
  await mkdir(dirname(destination), { recursive: true });
  await cp(resolve(root, filename), destination);
}

console.log(`GitHub Pages artifact: ${output}`);
