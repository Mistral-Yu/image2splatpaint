import { cp, mkdir, rm, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve, sep } from "node:path";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const output = resolve(root, process.argv[2] || "_site");
if (output === root || !output.startsWith(`${root}${sep}`)) {
  throw new Error("Pages output must be a child directory of the repository.");
}

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
await cp(resolve(root, "index.html"), resolve(output, "index.html"));
await cp(resolve(root, "LICENSE"), resolve(output, "LICENSE"));
await mkdir(resolve(output, "web"), { recursive: true });
for (const filename of ["index.html", "app.js", "styles.css"]) {
  await cp(resolve(root, "web", filename), resolve(output, "web", filename));
}
await writeFile(resolve(output, ".nojekyll"), "", "utf8");

console.log(`GitHub Pages artifact: ${output}`);
