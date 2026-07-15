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
await cp(resolve(root, "THIRD_PARTY_NOTICES.md"), resolve(output, "THIRD_PARTY_NOTICES.md"));
await mkdir(resolve(output, "web"), { recursive: true });
for (const filename of ["index.html", "app.js", "styles.css", "tilt-camera.mjs", "tilt-viewer.mjs"]) {
  await cp(resolve(root, "web", filename), resolve(output, "web", filename));
}
await cp(resolve(root, "web/vendor"), resolve(output, "web/vendor"), { recursive: true });
await mkdir(resolve(output, "assets/source-images"), { recursive: true });
for (const filename of ["README.md", "ramen-photo.jpg"]) {
  await cp(resolve(root, "assets/source-images", filename), resolve(output, "assets/source-images", filename));
}
await writeFile(resolve(output, ".nojekyll"), "", "utf8");

console.log(`GitHub Pages artifact: ${output}`);
