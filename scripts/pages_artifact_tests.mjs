import { access, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { createStaticServer } from "./static-server-lib.mjs";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const site = resolve(root, process.argv[2] || "_site");
const basePath = "/image2gaussianpaint/";
const required = [
  "index.html",
  "LICENSE",
  ".nojekyll",
  "web/index.html",
  "web/app.js",
  "web/styles.css",
];

for (const path of required) await access(resolve(site, path));

const rootHtml = await readFile(resolve(site, "index.html"), "utf8");
const webHtml = await readFile(resolve(site, "web/index.html"), "utf8");
const workflow = await readFile(resolve(root, ".github/workflows/pages.yml"), "utf8");
const contracts = {
  relativeLauncher: rootHtml.includes('url=./web/index.html') && rootHtml.includes('href="./web/index.html"'),
  contentSecurityPolicy: rootHtml.includes("Content-Security-Policy") && webHtml.includes("Content-Security-Policy"),
  noRootRelativeAssets: !/(?:src|href)=["']\//.test(webHtml),
  noRuntimeImportMap: !webHtml.includes('type="importmap"') && !webHtml.includes("playcanvas.mjs"),
  licenseLinks:
    webHtml.includes('href="../LICENSE"') &&
    !webHtml.includes("THIRD_PARTY_NOTICES"),
  simpleExports: !webHtml.includes("Compressed PLY") && !webHtml.includes("PlayCanvas SOG"),
  pagesWorkflow:
    workflow.includes("actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0 # v7") &&
    workflow.includes("actions/configure-pages@45bfe0192ca1faeb007ade9deae92b16b8254a0d # v6") &&
    workflow.includes("actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9 # v5") &&
    workflow.includes("actions/deploy-pages@cd2ce8fcbc39b97be8ca5fce6e763baed58fa128 # v5") &&
    workflow.includes("node scripts/build-pages.mjs"),
};
if (Object.values(contracts).some((value) => !value)) throw new Error(`Pages artifact contract failed: ${JSON.stringify(contracts)}`);

const server = createStaticServer({ root: site, basePath });
await new Promise((resolveListen, reject) => {
  server.once("error", reject);
  server.listen(0, "127.0.0.1", resolveListen);
});

try {
  const address = server.address();
  const origin = `http://127.0.0.1:${address.port}${basePath}`;
  const checks = [
    ["", "text/html"],
    ["web/index.html", "text/html"],
    ["web/app.js", "text/javascript"],
    ["LICENSE", "application/octet-stream"],
  ];
  const http = {};
  for (const [path, expectedType] of checks) {
    const response = await fetch(`${origin}${path}`);
    const type = response.headers.get("content-type") || "";
    http[path || "index.html"] = { status: response.status, type };
    if (!response.ok || !type.startsWith(expectedType)) throw new Error(`${path || "index.html"}: ${response.status} ${type}`);
  }
  const outside = await fetch(`http://127.0.0.1:${address.port}/web/index.html`);
  if (outside.status !== 404) throw new Error(`base path isolation failed: ${outside.status}`);
  console.log(JSON.stringify({ ok: true, basePath, contracts, http }, null, 2));
} finally {
  await new Promise((resolveClose) => server.close(resolveClose));
}
