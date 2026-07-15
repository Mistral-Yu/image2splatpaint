import { resolve } from "node:path";
import { createStaticServer, normalizeBasePath } from "./static-server-lib.mjs";

const port = Number(process.argv[2] || 8765);
const root = resolve(process.argv[3] || process.cwd());
const basePath = normalizeBasePath(process.argv[4] || "/");

createStaticServer({ root, basePath }).listen(port, "127.0.0.1", () => {
  console.log(`Image2SplatPaint: http://127.0.0.1:${port}${basePath}`);
});
