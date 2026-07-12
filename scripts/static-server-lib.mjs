import { createReadStream, realpathSync } from "node:fs";
import { realpath, stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, resolve, sep } from "node:path";

const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".png": "image/png",
  ".wasm": "application/wasm",
  ".webp": "image/webp",
};

function normalizeBasePath(value = "/") {
  const withLeadingSlash = value.startsWith("/") ? value : `/${value}`;
  return withLeadingSlash.endsWith("/") ? withLeadingSlash : `${withLeadingSlash}/`;
}

function createStaticServer({ root, basePath = "/", cacheControl = "no-store" }) {
  const resolvedRoot = realpathSync(resolve(root));
  const normalizedBase = normalizeBasePath(basePath);
  return createServer(async (request, response) => {
    try {
      const requestPath = decodeURIComponent(new URL(request.url, "http://127.0.0.1").pathname);
      if (!requestPath.startsWith(normalizedBase)) throw new Error("outside base path");
      const localPath = `/${requestPath.slice(normalizedBase.length)}`;
      let filePath = resolve(resolvedRoot, `.${localPath}`);
      if (filePath !== resolvedRoot && !filePath.startsWith(`${resolvedRoot}${sep}`)) throw new Error("invalid path");
      let info = await stat(filePath);
      if (info.isDirectory()) {
        filePath = resolve(filePath, "index.html");
        info = await stat(filePath);
      }
      if (!info.isFile()) throw new Error("not a file");
      filePath = await realpath(filePath);
      if (filePath !== resolvedRoot && !filePath.startsWith(`${resolvedRoot}${sep}`)) throw new Error("symlink outside root");
      response.writeHead(200, {
        "Content-Type": mimeTypes[extname(filePath).toLowerCase()] || "application/octet-stream",
        "Cache-Control": cacheControl,
      });
      createReadStream(filePath).pipe(response);
    } catch {
      response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      response.end("Not found\n");
    }
  });
}

export { createStaticServer, mimeTypes, normalizeBasePath };
