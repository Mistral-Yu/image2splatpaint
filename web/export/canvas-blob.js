(function installCanvasBlobExport(global) {
  function canvasToBlob(canvas, type = "image/png") {
    return new Promise((resolve, reject) => {
      canvas.toBlob((blob) => {
        if (blob) resolve(blob);
        else reject(new Error("Canvas image encoding failed."));
      }, type);
    });
  }

  global.Image2SplatPaintCanvasBlob = Object.freeze({ canvasToBlob });
})(globalThis);
