(function installNumericUtilities(global) {
  function clampNumber(value, min, max, fallback) {
    const number = Number(value);
    if (!Number.isFinite(number)) return fallback;
    return Math.min(max, Math.max(min, number));
  }

  function normalizeStepInteger(value, { min, max, fallback, step = 1 }) {
    const safeMin = Number.isSafeInteger(min) ? min : 0;
    const safeMax = Number.isSafeInteger(max) && max >= safeMin ? max : Number.MAX_SAFE_INTEGER;
    const safeStep = Number.isSafeInteger(step) && step > 0 ? step : 1;
    const fallbackNumber = Number(fallback);
    const safeFallback = Number.isFinite(fallbackNumber) ? fallbackNumber : safeMin;
    const parsed = Number(value);
    const finite = Number.isFinite(parsed) ? parsed : safeFallback;
    const clamped = Math.min(safeMax, Math.max(safeMin, finite));
    const snapped = safeMin + Math.round((clamped - safeMin) / safeStep) * safeStep;
    return Math.min(safeMax, Math.max(safeMin, Math.trunc(snapped)));
  }

  function hexColorToRgb(value, fallback = [0, 0, 0]) {
    const match = /^#([0-9a-f]{6})$/i.exec(String(value || ""));
    if (!match) return [...fallback];
    const packed = Number.parseInt(match[1], 16);
    return [((packed >> 16) & 255) / 255, ((packed >> 8) & 255) / 255, (packed & 255) / 255];
  }

  function limitNumber(limits, name, fallback) {
    const value = limits?.[name];
    return Number.isFinite(value) ? value : fallback;
  }

  global.Image2SplatPaintNumeric = Object.freeze({
    clampNumber,
    normalizeStepInteger,
    hexColorToRgb,
    limitNumber,
  });
})(globalThis);
