(function installGpuMetricReducers(global) {
  function profileDistributionSummary(histogramValues, pixelCount, total) {
    const labels = ["0", "1", "2-3", "4-7", "8-15", "16-31", "32-63", "64+"];
    const counts = Array.from(histogramValues, (value) => Number(value) || 0);
    const samples = Math.max(0, Number(pixelCount) || 0);
    const percentileBin = (fraction) => {
      if (samples <= 0) return null;
      const target = Math.max(1, Math.ceil(samples * fraction));
      let cumulative = 0;
      for (let index = 0; index < counts.length; index += 1) {
        cumulative += counts[index];
        if (cumulative >= target) return labels[index];
      }
      return labels.at(-1);
    };
    return {
      samples,
      mean: samples > 0 ? (Number(total) || 0) / samples : null,
      p50_bin: percentileBin(0.5),
      p90_bin: percentileBin(0.9),
      p99_bin: percentileBin(0.99),
      histogram: Object.fromEntries(labels.map((label, index) => [label, counts[index] || 0])),
    };
  }

  global.Image2SplatPaintGpuMetrics = Object.freeze({ profileDistributionSummary });
})(globalThis);
