(function installTrainingSession(global) {
  class TrainingSession {
    constructor({ generation, renderer, image }) {
      this.generation = generation;
      this.renderer = renderer;
      this.image = image;
      this.params = null;
      this.metrics = null;
      this.cancelled = false;
    }

    updateOwnership({ image, params, metrics } = {}) {
      if (image !== undefined) this.image = image;
      if (params !== undefined) this.params = params;
      if (metrics !== undefined) this.metrics = metrics;
    }

    owns(current) {
      return Boolean(
        !this.cancelled &&
        current.trainingRun === this &&
        current.trainingGeneration === this.generation &&
        current.webgpu.renderer === this.renderer &&
        current.image === this.image &&
        (!this.params || current.params === this.params) &&
        (!this.metrics || current.metrics === this.metrics) &&
        !this.renderer?.deviceLost
      );
    }

    assertCurrent(current) {
      if (!this.owns(current)) throw TrainingSession.cancelledError();
    }

    async awaitCurrent(current, promise) {
      this.assertCurrent(current);
      const value = await promise;
      this.assertCurrent(current);
      return value;
    }

    cancel() {
      this.cancelled = true;
    }

    static cancelledError() {
      const error = new Error("Training run was cancelled because its WebGPU lifecycle changed.");
      error.trainingRunCancelled = true;
      return error;
    }
  }

  global.Image2SplatPaintTrainingSession = TrainingSession;
})(globalThis);
