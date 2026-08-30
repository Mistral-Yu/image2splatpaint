const ROW_BYTES = 32;
const SH_C0 = 0.28209479177387814;
const MB = 1024 * 1024;
const GB = 1024 * MB;
const MEMORY_LIMITER_UNLOCK_MULTIPLIER = 4;
const MEMORY_LIMITER_UNLOCK_MAX_BYTES = 8 * GB;
const MAX_INPUT_FILE_BYTES = 128 * MB;
// Keep the cached source independent from the smaller training resolution.
// Large originals are decoded once, reduced to this long side, then the
// original decoder surface is released before any WebGPU work begins.
const INPUT_CACHE_MAX_SIDE = 4096;
const MAX_INPUT_DECODED_PIXELS = 400_000_000;
const MAX_IMAGE_HEADER_PROBE_BYTES = 4 * MB;
const DEFAULT_ITERATIONS = 3000;
const DEFAULT_MAX_SIDE = 512;
const DEFAULT_PNG_EXPORT_RESOLUTION = "training";
const MIN_PNG_EXPORT_LONG_SIDE = 32;
const MAX_PNG_EXPORT_LONG_SIDE = 8192;
const DEFAULT_INITIAL_SPLATS = 128;
const DEFAULT_FINAL_SPLATS = 8192;
const MANUAL_SPLATS_MAX = 1048576;
const CAPACITY_PROBE_FAST_PATH_MAX = 262144;
const CAPACITY_PROBE_TIERS = [262144, 524288, 786432, 1048576];
const DEFAULT_LR_SCALE = 1;
const DEFAULT_POSITION_LR = 0.00016;
const DEFAULT_COLOR_LR = 0.0025;
const DEFAULT_OPACITY_LR = 0.05;
const DEFAULT_SCALE_LR = 0.002;
const DEFAULT_ROTATION_LR = 0.001;
const DEFAULT_THETA_ALIGN_LR = 0.005;
const DEFAULT_MAX_ANISOTROPY = 8;
const DEFAULT_SURFACE_ANISOTROPY = 8;
const DEFAULT_BOUNDARY_SIGMA = 0;
const DEFAULT_DSSIM_WEIGHT = 0.2;
const DEFAULT_SGLD_NOISE_LR = 0.02;
const BOUNDARY_SIGMA = 2.5;
const RENDER_SIGMA = 4;
const MIN_SPLAT_SCALE = 0.0015;
// P1/P2 must reserve detail capacity relative to the initial point density,
// rather than to a fixed display-pixel size. P3 removes this extra floor.
const DEFAULT_STAGE_MIN_SCALE_RATIO = 0.05;
const DEFAULT_P1_BASE_SCALE_FLOOR_RATIO = 0.50;
const DEFAULT_P2_BASE_SCALE_FLOOR_RATIO = 0.35;
const DEFAULT_P3_BASE_SCALE_FLOOR_RATIO = 0.35;
const DEFAULT_P1_RELATIVE_SCALE_FLOOR_RATIO = 0.50;
const DEFAULT_P2_RELATIVE_SCALE_FLOOR_RATIO = 0.20;
const DEFAULT_P3_RELATIVE_SCALE_FLOOR_RATIO = 0.09;
const DEFAULT_RELATIVE_SCALE_GUARD_STRENGTH = 0.20;
const BACKGROUND_EXPOSURE_EPSILON = 1e-8;
const MIP_PIXEL_SIGMA = 0.35;
// Start with overlapping but not over-expanded footprints. Density growth can
// add coverage where the image needs it instead of beginning with broad blobs.
const INITIAL_SPLAT_COVERAGE_MULTIPLIER = 2.0;
const PHASE_ONE_MAX_PLANAR_SCALE = 0.32;
const PHASE_ONE_SHAPE_LR_MULTIPLIER = 2.5;
const DENSITY_EVENT_SLOTS = 35;
const PHASE33_IMPORTANCE_EMA = 0.05;
const PHASE33_COVERAGE_TARGET = 0.05;
const PHASE33_COVERAGE_LOSS_WEIGHT = 0.02;
const PHASE33_COVERAGE_DENSITY_STRENGTH = 0.15;
const PHASE33_COARSE_MAX_SIDE = 512;
const CURRICULUM_COARSE_MIN_SIDE = 1;
const CURRICULUM_COARSE_DIVISOR = 4;
const CURRICULUM_COARSE_FRACTION = 1 / 7;
const CURRICULUM_DENSITY_FRACTION = 3 / 7;
const DEFAULT_GROWTH_APPLY_UNTIL_FRACTION = 0.90;
const CURRICULUM_TILT_LATE_FRACTION = 6 / 7;
const DEFAULT_ADC_RECYCLE_RATE = 0.25;
const DEFAULT_ADC_LATE_RECYCLE_RATE = 0.10;
const DEFAULT_ADC_SPLIT_SIGNAL_THRESHOLD = 0.0003;
const DEFAULT_ADC_SPLIT_RESIDUAL_THRESHOLD = 0.0025;
const DEFAULT_ADC_WINDOW_EVENTS = 5;
const ADC_RECOVERY_DECAY_STEPS = 250;
const EXPERIMENTAL_REFINE_EVERY = 50;
const EXPERIMENTAL_ADC_INTERVAL_FOR_7000 = 3000;
const DENSIFY_WARMUP_FRACTION = 0.1;
const DENSIFY_WARMUP_MAX_STEPS = 700;
// Product training keeps GPU work queued and only waits at bounded health or
// structural checkpoints. Full-image quality evaluation is final-only.
const DEFAULT_TRAIN_SYNC_INTERVAL = 64;
const HIGH_SPLAT_SYNC_THRESHOLD = 65536;
const VERY_HIGH_SPLAT_SYNC_THRESHOLD = 262144;
const REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE = 8;
const PREFERRED_STORAGE_BUFFERS_PER_SHADER_STAGE = 9;
const DEFAULT_MAX_METRIC_INTERVAL = 100;
const MAX_PREVIEW_PADDING_PX = 256;
const MAX_PREVIEW_PADDING_FRACTION = 0.2;
const DEFAULT_LOCAL_COLOR_ANCHOR_WEIGHT = 0.02;
const DEFAULT_VIRTUAL_LOCAL_COLOR_ANCHOR_WEIGHT = 0.05;
const DEFAULT_ALPHA_LOSS_WEIGHT = 0.2;
const DEFAULT_ALPHA_TARGET = 0.99;
const LAYER_CODE_RANGE = 0.24;
const VIRTUAL_DEPTH_RAW_LIMIT = 4;
const DEFAULT_VIRTUAL_DEPTH_THICKNESS = 0.005;
const DEFAULT_VIRTUAL_DEPTH_CENTER_WEIGHT = 0.02;
const DEFAULT_VIRTUAL_DEPTH_SMOOTHNESS_WEIGHT = 0.01;
const DEFAULT_VIRTUAL_DEPTH_LEARNING_RATE = 0.05;
const DEFAULT_VIRTUAL_DEPTH_UPDATE_INTERVAL = 16;
const DEFAULT_VIRTUAL_DEPTH_SOFT_CONSTRAINT = false;
const DEFAULT_VIRTUAL_DEPTH_PRIOR_DELTA = 0.001;
const MIN_VIRTUAL_DEPTH_CAMERA_CONFIDENCE = 0.05;
// Large enough to survive common 3DGS depth-sort quantization, but only 0.5%
// of the exported plane's two-unit long side at the extrema.
const PLY_LAYER_DEPTH_SPAN = 1e-2;
const DEFAULT_TILT_SPLIT_ANGLE_DEGREES = 5;
const DEFAULT_TILT_SPLIT_COLOR_THRESHOLD = 0.08;
const DEFAULT_TILT_SPLIT_SHRINK = 0.8;
const DEFAULT_MAX_PLANAR_SCALE = 0.1;
const DEFAULT_VIRTUAL_TILT_INTERVAL = 32;
const DEFAULT_VIRTUAL_TILT_WEIGHT = 0.25;
const DEFAULT_VIRTUAL_ORDER_PENALTY_WEIGHT = 0;
const DEFAULT_VIRTUAL_TILT_CAMERA_DISTANCE = 4;
const DEFAULT_SHARED_CAMERA_FOV_DEGREES = 50;
const MIN_SHARED_CAMERA_FOV_DEGREES = 25;
const MAX_SHARED_CAMERA_FOV_DEGREES = 55;
const VIRTUAL_TILT_FOV_DEGREES = DEFAULT_SHARED_CAMERA_FOV_DEGREES;
const VIRTUAL_TILT_DIRECTIONS = Object.freeze([
  [1, 0], [-1, 0], [0, 1], [0, -1],
]);
const DEFAULT_VIRTUAL_CAMERA_POOL_SLOTS = 128;
const DEFAULT_VIRTUAL_CAMERA_SLOTS = 64;
const DEFAULT_VIRTUAL_CAMERA_SHARE_PERCENT = 50;
const DEFAULT_VIRTUAL_CAMERA_COUNT = 24;
const MAX_VIRTUAL_CAMERA_COUNT = 128;
const DEFAULT_VIRTUAL_CAMERA_MAX_ANGLE_DEGREES = 75;
// Keep virtual cameras inside the open front hemisphere. Exactly 90 degrees is
// singular for the planar inverse projection, but 75 degrees is no longer a
// product limit.
const MAX_VIRTUAL_CAMERA_ANGLE_DEGREES = 89;
const DEFAULT_VIRTUAL_CAMERA_SEED = 0x2f6e2b1;
const DEFAULT_VIRTUAL_CAMERA_REGULARIZATION_WEIGHT = 0.1;
const DEFAULT_VIRTUAL_CAMERA_REGULARIZATION_RAMP_STEPS = 200;
const DEFAULT_VIRTUAL_CAMERA_MID_ANGLE_DEGREES = 2;
const DEFAULT_VIRTUAL_CAMERA_FULL_ANGLE_DEGREES = 5;
const VIRTUAL_CAMERA_GOLDEN_ANGLE_RADIANS = Math.PI * (3 - Math.sqrt(5));
// Keep the shared config buffer 16-byte aligned. Slot 88 carries the paint
// minimum opacity and slot 90 enables learning above that floor for opaque
// Paint. Slot 87 carries the Brush taper learning rate; retired slot 89 stays
// zero, and slot 92 carries Rectangle orientation.
// Slot 93 enables the shared monochrome-underpainting workflow; slots 94-95 enable
// Brush opacity and trainable-width effects. Slots 96 and
// 98 carry Rectangle's minimum and maximum parallel-edge width ratios. Slot 97
// carries Rectangle shape flags. Slot 100 carries the phase-independent shared
// monochrome-underpainting cutoff; slots 101-104 carry Brush effect endpoints.
// Slots 105-106 are the shared scale-biased surface-layer sort: scheduled
// full-sort trigger and requested layer count. Slot 107 packs the bounded
// front-footprint parent replacement mode in 0..2, the QA-only front-split
// child experiment in bit 4, scheduled color-aware promotion in bit 8, and
// the train-layer RGB guard in bit 16.
// Virtual soft-depth uses slots 108-110 for camera confidence, robust-prior
// delta, and its enable flag. Slot 111 carries the default-OFF Brush local
// color-flow orientation strength. Slot 112 enables Brush-only directional
// stroke aspect persistence in the optimizer. Slots 113-114 carry
// its Ribbon and Accent floors. Slot 115 carries Rectangle's minimum long/short
// aspect ratio. Slots 116-117 carry Rectangle's short-side and long-side
// opacity-gradient multipliers. Slot 118 carries the learned paint-opacity
// maximum and slot 119 carries Brush's minimum long/short aspect ratio.
// Slots 120-122 carry the phase-relative geometric scale floor, its soft
// correction strength, and the optimizer enable flag. Slots 123-126 carry
// Rectangle and Brush center-to-edge opacity multiplier Min/Max pairs.
const TRAIN_CONFIG_FLOATS = 128;
const TRAIN_CONFIG_BYTES = TRAIN_CONFIG_FLOATS * 4;
const MAX_TRAIN_BATCH_SIZE = 16;
const TRAIN_BATCH_CONFIG_BYTES = TRAIN_CONFIG_BYTES * MAX_TRAIN_BATCH_SIZE;
const ALPHA_STATE_BYTES_PER_PIXEL = 16;
const LAYER_TRAIN_INTERVAL = 500;
const DEFAULT_DISCRETE_LAYER_COUNT = 16;
const DEFAULT_DISCRETE_LAYER_MOVE_RADIUS = 0;
const DEFAULT_SCALE_BIASED_SURFACE_LAYER_SORT_LAYERS = 32;
const DEFAULT_SCALE_BIASED_SURFACE_LAYER_SORT_INTERVAL = 100;
const DEFAULT_SCALE_BIASED_SURFACE_LAYER_SORT_UNTIL = 0.90;
const MAX_SCALE_BIASED_SURFACE_LAYER_SORT_INTERVAL = 100_000;
const MIN_DISCRETE_LAYER_COUNT = 2;
const MAX_DISCRETE_LAYER_COUNT = 32;
const MIN_COMPACTION_SPLATS = 256;
const LAYER_DIAGNOSTIC_DEEP_FRACTION = 0.5;
const EXACT_GRADIENT_STRIDE = 16;
const OVERLAP_METRIC_STRIDE = 19;
const TILE_SIZE = 16;
const TILE_INDEX_FACTOR = 64;
const TILE_INDEX_INITIAL_HEADROOM = 1.5;
const TILE_INDEX_GROWTH_HEADROOM = 1.25;
const TILE_OFFSET_OVERFLOW_BIT = 0x80000000;
const TILE_OFFSET_VALUE_MASK = 0x7fffffff;
const DEFAULT_GROWTH_FRACTION = 0.35;
const DEFAULT_GROWTH_SIGNAL_THRESHOLD = 0.0003;
const DEFAULT_STAGE_GROWTH_SHARES = Object.freeze({ p1: 18.15, p2: 51.85, p3: 30 });
const METRIC_TILE_STRIDE = 36;
const PSNR_MSE_FLOOR = 1e-12;
// Final-only virtual-camera readback. Keep this separate from the training
// metric layout so front-only training retains its established fast path.
const VIRTUAL_CAMERA_METRIC_TILE_STRIDE = 20;
// Four output vec4s (backward coefficients / SSIM map) plus five reusable
// separable-filter scratch vec4s. This keeps the exact 11x11 derivative
// separable instead of issuing 121 neighborhood reads per training pixel.
const SSIM_WORKING_BYTES_PER_PIXEL = 144;
const DEFAULT_STRUCTURE_REGION_GRID = 8;
const MAX_STRUCTURE_REGION_GRID = 16;
const DEFAULT_OVERDENSITY_DONOR_FRACTION = 1 / 32;
const DEFAULT_OVERDENSITY_CORRECTION_INTERVAL = 1500;
const PHASE45_REGION_GRID = MAX_STRUCTURE_REGION_GRID;
const PHASE45_REGION_COUNT = PHASE45_REGION_GRID * PHASE45_REGION_GRID;
const PHASE45_REGION_STRIDE = 24;
const HIGH_ITERATION_CONFIRM = 50000;
const PERFORMANCE_PROFILE_QUERY_CAPACITY = 32;
const EXACT_BACKWARD_TELEMETRY_BYTES = 96;
const SEGMENTED_EXACT_BACKWARD_MAX_BYTES = 64 * MB;
const FIXED_POINT_EXACT_GRADIENT_SCALE = 8192;
const FIXED_POINT_EXACT_GRADIENT_CONTROL_BYTES = 16;
const PRODUCT_NAME = "Image2SplatPaint";
const PRODUCT_FORMAT = "image2splatpaint-web";
const PLANAR_GAUSSIAN_ALGORITHM_ID = "planar-gaussian";
const RECTANGLE_SPLATS_ALGORITHM_ID = "rectangle-splats";
const LAYERED_OPAQUE_BRUSH_ALGORITHM_ID = "layered-opaque-brush";
const GS_VIRTUAL_CAMERA_ALGORITHM_ID = "gs-virtual-camera-sampling";
const FLOW_SPLAT_FUSION_ALGORITHM_ID = "flow-splat-fusion";
const CURVE_SPLAT_CHAIN_ALGORITHM_ID = "curve-splat-chain";
const RECTANGLE_KERNEL_EXTENT = 1.5;
const RECTANGLE_EDGE_SOFTNESS = 0.15;
const DEFAULT_RECTANGLE_TOP_RATIO = 1;
const DEFAULT_RECTANGLE_TOP_RATIO_MAX = 1;
const MIN_RECTANGLE_TOP_RATIO = 0;
const MAX_RECTANGLE_TOP_RATIO = 1;
const MIN_RECTANGLE_ASPECT_RATIO = 1;
const MAX_RECTANGLE_ASPECT_RATIO = 32;
const DEFAULT_RECTANGLE_MIN_ASPECT_RATIO = 1;
const DEFAULT_RECTANGLE_ASPECT_RATIO = 4;
const DEFAULT_RECTANGLE_ORIENTATION = "free";
const DEFAULT_RECTANGLE_ORIENTATION_TOLERANCE = 10;
const DEFAULT_RECTANGLE_PRESERVE_AREA = true;
const DEFAULT_RECTANGLE_EDGE_DIRECTED_TAPER = true;
const DEFAULT_RECTANGLE_STRUCTURE_AWARE_RATIO = true;
const DEFAULT_RECTANGLE_ASYMMETRIC_SOFTNESS = false;
const RECTANGLE_FLAG_PRESERVE_AREA = 1;
const RECTANGLE_FLAG_STRUCTURE_AWARE_RATIO = 2;
const RECTANGLE_FLAG_ASYMMETRIC_SOFTNESS = 4;
const RECTANGLE_FLAG_EDGE_DIRECTED_TAPER = 8;
const RECTANGLE_STRUCTURE_MIN_COHERENCE = 0.2;
const RECTANGLE_STRUCTURE_MIN_ENERGY = 0.00012;
const LAYERED_OPAQUE_BRUSH_KERNEL_EXTENT = 1.5;
const LAYERED_OPAQUE_BRUSH_EDGE_SOFTNESS = 0.18;
const LAYERED_OPAQUE_BRUSH_TIP_WIDTH = 0.18;
const ILLUSTRATIVE_OIL_DETAIL_THRESHOLD = 0.58;
const ILLUSTRATIVE_OIL_RIBBON_ANISOTROPY = 2.15;
const LAYERED_OPAQUE_BRUSH_OPACITY = 0.995;
// BR-CAND-01 starts as a final-only QA diagnostic.  It measures the actual
// front-to-back alpha contributions instead of the Gaussian/raw-weight overlap
// metric, and does not participate in optimization or density decisions.
const BRUSH_CONTRIBUTION_FLAT_LINEAR_GRADIENT = 0.03;
const BRUSH_CONTRIBUTION_MIN_COMPOSITE_ALPHA = 0.5;
const BRUSH_CONTRIBUTION_HIGH_NEFF = 2;
const BRUSH_CONTRIBUTION_MIN_HIGH_NEFF_FRACTION = 0.02;
const BRUSH_LOCAL_COLOR_FLOW_STRENGTH = 0.015;
const BRUSH_STROKE_PERSISTENCE_DIRECTIONAL_RATIO = 1.8;
const BRUSH_STROKE_PERSISTENCE_RIBBON_MIN_RATIO = 2.2;
const BRUSH_STROKE_PERSISTENCE_ACCENT_MIN_RATIO = 2.8;
const BRUSH_STROKE_PERSISTENCE_PROXIMAL_RATE = 0.01;
const DEFAULT_COLOR_FINISH_START_PERCENT = 43;
const MIN_COLOR_FINISH_START_PERCENT = 0;
const MAX_COLOR_FINISH_START_PERCENT = 100;
const DEFAULT_LAYERED_BRUSH_TAPER = 1;
const DEFAULT_LAYERED_BRUSH_TAPER_LR = 0.01;
const MIN_LEARNED_PAINT_OPACITY = 0.005;
const MAX_LEARNED_PAINT_OPACITY = 0.995;
const LAYERED_OPAQUE_BRUSH_LAYER_MOVE_RADIUS = 1;
const OPAQUE_PAINT_DETAIL_RECOVERY_INTERVAL = 250;
const OPAQUE_PAINT_LATE_SETTLE_FRACTION = 0.10;
const MAX_OPAQUE_PAINT_LATE_SETTLE_FRACTION = 0.20;
const OPAQUE_PAINT_VISIBILITY_GRACE_STEPS = 64;
const OPAQUE_PAINT_VISIBILITY_MIN_GAP_STEPS = 8;
const OPAQUE_PAINT_HARD_ZERO_EPSILON = 1e-7;
const OPAQUE_PAINT_HARD_ZERO_MAX_FRACTION = 0.1;
// Current Contribution Compaction v2 only removes splats which are absent from
// the current forward pass, plus a user-capped one-pixel / near-zero cohort.
// The run keeps a minimum active population even when the user explicitly
// selects a broad removal cap.
const CURRENT_CONTRIBUTION_NEAR_ZERO_MAX_FRACTION = 0.15;
const CURRENT_CONTRIBUTION_MAX_FRACTION = 1;
const CURRENT_CONTRIBUTION_NEAR_ZERO_MAX_COVERAGE = 1;
const CURRENT_CONTRIBUTION_NEAR_ZERO_MAX_INFLUENCE = 0.01;
const CURRENT_CONTRIBUTION_COMPACTION_FRACTION = 0.50;
const CURRENT_CONTRIBUTION_COMPACTION_INTERVAL = 500;
const CURRENT_CONTRIBUTION_MAX_INTERVAL = 100_000;
const CURRENT_CONTRIBUTION_MIN_COMPACTION_FRACTION = 0;
const CURRENT_CONTRIBUTION_MAX_COMPACTION_FRACTION = 1;
const CURRENT_CONTRIBUTION_MAX_WINDOW_STEPS = 1_000_000_000;
const MAX_FINAL_DIAGNOSTIC_SAMPLES = 16384;
const MAX_THIN_LINE_DIAGNOSTIC_SAMPLES = 8192;
