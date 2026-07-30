pub mod engine;
pub mod kernels;
pub mod norm;
pub mod pool;
pub mod post;
pub mod recommend;
pub mod refimpl;
pub mod server;
pub mod simd;
pub mod weights;

pub const CORPUS: usize = 6000;
pub const IN_DIM: usize = CORPUS * 2;
pub const HIDDEN: usize = 2048;
pub const BOTTLENECK: usize = 512;
pub const DEC_MID: usize = HIDDEN / 2;
pub const DEFAULT_LOGIT_WEIGHT: f32 = 0.3;
