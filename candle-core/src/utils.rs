//! Useful functions for checking features.
use std::str::FromStr;

/// Extension trait that backports the nightly-only `is_multiple_of` helper
/// so we can keep using stable Rust compilers.
pub trait IsMultipleOf: Sized + Copy + PartialEq + std::ops::Rem<Output = Self> + From<u8> {
    #[inline]
    fn is_multiple_of(self, other: Self) -> bool {
        if other == Self::from(0) {
            return false;
        }
        self % other == Self::from(0)
    }
}

macro_rules! impl_is_multiple_of {
    ($($ty:ty),* $(,)?) => {
        $(impl IsMultipleOf for $ty {})*
    };
}

impl_is_multiple_of!(usize, u128, u64, u32, u16, u8);

pub fn get_num_threads() -> usize {
    // Respond to the same environment variable as rayon.
    match std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| usize::from_str(&s).ok())
    {
        Some(x) if x > 0 => x,
        Some(_) | None => num_cpus::get(),
    }
}

pub fn has_accelerate() -> bool {
    cfg!(feature = "accelerate")
}

pub fn has_mkl() -> bool {
    cfg!(feature = "mkl")
}

pub fn cuda_is_available() -> bool {
    cfg!(feature = "cuda")
}

pub fn metal_is_available() -> bool {
    cfg!(feature = "metal")
}

pub fn with_avx() -> bool {
    cfg!(target_feature = "avx2")
}

pub fn with_neon() -> bool {
    cfg!(target_feature = "neon")
}

pub fn with_simd128() -> bool {
    cfg!(target_feature = "simd128")
}

pub fn with_f16c() -> bool {
    cfg!(target_feature = "f16c")
}
