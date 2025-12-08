
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::fmt;

/// Represents the game board using bit-packing for efficiency.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BitBoard {
    Small(u32),       // For N=3 (27 bits)
    Medium(u128),     // For N=4 (81 bits)
    Large(Vec<u64>),  // For N>=5 (variable size)
}

/// Holds the pre-calculated winning masks in a format optimized for the specific board size.
#[derive(Clone, Debug)]
pub enum WinningMasks {
    Small(Vec<u32>),
    Medium(Vec<u128>),
    Large(Vec<Vec<u64>>),
}

impl BitBoard {
    pub fn new(dimension: usize, side: usize) -> Self {
        let total_cells = side.pow(dimension as u32);
        if total_cells <= 32 {
            BitBoard::Small(0)
        } else if total_cells <= 128 {
            BitBoard::Medium(0)
        } else {
            let num_u64s = (total_cells + 63) / 64;
            BitBoard::Large(vec![0; num_u64s])
        }
    }

    pub fn set_bit(&mut self, index: usize) {
        match self {
            BitBoard::Small(b) => *b |= 1 << index,
            BitBoard::Medium(b) => *b |= 1 << index,
            BitBoard::Large(v) => {
                let vec_idx = index / 64;
                let bit_idx = index % 64;
                if vec_idx < v.len() {
                    v[vec_idx] |= 1 << bit_idx;
                }
            }
        }
    }

    pub fn clear_bit(&mut self, index: usize) {
        match self {
            BitBoard::Small(b) => *b &= !(1 << index),
            BitBoard::Medium(b) => *b &= !(1 << index),
            BitBoard::Large(v) => {
                let vec_idx = index / 64;
                let bit_idx = index % 64;
                if vec_idx < v.len() {
                    v[vec_idx] &= !(1 << bit_idx);
                }
            }
        }
    }

    pub fn get_bit(&self, index: usize) -> bool {
        match self {
            BitBoard::Small(b) => (*b & (1 << index)) != 0,
            BitBoard::Medium(b) => (*b & (1 << index)) != 0,
            BitBoard::Large(v) => {
                let vec_idx = index / 64;
                let bit_idx = index % 64;
                if let Some(chunk) = v.get(vec_idx) {
                    (*chunk & (1 << bit_idx)) != 0
                } else {
                    false
                }
            }
        }
    }

    pub fn check_win(&self, masks: &WinningMasks) -> bool {
        match (self, masks) {
            (BitBoard::Small(board_val), WinningMasks::Small(masks_vec)) => {
                #[cfg(target_arch = "x86_64")]
                if is_x86_feature_detected!("avx2") {
                   return unsafe { check_win_u32_avx2(*board_val, masks_vec) };
                }
                // Scalar fallback
                masks_vec.iter().any(|&m| (board_val & m) == m)
            }
            (BitBoard::Medium(board_val), WinningMasks::Medium(masks_vec)) => {
                // N=4 (81 bits) fits in u128. AVX2 deals with 256 bits (two u128s).
                #[cfg(target_arch = "x86_64")]
                if is_x86_feature_detected!("avx2") {
                    return unsafe { check_win_u128_avx2(*board_val, masks_vec) };
                }
                masks_vec.iter().any(|&m| (board_val & m) == m)
            }
            (BitBoard::Large(board_vec), WinningMasks::Large(masks_vec)) => {
                // Scalar check for large boards
                masks_vec.iter().any(|mask_chunks| {
                     if board_vec.len() != mask_chunks.len() { return false; }
                     board_vec.iter().zip(mask_chunks.iter()).all(|(b, m)| (*b & *m) == *m)
                })
            }
            _ => false, // Mismatched types
        }
    }
}

// --- SIMD Implementations ---

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn check_win_u32_avx2(board: u32, masks: &[u32]) -> bool {
    let board_vec = _mm256_set1_epi32(board as i32);
    
    // Process 8 masks at a time
    let chunks = masks.chunks_exact(8);
    let remainder = chunks.remainder();
    
    for chunk in chunks {
        unsafe {
            let mask_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            // (board & mask)
            let and_res = _mm256_and_si256(board_vec, mask_vec);
            // (board & mask) == mask
            let cmp = _mm256_cmpeq_epi32(and_res, mask_vec);
            // Check if any element equality was true (0xFFFFFFFF)
            if _mm256_movemask_epi8(cmp) != 0 {
                return true;
            }
        }
    }
    
    // Fallback for remainder
    for &m in remainder {
        if (board & m) == m { return true; }
    }
    
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn check_win_u128_avx2(board: u128, masks: &[u128]) -> bool {
    // 128-bit integers are a bit tricky in AVX2 which uses 256-bit registers (checking 2 masks at once).
    // We can cast u128 array to __m128i pointer, or load 2 u128s into __m256i.
    
    // _mm256_set1_epi64x is available but not set1_epi128. 
    // We can construct the board vector by broadcasting. 
    let board_low = board as u64;
    let board_high = (board >> 64) as u64;
    
    // Set 256 bit vector as [low, high, low, high] (two copies of the board)
    let board_vec = _mm256_set_epi64x(
        board_high as i64, board_low as i64,
        board_high as i64, board_low as i64 
    );
    
    let chunks = masks.chunks_exact(2);
    let remainder = chunks.remainder();
    
    for chunk in chunks {
        // chunk contains 2 u128s.
        // Load as 256-bit
        unsafe {
            let mask_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            
            let and_res = _mm256_and_si256(board_vec, mask_vec);
            // AVX2 has compare for 8, 16, 32, 64 bits. Not 128.
            // We use cmpeq_epi64.
            let cmp = _mm256_cmpeq_epi64(and_res, mask_vec);
            
            // cmp result is [q3, q2, q1, q0].
            // We want (q1 && q0) || (q3 && q2).
            // movemask gives us 32 bits (1 bit per byte). 
            
            let mask_bits = _mm256_movemask_epi8(cmp);
            
            if (mask_bits & 0xFFFF) == 0xFFFF { return true; }
            let mb_u32 = mask_bits as u32;
            if (mb_u32 & 0xFFFF0000) == 0xFFFF0000 { return true; }
        }
    }
    
    for &m in remainder {
        if (board & m) == m { return true; }
    }
    
    false
}


impl fmt::Display for BitBoard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BitBoard::Small(b) => write!(f, "{:032b}", b),
            BitBoard::Medium(b) => write!(f, "{:0128b}", b),
            BitBoard::Large(v) => write!(f, "{:?}", v),
        }
    }
}
