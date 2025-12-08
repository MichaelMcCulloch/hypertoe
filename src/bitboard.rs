// src/bitboard.rs

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BitBoard {
    Small(u32),
    Medium(u128),
    Large(Vec<u64>),
}

#[derive(Clone, Debug)]
pub enum WinningMasks {
    Small {
        masks: Vec<u32>,
        map: Vec<Vec<usize>>,
    },
    Medium {
        masks: Vec<u128>,
        map: Vec<Vec<usize>>,
    },
    Large {
        masks: Vec<Vec<u64>>,
        map: Vec<Vec<usize>>,
    },
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
                if vec_idx < v.len() {
                    v[vec_idx] |= 1 << (index % 64);
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
                if vec_idx < v.len() {
                    v[vec_idx] &= !(1 << (index % 64));
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
                if let Some(chunk) = v.get(vec_idx) {
                    (*chunk & (1 << (index % 64))) != 0
                } else {
                    false
                }
            }
        }
    }

    pub fn check_win(&self, winning_masks: &WinningMasks) -> bool {
        match (self, winning_masks) {
            (BitBoard::Small(board), WinningMasks::Small { masks, .. }) => {
                #[cfg(target_arch = "x86_64")]
                if is_x86_feature_detected!("avx2") {
                    return unsafe { check_win_u32_avx2(*board, masks) };
                }
                masks.iter().any(|&m| (board & m) == m)
            }
            (BitBoard::Medium(board), WinningMasks::Medium { masks, .. }) => {
                #[cfg(target_arch = "x86_64")]
                if is_x86_feature_detected!("avx2") {
                    return unsafe { check_win_u128_avx2(*board, masks) };
                }
                masks.iter().any(|&m| (board & m) == m)
            }
            (BitBoard::Large(board), WinningMasks::Large { masks, .. }) => {
                masks.iter().any(|mask_chunks| {
                    board.len() == mask_chunks.len()
                        && board
                            .iter()
                            .zip(mask_chunks.iter())
                            .all(|(b, m)| (*b & *m) == *m)
                })
            }
            _ => false,
        }
    }

    pub fn check_win_at(&self, winning_masks: &WinningMasks, index: usize) -> bool {
        match (self, winning_masks) {
            (BitBoard::Small(board), WinningMasks::Small { masks, map }) => {
                if let Some(indices) = map.get(index) {
                    for &i in indices {
                        let m = masks[i];
                        if (board & m) == m {
                            return true;
                        }
                    }
                }
                false
            }
            (BitBoard::Medium(board), WinningMasks::Medium { masks, map }) => {
                if let Some(indices) = map.get(index) {
                    for &i in indices {
                        let m = masks[i];
                        if (board & m) == m {
                            return true;
                        }
                    }
                }
                false
            }
            (BitBoard::Large(board), WinningMasks::Large { masks, map }) => {
                if let Some(indices) = map.get(index) {
                    for &i in indices {
                        let mask_chunks = &masks[i];
                        if board
                            .iter()
                            .zip(mask_chunks.iter())
                            .all(|(b, m)| (*b & *m) == *m)
                        {
                            return true;
                        }
                    }
                }
                false
            }
            _ => false,
        }
    }
}

// --- SIMD Implementations ---

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn check_win_u32_avx2(board: u32, masks: &[u32]) -> bool {
    let board_vec = unsafe { _mm256_set1_epi32(board as i32) }; // Wrapping unsafe op

    let chunks = masks.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        // FIX: Wrapped SIMD calls in unsafe block
        unsafe {
            let mask_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            let and_res = _mm256_and_si256(board_vec, mask_vec);
            let cmp = _mm256_cmpeq_epi32(and_res, mask_vec);
            if _mm256_movemask_epi8(cmp) != 0 {
                return true;
            }
        }
    }

    for &m in remainder {
        if (board & m) == m {
            return true;
        }
    }

    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn check_win_u128_avx2(board: u128, masks: &[u128]) -> bool {
    let board_low = board as u64;
    let board_high = (board >> 64) as u64;

    // FIX: Wrapped unsafe op
    let board_vec = unsafe {
        _mm256_set_epi64x(
            board_high as i64,
            board_low as i64,
            board_high as i64,
            board_low as i64,
        )
    };

    let chunks = masks.chunks_exact(2);
    let remainder = chunks.remainder();

    for chunk in chunks {
        // FIX: Wrapped SIMD calls in unsafe block
        unsafe {
            let mask_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            let and_res = _mm256_and_si256(board_vec, mask_vec);
            let cmp = _mm256_cmpeq_epi64(and_res, mask_vec);

            let mask_bits = _mm256_movemask_epi8(cmp);

            if (mask_bits & 0xFFFF) == 0xFFFF {
                return true;
            }
            let mb_u32 = mask_bits as u32;
            if (mb_u32 & 0xFFFF0000) == 0xFFFF0000 {
                return true;
            }
        }
    }

    for &m in remainder {
        if (board & m) == m {
            return true;
        }
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
