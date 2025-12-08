use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Flag {
    Exact = 0,
    LowerBound = 1,
    UpperBound = 2,
    None = 3,
}

impl Flag {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Flag::Exact,
            1 => Flag::LowerBound,
            2 => Flag::UpperBound,
            _ => Flag::None,
        }
    }
}

#[derive(Default)]
struct AtomicTranspositionEntry {
    key: AtomicU64,
    data: AtomicU64,
}

pub struct LockFreeTT {
    entries: Vec<AtomicTranspositionEntry>,
    mask: usize,
}

impl LockFreeTT {
    pub fn new(size_mb: usize) -> Self {
        let num_entries = (size_mb * 1024 * 1024) / 16;
        let size = num_entries.next_power_of_two();

        let mut entries = Vec::with_capacity(size);
        for _ in 0..size {
            entries.push(AtomicTranspositionEntry::default());
        }

        Self {
            entries,
            mask: size - 1,
        }
    }

    pub fn get(&self, hash: u64) -> Option<(i32, u8, Flag, Option<usize>)> {
        let idx = (hash as usize) & self.mask;
        let entry = &self.entries[idx];

        let entry_key = entry.key.load(Ordering::Relaxed);
        if entry_key != hash {
            return None;
        }

        let data = entry.data.load(Ordering::Relaxed);

        let score_u16 = (data & 0xFFFF) as u16;
        let score = (score_u16 as i32) - 10000;

        let depth = ((data >> 16) & 0xFF) as u8;
        let flag_u8 = ((data >> 24) & 0x3) as u8;
        let best_move_u16 = ((data >> 26) & 0xFFFF) as u16;

        let best_move = if best_move_u16 == 0xFFFF {
            None
        } else {
            Some(best_move_u16 as usize)
        };

        Some((score, depth, Flag::from_u8(flag_u8), best_move))
    }

    pub fn store(&self, hash: u64, score: i32, depth: u8, flag: Flag, best_move: Option<usize>) {
        let idx = (hash as usize) & self.mask;
        let entry = &self.entries[idx];

        let score_rebased = (score + 10000).clamp(0, 65535) as u64;
        let depth_bits = (depth as u64) << 16;
        let flag_bits = (flag as u64) << 24;
        let move_bits = match best_move {
            Some(m) => (m as u64) << 26,
            None => 0xFFFF << 26,
        };

        let new_data = score_rebased | depth_bits | flag_bits | move_bits;

        let existing_data = entry.data.load(Ordering::Relaxed);
        let existing_depth = ((existing_data >> 16) & 0xFF) as u8;

        let existing_key = entry.key.load(Ordering::Relaxed);

        if existing_key != hash || depth >= existing_depth {
            entry.key.store(hash, Ordering::Relaxed);
            entry.data.store(new_data, Ordering::Relaxed);
        }
    }
}
