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

// Packed atomic entry
// Word 1: Key (Full u64 hash)
// Word 2: Data packed as:
//   - Score: i16 (bits 0-15) - rebased to u16
//   - Depth: u8  (bits 16-23)
//   - Flag:  u8  (bits 24-25)
//   - BestMove: u16 (bits 26-41) (0xFFFF means None)
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
        // Each entry is 16 bytes.
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

        // Unpack
        let score_u16 = (data & 0xFFFF) as u16;
        let score = (score_u16 as i32) - 10000; // Offset back

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

        // Pack
        // Score: -10000 to 10000. Add 10000 to make it u16 compatible (0 to 20000)
        let score_rebased = (score + 10000).clamp(0, 65535) as u64;
        let depth_bits = (depth as u64) << 16;
        let flag_bits = (flag as u64) << 24;
        let move_bits = match best_move {
            Some(m) => (m as u64) << 26,
            None => 0xFFFF << 26,
        };

        let new_data = score_rebased | depth_bits | flag_bits | move_bits;

        // Simple replacement policy: Always replace if depth is greater or equal
        // Or if the slot is empty (key mismatch implicitly handled by overwrite)

        // For strict correctness in a race, we might want to check the existing depth,
        // but for a game engine concurrent access, "racy" overwrite is often acceptable and faster.
        // We will do a relaxed load to check depth to avoid thrashing valuable deep nodes with shallow ones.

        let existing_data = entry.data.load(Ordering::Relaxed);
        let existing_depth = ((existing_data >> 16) & 0xFF) as u8;

        let existing_key = entry.key.load(Ordering::Relaxed);

        if existing_key != hash || depth >= existing_depth {
            entry.key.store(hash, Ordering::Relaxed);
            entry.data.store(new_data, Ordering::Relaxed);
        }
    }
}
