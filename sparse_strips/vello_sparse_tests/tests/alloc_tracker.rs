// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tracks allocations and deallocations.

use core::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::{GlobalAlloc, Layout, System};

use serde::{Deserialize, Serialize};

pub(crate) static ALLOCATION_TRACKER: AllocationTracker = AllocationTracker {
    allocations: AtomicUsize::new(0),
    deallocations: AtomicUsize::new(0),
    reallocations: AtomicUsize::new(0),
    bytes_allocated: AtomicUsize::new(0),
    bytes_deallocated: AtomicUsize::new(0),
    bytes_reallocated: AtomicUsize::new(0),
};

#[global_allocator]
static GLOBAL: &AllocationTracker = &ALLOCATION_TRACKER;

#[derive(Default)]
pub(crate) struct AllocationTracker {
    allocations: AtomicUsize,
    deallocations: AtomicUsize,
    reallocations: AtomicUsize,
    bytes_allocated: AtomicUsize,
    bytes_deallocated: AtomicUsize,
    bytes_reallocated: AtomicUsize,
}

unsafe impl GlobalAlloc for &AllocationTracker {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        self.bytes_allocated
            .fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.deallocations.fetch_add(1, Ordering::Relaxed);
        self.bytes_deallocated
            .fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        self.bytes_allocated
            .fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        self.reallocations.fetch_add(1, Ordering::Relaxed);

        if new_size > layout.size() {
            self.bytes_allocated
                .fetch_add(new_size - layout.size(), Ordering::Relaxed);
        } else {
            self.bytes_deallocated
                .fetch_add(layout.size() - new_size, Ordering::Relaxed);
        }
        self.bytes_reallocated
            .fetch_add(new_size - layout.size(), Ordering::Relaxed);

        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct AllocationStats {
    pub allocations: usize,
    pub deallocations: usize,
    pub reallocations: usize,
    pub bytes_allocated: usize,
    pub bytes_deallocated: usize,
    pub bytes_reallocated: usize,
}

impl AllocationStats {
    fn sub(&self, other: &AllocationStats) -> AllocationStats {
        AllocationStats {
            allocations: self.allocations - other.allocations,
            deallocations: self.deallocations - other.deallocations,
            reallocations: self.reallocations - other.reallocations,
            bytes_allocated: self.bytes_allocated - other.bytes_allocated,
            bytes_deallocated: self.bytes_deallocated - other.bytes_deallocated,
            bytes_reallocated: self.bytes_reallocated - other.bytes_reallocated,
        }
    }
}

impl Into<AllocationStats> for &AllocationTracker {
    fn into(self) -> AllocationStats {
        AllocationStats {
            allocations: self.allocations.load(Ordering::Relaxed),
            deallocations: self.deallocations.load(Ordering::Relaxed),
            reallocations: self.reallocations.load(Ordering::Relaxed),
            bytes_allocated: self.bytes_allocated.load(Ordering::Relaxed),
            bytes_deallocated: self.bytes_deallocated.load(Ordering::Relaxed),
            bytes_reallocated: self.bytes_reallocated.load(Ordering::Relaxed),
        }
    }
}

pub(crate) struct AllocationSpan<'a> {
    allocator: &'a AllocationTracker,
    initial_stats: AllocationStats,
}

impl<'a> AllocationSpan<'a> {
    pub(crate) fn new(allocator: &'a AllocationTracker) -> Self {
        Self {
            allocator,
            initial_stats: allocator.into(),
        }
    }

    pub(crate) fn end(self) -> AllocationStats {
        let stats: AllocationStats = self.allocator.into();
        stats.sub(&self.initial_stats)
    }
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub(crate) struct AllocFile {
    pub test: String,
    pub units: String,
    pub instances: Vec<AllocInstance>,
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub(crate) struct AllocInstance {
    pub name: String,
    pub cold: AllocationStats,
    pub warm: AllocationStats,
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn save_alloc_stats(alloc_stats: AllocationStats, file_path: &str, instance_name: &str) {
    // Do nothing on wasm32.
}

pub(crate) enum RunType {
    Cold,
    Warm,
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn save_allocs_requested() -> bool {
    false
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn save_allocs_requested() -> bool {
    let save_allocs = std::env::var("SAVE_ALLOCS").map_or(false, |v| v == "1");
    if !save_allocs {
        return false;
    }
    assert!(
        !cfg!(debug_assertions),
        "Saving allocation stats requires --release",
    );
    let args: Vec<String> = std::env::args().collect();
    let mut test_threads_ok = false;
    for i in 0..args.len() {
        if args[i] == "--test-threads=1" {
            test_threads_ok = true;
            break;
        }
        if args[i] == "--test-threads" {
            if let Some(next) = args.get(i + 1) {
                if next == "1" {
                    test_threads_ok = true;
                    break;
                }
            }
        }
    }
    test_threads_ok |= std::env::var("RUST_TEST_THREADS").map_or(false, |v| v == "1");
    assert!(
        test_threads_ok,
        "Saving allocation stats requires running with -- --test-threads=1",
    );
    return true;
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn save_alloc_stats(
    run_type: RunType,
    alloc_stats: AllocationStats,
    file_path: &str,
    instance_name: &str,
) {
    use std::io::Write;
    use std::thread;
    use std::time::Duration;

    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../vello_sparse_tests/snapshots");
    if !dir.exists() {
        let _ = std::fs::create_dir_all(&dir);
    }
    let test_name = file_path;
    let file_path = dir.join(format!("{}.allocs.toml", file_path));
    let lock_path = dir.join(".allocs.lock");

    loop {
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
        {
            Ok(_) => break,
            Err(e) => {
                if e.kind() == std::io::ErrorKind::AlreadyExists {
                    thread::sleep(Duration::from_millis(10));
                    continue;
                } else {
                    break;
                }
            }
        }
    }

    let mut file_data: AllocFile = match std::fs::read_to_string(&file_path) {
        Ok(s) if !s.is_empty() => toml::from_str(&s).unwrap_or_default(),
        _ => AllocFile::default(),
    };
    file_data.test = test_name.to_string();
    file_data.units = "bytes".to_string();

    if let Some(inst) = file_data
        .instances
        .iter_mut()
        .find(|i| i.name == instance_name)
    {
        let stats = match run_type {
            RunType::Cold => &mut inst.cold,
            RunType::Warm => &mut inst.warm,
        };

        stats.allocations = alloc_stats.allocations;
        stats.deallocations = alloc_stats.deallocations;
        stats.reallocations = alloc_stats.reallocations;
        stats.bytes_allocated = alloc_stats.bytes_allocated;
        stats.bytes_deallocated = alloc_stats.bytes_deallocated;
        stats.bytes_reallocated = alloc_stats.bytes_reallocated;
    } else {
        let (cold_stats, warm_stats) = match run_type {
            RunType::Cold => (alloc_stats, AllocationStats::default()),
            RunType::Warm => (AllocationStats::default(), alloc_stats),
        };

        file_data.instances.push(AllocInstance {
            name: instance_name.to_string(),
            cold: cold_stats,
            warm: warm_stats,
        });
    }
    file_data.instances.sort_by(|a, b| a.name.cmp(&b.name));

    let out = toml::to_string_pretty(&file_data).unwrap();

    let tmp_path = file_path.with_extension("allocs.toml.tmp");
    let mut tmp = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tmp_path)
        .unwrap();
    let _ = tmp.write_all(out.as_bytes());
    let _ = std::fs::rename(&tmp_path, &file_path);

    let _ = std::fs::remove_file(&lock_path);
}
