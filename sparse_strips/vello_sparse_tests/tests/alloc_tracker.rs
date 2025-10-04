// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tracks allocations and deallocations.

use core::cell::Cell;
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

thread_local! {
    static TRACKING_ENABLED: Cell<bool> = Cell::new(false);
}

unsafe impl GlobalAlloc for &AllocationTracker {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING_ENABLED.get() {
            self.allocations.fetch_add(1, Ordering::Relaxed);
            self.bytes_allocated
                .fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if TRACKING_ENABLED.get() {
            self.deallocations.fetch_add(1, Ordering::Relaxed);
            self.bytes_deallocated
                .fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if TRACKING_ENABLED.get() {
            self.allocations.fetch_add(1, Ordering::Relaxed);
            self.bytes_allocated
                .fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACKING_ENABLED.get() {
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
        }

        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq)]
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

    fn approx_eq(
        &self,
        other: &AllocationStats,
        count_threshold: usize,
        bytes_threshold: usize,
    ) -> bool {
        let count_threshold = count_threshold as isize;
        let bytes_threshold = bytes_threshold as isize;
        (self.allocations as isize - other.allocations as isize).abs() <= count_threshold
            && (self.deallocations as isize - other.deallocations as isize).abs() <= count_threshold
            && (self.reallocations as isize - other.reallocations as isize).abs() <= count_threshold
            && (self.bytes_allocated as isize - other.bytes_allocated as isize).abs()
                <= bytes_threshold
            && (self.bytes_deallocated as isize - other.bytes_deallocated as isize).abs()
                <= bytes_threshold
            && (self.bytes_reallocated as isize - other.bytes_reallocated as isize).abs()
                <= bytes_threshold
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
        TRACKING_ENABLED.set(true);
        Self {
            allocator,
            initial_stats: allocator.into(),
        }
    }

    pub(crate) fn end(self) -> AllocationStats {
        TRACKING_ENABLED.set(false);
        let stats: AllocationStats = self.allocator.into();
        stats.sub(&self.initial_stats)
    }
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub(crate) struct AllocsFile {
    pub units: String,
    pub tests: Vec<AllocsTest>,
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub(crate) struct AllocsTest {
    pub test: String,
    pub cpu: AllocInstance,
    pub metal: AllocInstance,
    pub dx12: AllocInstance,
    pub vulkan: AllocInstance,
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub(crate) struct AllocInstance {
    pub cold: AllocationStats,
    pub warm: AllocationStats,
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn process_alloc_stats(
    alloc_stats: AllocationStats,
    file_path: &str,
    instance_name: &str,
) {
    // Do nothing on wasm32.
}

pub(crate) enum RunType {
    Cold,
    Warm,
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn should_process_allocs() -> bool {
    false
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn should_process_allocs() -> bool {
    let save_allocs = std::env::var("SAVE_ALLOCS").map_or(false, |v| v == "1");
    let test_allocs = std::env::var("TEST_ALLOCS").map_or(false, |v| v == "1");
    if !save_allocs && !test_allocs {
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Backend {
    Cpu,
    Vulkan,
    Metal,
    Dx12,
}

impl Into<Backend> for wgpu::Backend {
    fn into(self) -> Backend {
        match self {
            wgpu::Backend::Vulkan => Backend::Vulkan,
            wgpu::Backend::Metal => Backend::Metal,
            wgpu::Backend::Dx12 => Backend::Dx12,
            _ => unimplemented!("Backend {:?} not supported", self),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn process_alloc_stats(
    run_type: RunType,
    alloc_stats: AllocationStats,
    file_path: &str,
    backend: Backend,
) {
    use std::io::Write;

    if !should_process_allocs() {
        return;
    }

    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../vello_sparse_tests/snapshots");
    if !dir.exists() {
        let _ = std::fs::create_dir_all(&dir);
    }
    let summary_path = dir.join("allocs.toml");
    let test_name = file_path;

    let mut summary: AllocsFile = match std::fs::read_to_string(&summary_path) {
        Ok(s) if !s.is_empty() => toml::from_str(&s).unwrap_or_default(),
        _ => AllocsFile::default(),
    };
    if summary.units.is_empty() {
        summary.units = "bytes".to_string();
    }
    // Find or create the test entry
    let test_entry: &mut AllocsTest =
        if let Some(entry) = summary.tests.iter_mut().find(|t| t.test == test_name) {
            entry
        } else {
            summary.tests.push(AllocsTest {
                test: test_name.to_string(),
                ..Default::default()
            });
            summary.tests.last_mut().unwrap()
        };

    let is_test_mode = std::env::var("TEST_ALLOCS").map_or(false, |v| v == "1");
    if is_test_mode {
        // Load consolidated summary and look up expected stats for this test/instance
        let summary_path = dir.join("allocs.toml");
        let summary: AllocsFile = match std::fs::read_to_string(&summary_path) {
            Ok(s) if !s.is_empty() => toml::from_str(&s).unwrap_or_default(),
            _ => AllocsFile::default(),
        };

        let backend_stats = match backend {
            Backend::Cpu => &test_entry.cpu,
            Backend::Vulkan => &test_entry.vulkan,
            Backend::Metal => &test_entry.metal,
            Backend::Dx12 => &test_entry.dx12,
        };

        let expected_stats = match run_type {
            RunType::Cold => &backend_stats.cold,
            RunType::Warm => &backend_stats.warm,
        };

        assert!(
            alloc_stats.approx_eq(expected_stats, 0, 100),
            "{alloc_stats:?} should be approximately equal to {expected_stats:?}"
        );
        return;
    }

    let backend_stats = match backend {
        Backend::Cpu => &mut test_entry.cpu,
        Backend::Vulkan => &mut test_entry.vulkan,
        Backend::Metal => &mut test_entry.metal,
        Backend::Dx12 => &mut test_entry.dx12,
    };

    let stats = match run_type {
        RunType::Cold => &mut backend_stats.cold,
        RunType::Warm => &mut backend_stats.warm,
    };

    stats.allocations = alloc_stats.allocations;
    stats.deallocations = alloc_stats.deallocations;
    stats.reallocations = alloc_stats.reallocations;
    stats.bytes_allocated = alloc_stats.bytes_allocated;
    stats.bytes_deallocated = alloc_stats.bytes_deallocated;
    stats.bytes_reallocated = alloc_stats.bytes_reallocated;

    summary.tests.sort_by(|a, b| a.test.cmp(&b.test));

    let out = toml::to_string_pretty(&summary).unwrap();

    let tmp_path = summary_path.with_extension("allocs.toml.tmp");
    let mut tmp = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tmp_path)
        .unwrap();
    let _ = tmp.write_all(out.as_bytes());
    let _ = std::fs::rename(&tmp_path, &summary_path);
}
