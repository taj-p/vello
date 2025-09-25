use core::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::{GlobalAlloc, Layout, System};

use serde::{Deserialize, Serialize};

pub static ALLOCATION_TRACKER: AllocationTracker = AllocationTracker {
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
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.deallocations.fetch_add(1, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        self.allocations.fetch_add(1, Ordering::Relaxed);
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

#[derive(Debug, Default, Clone, Copy)]
pub struct AllocationStats {
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

pub struct AllocationSpan<'a> {
    allocator: &'a AllocationTracker,
    initial_stats: AllocationStats,
}

impl<'a> AllocationSpan<'a> {
    pub fn new(allocator: &'a AllocationTracker) -> Self {
        Self {
            allocator,
            initial_stats: allocator.into(),
        }
    }

    pub fn end(self) -> AllocationStats {
        let stats: AllocationStats = self.allocator.into();
        stats.sub(&self.initial_stats)
    }
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct AllocFile {
    pub test: String,
    pub units: String,
    pub instances: Vec<AllocInstance>,
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct AllocInstance {
    pub name: String,
    pub allocations: usize,
    pub deallocations: usize,
    pub reallocations: usize,
    pub bytes_allocated: usize,
    pub bytes_deallocated: usize,
    pub bytes_reallocated: usize,
    pub net_bytes: usize,
}

pub fn save_alloc_stats(alloc_stats: AllocationStats, file_path: &str, instance_name: &str) {
    #[cfg(not(target_arch = "wasm32"))]
    {
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

        let net_bytes = alloc_stats
            .bytes_allocated
            .saturating_sub(alloc_stats.bytes_deallocated);
        if let Some(inst) = file_data
            .instances
            .iter_mut()
            .find(|i| i.name == instance_name)
        {
            inst.allocations = alloc_stats.allocations;
            inst.deallocations = alloc_stats.deallocations;
            inst.reallocations = alloc_stats.reallocations;
            inst.bytes_allocated = alloc_stats.bytes_allocated;
            inst.bytes_deallocated = alloc_stats.bytes_deallocated;
            inst.bytes_reallocated = alloc_stats.bytes_reallocated;
            inst.net_bytes = net_bytes;
        } else {
            file_data.instances.push(AllocInstance {
                name: instance_name.to_string(),
                allocations: alloc_stats.allocations,
                deallocations: alloc_stats.deallocations,
                reallocations: alloc_stats.reallocations,
                bytes_allocated: alloc_stats.bytes_allocated,
                bytes_deallocated: alloc_stats.bytes_deallocated,
                bytes_reallocated: alloc_stats.bytes_reallocated,
                net_bytes,
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
}
