
// ============================================================================
// Opt-in Lazy GPU-Resident Write-Back (`VICTORY_LAZY_COPYBACK=1`, default OFF)
// ============================================================================
//
// Strict Acyclic Lock Hierarchy (deadlock-free with Windows VEH):
//   Level 0: `host.submitted`, `CARD_WORK`, `BUFFERS`, `LOOSE`, `EXACT`,
//            `CARDED`, `Shadow::{copied, pending, before, kept, card.*}`,
//            `watch::{JOURNAL, SPANS, GUARDS}`.
//   Level 1: `LAZY_TABLE` (metadata-only leaf table: armed/busy pages, pinned
//            fence pages, and flushed 64 KiB baselines). NEVER acquires Level 0
//            or Level 2 while held, and NEVER calls D3D12 or waits on GPU work.
//   Level 2: `LAZY_GPU` (dedicated D3D12 allocator, command list, fence, event,
//            and persistent 64 KiB `D3D12_HEAP_TYPE_READBACK` buffer). NEVER
//            acquires Level 0 or Level 1 while held.
//   Level 3: `FlushWait` (`Mutex<Option<FlushOutcome>>` + `Condvar`). Leaf only.
//
// VEH (`lazy_on_fault`) ONLY acquires Level 1 -> drops -> Level 2 -> drops ->
// `VirtualProtect` + `memcpy` -> Level 1 -> drops -> Level 3. It NEVER touches
// any Level 0 lock (`watch`, `Shadow`, `CARD_WORK`, `host.submitted`).

/// Whether opt-in Lazy GPU-Resident Write-Back is enabled
/// (`VICTORY_LAZY_COPYBACK=1`; default OFF).
pub(crate) fn lazy_copyback() -> bool {
    #[cfg(test)]
    {
        match LAZY_TEST_OVERRIDE.load(Ordering::Acquire) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAZY_COPYBACK").is_some_and(|v| v == "1"))
}

#[cfg(test)]
static LAZY_TEST_OVERRIDE: AtomicU8 = AtomicU8::new(0);

#[cfg(test)]
static LAZY_TEST_FRAME: AtomicU32 = AtomicU32::new(0);

#[cfg(test)]
struct LazyFrameGuard {
    prev: u32,
}

#[cfg(test)]
impl LazyFrameGuard {
    fn set(frame: u32) -> Self {
        let prev = LAZY_TEST_FRAME.swap(frame, Ordering::AcqRel);
        Self { prev }
    }

    fn update(frame: u32) {
        LAZY_TEST_FRAME.store(frame, Ordering::Release);
    }
}

#[cfg(test)]
impl Drop for LazyFrameGuard {
    fn drop(&mut self) {
        LAZY_TEST_FRAME.store(self.prev, Ordering::Release);
    }
}

fn lazy_current_frame() -> u32 {
    #[cfg(test)]
    {
        let f = LAZY_TEST_FRAME.load(Ordering::Acquire);
        if f > 0 {
            return f;
        }
        if LAZY_TEST_OVERRIDE.load(Ordering::Acquire) != 0 {
            return 0;
        }
    }
    let f = u64::from(crate::frame::current());
    if f == 0 {
        0
    } else {
        f.min((u32::MAX - 1) as u64) as u32
    }
}

fn lazy_hot_params() -> (u8, u32) {
    static PARAMS: OnceLock<(u8, u32)> = OnceLock::new();
    *PARAMS.get_or_init(|| {
        let strikes = std::env::var("VICTORY_LAZY_HOT_STRIKES")
            .ok()
            .and_then(|v| v.trim().parse::<u8>().ok())
            .unwrap_or(2)
            .max(1);
        let window = std::env::var("VICTORY_LAZY_HOT_WINDOW")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(60)
            .max(1);
        (strikes, window)
    })
}

#[cfg(test)]
type PostTransitionHook = Box<dyn FnOnce(usize) + Send>;

#[cfg(test)]
static TRY_DEFER_POST_TRANSITION_HOOK: Mutex<Option<PostTransitionHook>> = Mutex::new(None);

#[cfg(test)]
static LAZY_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
pub(crate) struct LazyTestGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    prev: u8,
}

#[cfg(test)]
impl LazyTestGuard {
    fn enable() -> Self {
        let lock = LAZY_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = LAZY_TEST_OVERRIDE.swap(2, Ordering::AcqRel);
        Self { _lock: lock, prev }
    }

    fn disable() -> Self {
        let lock = LAZY_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = LAZY_TEST_OVERRIDE.swap(1, Ordering::AcqRel);
        Self { _lock: lock, prev }
    }
}

#[cfg(test)]
pub(crate) fn test_disable_lazy_copyback() -> LazyTestGuard {
    LazyTestGuard::disable()
}

#[cfg(test)]
impl Drop for LazyTestGuard {
    fn drop(&mut self) {
        LAZY_TEST_OVERRIDE.store(self.prev, Ordering::Release);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FlushOutcome {
    /// Page has been flushed/unprotected (`PAGE_READWRITE`) and is accessible.
    Accessible,
    /// Arming committed (`PAGE_NOACCESS`): waiting faults must retry lookup.
    ArmedRetry,
}

struct FlushWait {
    state: Mutex<Option<FlushOutcome>>,
    ready: Condvar,
}

impl FlushWait {
    fn new() -> Self {
        Self {
            state: Mutex::new(None),
            ready: Condvar::new(),
        }
    }

    fn wait(&self) -> FlushOutcome {
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(outcome) = *guard {
                return outcome;
            }
            guard = self.ready.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn finish(&self, outcome: FlushOutcome) {
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        *guard = Some(outcome);
        self.ready.notify_all();
    }
}

struct ArmedPage {
    shadow: Weak<Shadow>,
    shadow_id: u64,
    offset: usize,
    page_idx: usize,
    /// When the armed page is a single-view `MEM_MAPPED` section (`inplace::Cover::Run`),
    /// stores `(section_handle, section_offset)` so VEH can write all 64 KiB via
    /// a temporary private alias view BEFORE flipping the title view from
    /// `PAGE_NOACCESS` to `PAGE_READWRITE` — eliminating any torn-read window!
    section_view: Option<(usize, u64)>,
}

enum LazyEntry {
    Armed(ArmedPage),
    Busy {
        shadow_id: u64,
        wait: Arc<FlushWait>,
    },
}

/// Per-page baseline flushed from GPU into title memory, paired with the
/// unapplied byte intervals inside `[0, PAGE)` so `lazy_apply_flushed_baseline`
/// applies each byte at most once (idempotent across consecutive relay merges
/// or overlapping ranges) while keeping `LAZY_RESYNC` active until the next
/// `Shadow::update` / `Shadow::writing`.
struct FlushedBaseline {
    bytes: Vec<u8>,
    remaining: Vec<(usize, usize)>,
}

impl FlushedBaseline {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            remaining: vec![(0, PAGE)],
        }
    }

    /// Copies any unapplied subranges intersecting `[src_lo, src_hi)` into `dst`
    /// (where `dst[0]` corresponds to `src_lo`), removing them from `remaining`.
    /// Returns `true` when every byte in `[0, PAGE)` has been applied.
    fn apply_once(&mut self, src_lo: usize, src_hi: usize, dst: &mut [u8]) -> bool {
        let mut next = Vec::with_capacity(self.remaining.len() + 1);
        for &(r_lo, r_hi) in &self.remaining {
            if r_hi <= src_lo || r_lo >= src_hi {
                next.push((r_lo, r_hi));
                continue;
            }
            if r_lo < src_lo {
                next.push((r_lo, src_lo));
            }
            let o_lo = r_lo.max(src_lo);
            let o_hi = r_hi.min(src_hi);
            let dst_off = o_lo - src_lo;
            let n = o_hi - o_lo;
            dst[dst_off..dst_off + n].copy_from_slice(&self.bytes[o_lo..o_hi]);
            if r_hi > src_hi {
                next.push((src_hi, r_hi));
            }
        }
        self.remaining = next;
        self.remaining.is_empty()
    }
}

struct LazyTable {
    pages: BTreeMap<usize, LazyEntry>,
    pinned: BTreeSet<usize>,
    /// Active `watch::title_memory_remapping` page intervals `(id, first_page, last_page)`.
    /// Stored as intervals (O(1) space even for full-address-space lengths).
    remapping: Vec<(u64, usize, usize)>,
    next_remap_id: u64,
    flushed_baselines: BTreeMap<(u64, usize), FlushedBaseline>,
}

impl LazyTable {
    const fn new() -> Self {
        Self {
            pages: BTreeMap::new(),
            pinned: BTreeSet::new(),
            remapping: Vec::new(),
            next_remap_id: 1,
            flushed_baselines: BTreeMap::new(),
        }
    }

    fn is_remapping(&self, page: usize) -> bool {
        self.remapping.iter().any(|&(_, lo, hi)| lo <= page && page < hi)
    }
}

/// Level 1 metadata table for lazy GPU-resident pages.
static LAZY_TABLE: Mutex<LazyTable> = Mutex::new(LazyTable::new());

/// Lock-free count of currently armed or busy lazy pages (0 when inactive).
static LAZY_ARMED_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Lock-free 4096-bucket filter indexing currently armed/busy 64 KiB pages
/// (`page & (LAZY_BUCKETS_LEN - 1)`), plus the last flushed page per bucket.
/// Eliminates `LAZY_TABLE.lock()` and `VirtualQuery` on the 99.9%+ of watch
/// faults, `probe_readable`, and `safecopy` reads that never touch a lazy page.
const LAZY_BUCKETS_LEN: usize = 4096;
const LAZY_BUCKETS_MASK: usize = LAZY_BUCKETS_LEN - 1;
static LAZY_BUCKETS: [AtomicU32; LAZY_BUCKETS_LEN] = [const { AtomicU32::new(0) }; LAZY_BUCKETS_LEN];
static LAZY_LAST_FLUSHED: [AtomicUsize; LAZY_BUCKETS_LEN] = [const { AtomicUsize::new(usize::MAX) }; LAZY_BUCKETS_LEN];

/// Lock-free per-owner page bounding-box summary published via an atomic
/// seqlock (single writer under `LAZY_TABLE`). Eliminates `LAZY_TABLE.lock()`
/// on `lazy_has_other_armed` / `lazy_flush_other_shadows` (~10,000+ calls/frame
/// across `victory-wide` threads in `Shadow::nothing_new`, `Shadow::writing`,
/// and `Shadow::update_free`), including for the sole armed 174 MiB heap itself.
const LAZY_OWNER_SLOTS: usize = 8;
static LAZY_SUMMARY_SEQ: AtomicU64 = AtomicU64::new(0);
static LAZY_SUMMARY_LEN: AtomicUsize = AtomicUsize::new(0);
static LAZY_SUMMARY_ID: [AtomicU64; LAZY_OWNER_SLOTS] = [const { AtomicU64::new(0) }; LAZY_OWNER_SLOTS];
static LAZY_SUMMARY_MIN: [AtomicUsize; LAZY_OWNER_SLOTS] = [const { AtomicUsize::new(0) }; LAZY_OWNER_SLOTS];
static LAZY_SUMMARY_MAX: [AtomicUsize; LAZY_OWNER_SLOTS] = [const { AtomicUsize::new(0) }; LAZY_OWNER_SLOTS];

#[inline]
fn lazy_entry_owner(entry: &LazyEntry) -> u64 {
    match entry {
        LazyEntry::Armed(a) => a.shadow_id,
        LazyEntry::Busy { shadow_id, .. } => *shadow_id,
    }
}

/// Recomputes and publishes the per-owner page bounding-box summary.
/// Caller MUST hold `LAZY_TABLE`.
fn lazy_summary_sync(table: &LazyTable) {
    let mut slots = [(0u64, 0usize, 0usize); LAZY_OWNER_SLOTS];
    let mut count = 0usize;
    let mut overflow = false;
    for (&page, entry) in &table.pages {
        let sid = lazy_entry_owner(entry);
        let end = page.saturating_add(1);
        if let Some(slot) = slots[..count].iter_mut().find(|s| s.0 == sid) {
            if page < slot.1 {
                slot.1 = page;
            }
            if end > slot.2 {
                slot.2 = end;
            }
        } else if count < LAZY_OWNER_SLOTS {
            slots[count] = (sid, page, end);
            count += 1;
        } else {
            overflow = true;
            break;
        }
    }
    let seq = LAZY_SUMMARY_SEQ.load(Ordering::Relaxed);
    LAZY_SUMMARY_SEQ.store(seq.wrapping_add(1), Ordering::Relaxed);
    std::sync::atomic::fence(Ordering::Release);
    if overflow {
        LAZY_SUMMARY_LEN.store(usize::MAX, Ordering::Relaxed);
    } else {
        for (i, &(sid, lo, hi)) in slots[..count].iter().enumerate() {
            LAZY_SUMMARY_ID[i].store(sid, Ordering::Relaxed);
            LAZY_SUMMARY_MIN[i].store(lo, Ordering::Relaxed);
            LAZY_SUMMARY_MAX[i].store(hi, Ordering::Relaxed);
        }
        LAZY_SUMMARY_LEN.store(count, Ordering::Relaxed);
    }
    LAZY_SUMMARY_SEQ.store(seq.wrapping_add(2), Ordering::Release);
}

/// Lock-free check whether any 64 KiB page overlapping `[at, at + len)` has a
/// non-empty bucket in `LAZY_BUCKETS`. Safe to call on hot `safecopy` paths.
#[inline]
pub(crate) fn lazy_has_any_in_range(at: usize, len: usize) -> bool {
    if len == 0 || LAZY_ARMED_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    let first = at / PAGE;
    let last = at.saturating_add(len).div_ceil(PAGE);
    if last.saturating_sub(first) >= LAZY_BUCKETS_LEN {
        return true;
    }
    for page in first..last {
        if LAZY_BUCKETS[page & LAZY_BUCKETS_MASK].load(Ordering::Acquire) != 0 {
            return true;
        }
    }
    false
}

/// Lock-free conservative filter: returns `false` only when guaranteed that NO
/// shadow other than `my_id` currently has an armed or busy lazy page in
/// `[from, to)`. Conservatively returns `true` on concurrent summary update,
/// >`LAZY_OWNER_SLOTS` distinct owners, or bucket hash collision.
#[inline]
fn lazy_may_have_other_in_range(my_id: u64, from: usize, to: usize) -> bool {
    if from >= to || LAZY_ARMED_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    let first = from / PAGE;
    let last = to.div_ceil(PAGE);
    let seq1 = LAZY_SUMMARY_SEQ.load(Ordering::Acquire);
    if seq1 & 1 == 0 {
        let n = LAZY_SUMMARY_LEN.load(Ordering::Relaxed);
        let mut hit = n > LAZY_OWNER_SLOTS;
        if !hit {
            for i in 0..n {
                let sid = LAZY_SUMMARY_ID[i].load(Ordering::Relaxed);
                if sid != my_id {
                    let lo = LAZY_SUMMARY_MIN[i].load(Ordering::Relaxed);
                    let hi = LAZY_SUMMARY_MAX[i].load(Ordering::Relaxed);
                    if lo < last && hi > first {
                        hit = true;
                        break;
                    }
                }
            }
        }
        std::sync::atomic::fence(Ordering::Acquire);
        if LAZY_SUMMARY_SEQ.load(Ordering::Relaxed) == seq1 && !hit {
            return false;
        }
    }
    lazy_has_any_in_range(from, to - from)
}

/// Monotonic mutation counter incremented under `LAZY_TABLE` on every insert,
/// transition, or removal in `LAZY_TABLE.pages`. Used by VEH fault handling to
/// detect concurrent flush/re-arm races without holding `GUARDS` across locks.
static LAZY_TABLE_EPOCH: AtomicU64 = AtomicU64::new(0);

#[inline]
pub(crate) fn lazy_table_epoch() -> u64 {
    LAZY_TABLE_EPOCH.load(Ordering::Acquire)
}

fn lazy_table_insert_busy(table: &mut LazyTable, page: usize, shadow_id: u64, wait: Arc<FlushWait>) {
    let prev = table.pages.insert(page, LazyEntry::Busy { shadow_id, wait });
    let owner_changed = prev.as_ref().is_none_or(|p| lazy_entry_owner(p) != shadow_id);
    if prev.is_none() {
        LAZY_BUCKETS[page & LAZY_BUCKETS_MASK].fetch_add(1, Ordering::AcqRel);
    }
    if owner_changed {
        lazy_summary_sync(table);
    }
    if prev.is_none() {
        LAZY_ARMED_COUNT.fetch_add(1, Ordering::AcqRel);
    }
    LAZY_TABLE_EPOCH.fetch_add(1, Ordering::AcqRel);
}

fn lazy_table_insert_armed(table: &mut LazyTable, page: usize, armed: ArmedPage) {
    let shadow_id = armed.shadow_id;
    let prev = table.pages.insert(page, LazyEntry::Armed(armed));
    let owner_changed = prev.as_ref().is_none_or(|p| lazy_entry_owner(p) != shadow_id);
    if prev.is_none() {
        LAZY_BUCKETS[page & LAZY_BUCKETS_MASK].fetch_add(1, Ordering::AcqRel);
    }
    if owner_changed {
        lazy_summary_sync(table);
    }
    if prev.is_none() {
        LAZY_ARMED_COUNT.fetch_add(1, Ordering::AcqRel);
    }
    LAZY_TABLE_EPOCH.fetch_add(1, Ordering::AcqRel);
}

fn lazy_table_remove_page(table: &mut LazyTable, page: usize) -> Option<LazyEntry> {
    let prev = table.pages.remove(&page);
    if prev.is_some() {
        LAZY_LAST_FLUSHED[page & LAZY_BUCKETS_MASK].store(page, Ordering::Release);
        LAZY_ARMED_COUNT.fetch_sub(1, Ordering::AcqRel);
        LAZY_BUCKETS[page & LAZY_BUCKETS_MASK].fetch_sub(1, Ordering::AcqRel);
        lazy_summary_sync(table);
        LAZY_TABLE_EPOCH.fetch_add(1, Ordering::AcqRel);
    }
    prev
}

/// Monotonic count of completed lazy flushes (for stale-fault disambiguation).
static LAZY_FLUSH_TOTAL: AtomicU64 = AtomicU64::new(0);

/// Counters: `[0]=newly_armed, [1]=steady_state_hits, [2]=veh_read_faults,
/// [3]=veh_write_faults, [4]=explicit_flushes, [5]=bytes_flushed`.
static LAZY_STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Dedicated single-piece GPU readback context (Level 2), pre-allocated before
/// any `PAGE_NOACCESS` page is ever armed so VEH never allocates D3D12 objects.
struct LazyReadback {
    allocator: windows::Win32::Graphics::Direct3D12::ID3D12CommandAllocator,
    list: windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList,
    fence: windows::Win32::Graphics::Direct3D12::ID3D12Fence,
    event: windows::Win32::Foundation::HANDLE,
    fence_value: u64,
    readback: ID3D12Resource,
    mapped: *const u8,
}

// SAFETY: accessed exclusively under `LAZY_GPU` mutex; D3D12 objects are
// free-threaded and `event` is only waited on while holding `LAZY_GPU`.
unsafe impl Send for LazyReadback {}

impl Drop for LazyReadback {
    fn drop(&mut self) {
        // SAFETY: `readback` was mapped in `LazyReadback::new` and `event` is ours.
        unsafe {
            self.readback.Unmap(0, None);
            let _ = windows::Win32::Foundation::CloseHandle(self.event);
        }
    }
}

impl LazyReadback {
    fn new() -> Option<Self> {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        use windows::Win32::System::Threading::CreateEventW;

        let host = host()?;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: PAGE as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        // SAFETY: plain D3D12 object creation on `host.device`.
        unsafe {
            let allocator: ID3D12CommandAllocator = host
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .ok()?;
            let list: ID3D12GraphicsCommandList = host
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .ok()?;
            list.Close().ok()?;
            let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            let mut readback: Option<ID3D12Resource> = None;
            if host
                .device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .is_err()
            {
                let _ = windows::Win32::Foundation::CloseHandle(event);
                return None;
            }
            let readback = match readback {
                Some(r) => r,
                None => {
                    let _ = windows::Win32::Foundation::CloseHandle(event);
                    return None;
                }
            };
            let mut mapped = std::ptr::null_mut();
            if readback.Map(0, None, Some(&mut mapped)).is_err() {
                let _ = windows::Win32::Foundation::CloseHandle(event);
                return None;
            }
            crate::host::name(&readback, || "lazy copy-back 64 KiB readback".into());
            Some(Self {
                allocator,
                list,
                fence,
                event,
                fence_value: 0,
                readback,
                mapped: mapped.cast(),
            })
        }
    }

    fn read_page(&mut self, shadow: &Shadow, offset: usize, dst: &mut [u8]) -> bool {
        use windows::Win32::Graphics::Direct3D12::ID3D12CommandList;
        use windows::Win32::System::Threading::{INFINITE, WaitForSingleObject};
        use windows::core::Interface;

        if dst.len() != PAGE || offset.saturating_add(PAGE) > shadow.len as usize {
            return false;
        }
        let Some(host) = host() else {
            return false;
        };
        if shadow.card.is_some() {
            // SAFETY: exclusive ownership under `LAZY_GPU`; `shadow.buffer` is
            // live and in `COMMON` (implicitly promotes to `COPY_SOURCE` and
            // decays back to `COMMON` when the command list finishes).
            let ok = unsafe {
                (|| -> windows::core::Result<()> {
                    self.allocator.Reset()?;
                    self.list.Reset(&self.allocator, None)?;
                    self.list.CopyBufferRegion(
                        &self.readback,
                        0,
                        &shadow.buffer,
                        offset as u64,
                        PAGE as u64,
                    );
                    self.list.Close()?;
                    let cmd: ID3D12CommandList = self.list.cast()?;
                    host.queue.ExecuteCommandLists(&[Some(cmd)]);
                    self.fence_value += 1;
                    host.queue.Signal(&self.fence, self.fence_value)?;
                    self.fence.SetEventOnCompletion(self.fence_value, self.event)?;
                    WaitForSingleObject(self.event, INFINITE);
                    std::ptr::copy_nonoverlapping(self.mapped, dst.as_mut_ptr(), PAGE);
                    Ok(())
                })()
            };
            ok.is_ok()
        } else {
            // PC-shared or ReBAR copy: wait for queued GPU work on `host.queue`
            // to complete so `shadow.data` holds all GPU writes, then copy once.
            // SAFETY: `self.fence` and `self.event` are exclusive under `LAZY_GPU`;
            // `offset + PAGE <= shadow.len` checked above.
            let ok = unsafe {
                (|| -> windows::core::Result<()> {
                    self.fence_value += 1;
                    host.queue.Signal(&self.fence, self.fence_value)?;
                    self.fence.SetEventOnCompletion(self.fence_value, self.event)?;
                    WaitForSingleObject(self.event, INFINITE);
                    std::ptr::copy_nonoverlapping(shadow.data.add(offset), dst.as_mut_ptr(), PAGE);
                    Ok(())
                })()
            };
            ok.is_ok()
        }
    }
}

/// Level 2 GPU readback serializer.
static LAZY_GPU: Mutex<Option<LazyReadback>> = Mutex::new(None);

fn ensure_lazy_readback() -> bool {
    let mut gpu = LAZY_GPU.lock().unwrap_or_else(|e| e.into_inner());
    if gpu.is_none() {
        *gpu = LazyReadback::new();
    }
    gpu.is_some()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PageAccess {
    Accessible,
    NoAccess,
    Other,
}

/// Queries the current commit/protection state of `addr` for `READ` vs `WRITE`.
/// Distinguishes committed `PAGE_NOACCESS` (so VEH retries when a concurrent
/// re-arm raced with table lookup) from already-accessible pages and genuine
/// foreign access violations.
fn query_page_access(addr: usize, is_write: bool) -> PageAccess {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
        PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain struct filled by `VirtualQuery`.
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid out pointer.
    let n = unsafe { VirtualQuery(addr as *const _, &mut mbi, size_of_val(&mbi)) };
    if n == 0 || mbi.State != MEM_COMMIT {
        return PageAccess::Other;
    }
    if mbi.Protect == PAGE_NOACCESS {
        return PageAccess::NoAccess;
    }
    let accessible = if is_write {
        matches!(mbi.Protect, PAGE_READWRITE | PAGE_EXECUTE_READWRITE)
    } else {
        matches!(
            mbi.Protect,
            PAGE_READWRITE | PAGE_READONLY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_READ
        )
    };
    if accessible {
        PageAccess::Accessible
    } else {
        PageAccess::Other
    }
}

/// Copies all `[sub_lo, sub_hi)` ranges inside `[0, PAGE)` that fall outside
/// `[excl_lo, excl_hi)` from `src` to `dst_base` (pass `excl_lo = PAGE, excl_hi = PAGE`
/// to copy the entire `[sub_lo, sub_hi)` slice).
///
/// # Safety
/// `dst_base + sub_lo .. dst_base + sub_hi` must be committed and writable, and
/// `src.len() == PAGE`.
unsafe fn copy_subregion_bytes(
    dst_base: *mut u8,
    src: &[u8],
    sub_lo: usize,
    sub_hi: usize,
    excl_lo: usize,
    excl_hi: usize,
) {
    debug_assert_eq!(src.len(), PAGE);
    if excl_lo >= sub_hi || excl_hi <= sub_lo {
        // SAFETY: caller guarantees `[sub_lo, sub_hi)` is writable.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr().add(sub_lo), dst_base.add(sub_lo), sub_hi - sub_lo);
        }
        return;
    }
    if sub_lo < excl_lo {
        // SAFETY: `[sub_lo, excl_lo)` lies inside `[sub_lo, sub_hi)`.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr().add(sub_lo), dst_base.add(sub_lo), excl_lo - sub_lo);
        }
    }
    if excl_hi < sub_hi {
        // SAFETY: `[excl_hi, sub_hi)` lies inside `[sub_lo, sub_hi)`.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr().add(excl_hi), dst_base.add(excl_hi), sub_hi - excl_hi);
        }
    }
}

/// Fallback for when OS protections within 64 KiB `[page_addr, page_addr + PAGE)`
/// were split into multiple VAD sub-regions (e.g., 4 KiB `VirtualProtect`): walks
/// every sub-region inside `[page_addr, page_addr + PAGE)`, restores
/// `PAGE_READWRITE` on every committed `PAGE_NOACCESS` sub-region, and populates
/// allowed bytes outside `[excl_lo, excl_hi)`.
fn unprotect_split_subregions(
    page_addr: usize,
    data: Option<&[u8]>,
    excl_lo: usize,
    excl_hi: usize,
    section_view: Option<(usize, u64)>,
) -> bool {
    use windows_sys::Win32::System::Memory::{
        FILE_MAP_WRITE, MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION,
        MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_NOACCESS, PAGE_READWRITE,
        UnmapViewOfFile, VirtualProtect, VirtualQuery,
    };
    let mut alias_base: Option<*mut u8> = None;
    let alias_raw = if let (Some(_), Some((section, off))) = (data, section_view) {
        // SAFETY: `section` backs this view at page-aligned offset `off`.
        let alias = unsafe {
            MapViewOfFile(
                section as *mut std::ffi::c_void,
                FILE_MAP_WRITE,
                (off >> 32) as u32,
                off as u32,
                PAGE,
            )
        };
        if !alias.Value.is_null() {
            alias_base = Some(alias.Value.cast::<u8>());
        }
        alias.Value
    } else {
        std::ptr::null_mut()
    };

    let page_end = page_addr.saturating_add(PAGE);
    let mut cursor = page_addr;
    let mut any_unprotected = false;
    let mut wrote_any = false;
    while cursor < page_end {
        // SAFETY: plain struct filled by `VirtualQuery`.
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: valid out pointer.
        let n = unsafe { VirtualQuery(cursor as *const _, &mut mbi, size_of_val(&mbi)) };
        if n == 0 {
            break;
        }
        let sub_end = (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize).min(page_end);
        if sub_end <= cursor {
            cursor = cursor.saturating_add(4096);
            continue;
        }
        if mbi.State == MEM_COMMIT && mbi.Protect == PAGE_NOACCESS {
            let sub_lo = cursor - page_addr;
            let sub_hi = sub_end - page_addr;
            if let (Some(bytes), Some(abase)) = (data, alias_base)
                && mbi.Type == MEM_MAPPED
            {
                // SAFETY: `abase` is a valid writable `PAGE`-byte alias for this section page.
                unsafe { copy_subregion_bytes(abase, bytes, sub_lo, sub_hi, excl_lo, excl_hi) };
                std::sync::atomic::fence(Ordering::Release);
                wrote_any = true;
            }
            let mut old = 0u32;
            // SAFETY: verified committed `PAGE_NOACCESS` sub-region within `[page_addr, page_end)`.
            if unsafe { VirtualProtect(cursor as *const _, sub_end - cursor, PAGE_READWRITE, &mut old) != 0 } {
                any_unprotected = true;
                if let Some(bytes) = data
                    && !(alias_base.is_some() && mbi.Type == MEM_MAPPED)
                {
                    // SAFETY: `[cursor, sub_end)` is now committed `PAGE_READWRITE`.
                    unsafe { copy_subregion_bytes(page_addr as *mut u8, bytes, sub_lo, sub_hi, excl_lo, excl_hi) };
                    wrote_any = true;
                }
            }
        }
        cursor = sub_end;
    }
    if wrote_any {
        std::sync::atomic::fence(Ordering::Release);
    }
    if !alias_raw.is_null() {
        // SAFETY: matching map returned above.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: alias_raw }) };
    }
    any_unprotected
}

/// Restores `PAGE_READWRITE` on `[page_addr, page_addr + PAGE)` only if it is
/// still committed `PAGE_NOACCESS` (handling both uniform 64 KiB regions and
/// split 4 KiB VAD sub-regions while defending against unnotified decommits),
/// and writes `data` into every unprotected sub-region.
fn unprotect_and_write_page(
    page_addr: usize,
    data: Option<&[u8]>,
    section_view: Option<(usize, u64)>,
) -> bool {
    use windows_sys::Win32::System::Memory::{
        FILE_MAP_WRITE, MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION,
        MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_NOACCESS, PAGE_READWRITE,
        UnmapViewOfFile, VirtualProtect, VirtualQuery,
    };
    // SAFETY: plain struct filled by `VirtualQuery`.
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid out pointer.
    let n = unsafe { VirtualQuery(page_addr as *const _, &mut mbi, size_of_val(&mbi)) };
    if n == 0 {
        return false;
    }
    if mbi.State != MEM_COMMIT
        || (mbi.BaseAddress as usize) > page_addr
        || (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize) < page_addr.saturating_add(PAGE)
        || mbi.Protect != PAGE_NOACCESS
    {
        return unprotect_split_subregions(page_addr, data, PAGE, PAGE, section_view);
    }
    if let (Some(bytes), Some((section, off))) = (data, section_view)
        && mbi.Type == MEM_MAPPED
    {
        debug_assert_eq!(bytes.len(), PAGE);
        // SAFETY: `section` is the live pagefile section backing `page_addr`
        // at `off` (recorded at arming time; `title_memory_remapping` flushes
        // before unmapping/closing). Mapping a temporary private view uses only
        // kernel `NtMapViewOfSection` (no loader/heap locks).
        let alias = unsafe {
            MapViewOfFile(
                section as *mut std::ffi::c_void,
                FILE_MAP_WRITE,
                (off >> 32) as u32,
                off as u32,
                PAGE,
            )
        };
        if !alias.Value.is_null() {
            // SAFETY: `alias.Value` maps `PAGE` writable bytes of the physical
            // page while `page_addr` remains `PAGE_NOACCESS`.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), alias.Value.cast::<u8>(), PAGE);
                std::sync::atomic::fence(Ordering::Release);
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: alias.Value });
            }
            let mut old = 0u32;
            // SAFETY: atomic PTE flip revealing fully populated physical page.
            return unsafe { VirtualProtect(page_addr as *const _, PAGE, PAGE_READWRITE, &mut old) != 0 };
        }
    }
    let mut old = 0u32;
    // SAFETY: verified committed `PAGE_NOACCESS` 64 KiB page we armed.
    let ok = unsafe { VirtualProtect(page_addr as *const _, PAGE, PAGE_READWRITE, &mut old) != 0 };
    if !ok {
        return false;
    }
    if let Some(bytes) = data {
        debug_assert_eq!(bytes.len(), PAGE);
        // SAFETY: `page_addr` was just verified committed and set `PAGE_READWRITE`
        // for `PAGE` bytes, and `bytes` is our disjoint heap buffer.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), page_addr as *mut u8, PAGE) };
        std::sync::atomic::fence(Ordering::Release);
    }
    true
}

/// Writes GPU bytes ONLY to the unwritten complement `[0..rel_lo)` and
/// `[rel_hi..PAGE)` of a partially overwritten deferred page `[page_addr, page_addr + PAGE)`
/// (used by [`lazy_release_written_range`] after DirectStorage writes into
/// backing physical pages), then restores `PAGE_READWRITE`. Never touches
/// `[rel_lo..rel_hi)`.
fn unprotect_and_write_complement(
    page_addr: usize,
    gpu_bytes: &[u8],
    rel_lo: usize,
    rel_hi: usize,
    section_view: Option<(usize, u64)>,
) -> bool {
    use windows_sys::Win32::System::Memory::{
        FILE_MAP_WRITE, MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION,
        MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_NOACCESS, PAGE_READWRITE,
        UnmapViewOfFile, VirtualProtect, VirtualQuery,
    };
    debug_assert_eq!(gpu_bytes.len(), PAGE);
    debug_assert!(rel_lo <= rel_hi && rel_hi <= PAGE);
    // SAFETY: plain struct filled by `VirtualQuery`.
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid out pointer.
    let n = unsafe { VirtualQuery(page_addr as *const _, &mut mbi, size_of_val(&mbi)) };
    if n == 0 {
        return false;
    }
    if mbi.State != MEM_COMMIT
        || (mbi.BaseAddress as usize) > page_addr
        || (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize) < page_addr.saturating_add(PAGE)
        || mbi.Protect != PAGE_NOACCESS
    {
        return unprotect_split_subregions(page_addr, Some(gpu_bytes), rel_lo, rel_hi, section_view);
    }
    if let Some((section, off)) = section_view
        && mbi.Type == MEM_MAPPED
    {
        // SAFETY: `section` is the live section backing `page_addr` at `off`.
        let alias = unsafe {
            MapViewOfFile(
                section as *mut std::ffi::c_void,
                FILE_MAP_WRITE,
                (off >> 32) as u32,
                off as u32,
                PAGE,
            )
        };
        if !alias.Value.is_null() {
            let base = alias.Value.cast::<u8>();
            // SAFETY: writing only the complement `[0..rel_lo)` and `[rel_hi..PAGE)`
            // inside our temporary `PAGE`-byte writable alias while `page_addr`
            // remains `PAGE_NOACCESS`.
            unsafe {
                if rel_lo > 0 {
                    std::ptr::copy_nonoverlapping(gpu_bytes.as_ptr(), base, rel_lo);
                }
                if rel_hi < PAGE {
                    std::ptr::copy_nonoverlapping(gpu_bytes.as_ptr().add(rel_hi), base.add(rel_hi), PAGE - rel_hi);
                }
                std::sync::atomic::fence(Ordering::Release);
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: alias.Value });
            }
            let mut old = 0u32;
            // SAFETY: atomic PTE flip revealing fully coherent page.
            return unsafe { VirtualProtect(page_addr as *const _, PAGE, PAGE_READWRITE, &mut old) != 0 };
        }
    }
    let mut old = 0u32;
    // SAFETY: verified committed `PAGE_NOACCESS` 64 KiB page we armed.
    if unsafe { VirtualProtect(page_addr as *const _, PAGE, PAGE_READWRITE, &mut old) == 0 } {
        return false;
    }
    // SAFETY: `page_addr` is committed and `PAGE_READWRITE`; copy only complement.
    unsafe {
        let base = page_addr as *mut u8;
        if rel_lo > 0 {
            std::ptr::copy_nonoverlapping(gpu_bytes.as_ptr(), base, rel_lo);
        }
        if rel_hi < PAGE {
            std::ptr::copy_nonoverlapping(gpu_bytes.as_ptr().add(rel_hi), base.add(rel_hi), PAGE - rel_hi);
        }
        std::sync::atomic::fence(Ordering::Release);
    }
    true
}

/// Flushes one 64 KiB page that has already been transitioned to `LazyEntry::Busy`
/// in `LAZY_TABLE` and `LAZY_FLUSHING` in `shadow.lazy_state`.
#[allow(clippy::too_many_arguments)]
fn flush_armed_page(
    shadow: &Shadow,
    page: usize,
    offset: usize,
    page_idx: usize,
    section_view: Option<(usize, u64)>,
    wait: &FlushWait,
    is_veh: bool,
    is_write: bool,
    save_baseline: bool,
) {
    // Heap allocation (never 64 KiB on a VEH stack).
    let mut buf = vec![0u8; PAGE];
    let ok = {
        let mut gpu = LAZY_GPU.lock().unwrap_or_else(|e| e.into_inner());
        gpu.as_mut().is_some_and(|rb| rb.read_page(shadow, offset, &mut buf))
    };
    let wrote = unprotect_and_write_page(page * PAGE, ok.then_some(buf.as_slice()), section_view);
    {
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        lazy_table_remove_page(&mut table, page);
        LAZY_FLUSH_TOTAL.fetch_add(1, Ordering::Release);
        if save_baseline {
            if ok && wrote {
                table
                    .flushed_baselines
                    .insert((shadow.id, page_idx), FlushedBaseline::new(buf));
            }
            shadow.record_lazy_flush_strike(page_idx);
            shadow.lazy_state[page_idx].store(LAZY_RESYNC, Ordering::Release);
            shadow.resync_count.fetch_add(1, Ordering::AcqRel);
        } else {
            shadow.lazy_state[page_idx].store(LAZY_EAGER, Ordering::Release);
        }
        shadow.deferred_count.fetch_sub(1, Ordering::AcqRel);
    }
    if is_veh {
        LAZY_STATS[if is_write { 3 } else { 2 }].fetch_add(1, Ordering::Relaxed);
    } else {
        LAZY_STATS[4].fetch_add(1, Ordering::Relaxed);
    }
    if ok {
        LAZY_STATS[5].fetch_add(PAGE as u64, Ordering::Relaxed);
    }
    wait.finish(FlushOutcome::Accessible);
}

/// Permanently pins title pages overlapping `[at, at + len)` (e.g., GPU fence
/// polling and `defer_write` targets) so they are NEVER protected `PAGE_NOACCESS`.
/// Any currently deferred page in the range is flushed first.
pub(crate) fn lazy_pin_range(at: usize, len: usize) {
    if len == 0 || !lazy_copyback() {
        return;
    }
    let first = at / PAGE;
    let last = at.saturating_add(len).div_ceil(PAGE);
    {
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        for page in first..last {
            table.pinned.insert(page);
        }
    }
    lazy_flush_range(at, len);
}

/// Whether `page` (`addr / PAGE`) is currently armed or mid-flush in `LAZY_TABLE`.
/// Checked by `watch.rs` so read-only guard re-arming never downgrades `PAGE_NOACCESS`.
pub(crate) fn is_lazy_deferred_page(page: usize) -> bool {
    if LAZY_ARMED_COUNT.load(Ordering::Acquire) == 0
        || LAZY_BUCKETS[page & LAZY_BUCKETS_MASK].load(Ordering::Acquire) == 0
    {
        return false;
    }
    LAZY_TABLE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .pages
        .contains_key(&page)
}

/// Deadlock-free VEH demand-fault handler called from `watch.rs::on_fault` on
/// both READ and WRITE access violations before `watch` acquires any lock.
/// Returns `true` iff the fault was handled and execution may resume.
pub(crate) fn lazy_on_fault(addr: usize, is_write: bool) -> bool {
    if LAZY_ARMED_COUNT.load(Ordering::Acquire) == 0
        && LAZY_FLUSH_TOTAL.load(Ordering::Acquire) == 0
        && LAZY_TABLE_EPOCH.load(Ordering::Acquire) == 0
    {
        return false;
    }
    let page = addr / PAGE;
    let bucket = page & LAZY_BUCKETS_MASK;
    // Lock-free fast reject for non-lazy pages (e.g. `watch` PAGE_READONLY
    // write-guard faults): skips BOTH `LAZY_TABLE.lock()` and `VirtualQuery`
    // (~2 ms/call in title pools) when the bucket is empty, this page was not
    // recently flushed, and `LAZY_TABLE_EPOCH` did not advance concurrently.
    let epoch_fast = LAZY_TABLE_EPOCH.load(Ordering::Acquire);
    if LAZY_BUCKETS[bucket].load(Ordering::Acquire) == 0
        && LAZY_LAST_FLUSHED[bucket].load(Ordering::Acquire) != page
        && LAZY_TABLE_EPOCH.load(Ordering::Acquire) == epoch_fast
    {
        return false;
    }
    if IN_LAZY_VEH.with(Cell::get) {
        return false;
    }
    struct InVehGuard;
    impl Drop for InVehGuard {
        fn drop(&mut self) {
            IN_LAZY_VEH.with(|c| c.set(false));
        }
    }
    IN_LAZY_VEH.with(|c| c.set(true));
    let _in_veh = InVehGuard;

    loop {
        let action = {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            match table.pages.get(&page) {
                Some(LazyEntry::Busy { wait, .. }) => Ok(Err(wait.clone())),
                Some(LazyEntry::Armed(armed)) => match armed.shadow.upgrade() {
                    Some(shadow) => {
                        let (shadow_id, offset, page_idx, section_view) =
                            (armed.shadow_id, armed.offset, armed.page_idx, armed.section_view);
                        let wait = Arc::new(FlushWait::new());
                        shadow.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                        lazy_table_insert_busy(&mut table, page, shadow_id, wait.clone());
                        Ok(Ok((shadow, offset, page_idx, section_view, wait)))
                    }
                    // `Shadow::drop` is actively executing and will acquire
                    // `LAZY_TABLE` momentarily to transition to `Busy` and flush.
                    None => Err(None),
                },
                None => Err(Some(LAZY_TABLE_EPOCH.load(Ordering::Acquire))),
            }
        };
        match action {
            Ok(Ok((shadow, offset, page_idx, section_view, wait))) => {
                flush_armed_page(&shadow, page, offset, page_idx, section_view, &wait, true, is_write, true);
                return true;
            }
            Ok(Err(wait)) => {
                if wait.wait() == FlushOutcome::Accessible {
                    return true;
                }
                continue;
            }
            Err(None) => {
                std::thread::yield_now();
                continue;
            }
            Err(Some(epoch_before)) => {
                // Skip `VirtualQuery` (~2 ms in large title heaps) when this
                // page was neither recently flushed nor raced with a table edit.
                if LAZY_LAST_FLUSHED[bucket].load(Ordering::Acquire) != page
                    && LAZY_TABLE_EPOCH.load(Ordering::Acquire) == epoch_before
                {
                    return false;
                }
                match query_page_access(addr, is_write) {
                    PageAccess::Accessible
                        if LAZY_FLUSH_TOTAL.load(Ordering::Acquire) > 0
                            || LAZY_TABLE_EPOCH.load(Ordering::Acquire) > 0 =>
                    {
                        return true;
                    }
                    PageAccess::NoAccess if LAZY_TABLE_EPOCH.load(Ordering::Acquire) != epoch_before => continue,
                    _ => return false,
                }
            }
        }
    }
}

/// Flushes every deferred GPU-resident page overlapping `[at, at + len)` into
/// title memory and restores `PAGE_READWRITE`, waiting on any in-flight flush.
/// Safe to call with zero Level 0 locks held (and never acquires Level 0 locks).
pub(crate) fn lazy_flush_range(at: usize, len: usize) {
    if !lazy_has_any_in_range(at, len) {
        return;
    }
    let first = at / PAGE;
    let last = at.saturating_add(len).div_ceil(PAGE);
    loop {
        enum Next {
            Done,
            Wait(Arc<FlushWait>),
            Yield,
            Flush(usize, Arc<Shadow>, usize, usize, Option<(usize, u64)>, Arc<FlushWait>),
        }
        let next = {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            let found = table
                .pages
                .range(first..last)
                .next()
                .map(|(&p, _)| p);
            match found {
                None => Next::Done,
                Some(page) => match table.pages.get(&page) {
                    Some(LazyEntry::Busy { wait, .. }) => Next::Wait(wait.clone()),
                    Some(LazyEntry::Armed(armed)) => match armed.shadow.upgrade() {
                        Some(shadow) => {
                            let (shadow_id, offset, page_idx, section_view) =
                                (armed.shadow_id, armed.offset, armed.page_idx, armed.section_view);
                            let wait = Arc::new(FlushWait::new());
                            shadow.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                            lazy_table_insert_busy(&mut table, page, shadow_id, wait.clone());
                            Next::Flush(page, shadow, offset, page_idx, section_view, wait)
                        }
                        None => Next::Yield,
                    },
                    None => Next::Done,
                },
            }
        };
        match next {
            Next::Done => return,
            Next::Wait(wait) => {
                let _ = wait.wait();
            }
            Next::Yield => std::thread::yield_now(),
            Next::Flush(page, shadow, offset, page_idx, section_view, wait) => {
                flush_armed_page(&shadow, page, offset, page_idx, section_view, &wait, false, false, true);
            }
        }
    }
}

/// Scoped active-remap guard held inside `watch::Remapping` from entry of
/// `watch::title_memory_remapping` until `Remapping::drop`: prevents any
/// concurrent relay/submit thread from re-arming `PAGE_NOACCESS` inside
/// `[at, at + len)` while views are being unmapped or replaced.
pub(crate) struct LazyRemapGuard {
    id: Option<u64>,
}

impl Drop for LazyRemapGuard {
    fn drop(&mut self) {
        if let Some(id) = self.id.take() {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            table.remapping.retain(|&(rid, _, _)| rid != id);
        }
    }
}

/// Registers an active remap barrier over `[at, at + len)` in `LAZY_TABLE` and
/// flushes all currently deferred pages overlapping `[at, at + len)` before
/// returning the guard. Safe to call with zero Level 0 locks held.
pub(crate) fn lazy_begin_remapping(at: usize, len: usize) -> LazyRemapGuard {
    lazy_flush_and_hold(at, len)
}

/// Registers a scoped non-arming range barrier for `[at, at + len)` in
/// `LAZY_TABLE.remapping` and flushes any currently armed or in-flight-arming
/// deferred pages in that range before returning. While the returned guard is
/// live, no concurrent relay/submit thread can arm `PAGE_NOACCESS` on any page
/// overlapping `[at, at + len)` (eliminating TOCTOU races with
/// `ReadProcessMemory` / `read_sparse` / `NtWriteVirtualMemory` /
/// `WriteProcessMemory`).
///
/// Fast path when `VICTORY_LAZY_COPYBACK` is OFF: returns an inert guard with
/// zero locking.
pub(crate) fn lazy_flush_and_hold(at: usize, len: usize) -> LazyRemapGuard {
    if len == 0 || !lazy_copyback() {
        return LazyRemapGuard { id: None };
    }
    let first = at / PAGE;
    let last = at.saturating_add(len).div_ceil(PAGE);
    let id = {
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        let id = table.next_remap_id;
        table.next_remap_id = table.next_remap_id.wrapping_add(1).max(1);
        table.remapping.push((id, first, last));
        id
    };
    let guard = LazyRemapGuard { id: Some(id) };
    lazy_flush_range(at, len);
    guard
}

/// Releases deferred `PAGE_NOACCESS` pages overlapping `[at, at + len)` AFTER
/// external/DirectStorage bytes have ALREADY been written directly into the
/// backing physical pages (`watch::title_memory_written`):
/// - For deferred pages wholly covered by `[at, at + len)`: restores
///   `PAGE_READWRITE` without copying stale GPU bytes over the freshly written
///   data, captures GPU bytes into `FlushedBaseline` only if `writes_pending()`
///   is true (so in-flight relay merges diff against the deferred GPU bytes and
///   preserve the freshly written title bytes), and marks `LAZY_RESYNC`.
/// - For partially covered 64 KiB deferred pages (`[w_lo, w_hi)`): copies GPU
///   bytes ONLY into the unwritten complement `[page_addr..w_lo)` and
///   `[w_hi..page_end)`, never touching `[w_lo..w_hi)`, restores `PAGE_READWRITE`,
///   and marks `LAZY_RESYNC`.
pub(crate) fn lazy_release_written_range(at: usize, len: usize) {
    if !lazy_has_any_in_range(at, len) {
        return;
    }
    let end = at.saturating_add(len);
    let first = at / PAGE;
    let last = end.div_ceil(PAGE);
    loop {
        enum Next {
            Done,
            Wait(Arc<FlushWait>),
            Yield,
            Release(usize, Arc<Shadow>, usize, usize, Option<(usize, u64)>, Arc<FlushWait>),
        }
        let next = {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            let found = table.pages.range(first..last).next().map(|(&p, _)| p);
            match found {
                None => Next::Done,
                Some(page) => match table.pages.get(&page) {
                    Some(LazyEntry::Busy { wait, .. }) => Next::Wait(wait.clone()),
                    Some(LazyEntry::Armed(armed)) => match armed.shadow.upgrade() {
                        Some(shadow) => {
                            let (shadow_id, offset, page_idx, section_view) =
                                (armed.shadow_id, armed.offset, armed.page_idx, armed.section_view);
                            let wait = Arc::new(FlushWait::new());
                            shadow.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                            lazy_table_insert_busy(&mut table, page, shadow_id, wait.clone());
                            Next::Release(page, shadow, offset, page_idx, section_view, wait)
                        }
                        None => Next::Yield,
                    },
                    None => Next::Done,
                },
            }
        };
        match next {
            Next::Done => return,
            Next::Wait(wait) => {
                let _ = wait.wait();
            }
            Next::Yield => std::thread::yield_now(),
            Next::Release(page, shadow, offset, page_idx, section_view, wait) => {
                let page_addr = page * PAGE;
                let page_end = page_addr.saturating_add(PAGE);
                let w_lo = at.max(page_addr);
                let w_hi = end.min(page_end);
                let full_cover = w_lo == page_addr && w_hi == page_end;
                let mut baseline = if full_cover {
                    // Fresh bytes cover all 64 KiB: restore `PAGE_READWRITE`
                    // without ever touching title memory. When a flight is
                    // still pending on this shadow (`writes_pending()`), capture
                    // the GPU bytes into `FlushedBaseline` (without writing
                    // them to title memory) so in-flight `card_merge_relay` /
                    // `relay_merge` diffs against the deferred GPU bytes rather
                    // than stale `self.data` and preserves 100% of the freshly
                    // written title bytes.
                    let captured = if shadow.writes_pending() {
                        let mut buf = vec![0u8; PAGE];
                        let ok = {
                            let mut gpu = LAZY_GPU.lock().unwrap_or_else(|e| e.into_inner());
                            gpu.as_mut().is_some_and(|rb| rb.read_page(&shadow, offset, &mut buf))
                        };
                        if ok {
                            LAZY_STATS[5].fetch_add(PAGE as u64, Ordering::Relaxed);
                        }
                        ok.then_some(buf)
                    } else {
                        None
                    };
                    let _ = unprotect_and_write_page(page_addr, None, None);
                    captured
                } else {
                    // Partial overwrite: fetch GPU page and write ONLY the
                    // unwritten complement `[page_addr..w_lo)` and `[w_hi..page_end)`.
                    let mut buf = vec![0u8; PAGE];
                    let ok = {
                        let mut gpu = LAZY_GPU.lock().unwrap_or_else(|e| e.into_inner());
                        gpu.as_mut().is_some_and(|rb| rb.read_page(&shadow, offset, &mut buf))
                    };
                    let wrote = if ok {
                        unprotect_and_write_complement(
                            page_addr,
                            &buf,
                            w_lo - page_addr,
                            w_hi - page_addr,
                            section_view,
                        )
                    } else {
                        unprotect_and_write_page(page_addr, None, None)
                    };
                    if ok {
                        LAZY_STATS[5].fetch_add(PAGE as u64, Ordering::Relaxed);
                    }
                    (ok && wrote).then_some(buf)
                };
                // Guard against a concurrent `Shadow::writing` setting
                // `has_pending` after the initial `full_cover` check.
                if full_cover && baseline.is_none() && shadow.writes_pending() {
                    let mut buf = vec![0u8; PAGE];
                    let ok = {
                        let mut gpu = LAZY_GPU.lock().unwrap_or_else(|e| e.into_inner());
                        gpu.as_mut().is_some_and(|rb| rb.read_page(&shadow, offset, &mut buf))
                    };
                    if ok {
                        LAZY_STATS[5].fetch_add(PAGE as u64, Ordering::Relaxed);
                        baseline = Some(buf);
                    }
                }
                {
                    let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                    lazy_table_remove_page(&mut table, page);
                    LAZY_FLUSH_TOTAL.fetch_add(1, Ordering::Release);
                    if let Some(buf) = baseline {
                        table
                            .flushed_baselines
                            .insert((shadow.id, page_idx), FlushedBaseline::new(buf));
                    }
                    shadow.record_lazy_flush_strike(page_idx);
                    shadow.lazy_state[page_idx].store(LAZY_RESYNC, Ordering::Release);
                    shadow.resync_count.fetch_add(1, Ordering::AcqRel);
                    shadow.deferred_count.fetch_sub(1, Ordering::AcqRel);
                }
                LAZY_STATS[4].fetch_add(1, Ordering::Relaxed);
                wait.finish(FlushOutcome::Accessible);
            }
        }
    }
}

/// Flushes deferred pages in `[from, to)` owned by ANY shadow other than `my_id`
/// (resolving overlapping `Shadow::exact` / nested sub-buffer coherence).
/// Returns the page numbers that were flushed/waited so the caller can force a
/// fresh copy from title memory even when `watch::ask_once` is active.
fn lazy_flush_other_shadows(my_id: u64, from: usize, to: usize) -> Vec<usize> {
    if !lazy_may_have_other_in_range(my_id, from, to) {
        return Vec::new();
    }
    let first = from / PAGE;
    let last = to.div_ceil(PAGE);
    let mut flushed = Vec::new();
    loop {
        enum Next {
            Done,
            Wait(usize, Arc<FlushWait>),
            Yield,
            Flush(usize, Arc<Shadow>, usize, usize, Option<(usize, u64)>, Arc<FlushWait>),
        }
        let next = {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            let found = table.pages.range(first..last).find_map(|(&p, entry)| match entry {
                LazyEntry::Armed(a) if a.shadow_id != my_id => Some(p),
                LazyEntry::Busy { shadow_id, .. } if *shadow_id != my_id => Some(p),
                _ => None,
            });
            match found {
                None => Next::Done,
                Some(page) => match table.pages.get(&page) {
                    Some(LazyEntry::Busy { wait, .. }) => Next::Wait(page, wait.clone()),
                    Some(LazyEntry::Armed(armed)) => match armed.shadow.upgrade() {
                        Some(shadow) => {
                            let (shadow_id, offset, page_idx, section_view) =
                                (armed.shadow_id, armed.offset, armed.page_idx, armed.section_view);
                            let wait = Arc::new(FlushWait::new());
                            shadow.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                            lazy_table_insert_busy(&mut table, page, shadow_id, wait.clone());
                            Next::Flush(page, shadow, offset, page_idx, section_view, wait)
                        }
                        None => Next::Yield,
                    },
                    None => Next::Done,
                },
            }
        };
        match next {
            Next::Done => return flushed,
            Next::Wait(page, wait) => {
                let _ = wait.wait();
                flushed.push(page);
            }
            Next::Yield => std::thread::yield_now(),
            Next::Flush(page, shadow, offset, page_idx, section_view, wait) => {
                flush_armed_page(&shadow, page, offset, page_idx, section_view, &wait, false, false, true);
                flushed.push(page);
            }
        }
    }
}

/// Drops deferred `PAGE_NOACCESS` protections overlapping `[at, at + len)` before
/// `VirtualFree` without wasting a GPU readback on memory being freed, restoring
/// `PAGE_READWRITE` while committed and waiting for any in-flight flush.
pub(crate) fn lazy_release_range(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let first = at / PAGE;
    let last = at.saturating_add(len).div_ceil(PAGE);
    {
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        let stale_pins: Vec<usize> = table.pinned.range(first..last).copied().collect();
        for p in stale_pins {
            table.pinned.remove(&p);
        }
    }
    if LAZY_ARMED_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    loop {
        enum Next {
            Done,
            Wait(Arc<FlushWait>),
            Release(usize, Weak<Shadow>, usize, Arc<FlushWait>),
        }
        let next = {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            let found = table.pages.range(first..last).next().map(|(&p, _)| p);
            match found {
                None => Next::Done,
                Some(page) => match table.pages.get(&page) {
                    Some(LazyEntry::Busy { wait, .. }) => Next::Wait(wait.clone()),
                    Some(LazyEntry::Armed(armed)) => {
                        let (shadow, shadow_id, page_idx) = (armed.shadow.clone(), armed.shadow_id, armed.page_idx);
                        let wait = Arc::new(FlushWait::new());
                        if let Some(s) = shadow.upgrade() {
                            s.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                        }
                        lazy_table_insert_busy(&mut table, page, shadow_id, wait.clone());
                        Next::Release(page, shadow, page_idx, wait)
                    }
                    None => Next::Done,
                },
            }
        };
        match next {
            Next::Done => return,
            Next::Wait(wait) => {
                let _ = wait.wait();
            }
            Next::Release(page, shadow, page_idx, wait) => {
                let _ = unprotect_and_write_page(page * PAGE, None, None);
                {
                    let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                    lazy_table_remove_page(&mut table, page);
                    LAZY_FLUSH_TOTAL.fetch_add(1, Ordering::Release);
                    if let Some(s) = shadow.upgrade() {
                        s.lazy_state[page_idx].store(LAZY_EAGER, Ordering::Release);
                        s.deferred_count.fetch_sub(1, Ordering::AcqRel);
                    }
                }
                wait.finish(FlushOutcome::Accessible);
            }
        }
    }
}

impl Shadow {
    fn lazy_mark_hot(&self, page_idx: usize, cool_until: u32) {
        if page_idx < self.lazy_cool_until.len() {
            self.lazy_cool_until[page_idx].fetch_max(cool_until, Ordering::AcqRel);
            let _ = self.lazy_state[page_idx].compare_exchange(
                LAZY_EAGER,
                LAZY_HOT,
                Ordering::Release,
                Ordering::Relaxed,
            );
        }
    }

    fn lazy_mark_ineligible(&self, page_idx: usize) {
        if page_idx < self.lazy_cool_until.len() {
            self.lazy_cool_until[page_idx].store(u32::MAX, Ordering::Release);
            let _ = self.lazy_state[page_idx].compare_exchange(
                LAZY_EAGER,
                LAZY_INELIGIBLE,
                Ordering::Release,
                Ordering::Relaxed,
            );
        }
    }

    fn record_lazy_flush_strike(&self, page_idx: usize) {
        let frame = lazy_current_frame();
        if frame == 0 || page_idx >= self.lazy_strikes.len() {
            return;
        }
        let (threshold, window) = lazy_hot_params();
        let prev_flush = self.lazy_last_flush_frame[page_idx].swap(frame, Ordering::Relaxed);
        let prev_cool = self.lazy_cool_until[page_idx].load(Ordering::Relaxed);
        let ref_frame = if prev_cool == u32::MAX {
            prev_flush
        } else {
            prev_flush.max(prev_cool)
        };
        let prev_strikes = self.lazy_strikes[page_idx].load(Ordering::Relaxed);
        let strikes = if prev_flush > 0 && frame.saturating_sub(ref_frame) <= window {
            prev_strikes.saturating_add(1)
        } else {
            1
        };
        self.lazy_strikes[page_idx].store(strikes, Ordering::Relaxed);
        if strikes >= threshold {
            let cool_until = match strikes - threshold {
                0 => frame.saturating_add(300).min(u32::MAX - 1),
                1 => frame.saturating_add(1800).min(u32::MAX - 1),
                _ => u32::MAX,
            };
            self.lazy_cool_until[page_idx].store(cool_until, Ordering::Release);
        } else if prev_cool != u32::MAX && prev_cool <= frame {
            self.lazy_cool_until[page_idx].store(0, Ordering::Release);
        }
    }

    fn lazy_post_resync_state(&self, page_idx: usize, cur_frame: u32) -> u8 {
        let cool = self.lazy_cool_until[page_idx].load(Ordering::Acquire);
        if cool == u32::MAX {
            LAZY_INELIGIBLE
        } else if cur_frame > 0 && cool > cur_frame {
            LAZY_HOT
        } else {
            LAZY_EAGER
        }
    }

    /// Attempts to leave the GPU-written piece `[lo, hi)` resident on the GPU
    /// instead of copying it back eagerly (`VICTORY_LAZY_COPYBACK=1`).
    ///
    /// Eligibility & tear-freedom guarantees:
    /// - Only whole 64 KiB title-page-aligned, previously whole-page-initialized
    ///   (`copied_at != 0`) pieces of watched (`MEM_MAPPED` single-view clean
    ///   `watch` guard, or `MEM_PRIVATE | MEM_WRITE_WATCH` `PAGE_READWRITE`)
    ///   memory are ever armed. Unwatched `MEM_PRIVATE` pages are never armed.
    /// - Never lowers `PAGE_NOACCESS` to `PAGE_READONLY` during arming (zero
    ///   stale pre-GPU exposure). On virgin demand-zero `MEM_WRITE_WATCH` pages
    ///   where `GetWriteWatch(0)` reports `count > 0`, defers cleanly to eager
    ///   copy-back on Frame 1 so Frame 2+ arms with `count == 0`.
    /// - Zero-tear concurrent multi-thread flush applies strictly to section-backed
    ///   `MEM_MAPPED` pages (console physical GPU heaps, via temporary writable
    ///   `MapViewOfFile` alias before atomic PTE flip); `MEM_PRIVATE` pages rely
    ///   on title fence/thread ordering for threads arriving after `VirtualProtect`.
    pub(crate) fn try_defer_piece(self: &Arc<Self>, lo: usize, hi: usize) -> bool {
        use windows_sys::Win32::System::Memory::{
            GetWriteWatch, MEM_COMMIT, MEM_MAPPED, MEM_PRIVATE, MEMORY_BASIC_INFORMATION,
            PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, VirtualProtect, VirtualQuery,
        };

        #[cfg(test)]
        if !self.lazy_opted_in {
            return false;
        }
        if !lazy_copyback() || self.place.is_some() {
            return false;
        }
        let page_addr = (self.start as usize).saturating_add(lo);
        if hi.saturating_sub(lo) != PAGE || !page_addr.is_multiple_of(PAGE) || hi > self.len as usize {
            return false;
        }
        let base = self.start as usize / PAGE;
        let page = page_addr / PAGE;
        let idx = page - base;
        if idx >= self.lazy_state.len() {
            return false;
        }

        let cur_frame = lazy_current_frame();
        // Steady-state O(1) lock-free fast path: already GPU-resident, hot-cooled,
        // or permanently ineligible.
        let state = self.lazy_state[idx].load(Ordering::Acquire);
        if state == LAZY_DEFERRED {
            LAZY_STATS[1].fetch_add(1, Ordering::Relaxed);
            return true;
        }
        if state == LAZY_INELIGIBLE {
            return false;
        }
        if state == LAZY_HOT {
            let cool = self.lazy_cool_until[idx].load(Ordering::Acquire);
            if cool == u32::MAX || cur_frame < cool {
                return false;
            }
            if self.lazy_state[idx]
                .compare_exchange(LAZY_HOT, LAZY_EAGER, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                return false;
            }
        } else if state != LAZY_EAGER {
            return false;
        }

        // Only arm watched memory (`MEM_MAPPED` clean guards or `MEM_WRITE_WATCH`).
        // Checked lock-free BEFORE acquiring `self.copied` or `card.dirty`.
        let watched_flag = self.watched.load(Ordering::Relaxed);
        if !matches!(watched_flag, 1 | GUARDED) {
            return false;
        }

        // Only defer pages whose entire 64 KiB were previously initialized into
        // the shadow (otherwise a later VEH flush would overwrite uncopied title
        // bytes with zeros).
        let copied_at = self.copied.lock().unwrap_or_else(|e| e.into_inner()).at.get(idx).copied().unwrap_or(0);
        if copied_at == 0 {
            return false;
        }

        // Pure userspace pre-checks BEFORE calling `VirtualQuery` (~1-2 ms in
        // large title pools) so ineligible or recently written pages exit in ns.
        if crate::watch::was_written_since(page, copied_at) {
            return false;
        }
        if watched_flag == GUARDED {
            if crate::inplace::pages_of(page_addr).len() > 1 {
                if cur_frame > 0 {
                    self.lazy_mark_hot(idx, cur_frame.saturating_add(120).min(u32::MAX - 1));
                }
                return false;
            }
            if !crate::watch::is_guarded(page) {
                return false;
            }
        }

        // Do not defer a carded piece that has un-uploaded CPU dirty bytes.
        if let Some(card) = &self.card {
            let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
            if dirty.iter().any(|&(a, b)| a < hi && b > lo) {
                return false;
            }
        }

        // Check OS page eligibility BEFORE touching `LAZY_TABLE` or protections.
        // SAFETY: plain struct filled by `VirtualQuery`.
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: valid out pointer.
        let n = unsafe { VirtualQuery(page_addr as *const _, &mut mbi, size_of_val(&mbi)) };
        if n == 0
            || mbi.State != MEM_COMMIT
            || (mbi.BaseAddress as usize) > page_addr
            || (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize) < page_addr.saturating_add(PAGE)
        {
            if cur_frame > 0 {
                self.lazy_mark_hot(idx, cur_frame.saturating_add(60).min(u32::MAX - 1));
            }
            return false;
        }

        let is_mapped = mbi.Type == MEM_MAPPED;
        if is_mapped {
            if crate::inplace::pages_of(page_addr).len() > 1 {
                if cur_frame > 0 {
                    self.lazy_mark_hot(idx, cur_frame.saturating_add(120).min(u32::MAX - 1));
                }
                return false;
            }
            if mbi.Protect != PAGE_READONLY
                || !crate::watch::is_guarded(page)
                || crate::watch::was_written_since(page, copied_at)
            {
                if cur_frame > 0 {
                    self.lazy_mark_hot(idx, cur_frame.saturating_add(60).min(u32::MAX - 1));
                }
                return false;
            }
        } else if mbi.Type == MEM_PRIVATE {
            if mbi.Protect != PAGE_READWRITE || crate::watch::was_written_since(page, copied_at) {
                if cur_frame > 0 {
                    self.lazy_mark_hot(idx, cur_frame.saturating_add(60).min(u32::MAX - 1));
                }
                return false;
            }
        } else {
            self.lazy_mark_ineligible(idx);
            return false;
        }

        // Install VEH and pre-allocate the 64 KiB readback channel BEFORE ever
        // setting `PAGE_NOACCESS` on a title page.
        crate::watch::install_veh();
        if !ensure_lazy_readback() {
            return false;
        }

        // Reserve the page slot as `Busy` BEFORE calling `VirtualProtect` so any
        // concurrent thread faulting during arming blocks on `wait` instead of
        // observing an unhandled `PAGE_NOACCESS` exception.
        let wait = Arc::new(FlushWait::new());
        let pinned_before = {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            if table.pinned.contains(&page) {
                true
            } else if table.is_remapping(page) || table.pages.contains_key(&page) {
                return false;
            } else {
                lazy_table_insert_busy(&mut table, page, self.id, wait.clone());
                false
            }
        };
        if pinned_before {
            self.lazy_mark_ineligible(idx);
            return false;
        }

        let mut armed_ok = false;
        let mut restore_rw = false;
        let mut section_view = None;
        let mut perm_ineligible = false;

        if is_mapped {
            section_view = match crate::inplace::cover(page_addr, PAGE) {
                crate::inplace::Cover::Run(section, off) => Some((section, off)),
                _ => None,
            };
            // Atomically transition single-view clean `watch`-guarded page
            // directly `PAGE_READONLY -> PAGE_NOACCESS`. Once transitioned,
            // mark `restore_rw = true` immediately so any subsequent rejection
            // (e.g. concurrent `watch::wrote` advancing the journal stamp)
            // always rolls back `PAGE_NOACCESS -> PAGE_READWRITE`.
            if crate::watch::transition_clean_guard_to_noaccess(page) {
                restore_rw = true;
                #[cfg(test)]
                if let Some(hook) = TRY_DEFER_POST_TRANSITION_HOOK
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take()
                {
                    hook(page);
                }
                if !crate::watch::was_written_since(page, copied_at) {
                    armed_ok = true;
                }
            }
        } else {
            // `MEM_WRITE_WATCH` private memory: set `PAGE_NOACCESS` directly
            // (never exposing `PAGE_READONLY`), then prove via non-resetting
            // `GetWriteWatch(0)` that zero 4 KiB subpages were written between
            // `copied_at` and the protection change. If `count > 0` (concurrent
            // CPU write or virgin demand-zero fault on Frame 1), roll back to
            // `PAGE_READWRITE` and stay eager so Frame 2 can arm cleanly.
            let mut old = 0u32;
            // SAFETY: verified committed `MEM_PRIVATE` `PAGE_READWRITE` page.
            let prot_ok = unsafe { VirtualProtect(page_addr as *const _, PAGE, PAGE_NOACCESS, &mut old) != 0 };
            if prot_ok && old == PAGE_READWRITE {
                restore_rw = true;
                let mut touched = [0usize; PAGE / 4096];
                let mut count = touched.len();
                let mut granularity = 0u32;
                // SAFETY: non-resetting write-watch query over the 64 KiB page.
                let ww_ok = unsafe {
                    GetWriteWatch(
                        0,
                        page_addr as *const _,
                        PAGE,
                        touched.as_mut_ptr().cast(),
                        &mut count,
                        &mut granularity,
                    ) == 0
                };
                if !ww_ok {
                    perm_ineligible = true;
                } else if count == 0 && !crate::watch::was_written_since(page, copied_at) {
                    armed_ok = true;
                }
            } else if prot_ok {
                let mut _rev = 0u32;
                // SAFETY: reverting unexpected prior protection.
                unsafe { VirtualProtect(page_addr as *const _, PAGE, old, &mut _rev) };
            }
        }

        if armed_ok {
            let mut pinned_now = false;
            let committed = {
                let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                if table.pinned.contains(&page) {
                    pinned_now = true;
                    false
                } else if table.is_remapping(page) {
                    false
                } else {
                    lazy_table_insert_armed(
                        &mut table,
                        page,
                        ArmedPage {
                            shadow: Arc::downgrade(self),
                            shadow_id: self.id,
                            offset: lo,
                            page_idx: idx,
                            section_view,
                        },
                    );
                    self.lazy_state[idx].store(LAZY_DEFERRED, Ordering::Release);
                    self.deferred_count.fetch_add(1, Ordering::AcqRel);
                    true
                }
            };
            if committed {
                LAZY_STATS[0].fetch_add(1, Ordering::Relaxed);
                wait.finish(FlushOutcome::ArmedRetry);
                return true;
            }
            if pinned_now {
                perm_ineligible = true;
            }
            // Pinned or remapped concurrently while arming: roll back to `PAGE_READWRITE`.
            restore_rw = true;
        }

        if restore_rw {
            let mut _rev = 0u32;
            // SAFETY: restoring `PAGE_READWRITE` on the page we temporarily protected.
            if unsafe { VirtualProtect(page_addr as *const _, PAGE, PAGE_READWRITE, &mut _rev) } == 0 {
                let _ = unprotect_and_write_page(page_addr, None, None);
            }
        }
        if perm_ineligible {
            self.lazy_mark_ineligible(idx);
        }
        {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            lazy_table_remove_page(&mut table, page);
        }
        wait.finish(FlushOutcome::Accessible);
        false
    }

    /// Relay variant of [`Self::try_defer_piece`]: also refuses to defer any
    /// newly-armed piece that has pending/later CPU uploads (`dirty` / `sent`).
    pub(crate) fn try_defer_piece_relay(self: &Arc<Self>, lo: usize, hi: usize, fetched_after: u64) -> bool {
        // Lock-free fast reject BEFORE touching `card.dirty` / `card.sent` so
        // ineligible, cooled-down, partial, or unwatched pieces (~75% of pieces)
        // never block `victory-relay` behind `victory-submit::card_dirty`.
        let start = self.start as usize;
        if lo >= hi || hi > self.len as usize || (start + lo) % PAGE != 0 || hi - lo != PAGE {
            return false;
        }
        let idx = ((start + lo) / PAGE) - (start / PAGE);
        if idx >= self.lazy_state.len() {
            return false;
        }
        let cur_state = self.lazy_state[idx].load(Ordering::Acquire);
        if cur_state == LAZY_INELIGIBLE || cur_state == LAZY_RESYNC || cur_state == LAZY_FLUSHING {
            return false;
        }
        if cur_state != LAZY_DEFERRED {
            let cur_frame = crate::frame::current().min(u32::MAX as u64) as u32;
            if cur_frame > 0 {
                let cool_until = self.lazy_cool_until[idx].load(Ordering::Relaxed);
                if cool_until == u32::MAX || cur_frame < cool_until {
                    return false;
                }
            }
            if !matches!(self.watched.load(Ordering::Relaxed), 1 | GUARDED) {
                return false;
            }
        }
        if let Some(card) = &self.card {
            let touched = card.touched.load(Ordering::Acquire);
            let has_sent = card.has_sent.load(Ordering::Acquire);
            if touched || has_sent {
                let dirty_hit = touched && {
                    let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
                    dirty.iter().any(|&(a, b)| a < hi && b > lo)
                };
                if dirty_hit {
                    return false;
                }
                let sent_hit = has_sent && {
                    let sent = card.sent.lock().unwrap_or_else(|e| e.into_inner());
                    sent.has_near(fetched_after, lo, hi)
                };
                if sent_hit {
                    return false;
                }
            }
        }
        self.try_defer_piece(lo, hi)
    }

    /// Applies any unapplied bytes of a VEH-flushed 64 KiB baseline into `dst`
    /// (covering shadow offsets `[at, at + dst.len())`) before eager 3-way merge
    /// so CPU writes made after a mid-flight VEH flush are never overwritten by
    /// stale pre-GPU staging/snapshot bytes. Consumes applied byte subranges so
    /// back-to-back merges (`card_merge_relay` / `relay_merge`) are idempotent
    /// and never rewind `self.data`/`before` to an older baseline.
    fn lazy_apply_flushed_baseline(&self, at: usize, dst: &mut [u8]) {
        if dst.is_empty() {
            return;
        }
        let start = self.start as usize;
        let base = start / PAGE;
        let first = (start + at) / PAGE;
        let last = (start + at + dst.len()).div_ceil(PAGE);
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        for page in first..last {
            let idx = page - base;
            if idx >= self.lazy_state.len() || self.lazy_state[idx].load(Ordering::Acquire) != LAZY_RESYNC {
                continue;
            }
            let page_addr = page * PAGE;
            let lo = page_addr.max(start + at);
            let hi = (page_addr + PAGE).min(start + at + dst.len());
            if lo >= hi {
                continue;
            }
            let src_lo = lo - page_addr;
            let src_hi = hi - page_addr;
            let dst_off = lo - (start + at);
            let n = hi - lo;
            let remove = if let Some(baseline) = table.flushed_baselines.get_mut(&(self.id, idx)) {
                baseline.apply_once(src_lo, src_hi, &mut dst[dst_off..dst_off + n])
            } else {
                false
            };
            if remove {
                table.flushed_baselines.remove(&(self.id, idx));
            }
        }
    }

    /// Copies specific flushed pages from title memory into this shadow (used by
    /// [`Shadow::writing`] after [`lazy_flush_other_shadows`] materializes another
    /// overlapping shadow's deferred pages), skipping any page intersecting
    /// `skip_busy` (`self.pending`).
    fn resync_pages_from_title(&self, pages: &[usize], skip_busy: &[(usize, usize)]) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let base = start / PAGE;
        let watched = matches!(self.watched.load(Ordering::Relaxed), 1 | GUARDED);
        for &page in pages {
            if page < base {
                continue;
            }
            let idx = page - base;
            if idx >= self.lazy_state.len() {
                continue;
            }
            let page_lo = page * PAGE;
            let page_hi = page_lo.saturating_add(PAGE);
            if skip_busy.iter().any(|&(ba, be)| ba < page_hi && be > page_lo) {
                continue;
            }
            let p_lo = page_lo.max(start);
            let p_hi = page_hi.min(end);
            if p_lo >= p_hi {
                continue;
            }
            let whole = p_lo == page_lo && p_hi == page_hi;
            let now = if watched {
                crate::watch::writes(p_lo, p_hi, true).map_or(0, |(now, _)| now)
            } else {
                0
            };
            self.copy(p_lo, p_hi);
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            copied.set(idx, if whole { now } else { 0 });
            if !whole || now == 0 {
                self.clear_verified();
            }
        }
    }

    /// Transitions `LAZY_RESYNC` pages in `[from, to)` back to `LAZY_EAGER` /
    /// `LAZY_HOT` / `LAZY_INELIGIBLE` (skipping any page overlapping `skip_busy`)
    /// and clears their `copied` entry so [`Shadow::update_free`] copies fresh
    /// title memory (including any CPU writes made after VEH flush) into staging
    /// and schedules VRAM upload (`card_changed`).
    fn lazy_prepare_update(&self, from: usize, to: usize, skip_busy: &[(usize, usize)]) {
        if self.resync_count.load(Ordering::Acquire) == 0 || from >= to {
            return;
        }
        let base = self.start as usize / PAGE;
        let first = from / PAGE;
        let last = to.div_ceil(PAGE);
        let cur_frame = lazy_current_frame();
        let mut resynced = Vec::new();
        for page in first..last {
            let page_lo = page * PAGE;
            let page_hi = page_lo.saturating_add(PAGE);
            if skip_busy.iter().any(|&(ba, be)| ba < page_hi && be > page_lo) {
                continue;
            }
            let idx = page - base;
            if idx < self.lazy_state.len() && self.lazy_state[idx].load(Ordering::Acquire) == LAZY_RESYNC {
                let next_state = self.lazy_post_resync_state(idx, cur_frame);
                if self.lazy_state[idx]
                    .compare_exchange(LAZY_RESYNC, next_state, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    self.resync_count.fetch_sub(1, Ordering::AcqRel);
                    resynced.push(idx);
                }
            }
        }
        if resynced.is_empty() {
            return;
        }
        {
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            for &idx in &resynced {
                copied.set(idx, 0);
            }
            copied.ranged = Default::default();
            self.clear_verified();
            self.all_copied.store(copied.uncopied == 0, Ordering::Release);
        }
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        for idx in resynced {
            table.flushed_baselines.remove(&(self.id, idx));
        }
    }

    /// Resyncs `LAZY_RESYNC` pages in `[from, to)` (skipping any page
    /// overlapping `skip_busy`) directly from title memory into `self.data`
    /// (and queues `card_changed`) before a UAV-only submission declared via
    /// [`Shadow::writing`] runs without a preceding `refresh`.
    fn lazy_resync_range(&self, from: usize, to: usize, skip_busy: &[(usize, usize)]) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if from >= to {
            return;
        }
        let base = start / PAGE;
        let first = from / PAGE;
        let last = to.div_ceil(PAGE);
        let cur_frame = lazy_current_frame();
        let mut pages = Vec::new();
        for page in first..last {
            let page_lo = page * PAGE;
            let page_hi = page_lo.saturating_add(PAGE);
            if skip_busy.iter().any(|&(ba, be)| ba < page_hi && be > page_lo) {
                continue;
            }
            let idx = page - base;
            if idx < self.lazy_state.len() && self.lazy_state[idx].load(Ordering::Acquire) == LAZY_RESYNC {
                let next_state = self.lazy_post_resync_state(idx, cur_frame);
                if self.lazy_state[idx]
                    .compare_exchange(LAZY_RESYNC, next_state, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    self.resync_count.fetch_sub(1, Ordering::AcqRel);
                    pages.push((page, idx));
                }
            }
        }
        if pages.is_empty() {
            return;
        }
        {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            for &(_, idx) in &pages {
                table.flushed_baselines.remove(&(self.id, idx));
            }
        }
        let watched = matches!(self.watched.load(Ordering::Relaxed), 1 | GUARDED);
        for (page, idx) in pages {
            let p_lo = (page * PAGE).max(start);
            let p_hi = ((page + 1) * PAGE).min(end);
            if p_lo >= p_hi {
                continue;
            }
            let whole = p_lo == page * PAGE && p_hi == (page + 1) * PAGE;
            let now = if watched {
                crate::watch::writes(p_lo, p_hi, true).map_or(0, |(now, _)| now)
            } else {
                u64::from(whole)
            };
            self.copy(p_lo, p_hi);
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            copied.set(idx, if whole { now } else { 0 });
            if !whole || now == 0 {
                self.clear_verified();
            }
        }
    }

    /// Unwatched copy of `[from, to)` that skips `LAZY_DEFERRED` / `LAZY_FLUSHING`
    /// pages so GPU-resident pages are never spuriously flushed or overwritten.
    fn copy_unwatched_skipping_deferred(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if from >= to {
            return 0;
        }
        let base = start / PAGE;
        let first = from / PAGE;
        let last = to.div_ceil(PAGE);
        let mut bytes = 0;
        let mut seg_start: Option<usize> = None;
        for page in first..last {
            let idx = page - base;
            let skip = idx < self.lazy_state.len() && is_lazy_held(self.lazy_state[idx].load(Ordering::Acquire));
            let p_lo = (page * PAGE).max(from);
            let p_hi = ((page + 1) * PAGE).min(to);
            if skip {
                if let Some(s) = seg_start.take() {
                    bytes += self.copy(s, p_lo);
                }
            } else {
                seg_start.get_or_insert(p_lo);
                if page + 1 == last
                    && let Some(s) = seg_start.take()
                {
                    bytes += self.copy(s, p_hi);
                }
            }
        }
        bytes
    }

    /// Flushes every deferred page owned by this `Shadow`. Pass
    /// `save_baseline = true` from `unregister` (where in-flight lists may still
    /// hold `Arc<Shadow>` and settle via `card_merge_relay`), or `false` from
    /// `Drop::drop` (which also removes any remaining `flushed_baselines`).
    fn lazy_flush_all(&self, save_baseline: bool) {
        let first = self.start as usize / PAGE;
        let last = (self.start.saturating_add(self.len) as usize).div_ceil(PAGE);
        loop {
            enum Next {
                Done,
                Wait(Arc<FlushWait>),
                Flush(usize, usize, usize, Option<(usize, u64)>, Arc<FlushWait>),
            }
            let next = {
                let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                let found = table
                    .pages
                    .range(first..last)
                    .find_map(|(&page, entry)| match entry {
                        LazyEntry::Armed(a) if a.shadow_id == self.id => Some(page),
                        LazyEntry::Busy { shadow_id, .. } if *shadow_id == self.id => Some(page),
                        _ => None,
                    });
                match found {
                    None => Next::Done,
                    Some(page) => match table.pages.get(&page) {
                        Some(LazyEntry::Busy { wait, .. }) => Next::Wait(wait.clone()),
                        Some(LazyEntry::Armed(armed)) => {
                            let (offset, page_idx, section_view) =
                                (armed.offset, armed.page_idx, armed.section_view);
                            let wait = Arc::new(FlushWait::new());
                            self.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                            lazy_table_insert_busy(&mut table, page, self.id, wait.clone());
                            Next::Flush(page, offset, page_idx, section_view, wait)
                        }
                        None => Next::Done,
                    },
                }
            };
            match next {
                Next::Done => break,
                Next::Wait(wait) => {
                    let _ = wait.wait();
                }
                Next::Flush(page, offset, page_idx, section_view, wait) => {
                    flush_armed_page(
                        self,
                        page,
                        offset,
                        page_idx,
                        section_view,
                        &wait,
                        false,
                        false,
                        save_baseline,
                    );
                }
            }
        }
        if !save_baseline {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            table.flushed_baselines.retain(|&(sid, _), _| sid != self.id);
        }
    }

    /// Flushes every deferred page owned by this `Shadow` and removes all of its
    /// `flushed_baselines` during `Drop::drop` without relying on `Weak::upgrade()`.
    fn lazy_flush_all_on_drop(&mut self) {
        self.lazy_flush_all(false);
    }
}


/// Whether any shadow other than `my_id` currently has armed/busy lazy pages in `[from, to)`.
pub(crate) fn lazy_has_other_armed(my_id: u64, from: usize, to: usize) -> bool {
    if !lazy_may_have_other_in_range(my_id, from, to) {
        return false;
    }
    let first = from / PAGE;
    let last = to.div_ceil(PAGE);
    let table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
    table.pages.range(first..last).any(|(_, entry)| match entry {
        LazyEntry::Armed(a) => a.shadow_id != my_id,
        LazyEntry::Busy { shadow_id, .. } => *shadow_id != my_id,
    })
}

/// Periodic diagnostic line when `VICTORY_LAZY_COPYBACK=1`.
pub(crate) fn lazy_stats_line(frames: f64) {
    if !lazy_copyback() {
        return;
    }
    let [armed, hits, veh_r, veh_w, explicit, bytes] =
        LAZY_STATS.each_ref().map(|c| c.swap(0, Ordering::Relaxed));
    let live = LAZY_ARMED_COUNT.load(Ordering::Relaxed);
    tracing::info!(
        "lazy copy-back per frame (VICTORY_LAZY_COPYBACK=1): {:.1} newly armed, {:.1} steady hits,          {:.1} VEH read faults, {:.1} VEH write faults, {:.1} explicit flushes ({:.2} MB),          {live} pages armed now",
        armed as f64 / frames,
        hits as f64 / frames,
        veh_r as f64 / frames,
        veh_w as f64 / frames,
        explicit as f64 / frames,
        bytes as f64 / (1024.0 * 1024.0) / frames,
    );
}

#[cfg(test)]
mod lazy_copyback_tests {
    use super::*;

    struct VirtualBlock {
        ptr: *mut u8,
        len: usize,
        freed: bool,
    }

    impl VirtualBlock {
        fn new(len: usize, write_watch: bool) -> Self {
            use windows_sys::Win32::System::Memory::{
                MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
            };
            const MEM_WRITE_WATCH: u32 = 0x0020_0000;
            let flags = MEM_COMMIT | MEM_RESERVE | if write_watch { MEM_WRITE_WATCH } else { 0 };
            // SAFETY: fresh private virtual allocation.
            let ptr = unsafe { VirtualAlloc(std::ptr::null(), len, flags, PAGE_READWRITE) } as *mut u8;
            assert!(!ptr.is_null(), "VirtualAlloc({len}) failed");
            assert_eq!((ptr as usize) % PAGE, 0);
            Self {
                ptr,
                len,
                freed: false,
            }
        }

        fn addr(&self) -> usize {
            self.ptr as usize
        }

        fn free_raw(&mut self) {
            use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
            if !self.freed {
                self.freed = true;
                // SAFETY: freeing our own allocation.
                let ok = unsafe { VirtualFree(self.ptr.cast(), 0, MEM_RELEASE) };
                assert_ne!(ok, 0);
            }
        }
    }

    impl Drop for VirtualBlock {
        fn drop(&mut self) {
            if !self.freed {
                crate::watch::before_virtual_free(self.addr(), self.len);
                self.free_raw();
            }
        }
    }

    /// Tiny compute shader: adds `delta` (root constant 0) to every `uint` in
    /// UAV `u0` from `start_u32` (root constant 1) for `count_u32` elements
    /// (root constant 2).
    struct GpuAddPipeline {
        root: windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
        pso: windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
    }

    impl GpuAddPipeline {
        fn new(h: &crate::host::Host) -> Option<Self> {
            use windows::Win32::Graphics::Direct3D::ID3DBlob;
            use windows::Win32::Graphics::Direct3D12::*;
            const HLSL: &str = r#"
cbuffer Params : register(b0) {
    uint delta;
    uint start_u32;
    uint count_u32;
    uint mode;
};
RWByteAddressBuffer buf : register(u0);
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= count_u32) return;
    uint byte_off = (start_u32 + idx) * 4;
    if (mode == 0) {
        uint v = buf.Load(byte_off);
        buf.Store(byte_off, v + delta);
    } else {
        buf.Store(byte_off, delta ^ idx);
    }
}
"#;
            let bytecode = crate::host::compile(HLSL, windows::core::s!("main"), windows::core::s!("cs_5_0")).ok()?;
            let params = [
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                            Num32BitValues: 4,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
            ];
            let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                ..Default::default()
            };
            // SAFETY: valid root signature and compute PSO creation on `h.device`.
            unsafe {
                let mut sig_blob: Option<ID3DBlob> = None;
                let mut err_blob: Option<ID3DBlob> = None;
                D3D12SerializeRootSignature(
                    &rs_desc,
                    D3D_ROOT_SIGNATURE_VERSION_1,
                    &mut sig_blob,
                    Some(&mut err_blob),
                )
                .ok()?;
                let sig_blob = sig_blob?;
                let sig_bytes = std::slice::from_raw_parts(
                    sig_blob.GetBufferPointer().cast::<u8>(),
                    sig_blob.GetBufferSize(),
                );
                let root: ID3D12RootSignature = h.device.CreateRootSignature(0, sig_bytes).ok()?;
                let pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(root.clone())),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytecode.as_ptr().cast(),
                        BytecodeLength: bytecode.len(),
                    },
                    ..Default::default()
                };
                let pso: ID3D12PipelineState = h.device.CreateComputePipelineState(&pso_desc).ok()?;
                std::mem::ManuallyDrop::into_inner(pso_desc.pRootSignature);
                Some(Self { root, pso })
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn dispatch_on(
            &self,
            h: &crate::host::Host,
            list: &HostList,
            shadow: &Arc<Shadow>,
            start_byte: usize,
            byte_len: usize,
            delta: u32,
            mode: u32,
        ) {
            let addr = shadow.start + start_byte as u64;
            shadow.refresh(list, addr, byte_len as u64);
            let total_u32 = (byte_len / 4) as u32;
            let base_u32 = (start_byte / 4) as u32;
            // SAFETY: open command list, live pipeline, root signature, and shadow buffer.
            unsafe {
                list.list.SetComputeRootSignature(&self.root);
                list.list.SetPipelineState(&self.pso);
                list.list.SetComputeRootUnorderedAccessView(1, shadow.gpu(shadow.start));
                // Dispatch in chunks of up to 65535 * 64 u32s (~16 MiB per dispatch).
                let max_per_dispatch = 65535u32 * 64;
                let mut done = 0u32;
                while done < total_u32 {
                    let count = (total_u32 - done).min(max_per_dispatch);
                    let constants = [delta, base_u32 + done, count, mode];
                    list.list
                        .SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                    list.list.Dispatch(count.div_ceil(64), 1, 1);
                    done += count;
                }
            }
            shadow.gpu_writes(list, addr, byte_len as u64);
            list.close();
            h.execute(&[list]);
            list.reset();
        }
    }

    fn make_test_shadow(h: &crate::host::Host, addr: u64, len: u64, carded: bool) -> Option<Arc<Shadow>> {
        let size = len.next_multiple_of(PAGE as u64);
        let (staging, data) = h.shared_buffer(size)?;
        let (buffer, card) = if carded {
            let on_card = h.uav_buffer(size)?;
            CARD_BYTES[0].fetch_add(size, Ordering::Relaxed);
            let card = Card {
                staging,
                before: Mutex::default(),
                dirty: Mutex::default(),
                sent: Mutex::default(),
                touched: Default::default(),
                has_sent: Default::default(),
                size,
            };
            (on_card, Some(card))
        } else {
            (staging, None)
        };
        // SAFETY: live D3D12 buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let mut shadow = Shadow::new(buffer, data, gpu, addr, len);
        shadow.card = card;
        shadow.lazy_opted_in = true;
        let shadow = Arc::new(shadow);
        if shadow.card.is_some() {
            register_carded(&shadow);
        }
        Some(shadow)
    }

    fn page_protect(addr: usize) -> u32 {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain struct filled by `VirtualQuery`.
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: valid out pointer.
        let n = unsafe { VirtualQuery(addr as *const _, &mut mbi, size_of_val(&mbi)) };
        assert_ne!(n, 0);
        mbi.Protect
    }

    /// Live GPU test: byte-for-byte GPU->CPU fault-on-demand coherence on BOTH
    /// carded VRAM and PC-shared shadows, CPU write fault -> GPU read coherence,
    /// partial unaligned boundaries staying eager, and pinned/fence pages NEVER
    /// becoming `PAGE_NOACCESS`.
    #[test]
    fn lazy_copyback_gpu_to_cpu_fault_on_demand_coherence() {
        use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READWRITE};
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        for &carded in &[true, false] {
            // 4 aligned 64 KiB pages:
            //   page 0: GPU write -> CPU READ fault
            //   page 1: GPU write -> CPU WRITE fault -> GPU add -> CPU READ fault
            //   page 2: pinned (fence/sync range) -> MUST stay PAGE_READWRITE
            //   page 3: partial unaligned subrange -> MUST stay eager PAGE_READWRITE
            let block = VirtualBlock::new(4 * PAGE, true);
            // SAFETY: fresh writable allocation of 4 * PAGE bytes.
            unsafe {
                let words = std::slice::from_raw_parts_mut(block.ptr.cast::<u32>(), (4 * PAGE) / 4);
                for (i, w) in words.iter_mut().enumerate() {
                    *w = (i as u32).wrapping_mul(3).wrapping_add(7);
                }
            }
            // Pin page 2 BEFORE any GPU write (simulating fence polling range).
            lazy_pin_range(block.addr() + 2 * PAGE, 64);

            // Create a shadow covering 3 whole pages + half of page 3 (unaligned end!).
            let shadow_len = (3 * PAGE + PAGE / 2) as u64;
            let shadow = make_test_shadow(h, block.addr() as u64, shadow_len, carded).expect("shadow");

            // Frame 1: GPU adds +0x100 across the entire shadow.
            pipe.dispatch_on(h, &list, &shadow, 0, shadow_len as usize, 0x100, 0);

            // Verify protections:
            // - pages 0 and 1 are deferred (`PAGE_NOACCESS`),
            // - page 2 (pinned) is NEVER `PAGE_NOACCESS` and already holds GPU writes eagerly,
            // - page 3 (partial boundary) stays eager `PAGE_READWRITE` and holds GPU writes.
            assert_eq!(page_protect(block.addr()), PAGE_NOACCESS, "carded={carded} page 0");
            assert_eq!(page_protect(block.addr() + PAGE), PAGE_NOACCESS, "carded={carded} page 1");
            assert_eq!(page_protect(block.addr() + 2 * PAGE), PAGE_READWRITE, "pinned page 2");
            assert_eq!(page_protect(block.addr() + 3 * PAGE), PAGE_READWRITE, "partial page 3");

            // Verify pinned page 2 and partial page 3 eagerly received GPU writes.
            // SAFETY: verified PAGE_READWRITE above.
            unsafe {
                let p2 = std::slice::from_raw_parts((block.addr() + 2 * PAGE) as *const u32, PAGE / 4);
                for (i, &w) in p2.iter().enumerate() {
                    let idx = (2 * PAGE / 4 + i) as u32;
                    assert_eq!(w, idx.wrapping_mul(3).wrapping_add(7).wrapping_add(0x100));
                }
                let p3 = std::slice::from_raw_parts((block.addr() + 3 * PAGE) as *const u32, PAGE / 8);
                for (i, &w) in p3.iter().enumerate() {
                    let idx = (3 * PAGE / 4 + i) as u32;
                    assert_eq!(w, idx.wrapping_mul(3).wrapping_add(7).wrapping_add(0x100));
                }
            }

            // CPU READ fault on page 0 -> VEH flushes ONLY page 0 (page 1 stays PAGE_NOACCESS!).
            let read_faults_before = LAZY_STATS[2].load(Ordering::Relaxed);
            // SAFETY: VEH handles PAGE_NOACCESS and restores PAGE_READWRITE with GPU data.
            unsafe {
                let p0 = std::slice::from_raw_parts(block.ptr.cast::<u32>(), PAGE / 4);
                for (i, &w) in p0.iter().enumerate() {
                    let expected = (i as u32).wrapping_mul(3).wrapping_add(7).wrapping_add(0x100);
                    assert_eq!(w, expected, "carded={carded} page 0 word {i}");
                }
            }
            assert!(
                LAZY_STATS[2].load(Ordering::Relaxed) > read_faults_before,
                "expected VEH read fault"
            );
            assert_eq!(page_protect(block.addr()), PAGE_READWRITE);
            assert_eq!(
                page_protect(block.addr() + PAGE),
                PAGE_NOACCESS,
                "flushing page 0 must NOT flush page 1"
            );

            // CPU WRITE fault on page 1 -> VEH flushes page 1, CPU writes sentinel,
            // then Frame 2 GPU adds +0x200 -> GPU MUST observe CPU's sentinel!
            let write_faults_before = LAZY_STATS[3].load(Ordering::Relaxed);
            // SAFETY: VEH handles write fault on page 1 and restores PAGE_READWRITE.
            unsafe {
                let w_ptr = (block.addr() + PAGE + 128) as *mut u32;
                w_ptr.write_volatile(0xCAFE_0000);
            }
            assert!(
                LAZY_STATS[3].load(Ordering::Relaxed) > write_faults_before,
                "expected VEH write fault"
            );
            assert_eq!(page_protect(block.addr() + PAGE), PAGE_READWRITE);

            // Frame 2: GPU adds +0x200 to all pages.
            pipe.dispatch_on(h, &list, &shadow, 0, shadow_len as usize, 0x200, 0);
            assert_eq!(page_protect(block.addr()), PAGE_NOACCESS, "page 0 re-armed");
            assert_eq!(page_protect(block.addr() + PAGE), PAGE_NOACCESS, "page 1 re-armed");

            // Read page 1: word at byte offset 128 MUST be 0xCAFE_0000 + 0x200,
            // and other words in page 1 MUST be initial + 0x100 + 0x200!
            // SAFETY: VEH handles read fault on page 1.
            unsafe {
                let p1 = std::slice::from_raw_parts((block.addr() + PAGE) as *const u32, PAGE / 4);
                for (i, &w) in p1.iter().enumerate() {
                    let expected = if i == 32 {
                        0xCAFE_0200
                    } else {
                        let idx = (PAGE / 4 + i) as u32;
                        idx.wrapping_mul(3).wrapping_add(7).wrapping_add(0x300)
                    };
                    assert_eq!(w, expected, "carded={carded} page 1 word {i}");
                }
            }
            drop(shadow);
        }
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// Live GPU test: GPU->GPU steady-state zero-readback invariance.
    /// Runs 10 consecutive GPU read-modify-write compute frames on a carded VRAM
    /// shadow with `VICTORY_LAZY_COPYBACK=1` and asserts ZERO PCIe readback bytes,
    /// ZERO PCIe re-upload bytes, and ZERO VEH faults in frames 2..=10, then
    /// verifies the exact accumulated 10-frame result on a single CPU read.
    #[test]
    fn lazy_copyback_gpu_to_gpu_zero_readback_invariance() {
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        const PAGES: usize = 32; // 2 MiB carded VRAM buffer
        const LEN: usize = PAGES * PAGE;
        let block = VirtualBlock::new(LEN, true);
        // SAFETY: writable allocation of `LEN` bytes.
        unsafe {
            let words = std::slice::from_raw_parts_mut(block.ptr.cast::<u32>(), LEN / 4);
            for (i, w) in words.iter_mut().enumerate() {
                *w = i as u32;
            }
        }
        let shadow = make_test_shadow(h, block.addr() as u64, LEN as u64, true).expect("shadow");

        // Frame 1 seeds VRAM and arms all 32 pages as LAZY_DEFERRED.
        pipe.dispatch_on(h, &list, &shadow, 0, LEN, 1, 0);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), PAGES);

        // Snapshot counters before steady-state frames 2..=10.
        let upload_before = CARD_STATS[0].load(Ordering::Relaxed);
        let fetched_pieces_before = CARD_STATS[3].load(Ordering::Relaxed);
        let fetched_bytes_before = CARD_STATS[4].load(Ordering::Relaxed);
        let veh_reads_before = LAZY_STATS[2].load(Ordering::Relaxed);
        let veh_writes_before = LAZY_STATS[3].load(Ordering::Relaxed);
        let hits_before = LAZY_STATS[1].load(Ordering::Relaxed);

        for _ in 0..9 {
            pipe.dispatch_on(h, &list, &shadow, 0, LEN, 1, 0);
        }

        assert_eq!(
            CARD_STATS[0].load(Ordering::Relaxed) - upload_before,
            0,
            "steady-state GPU->GPU must do ZERO PCIe uploads"
        );
        assert_eq!(
            CARD_STATS[3].load(Ordering::Relaxed) - fetched_pieces_before,
            0,
            "steady-state GPU->GPU must fetch ZERO pieces over PCIe"
        );
        assert_eq!(
            CARD_STATS[4].load(Ordering::Relaxed) - fetched_bytes_before,
            0,
            "steady-state GPU->GPU must read back ZERO bytes over PCIe"
        );
        assert_eq!(LAZY_STATS[2].load(Ordering::Relaxed) - veh_reads_before, 0);
        assert_eq!(LAZY_STATS[3].load(Ordering::Relaxed) - veh_writes_before, 0);
        assert_eq!(
            LAZY_STATS[1].load(Ordering::Relaxed) - hits_before,
            (9 * PAGES) as u64,
            "every steady-state piece must hit the O(1) already-deferred fast path"
        );

        // Final CPU read faults every page on demand and sees exact initial + 10.
        // SAFETY: VEH flushes each page on demand.
        unsafe {
            let words = std::slice::from_raw_parts(block.ptr.cast::<u32>(), LEN / 4);
            for (i, &w) in words.iter().enumerate() {
                assert_eq!(w, i as u32 + 10, "word {i}");
            }
        }
        drop(shadow);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    struct MappedSectionBlock {
        section: usize,
        ptr: *mut u8,
        len: usize,
    }

    impl MappedSectionBlock {
        fn new(len: usize) -> Self {
            use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
            use windows_sys::Win32::System::Memory::{
                CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE,
            };
            // SAFETY: fresh pagefile section mapped into a single writable view,
            // registered with `inplace` just like `xmem::XMemAllocatePhysicalPages`.
            unsafe {
                let section = CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    std::ptr::null(),
                    PAGE_READWRITE,
                    (len as u64 >> 32) as u32,
                    len as u32,
                    std::ptr::null(),
                );
                assert!(!section.is_null(), "CreateFileMappingW({len}) failed");
                let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len);
                assert!(!view.Value.is_null(), "MapViewOfFile({len}) failed");
                let addr = view.Value as usize;
                assert_eq!(addr % PAGE, 0);
                crate::inplace::title_memory_section(section as usize, len as u64, true);
                crate::inplace::title_memory_view(addr, len, section as usize, 0);
                Self {
                    section: section as usize,
                    ptr: view.Value.cast(),
                    len,
                }
            }
        }

        fn addr(&self) -> usize {
            self.ptr as usize
        }
    }

    impl Drop for MappedSectionBlock {
        fn drop(&mut self) {
            use windows_sys::Win32::Foundation::CloseHandle;
            use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
            crate::watch::title_memory_remapped(self.addr(), self.len);
            crate::watch::before_virtual_free(self.addr(), self.len);
            // SAFETY: unmapping our view and closing our section handle.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: self.ptr.cast(),
                });
                CloseHandle(self.section as *mut std::ffi::c_void);
            }
        }
    }

    /// Live GPU stress test: 8 threads concurrently faulting (reads AND writes)
    /// on the SAME and overlapping deferred 64 KiB pieces across multiple rounds:
    /// 1. Section-backed `MEM_MAPPED` (Xbox physical GPU heap model) with atomic
    ///    dual-view flush (`MapViewOfFile` -> `memcpy` -> `UnmapViewOfFile` ->
    ///    `VirtualProtect`): zero torn-read window across staggered multi-page
    ///    waves of 8 colliding threads over 8 rounds.
    /// 2. `MEM_PRIVATE | MEM_WRITE_WATCH` concurrent 8-thread collision on the
    ///    same page exercising `FlushWait` serialization.
    #[test]
    fn lazy_copyback_multithread_concurrent_fault_stress() {
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        const PAGES: usize = 8;
        const LEN: usize = PAGES * PAGE;
        const THREADS: usize = 8;

        // Part 1: Section-backed `MEM_MAPPED` atomic dual-view stress (0 ns tear window).
        {
            let block = MappedSectionBlock::new(LEN);
            let base_addr = block.addr();
            let shadow = make_test_shadow(h, base_addr as u64, LEN as u64, true).expect("shadow");

            // Initial seed so `watch::guard` protects all 8 pages cleanly.
            pipe.dispatch_on(h, &list, &shadow, 0, LEN, 0, 1);

            for round in 1u32..=8 {
                let tag = round.wrapping_mul(0x1111_0001);
                pipe.dispatch_on(h, &list, &shadow, 0, LEN, tag, 1);
                assert_eq!(shadow.deferred_count.load(Ordering::Acquire), PAGES, "mapped round={round}");

                let barrier = std::sync::Barrier::new(THREADS);
                std::thread::scope(|scope| {
                    for tid in 0..THREADS {
                        let barrier = &barrier;
                        scope.spawn(move || {
                            barrier.wait();
                            for p in 0..PAGES {
                                let page_idx = (p + (tid % 2)) % PAGES;
                                let page_ptr = (base_addr + page_idx * PAGE) as *mut u32;
                                for sample in [0usize, 17, 511, 4096, 12_345, PAGE / 4 - 1] {
                                    let global_u32 = (page_idx * (PAGE / 4) + sample) as u32;
                                    // SAFETY: VEH atomic dual-view flushes page before returning.
                                    let got = unsafe { page_ptr.add(sample).read_volatile() };
                                    assert_eq!(
                                        got,
                                        tag ^ global_u32,
                                        "mapped round={round} tid={tid} page={page_idx} sample={sample}"
                                    );
                                }
                                let slot = 100 + tid;
                                let marker = 0xDEAD_0000 ^ (round << 8) ^ (page_idx as u32) ^ (tid as u32);
                                // SAFETY: disjoint word per thread within committed page.
                                unsafe { page_ptr.add(slot).write_volatile(marker) };
                            }
                        });
                    }
                });

                for p in 0..PAGES {
                    let page_ptr = (base_addr + p * PAGE) as *const u32;
                    for tid in 0..THREADS {
                        let marker = 0xDEAD_0000 ^ (round << 8) ^ (p as u32) ^ (tid as u32);
                        // SAFETY: all pages are now PAGE_READWRITE.
                        let got = unsafe { page_ptr.add(100 + tid).read_volatile() };
                        assert_eq!(got, marker, "mapped round={round} page={p} tid={tid} write lost");
                    }
                }
            }
            drop(shadow);
        }

        // Part 2: `MEM_PRIVATE | MEM_WRITE_WATCH` synchronized 8-thread collision on the same page.
        {
            let block = VirtualBlock::new(LEN, true);
            let base_addr = block.addr();
            // SAFETY: fresh writable allocation of `LEN` bytes.
            unsafe {
                std::ptr::write_bytes(block.ptr, 0x11, LEN);
            }
            let shadow = make_test_shadow(h, base_addr as u64, LEN as u64, true).expect("shadow");
            for round in 1u32..=4 {
                let tag = round.wrapping_mul(0x2222_0003);
                pipe.dispatch_on(h, &list, &shadow, 0, LEN, tag, 1);
                assert_eq!(shadow.deferred_count.load(Ordering::Acquire), PAGES, "private round={round}");

                for page_idx in 0..PAGES {
                    let barrier = std::sync::Barrier::new(THREADS);
                    std::thread::scope(|scope| {
                        for tid in 0..THREADS {
                            let barrier = &barrier;
                            scope.spawn(move || {
                                barrier.wait();
                                let page_ptr = (base_addr + page_idx * PAGE) as *mut u32;
                                // First touch collides all 8 threads on `PAGE_NOACCESS` during GPU readback.
                                let first = unsafe { page_ptr.read_volatile() };
                                assert_eq!(first, tag ^ (page_idx * (PAGE / 4)) as u32);
                                for sample in [17usize, 511, 4096, 12_345, PAGE / 4 - 1] {
                                    let global_u32 = (page_idx * (PAGE / 4) + sample) as u32;
                                    // SAFETY: page is already flushed to PAGE_READWRITE.
                                    let got = unsafe { page_ptr.add(sample).read_volatile() };
                                    assert_eq!(got, tag ^ global_u32);
                                }
                                let slot = 100 + tid;
                                let marker = 0xBEEF_0000 ^ (round << 8) ^ (page_idx as u32) ^ (tid as u32);
                                // SAFETY: disjoint word per thread.
                                unsafe { page_ptr.add(slot).write_volatile(marker) };
                            });
                        }
                    });
                }
            }
            drop(shadow);
        }

        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// Live GPU lifecycle test: `title_memory_remapped`, `before_virtual_protect`,
    /// `before_virtual_free`, `shadow::unregister`, `Shadow::drop`, AND unnotified
    /// raw `VirtualFree` + `VirtualAlloc` reuse. Asserts exact data preservation
    /// and `LAZY_ARMED_COUNT == 0` after teardown.
    #[test]
    fn lazy_copyback_remap_drop_unregister_virtual_lifecycle() {
        use windows_sys::Win32::System::Memory::PAGE_READWRITE;
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        // 1. Virgin demand-zero `MEM_WRITE_WATCH` stays safely eager on Frame 1
        //    (no `PAGE_READONLY` downgrade during arming!), arms cleanly on Frame 2,
        //    then exercises `title_memory_remapped`, `before_virtual_protect`, and `Shadow::drop`.
        {
            let block = VirtualBlock::new(3 * PAGE, true);
            let shadow = make_test_shadow(h, block.addr() as u64, (3 * PAGE) as u64, true).expect("shadow");
            // Frame 1 on untouched virgin demand-zero pages: stays safely eager.
            pipe.dispatch_on(h, &list, &shadow, 0, 3 * PAGE, 0x1111_0000, 1);
            assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 0);
            // Frame 2: pages are now resident and clean -> arms all 3 with `GetWriteWatch(0) == 0`.
            pipe.dispatch_on(h, &list, &shadow, 0, 3 * PAGE, 0xABCD_1234, 1);
            assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 3);

            // Page 0: `title_memory_remapped` flushes GPU bytes & restores PAGE_READWRITE.
            crate::watch::title_memory_remapped(block.addr(), PAGE);
            assert_eq!(page_protect(block.addr()), PAGE_READWRITE);
            // SAFETY: flushed to PAGE_READWRITE above.
            assert_eq!(unsafe { *(block.ptr.cast::<u32>()) }, 0xABCD_1234);

            // Page 1: `before_virtual_protect` flushes GPU bytes & restores PAGE_READWRITE.
            crate::watch::before_virtual_protect(block.addr() + PAGE, PAGE);
            assert_eq!(page_protect(block.addr() + PAGE), PAGE_READWRITE);
            // SAFETY: flushed to PAGE_READWRITE above.
            assert_eq!(
                unsafe { *((block.addr() + PAGE) as *const u32) },
                0xABCD_1234 ^ (PAGE / 4) as u32
            );

            // Page 2: `Shadow::drop` flushes remaining deferred page & restores PAGE_READWRITE.
            drop(shadow);
            assert_eq!(page_protect(block.addr() + 2 * PAGE), PAGE_READWRITE);
            // SAFETY: flushed by `Shadow::drop` above.
            assert_eq!(
                unsafe { *((block.addr() + 2 * PAGE) as *const u32) },
                0xABCD_1234 ^ (2 * PAGE / 4) as u32
            );
        }

        // 2. `shadow::unregister` flushes deferred pages before title frees buffer.
        {
            let block = VirtualBlock::new(PAGE, true);
            // SAFETY: seed physical page before first dispatch.
            unsafe { std::ptr::write_bytes(block.ptr, 0x22, PAGE) };
            let fake_res = block.addr();
            register(fake_res, block.addr() as u64, PAGE as u64);
            let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");
            buffers()
                .get_mut(&(block.addr() as u64, fake_res))
                .expect("registered")
                .shadow = Some(shadow.clone());
            pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x55AA_0000, 1);
            assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);

            unregister(fake_res, block.addr() as u64);
            assert_eq!(page_protect(block.addr()), PAGE_READWRITE);
            // SAFETY: flushed by `unregister`.
            assert_eq!(unsafe { *(block.ptr.cast::<u32>()) }, 0x55AA_0000);
            drop(shadow);
        }

        // 3. `before_virtual_free` cleanly drops deferred protection without crashing.
        {
            let mut block = VirtualBlock::new(PAGE, true);
            // SAFETY: seed physical page before first dispatch.
            unsafe { std::ptr::write_bytes(block.ptr, 0x33, PAGE) };
            let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");
            pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x3333_0000, 1);
            assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);
            crate::watch::before_virtual_free(block.addr(), PAGE);
            assert_eq!(page_protect(block.addr()), PAGE_READWRITE);
            block.free_raw();
            drop(shadow);
        }

        // 4. Unnotified raw `VirtualFree(..., MEM_DECOMMIT)` followed by
        //    `Shadow::drop` MUST NOT crash or clobber memory while VA stays reserved!
        {
            use windows_sys::Win32::System::Memory::{MEM_DECOMMIT, VirtualFree};
            let block = VirtualBlock::new(PAGE, true);
            // SAFETY: seed physical page before first dispatch.
            unsafe { std::ptr::write_bytes(block.ptr, 0x44, PAGE) };
            let raw_addr = block.addr();
            let shadow = make_test_shadow(h, raw_addr as u64, PAGE as u64, true).expect("shadow");
            pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x7777_0000, 1);
            assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);
            // SAFETY: decommit without notifying `watch` while VA stays reserved.
            assert_ne!(unsafe { VirtualFree(block.ptr.cast(), PAGE, MEM_DECOMMIT) }, 0);
            drop(shadow);
            crate::watch::before_virtual_free(raw_addr, PAGE);
        }

        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// F1 regression: DirectStorage writes directly into backing physical section
    /// pages and then calls `watch::title_memory_written(at, len, true)`.
    /// Verifies:
    /// - Full-page write preserves 100% of DirectStorage bytes (never clobbered by GPU),
    /// - Partial-page write preserves DirectStorage bytes in `[w_lo, w_hi)` AND
    ///   restores exact GPU bytes in the unwritten complement `[0..w_lo)` & `[w_hi..PAGE)`,
    /// - `pages == false` also releases deferred pages even when `page_writes() & 4 == 0`,
    /// - Next GPU submission sees the exact DirectStorage + complement bytes.
    #[test]
    fn lazy_copyback_f1_directstorage_title_memory_written_preserves_full_and_partial() {
        use windows_sys::Win32::System::Memory::{
            FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_NOACCESS,
            PAGE_READWRITE, UnmapViewOfFile,
        };
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        const PAGES: usize = 3;
        const LEN: usize = PAGES * PAGE;
        let block = MappedSectionBlock::new(LEN);
        let shadow = make_test_shadow(h, block.addr() as u64, LEN as u64, true).expect("shadow");

        // GPU fills all 3 pages with `0x1000_0000 ^ idx` and arms them `PAGE_NOACCESS`.
        pipe.dispatch_on(h, &list, &shadow, 0, LEN, 0x1000_0000, 1);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), PAGES);
        for p in 0..PAGES {
            assert_eq!(page_protect(block.addr() + p * PAGE), PAGE_NOACCESS);
        }

        // Write DirectStorage bytes via a separate temporary alias view of the
        // backing section (exactly like `xmem::PageRuns::write`):
        //   - Page 0: FULL 64 KiB overwritten with `0xD000_0000 + i`
        //   - Page 1: PARTIAL middle `[16 KiB..48 KiB)` overwritten with `0xD100_0000 + i`
        //   - Page 2: FULL 64 KiB overwritten with `0xD200_0000 + i` (tested with `pages = false`)
        // SAFETY: temporary writable view of our live pagefile section.
        unsafe {
            let alias = MapViewOfFile(block.section as *mut std::ffi::c_void, FILE_MAP_WRITE, 0, 0, LEN);
            assert!(!alias.Value.is_null());
            let words = std::slice::from_raw_parts_mut(alias.Value.cast::<u32>(), LEN / 4);
            for (i, w) in words[0..PAGE / 4].iter_mut().enumerate() {
                *w = 0xD000_0000 + i as u32;
            }
            for (i, w) in words[(PAGE + 16_384) / 4..(PAGE + 49_152) / 4].iter_mut().enumerate() {
                *w = 0xD100_0000 + i as u32;
            }
            for (i, w) in words[2 * PAGE / 4..3 * PAGE / 4].iter_mut().enumerate() {
                *w = 0xD200_0000 + i as u32;
            }
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: alias.Value });
        }

        crate::watch::title_memory_written(block.addr(), PAGE, true);
        crate::watch::title_memory_written(block.addr() + PAGE + 16_384, 32_768, true);
        crate::watch::title_memory_written(block.addr() + 2 * PAGE, PAGE, false);

        for p in 0..PAGES {
            assert_eq!(page_protect(block.addr() + p * PAGE), PAGE_READWRITE);
        }
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 0);

        // Verify page 0 (full DirectStorage), page 1 (GPU complement + DirectStorage middle),
        // and page 2 (`pages = false` full DirectStorage).
        // SAFETY: all 3 pages are `PAGE_READWRITE`.
        unsafe {
            let p0 = std::slice::from_raw_parts(block.ptr.cast::<u32>(), PAGE / 4);
            for (i, &w) in p0.iter().enumerate() {
                assert_eq!(w, 0xD000_0000 + i as u32, "page 0 word {i}");
            }
            let p1 = std::slice::from_raw_parts((block.addr() + PAGE) as *const u32, PAGE / 4);
            for (i, &w) in p1.iter().enumerate() {
                let expected = if (4096..12_288).contains(&i) {
                    0xD100_0000 + (i - 4096) as u32
                } else {
                    0x1000_0000 ^ (PAGE / 4 + i) as u32
                };
                assert_eq!(w, expected, "page 1 word {i}");
            }
            let p2 = std::slice::from_raw_parts((block.addr() + 2 * PAGE) as *const u32, PAGE / 4);
            for (i, &w) in p2.iter().enumerate() {
                assert_eq!(w, 0xD200_0000 + i as u32, "page 2 word {i}");
            }
        }

        // Next GPU dispatch adds +1 (`mode = 0`): GPU MUST observe the exact DirectStorage
        // + complement bytes, NOT stale pre-DirectStorage GPU bytes!
        pipe.dispatch_on(h, &list, &shadow, 0, LEN, 1, 0);
        // SAFETY: VEH demand-flushes on read.
        unsafe {
            assert_eq!(*block.ptr.cast::<u32>(), 0xD000_0001);
            assert_eq!(*((block.addr() + PAGE) as *const u32), (0x1000_0000 ^ (PAGE / 4) as u32) + 1);
            assert_eq!(*((block.addr() + PAGE + 16_384) as *const u32), 0xD100_0001);
            assert_eq!(*((block.addr() + 2 * PAGE) as *const u32), 0xD200_0001);
        }

        drop(shadow);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// F2 regression: `Shadow::merge_under_writes` and `Shadow::stale_under_writes`
    /// MUST return `true` (forcing clean flight settle) for any `LAZY_RESYNC` page
    /// under pending GPU writes, instead of diffing VEH-flushed GPU bytes `B`
    /// against stale staging `A` and queuing bogus `card_changed` uploads.
    #[test]
    fn lazy_copyback_f2_merge_under_writes_settles_resync_page() {
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        let block = VirtualBlock::new(PAGE, true);
        // SAFETY: seed initial bytes `A`.
        unsafe { std::ptr::write_bytes(block.ptr, 0x11, PAGE) };
        let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");

        // Flight 1 writes `B` and arms `LAZY_DEFERRED` (`self.data` still holds `A`).
        pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x2222_0000, 1);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);

        // Simulate Flight 2 in-flight (`pending` covers page).
        shadow.writing(block.addr(), PAGE);
        assert!(shadow.writes_pending());

        // CPU read faults via VEH -> flushes `B` to title, sets `LAZY_RESYNC`,
        // keeps `self.data` at `A`.
        // SAFETY: VEH demand-flushes page.
        let got = unsafe { *block.ptr.cast::<u32>() };
        assert_eq!(got, 0x2222_0000);
        assert_eq!(shadow.resync_count.load(Ordering::Acquire), 1);

        let card_dirty_before = !shadow.card.as_ref().unwrap().dirty.lock().unwrap_or_else(|e| e.into_inner()).is_empty();
        assert!(!card_dirty_before);

        // Flight 3 checks `stale_under_writes` and `merge_under_writes`:
        // BOTH must return `true` (settle) and must NOT dirty `card`!
        assert!(shadow.stale_under_writes(block.addr(), PAGE));
        let mut runs_left = 1024;
        assert!(shadow.merge_under_writes(block.addr(), PAGE, &mut runs_left));
        assert!(
            !!shadow.card.as_ref().unwrap().dirty.lock().unwrap_or_else(|e| e.into_inner()).is_empty(),
            "merge_under_writes must NOT upload stale Flight 1 bytes over in-flight Flight 2"
        );

        shadow.copy_back_done(&[(block.addr(), block.addr() + PAGE)]);
        drop(shadow);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// F3 regression: `lazy_apply_flushed_baseline` MUST be idempotent across
    /// back-to-back `card_merge_relay` flights on a `LAZY_RESYNC` page (never
    /// rewinding `self.data` to the stale VEH baseline on Flight 3), AND
    /// `Shadow::drop` MUST clean up `LAZY_TABLE.flushed_baselines` even when
    /// `deferred_count == 0 && resync_count > 0`.
    #[test]
    fn lazy_copyback_f3_idempotent_baseline_and_drop_cleanup() {
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        let block = VirtualBlock::new(PAGE, true);
        // SAFETY: seed initial bytes `A`.
        unsafe { std::ptr::write_bytes(block.ptr, 0x11, PAGE) };
        let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");
        let shadow_id = shadow.id;

        // Flight 1 writes `B` (`0x3333_0000 ^ i`) and defers page.
        pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x3333_0000, 1);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);

        // VEH read fault flushes `B` to title and records baseline `B` in `LAZY_TABLE`.
        // SAFETY: VEH demand-flushes page.
        assert_eq!(unsafe { *block.ptr.cast::<u32>() }, 0x3333_0000);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 0);
        assert_eq!(shadow.resync_count.load(Ordering::Acquire), 1);

        // Capture `B` snapshot, then simulate consecutive `card_merge_relay` landings:
        //   - Flight 2 GPU changes word 10 to `0x2222_2222`
        //   - CPU writes word 20 to `0xCCCC_CCCC`
        //   - Flight 3 GPU changes word 30 to `0x4444_4444` (still holding word 10 = `0x2222_2222`)
        // SAFETY: page is `PAGE_READWRITE`.
        let mut flight2_gpu = unsafe { std::slice::from_raw_parts(block.ptr, PAGE) }.to_vec();
        flight2_gpu[40..44].copy_from_slice(&0x2222_2222u32.to_le_bytes());
        let mut scratch = Vec::new();
        let _ = shadow.card_merge_relay(0, &flight2_gpu, 1, &mut scratch);

        // CPU modifies word 10 AFTER Flight 2 landed (word 10 -> `0xAAAA_AAAA`) and word 20 -> `0xCCCC_CCCC`.
        // SAFETY: page is `PAGE_READWRITE`.
        unsafe {
            block.ptr.cast::<u32>().add(10).write_volatile(0xAAAA_AAAA);
            block.ptr.cast::<u32>().add(20).write_volatile(0xCCCC_CCCC);
        }

        // Flight 3 lands with Flight 2's word 10 (`0x2222_2222`) + new GPU word 30 (`0x4444_4444`).
        // If `lazy_apply_flushed_baseline` re-applied `B`, `held[10]` would rewind to `B[10]`
        // and Flight 3 would falsely think GPU changed word 10 and clobber CPU's `0xAAAA_AAAA`!
        let mut flight3_gpu = flight2_gpu.clone();
        flight3_gpu[120..124].copy_from_slice(&0x4444_4444u32.to_le_bytes());
        let _ = shadow.card_merge_relay(0, &flight3_gpu, 2, &mut scratch);

        // SAFETY: page is `PAGE_READWRITE`.
        unsafe {
            assert_eq!(
                block.ptr.cast::<u32>().add(10).read_volatile(),
                0xAAAA_AAAA,
                "CPU write to word 10 must NOT be clobbered by non-idempotent baseline rewind"
            );
            assert_eq!(block.ptr.cast::<u32>().add(20).read_volatile(), 0xCCCC_CCCC);
            assert_eq!(block.ptr.cast::<u32>().add(30).read_volatile(), 0x4444_4444);
        }

        // Also test `Shadow::drop` with `deferred_count == 0 && resync_count > 0` cleans up
        // `LAZY_TABLE.flushed_baselines` (zero leaked 64 KiB baseline entries).
        drop(shadow);
        {
            let block2 = VirtualBlock::new(PAGE, true);
            // SAFETY: seed physical page.
            unsafe { std::ptr::write_bytes(block2.ptr, 0x55, PAGE) };
            let shadow2 = make_test_shadow(h, block2.addr() as u64, PAGE as u64, true).expect("shadow2");
            let sid2 = shadow2.id;
            pipe.dispatch_on(h, &list, &shadow2, 0, PAGE, 0x6666_0000, 1);
            // Trigger VEH flush -> `deferred_count == 0`, `resync_count == 1`, baseline present.
            // SAFETY: VEH demand-flushes page.
            assert_eq!(unsafe { *block2.ptr.cast::<u32>() }, 0x6666_0000);
            assert_eq!(shadow2.deferred_count.load(Ordering::Acquire), 0);
            assert_eq!(shadow2.resync_count.load(Ordering::Acquire), 1);
            assert!(
                LAZY_TABLE
                    .lock()
                    .unwrap()
                    .flushed_baselines
                    .contains_key(&(sid2, 0))
            );
            drop(shadow2);
            assert!(
                !LAZY_TABLE
                    .lock()
                    .unwrap()
                    .flushed_baselines
                    .keys()
                    .any(|&(sid, _)| sid == sid2 || sid == shadow_id),
                "Shadow::drop leaked flushed_baselines entry"
            );
        }
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// F4 regression: overlapping `Shadow2` (`Shadow::exact` / nested buffer)
    /// MUST observe GPU writes deferred by `Shadow1` when `Shadow2::update` runs
    /// (even with `watch::ask_once` active and `Shadow2.copied[0] > 0`).
    #[test]
    fn lazy_copyback_f4_overlapping_shadow_coherence() {
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        const LEN: usize = 2 * PAGE;
        let block = MappedSectionBlock::new(LEN);
        // SAFETY: seed initial bytes.
        unsafe { std::ptr::write_bytes(block.ptr, 0x11, LEN) };
        let shadow1 = make_test_shadow(h, block.addr() as u64, LEN as u64, true).expect("shadow1");
        let shadow2 = make_test_shadow(h, block.addr() as u64, PAGE as u64, false).expect("shadow2");

        // Initialize `shadow2` once so `shadow2.copied[0] > 0` and `shadow2.deferred_count == 0`.
        assert_eq!(shadow2.update(block.addr(), PAGE), PAGE);

        // GPU writes `0x7777_0000 ^ idx` via `shadow1` -> arms page 0 & 1 under `shadow1`.
        pipe.dispatch_on(h, &list, &shadow1, 0, LEN, 0x7777_0000, 1);
        assert_eq!(shadow1.deferred_count.load(Ordering::Acquire), 2);
        assert_eq!(shadow2.deferred_count.load(Ordering::Acquire), 0);

        // Simulate `watch::ask_once` over `[block, block+2*PAGE)` BEFORE `shadow2.update`.
        crate::watch::ask_once(&[
            (block.addr(), block.addr() + PAGE),
            (block.addr() + PAGE, block.addr() + 2 * PAGE),
        ]);
        let copied = shadow2.update(block.addr(), PAGE);
        crate::watch::ask_once_end();
        assert_eq!(copied, PAGE, "overlapping shadow2 must flush shadow1 and copy fresh GPU bytes");

        // Verify `shadow2.data` holds the exact GPU bytes written via `shadow1`!
        // SAFETY: `shadow2.data` is mapped for `PAGE` bytes.
        unsafe {
            let s2_words = std::slice::from_raw_parts(shadow2.data.cast::<u32>(), PAGE / 4);
            for (i, &w) in s2_words.iter().enumerate() {
                assert_eq!(w, 0x7777_0000 ^ i as u32, "shadow2 word {i}");
            }
        }
        // Page 1 (not covered by `shadow2`) remains deferred under `shadow1`.
        assert_eq!(shadow1.deferred_count.load(Ordering::Acquire), 1);

        drop(shadow2);
        drop(shadow1);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// F5 regression:
    /// 1. Holding `watch::title_memory_remapping(at, len)` flushes deferred pages
    ///    AND blocks concurrent `try_defer_piece` re-arming until `Remapping::drop`.
    /// 2. `shadow::unregister` flushes deferred pages with `save_baseline = true`
    ///    so an in-flight `Arc<Shadow>` landing `card_merge_relay` afterward never
    ///    clobbers subsequent CPU writes using stale pre-deferred staging bytes.
    #[test]
    fn lazy_copyback_f5_remapping_blocks_rearm_and_unregister_preserves_baseline() {
        use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READWRITE};
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        // Part 1: Active `Remapping` blocks re-arming until dropped.
        {
            let block = MappedSectionBlock::new(PAGE);
            let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");
            pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x8888_0000, 1);
            assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);
            assert_eq!(page_protect(block.addr()), PAGE_NOACCESS);

            let remap = crate::watch::title_memory_remapping();
            crate::watch::title_memory_unmapping(block.addr(), PAGE);
            assert_eq!(page_protect(block.addr()), PAGE_READWRITE);
            assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 0);

            // Simulate a concurrent relay/submit thread trying to re-arm while `remap` is held.
            pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x8888_1111, 1);
            assert_eq!(
                shadow.deferred_count.load(Ordering::Acquire),
                0,
                "active Remapping MUST block re-arming"
            );
            assert!(!shadow.try_defer_piece(0, PAGE));
            assert_eq!(page_protect(block.addr()), PAGE_READWRITE);

            drop(remap);
            // After `Remapping::drop`, the next GPU write arms cleanly again.
            pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x8888_2222, 1);
            assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);
            assert_eq!(page_protect(block.addr()), PAGE_NOACCESS);
            drop(shadow);
        }

        // Part 2: `shadow::unregister` preserves flushed baseline for in-flight `Arc<Shadow>`.
        {
            let block = VirtualBlock::new(PAGE, true);
            // SAFETY: seed initial bytes `A`.
            unsafe { std::ptr::write_bytes(block.ptr, 0x11, PAGE) };
            let fake_res = block.addr();
            register(fake_res, block.addr() as u64, PAGE as u64);
            let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");
            buffers()
                .get_mut(&(block.addr() as u64, fake_res))
                .expect("registered")
                .shadow = Some(shadow.clone());

            pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x9999_0000, 1);
            assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);

            // Unregister while `shadow` (`Arc<Shadow>`) is still held by an in-flight list.
            unregister(fake_res, block.addr() as u64);
            assert_eq!(page_protect(block.addr()), PAGE_READWRITE);
            assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 0);
            assert_eq!(shadow.resync_count.load(Ordering::Acquire), 1);

            // Snapshot flushed GPU bytes `B`, then CPU writes word 5 = `0xF00D_BEEF`.
            // SAFETY: flushed to `PAGE_READWRITE` by `unregister`.
            let gpu_b = unsafe { std::slice::from_raw_parts(block.ptr, PAGE) }.to_vec();
            unsafe { block.ptr.cast::<u32>().add(5).write_volatile(0xF00D_BEEF) };

            // In-flight `card_merge_relay` lands with `gpu_b`: MUST see `gpu_b == baseline`
            // and MUST NOT overwrite CPU's `0xF00D_BEEF`!
            let mut scratch = Vec::new();
            let written = shadow.card_merge_relay(0, &gpu_b, 1, &mut scratch);
            assert_eq!(written, 0);
            // SAFETY: page is `PAGE_READWRITE`.
            assert_eq!(unsafe { block.ptr.cast::<u32>().add(5).read_volatile() }, 0xF00D_BEEF);
            drop(shadow);
        }

        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// Regression: when `transition_clean_guard_to_noaccess` succeeds on a
    /// `MEM_MAPPED` page and `watch::was_written_since(page, copied_at)`
    /// subsequently rejects arming (e.g., concurrent `watch::wrote` advancing
    /// the journal stamp without removing `GUARDS`), `try_defer_piece` MUST
    /// roll the page back to `PAGE_READWRITE` and leave `LAZY_ARMED_COUNT == 0`.
    #[test]
    fn lazy_copyback_mem_mapped_write_race_rolls_back_to_readwrite() {
        use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE};

        struct AskOnceGuard;
        impl Drop for AskOnceGuard {
            fn drop(&mut self) {
                crate::watch::ask_once_end();
            }
        }

        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }

        let block = MappedSectionBlock::new(PAGE);
        // SAFETY: seed initial bytes inside live mapped view.
        unsafe { std::ptr::write_bytes(block.ptr, 0x3A, PAGE) };
        let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");
        let page = block.addr() / PAGE;

        // Initialize shadow and arm `watch`'s clean `PAGE_READONLY` guard.
        let _ = shadow.update(block.addr(), PAGE);
        let _ = shadow.update(block.addr(), PAGE);
        let copied_at = shadow.copied.lock().unwrap_or_else(|e| e.into_inner()).at.first().copied().unwrap_or(0);
        assert_ne!(copied_at, 0);
        assert!(crate::watch::is_guarded(page), "page should be guarded by watch");
        assert_eq!(page_protect(block.addr()), PAGE_READONLY);

        // Keep `watch`'s standing asking open on a disjoint write-watched page
        // so `watch::wrote` advances the journal stamp for `block` right after
        // `PAGE_READONLY -> PAGE_NOACCESS` and before `was_written_since`.
        let watch_keeper = VirtualBlock::new(PAGE, true);
        crate::watch::ask_once(&[(watch_keeper.addr(), watch_keeper.addr() + PAGE)]);
        let _ask_once = AskOnceGuard;

        // Clear any pending card dirty flags so `try_defer_piece` reaches
        // `transition_clean_guard_to_noaccess`, then inject a concurrent
        // `watch::wrote` immediately after the protection transition.
        let _ = card_dirty(1);
        *TRY_DEFER_POST_TRANSITION_HOOK
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(Box::new(|p| {
            crate::watch::wrote(p * PAGE, PAGE);
        }));
        assert!(
            !shadow.try_defer_piece(0, PAGE),
            "try_defer_piece must refuse when journal stamp advances after transition"
        );
        assert!(
            TRY_DEFER_POST_TRANSITION_HOOK
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_none(),
            "post-transition hook must have fired after transition_clean_guard_to_noaccess"
        );
        assert!(
            crate::watch::was_written_since(page, copied_at),
            "post-transition hook must have advanced journal stamp past copied_at"
        );
        assert_eq!(
            page_protect(block.addr()),
            PAGE_READWRITE,
            "rejected MEM_MAPPED arming must roll back PAGE_NOACCESS -> PAGE_READWRITE"
        );
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 0);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);

        // Direct CPU read and write MUST succeed without unhandled AV.
        // SAFETY: page was rolled back to `PAGE_READWRITE`.
        unsafe {
            assert_eq!(*block.ptr, 0x3A);
            *block.ptr = 0x7C;
            assert_eq!(*block.ptr, 0x7C);
        }

        drop(shadow);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// Regression: `Shadow::writing` MUST NOT touch `self.data`, `LAZY_RESYNC`,
    /// `FlushedBaseline`, or `card.has_dirty` for subranges still in
    /// `self.pending`, and `Shadow::has_lazy_resync_under_writes` MUST report
    /// `true` so `host::stale_under_flights` settles UAV-only `Step::GpuWrite`
    /// flights cleanly.
    #[test]
    fn lazy_copyback_writing_skips_pending_resync_and_stale_under_flights_settles_gpu_write() {
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        let block = VirtualBlock::new(PAGE, true);
        // SAFETY: seed initial bytes `A`.
        unsafe { std::ptr::write_bytes(block.ptr, 0x11, PAGE) };
        let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");
        let shadow_id = shadow.id;

        // Flight 0 writes `B` (`0x2424_0000 ^ i`) and arms `LAZY_DEFERRED`.
        pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x2424_0000, 1);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);

        // Flight 1 declares the page in `self.pending`.
        shadow.writing(block.addr(), PAGE);
        assert!(shadow.writes_pending());

        // CPU read faults via VEH -> flushes `B`, marks `LAZY_RESYNC`, records `FlushedBaseline(B)`.
        // Then CPU writes fresh bytes `0xCAFE_BABE` at dword 4.
        // SAFETY: VEH demand-flushes page to `PAGE_READWRITE`.
        unsafe {
            assert_eq!(*block.ptr.cast::<u32>(), 0x2424_0000);
            block.ptr.cast::<u32>().add(4).write_volatile(0xCAFE_BABE);
        }
        assert_eq!(shadow.resync_count.load(Ordering::Acquire), 1);
        assert!(
            shadow.has_lazy_resync_under_writes(block.addr(), PAGE),
            "UAV-only Step::GpuWrite check must detect pending LAZY_RESYNC page"
        );

        // Calling `Shadow::writing` again while range is STILL in `self.pending`
        // MUST skip resync so `FlushedBaseline`, `self.data`, and `card.has_dirty`
        // stay intact for Flight 1's `card_merge_relay`.
        shadow.writing(block.addr(), PAGE);
        assert_eq!(shadow.resync_count.load(Ordering::Acquire), 1);
        assert!(
            !!shadow.card.as_ref().unwrap().dirty.lock().unwrap_or_else(|e| e.into_inner()).is_empty(),
            "Shadow::writing must not dirty card for a pending range"
        );
        {
            let table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            assert!(
                table.flushed_baselines.contains_key(&(shadow_id, 0)),
                "FlushedBaseline must remain intact while range is pending"
            );
        }

        // Flight 1 lands in `card_merge_relay` with `gpu == B`: must write 0 bytes
        // and preserve CPU's `0xCAFE_BABE`.
        let mut gpu_b = vec![0u8; PAGE];
        for i in 0..PAGE / 4 {
            let v = 0x2424_0000u32 ^ i as u32;
            gpu_b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let mut scratch = Vec::new();
        assert_eq!(shadow.card_merge_relay(0, &gpu_b, 1, &mut scratch), 0);
        // SAFETY: page is `PAGE_READWRITE`.
        assert_eq!(unsafe { block.ptr.cast::<u32>().add(4).read_volatile() }, 0xCAFE_BABE);

        // Once Flight 1 retires from `self.pending`, `Shadow::writing` resyncs cleanly.
        shadow.copy_back_done(&[(block.addr(), block.addr() + PAGE)]);
        assert!(!shadow.has_lazy_resync_under_writes(block.addr(), PAGE));
        shadow.writing(block.addr(), PAGE);
        assert_eq!(shadow.resync_count.load(Ordering::Acquire), 0);
        assert!(!shadow.card.as_ref().unwrap().dirty.lock().unwrap_or_else(|e| e.into_inner()).is_empty());

        shadow.copy_back_done(&[(block.addr(), block.addr() + PAGE)]);
        drop(shadow);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// Regression: if a page transitions `LAZY_DEFERRED -> LAZY_RESYNC` AFTER
    /// `merge_under_writes`'s upfront `resync_count > 0` check, the in-loop
    /// `LAZY_RESYNC` re-check MUST still return `true` (forcing settle)
    /// without diffing VEH-flushed GPU bytes against stale `self.data`.
    #[test]
    fn lazy_copyback_merge_under_writes_settles_mid_loop_resync() {
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        let block = VirtualBlock::new(PAGE, true);
        // SAFETY: seed initial bytes `A`.
        unsafe { std::ptr::write_bytes(block.ptr, 0x11, PAGE) };
        let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");

        // Flight 0 writes `B` (`0x5555_0000 ^ i`) and arms `LAZY_DEFERRED`.
        pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x5555_0000, 1);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);

        // Flight 1 marks range pending; VEH flushes `B` -> `LAZY_RESYNC`.
        shadow.writing(block.addr(), PAGE);
        // SAFETY: VEH demand-flushes page to `PAGE_READWRITE`.
        assert_eq!(unsafe { *block.ptr.cast::<u32>() }, 0x5555_0000);
        assert_eq!(shadow.resync_count.load(Ordering::Acquire), 1);
        assert_eq!(shadow.lazy_state[0].load(Ordering::Acquire), LAZY_RESYNC);

        // Simulate VEH racing right after `merge_under_writes`'s upfront
        // `resync_count > 0` check by temporarily hiding `resync_count == 0`
        // while `lazy_state[0]` is `LAZY_RESYNC`.
        shadow.resync_count.store(0, Ordering::Release);
        let mut runs_left = 1024;
        let must_settle = shadow.merge_under_writes(block.addr(), PAGE, &mut runs_left);
        shadow.resync_count.store(1, Ordering::Release);

        assert!(must_settle, "in-loop LAZY_RESYNC check must force flight settle");
        assert!(
            !!shadow.card.as_ref().unwrap().dirty.lock().unwrap_or_else(|e| e.into_inner()).is_empty(),
            "merge_under_writes must NOT queue bogus card_changed uploads on LAZY_RESYNC"
        );
        // SAFETY: `self.data` still holds pre-GPU `0x11` bytes.
        assert_eq!(unsafe { *shadow.data }, 0x11);

        shadow.copy_back_done(&[(block.addr(), block.addr() + PAGE)]);
        drop(shadow);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// Regression: `lazy_release_written_range` with `full_cover == true`
    /// while `shadow.writes_pending() == true` MUST capture `FlushedBaseline(B)`
    /// (without writing `B` to title memory) so an in-flight `card_merge_relay`
    /// diffs against `B` rather than stale `self.data` (`A`) and preserves 100%
    /// of the freshly written 64 KiB DirectStorage data `D`.
    #[test]
    fn lazy_copyback_f1_full_cover_written_while_pending_preserves_fresh_data() {
        use windows_sys::Win32::System::Memory::{
            FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_NOACCESS, PAGE_READWRITE,
            UnmapViewOfFile,
        };

        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        let block = MappedSectionBlock::new(PAGE);
        // SAFETY: seed initial bytes `A`.
        unsafe { std::ptr::write_bytes(block.ptr, 0x11, PAGE) };
        let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");
        let shadow_id = shadow.id;

        // Flight 0 writes `B` (`0x6666_0000 ^ i`) and arms `LAZY_DEFERRED` (`self.data` still `0x11`).
        pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x6666_0000, 1);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);
        assert_eq!(page_protect(block.addr()), PAGE_NOACCESS);

        // Flight 1 is in flight (`shadow.writes_pending() == true`).
        shadow.writing(block.addr(), PAGE);
        assert!(shadow.writes_pending());

        // DirectStorage overwrites all 64 KiB with fresh `D` (`0x7777_0000 + i`)
        // via an independent writable section mapping while title view is `PAGE_NOACCESS`.
        // SAFETY: temporary writable view of our live pagefile section.
        unsafe {
            let alias = MapViewOfFile(block.section as *mut std::ffi::c_void, FILE_MAP_WRITE, 0, 0, PAGE);
            assert!(!alias.Value.is_null());
            let words = std::slice::from_raw_parts_mut(alias.Value.cast::<u32>(), PAGE / 4);
            for (i, w) in words.iter_mut().enumerate() {
                *w = 0x7777_0000 + i as u32;
            }
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: alias.Value });
        }

        // Notify watch (full 64 KiB cover while `writes_pending() == true`).
        crate::watch::title_memory_written(block.addr(), PAGE, true);
        assert_eq!(page_protect(block.addr()), PAGE_READWRITE);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 0);
        assert_eq!(shadow.resync_count.load(Ordering::Acquire), 1);
        {
            let table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            assert!(
                table.flushed_baselines.contains_key(&(shadow_id, 0)),
                "full_cover title_memory_written while pending must capture FlushedBaseline"
            );
        }

        // Flight 1 lands in `card_merge_relay`: VRAM has Flight 0's `B` everywhere
        // except dword 0 which Flight 1 updated to `0x9999_0001`.
        let mut gpu_f1 = vec![0u8; PAGE];
        for i in 0..PAGE / 4 {
            let v = if i == 0 { 0x9999_0001 } else { 0x6666_0000u32 ^ i as u32 };
            gpu_f1[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let mut scratch = Vec::new();
        let written = shadow.card_merge_relay(0, &gpu_f1, 1, &mut scratch);
        assert!(written > 0 && written <= 4, "only dword 0 differed from FlushedBaseline(B)");

        // Verify dword 0 got Flight 1's write and all 16,383 remaining dwords
        // kept DirectStorage's fresh `D` (NOT overwritten with stale Flight 0 `B`!).
        // SAFETY: page is `PAGE_READWRITE`.
        unsafe {
            let words = std::slice::from_raw_parts(block.ptr.cast::<u32>(), PAGE / 4);
            assert_eq!(words[0], 0x9999_0001);
            for (i, &w) in words.iter().enumerate().skip(1) {
                assert_eq!(w, 0x7777_0000 + i as u32, "word {i} must preserve DirectStorage D");
            }
        }

        shadow.copy_back_done(&[(block.addr(), block.addr() + PAGE)]);
        drop(shadow);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// Regression: `lazy_flush_and_hold` held across `host::copy_mapped`,
    /// `host::read_into`, and `host::write_mapped` prevents concurrent relay
    /// `try_defer_piece_relay` from re-arming `PAGE_NOACCESS` mid-access (no
    /// silent `read_sparse` zero-fills, no spurious `read_into == false`, and no
    /// dropped `write_mapped` writes).
    #[test]
    fn lazy_copyback_host_access_blocks_concurrent_relay_rearm() {
        use std::sync::atomic::AtomicBool;
        use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READWRITE};

        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        let block = VirtualBlock::new(PAGE, true);
        // SAFETY: seed initial bytes so `MEM_WRITE_WATCH` is committed before Frame 1.
        unsafe { std::ptr::write_bytes(block.ptr, 0x11, PAGE) };
        let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");

        // GPU writes non-zero pattern `0x8888_0000 ^ i` and arms `LAZY_DEFERRED`.
        pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x8888_0000, 1);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);
        assert_eq!(page_protect(block.addr()), PAGE_NOACCESS);

        // Deterministic barrier check: while `lazy_flush_and_hold` is live,
        // concurrent `try_defer_piece_relay` MUST fail to arm.
        {
            let _hold = lazy_flush_and_hold(block.addr(), PAGE);
            assert_eq!(page_protect(block.addr()), PAGE_READWRITE);
            assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
            assert!(
                !shadow.try_defer_piece_relay(0, PAGE, 0),
                "concurrent relay re-arm must be blocked while lazy_flush_and_hold is held"
            );
            assert_eq!(page_protect(block.addr()), PAGE_READWRITE);
        }

        // Concurrent stress: relay thread hammers `try_defer_piece_relay` while
        // reader/writer exercises `copy_mapped`, `read_into`, and `write_mapped`.
        pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x8888_0000, 1);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_bg = stop.clone();
        let fetched_after = Arc::new(AtomicU64::new(1));
        let fetched_bg = fetched_after.clone();
        let shadow_bg = shadow.clone();
        let bg = std::thread::spawn(move || {
            while !stop_bg.load(Ordering::Acquire) {
                let fa = fetched_bg.load(Ordering::Acquire);
                let _ = shadow_bg.try_defer_piece_relay(0, PAGE, fa);
                std::hint::spin_loop();
            }
        });

        let mut dst = vec![0u8; PAGE];
        let mut rbuf = vec![0u8; PAGE];
        for iter in 0u32..80 {
            // Re-enable arming eligibility each iteration so the relay thread
            // genuinely races `try_defer_piece_relay` against host accessors.
            let _ = shadow.update(block.addr(), PAGE);
            let seq = u64::from(iter) + 2;
            let _ = card_dirty(seq);
            fetched_after.store(seq, Ordering::Release);

            crate::host::copy_mapped(dst.as_mut_ptr(), block.ptr, PAGE, false);
            let w0 = u32::from_le_bytes(dst[0..4].try_into().unwrap());
            assert_ne!(w0, 0, "iter {iter}: copy_mapped must never zero-fill live deferred page");

            assert!(
                crate::host::read_into(block.ptr, &mut rbuf),
                "iter {iter}: read_into must never fail due to concurrent relay arming"
            );
            let r0 = u32::from_le_bytes(rbuf[0..4].try_into().unwrap());
            assert_ne!(r0, 0, "iter {iter}: read_into must observe live non-zero bytes");

            let marker = (0xA000_0000u32 | (iter + 1)).to_le_bytes();
            crate::host::write_mapped(block.ptr, marker.as_ptr(), 4);
            let mut verify = [0u8; 4];
            assert!(crate::host::read_into(block.ptr, &mut verify));
            assert_eq!(verify, marker, "iter {iter}: write_mapped must never drop writes");
        }

        stop.store(true, Ordering::Release);
        bg.join().expect("relay stress thread");
        lazy_flush_range(block.addr(), PAGE);
        drop(shadow);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// `--release` hardware GPU benchmark comparing eager copy-back vs
    /// `VICTORY_LAZY_COPYBACK=1` on a 64 MiB GPU-written carded VRAM workload
    /// (both 0% CPU reads and 2% sparse CPU reads = 20 of 1,024 pieces/frame).
    #[test]
    #[ignore = "benchmark"]
    fn bench_lazy_copyback() {
        use std::time::Instant;
        let Some(h) = host() else {
            eprintln!("no host GPU available");
            return;
        };
        if !h.card_ready() {
            eprintln!("host card buffers not ready");
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            eprintln!("failed to compile compute shader");
            return;
        };
        let list = h.new_list().expect("new_list");

        const MIB: usize = 1 << 20;
        const LEN: usize = 64 * MIB;
        const PIECES: usize = LEN / PAGE; // 1,024 pieces
        const WARMUP: usize = 5;
        const FRAMES: usize = 30;
        const SPARSE_READS: usize = 20; // ~2% of 1,024 pieces/frame

        let run_case = |label: &str, lazy_on: bool, sparse_reads: usize| -> (f64, f64, f64, f64) {
            let _guard = if lazy_on {
                LazyTestGuard::enable()
            } else {
                LazyTestGuard::disable()
            };
            let block = VirtualBlock::new(LEN, true);
            let shadow = make_test_shadow(h, block.addr() as u64, LEN as u64, true).expect("shadow");

            // Seed VRAM + warmup frames.
            for f in 0..WARMUP {
                pipe.dispatch_on(h, &list, &shadow, 0, LEN, f as u32 + 1, 0);
                for k in 0..sparse_reads {
                    let p = (k * 51) % PIECES;
                    // SAFETY: committed allocation; VEH flushes on demand when lazy.
                    let _ = unsafe { ((block.addr() + p * PAGE) as *const u32).read_volatile() };
                }
            }

            let up0 = CARD_STATS[0].load(Ordering::Relaxed);
            let rb_bytes0 = CARD_STATS[4].load(Ordering::Relaxed);
            let veh_r0 = LAZY_STATS[2].load(Ordering::Relaxed);
            let veh_bytes0 = LAZY_STATS[5].load(Ordering::Relaxed);

            let t0 = Instant::now();
            let mut sink = 0u32;
            for f in 0..FRAMES {
                pipe.dispatch_on(h, &list, &shadow, 0, LEN, (f + 10) as u32, 0);
                for k in 0..sparse_reads {
                    let p = (k * 51) % PIECES;
                    // SAFETY: committed allocation; VEH flushes on demand when lazy.
                    sink = sink.wrapping_add(unsafe { ((block.addr() + p * PAGE) as *const u32).read_volatile() });
                }
            }
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(sink);

            let ms_per_frame = elapsed_ms / FRAMES as f64;
            let upload_mb = (CARD_STATS[0].load(Ordering::Relaxed) - up0) as f64 / (FRAMES as f64 * MIB as f64);
            let eager_rb_mb = (CARD_STATS[4].load(Ordering::Relaxed) - rb_bytes0) as f64 / (FRAMES as f64 * MIB as f64);
            let veh_rb_mb = (LAZY_STATS[5].load(Ordering::Relaxed) - veh_bytes0) as f64 / (FRAMES as f64 * MIB as f64);
            let total_rb_mb = eager_rb_mb + veh_rb_mb;
            let veh_faults = (LAZY_STATS[2].load(Ordering::Relaxed) - veh_r0) as f64 / FRAMES as f64;

            eprintln!(
                "{label:<36} | {ms_per_frame:7.3} ms/frame | readback {total_rb_mb:6.2} MB/f | upload {upload_mb:6.2} MB/f | veh_faults {veh_faults:5.1}/f"
            );
            drop(shadow);
            (ms_per_frame, total_rb_mb, upload_mb, veh_faults)
        };

        eprintln!("\n=== 64 MiB GPU-Written Carded VRAM Copy-Back Benchmark (RTX 4070 SUPER) ===");
        let (eager_0, ..) = run_case("Eager (0% CPU reads)", false, 0);
        let (lazy_0, ..) = run_case("VICTORY_LAZY_COPYBACK=1 (0% reads)", true, 0);
        let (eager_2, ..) = run_case("Eager (2% sparse CPU reads)", false, SPARSE_READS);
        let (lazy_2, ..) = run_case("VICTORY_LAZY_COPYBACK=1 (2% reads)", true, SPARSE_READS);
        eprintln!(
            "Speedup (0% reads): {:.2}x | Speedup (2% sparse reads): {:.2}x",
            eager_0 / lazy_0.max(1e-6),
            eager_2 / lazy_2.max(1e-6)
        );
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// Regression test 1 (`0xc0000005` VEH escape on concurrent re-arm / guard
    /// transition): hammers `watch` `PAGE_READONLY -> PAGE_NOACCESS` transitions
    /// and repeated `try_defer_piece` / `try_defer_piece_relay` re-arming while
    /// concurrent reader/writer threads access trailing offsets `0xf804` and
    /// `0xfc04`. Must never escape the VEH handler or tear data.
    #[test]
    fn lazy_copyback_veh_rearm_and_guard_transition_race_never_escapes() {
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        const M_SEED_F804: u32 = 0x4000_0000 ^ (0xf804 / 4) as u32;
        const M_SEED_FC04: u32 = 0x4000_0000 ^ (0xfc04 / 4) as u32;
        const P_SEED_F804: u32 = 0x5000_0000 ^ (0xf804 / 4) as u32;
        const P_SEED_FC04: u32 = 0x5000_0000 ^ (0xfc04 / 4) as u32;

        // Exercise both section-backed `MEM_MAPPED` (`PAGE_READONLY -> PAGE_NOACCESS`)
        // and `MEM_PRIVATE | MEM_WRITE_WATCH` paths across concurrent threads.
        let mblock = MappedSectionBlock::new(PAGE);
        // SAFETY: seed mapped page.
        unsafe { std::ptr::write_bytes(mblock.ptr, 0x11, PAGE) };
        let mshadow = make_test_shadow(h, mblock.addr() as u64, PAGE as u64, false).expect("mshadow");
        pipe.dispatch_on(h, &list, &mshadow, 0, PAGE, 0x4000_0000, 1);

        let pblock = VirtualBlock::new(PAGE, true);
        // SAFETY: seed private write-watch page.
        unsafe { std::ptr::write_bytes(pblock.ptr, 0x22, PAGE) };
        let pshadow = make_test_shadow(h, pblock.addr() as u64, PAGE as u64, false).expect("pshadow");
        pipe.dispatch_on(h, &list, &pshadow, 0, PAGE, 0x5000_0000, 1);

        // Complete initial Flush #1 on `pblock` before spawning the concurrent
        // accessor so `+0xf804` physical backing is initialized before re-arming.
        // SAFETY: triggers VEH demand flush on both blocks.
        unsafe {
            assert_eq!(std::ptr::read_volatile((mblock.addr() + 0xf804) as *const u32), M_SEED_F804);
            assert_eq!(std::ptr::read_volatile((pblock.addr() + 0xf804) as *const u32), P_SEED_F804);
        }
        let _ = mshadow.update(mblock.addr(), PAGE);
        let _ = pshadow.update(pblock.addr(), PAGE);
        assert!(mshadow.try_defer_piece(0, PAGE));
        assert!(pshadow.try_defer_piece(0, PAGE));

        let flushes_before = LAZY_FLUSH_TOTAL.load(Ordering::Acquire);
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_r = stop.clone();
        let accessor_iters = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let accessor_iters_w = accessor_iters.clone();
        let m_addr = mblock.addr();
        let p_addr = pblock.addr();

        let accessor = std::thread::spawn(move || {
            let mut i = 0u32;
            while !stop_r.load(Ordering::Relaxed) {
                i = i.wrapping_add(1);
                for &(base, seed_f8, seed_fc) in &[
                    (m_addr, M_SEED_F804, M_SEED_FC04),
                    (p_addr, P_SEED_F804, P_SEED_FC04),
                ] {
                    // SAFETY: committed test pages; VEH/guard handler resolves all faults.
                    unsafe {
                        let p1 = (base + 0xf804) as *const u32;
                        let p2 = (base + 0xfc04) as *mut u32;
                        assert_eq!(std::ptr::read_volatile(p1), seed_f8);
                        if i.is_multiple_of(2) {
                            std::ptr::write_volatile(p2, 0xDEAD_0000 | (i & 0xFFFF));
                        }
                        let v2 = std::ptr::read_volatile(p2);
                        assert!(
                            v2 == seed_fc || (v2 & 0xFFFF_0000) == 0xDEAD_0000,
                            "torn read at +0xfc04: {v2:#x}"
                        );
                    }
                }
                accessor_iters_w.fetch_add(1, Ordering::Relaxed);
                if i.is_multiple_of(4) {
                    std::thread::yield_now();
                }
            }
        });

        let mut m_rearms = 0u32;
        let mut p_rearms = 0u32;
        for iter in 0u32..2000 {
            let m_was_def = mshadow.lazy_state[0].load(Ordering::Acquire) == LAZY_DEFERRED;
            let _ = mshadow.update(mblock.addr(), PAGE);
            let m_ok = if iter.is_multiple_of(2) {
                mshadow.try_defer_piece(0, PAGE)
            } else {
                mshadow.try_defer_piece_relay(0, PAGE, iter as u64)
            };
            if m_ok && !m_was_def && mshadow.lazy_state[0].load(Ordering::Acquire) == LAZY_DEFERRED {
                m_rearms += 1;
            }

            let p_was_def = pshadow.lazy_state[0].load(Ordering::Acquire) == LAZY_DEFERRED;
            let _ = pshadow.update(pblock.addr(), PAGE);
            let p_ok = if iter.is_multiple_of(2) {
                pshadow.try_defer_piece(0, PAGE)
            } else {
                pshadow.try_defer_piece_relay(0, PAGE, iter as u64)
            };
            if p_ok && !p_was_def && pshadow.lazy_state[0].load(Ordering::Acquire) == LAZY_DEFERRED {
                p_rearms += 1;
            }

            // Also contend directly on `watch` write-guard re-arming vs `try_defer_piece`.
            let _ = crate::watch::writes(mblock.addr(), mblock.addr() + PAGE, true);
            if iter.is_multiple_of(4) {
                std::thread::yield_now();
            }
            if iter >= 250
                && m_rearms >= 2
                && p_rearms >= 2
                && LAZY_FLUSH_TOTAL.load(Ordering::Acquire) >= flushes_before + 4
                && accessor_iters.load(Ordering::Acquire) >= 32
            {
                break;
            }
        }

        stop.store(true, Ordering::Release);
        accessor.join().expect("accessor thread must complete without unhandled AV");
        assert!(m_rearms >= 2, "expected repeated mapped re-arms, got {m_rearms}");
        assert!(p_rearms >= 2, "expected repeated private re-arms, got {p_rearms}");
        assert!(accessor_iters.load(Ordering::Acquire) >= 32);
        assert!(LAZY_FLUSH_TOTAL.load(Ordering::Acquire) >= flushes_before + 4);

        drop(pshadow);
        drop(mshadow);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// Regression test 2: splitting VAD protections inside an armed 64 KiB page
    /// across single-page VEH fault (`MEM_PRIVATE`), `lazy_flush_range`
    /// (`MEM_MAPPED`), `lazy_release_range`, and `lazy_release_written_range`.
    #[test]
    fn lazy_copyback_split_subpage_unprotect_restores_all_4k_subpages() {
        use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        // 2A: `MEM_PRIVATE` single-page VEH fault after splitting `[0..4096)` `PAGE_READWRITE`.
        let block = VirtualBlock::new(PAGE, true);
        // SAFETY: seed physical page before first dispatch.
        unsafe { std::ptr::write_bytes(block.ptr, 0x3C, PAGE) };
        let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");
        pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x6161_0000, 1);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);
        assert_eq!(page_protect(block.addr()), PAGE_NOACCESS);

        let mut old = 0u32;
        // SAFETY: splitting protection on first 4 KiB subpage of our test allocation.
        assert_ne!(
            unsafe { VirtualProtect(block.ptr.cast(), 4096, PAGE_READWRITE, &mut old) },
            0
        );
        assert_eq!(page_protect(block.addr()), PAGE_READWRITE);
        assert_eq!(page_protect(block.addr() + 0xf804), PAGE_NOACCESS);

        // SAFETY: fault is handled by VEH + `unprotect_split_subregions`.
        unsafe {
            let v1 = std::ptr::read_volatile((block.addr() + 0xf804) as *const u32);
            assert_eq!(v1, 0x6161_0000 ^ (0xf804 / 4) as u32);
            let p2 = (block.addr() + 0xfc04) as *mut u32;
            assert_eq!(std::ptr::read_volatile(p2), 0x6161_0000 ^ (0xfc04 / 4) as u32);
            std::ptr::write_volatile(p2, 0xCAFE_BEEF);
            assert_eq!(std::ptr::read_volatile(p2), 0xCAFE_BEEF);
        }
        assert_eq!(page_protect(block.addr() + 0xf804), PAGE_READWRITE);
        assert_eq!(page_protect(block.addr() + 0xfc04), PAGE_READWRITE);
        drop(shadow);

        // 2B: `MEM_MAPPED` split-4K via `lazy_flush_range`, with `[0..4096)` set
        // `PAGE_READONLY` so any blind full-page copy would AV immediately.
        let mblock = MappedSectionBlock::new(PAGE);
        // SAFETY: seed physical section page.
        unsafe { std::ptr::write_bytes(mblock.ptr, 0x4D, PAGE) };
        let mshadow = make_test_shadow(h, mblock.addr() as u64, PAGE as u64, true).expect("mshadow");
        pipe.dispatch_on(h, &list, &mshadow, 0, PAGE, 0x7272_0000, 1);
        assert_eq!(mshadow.deferred_count.load(Ordering::Acquire), 1);
        // SAFETY: split first 4 KiB to `PAGE_READONLY`.
        assert_ne!(
            unsafe { VirtualProtect(mblock.ptr.cast(), 4096, PAGE_READONLY, &mut old) },
            0
        );
        lazy_flush_range(mblock.addr(), PAGE);
        assert_eq!(mshadow.deferred_count.load(Ordering::Acquire), 0);
        assert_eq!(page_protect(mblock.addr()), PAGE_READONLY);
        assert_eq!(page_protect(mblock.addr() + 0xf804), PAGE_READWRITE);
        // SAFETY: trailing subpage is now `PAGE_READWRITE` with GPU bytes.
        assert_eq!(
            unsafe { std::ptr::read_volatile((mblock.addr() + 0xf804) as *const u32) },
            0x7272_0000 ^ (0xf804 / 4) as u32
        );
        drop(mshadow);

        // 2C(i): `lazy_release_range` on split-4K page (`[0..4096)` = `PAGE_READONLY`).
        let rblock = VirtualBlock::new(PAGE, true);
        // SAFETY: seed physical page.
        unsafe { std::ptr::write_bytes(rblock.ptr, 0x5E, PAGE) };
        let rshadow = make_test_shadow(h, rblock.addr() as u64, PAGE as u64, true).expect("rshadow");
        pipe.dispatch_on(h, &list, &rshadow, 0, PAGE, 0x8181_0000, 1);
        // SAFETY: split first 4 KiB to `PAGE_READONLY`.
        assert_ne!(
            unsafe { VirtualProtect(rblock.ptr.cast(), 4096, PAGE_READONLY, &mut old) },
            0
        );
        lazy_release_range(rblock.addr(), PAGE);
        assert_eq!(rshadow.deferred_count.load(Ordering::Acquire), 0);
        assert_eq!(page_protect(rblock.addr()), PAGE_READONLY);
        assert_eq!(page_protect(rblock.addr() + 0xf804), PAGE_READWRITE);
        drop(rshadow);

        // 2C(ii): `lazy_release_written_range` preserves `[+0x2000..+0x4000)` while
        // restoring and populating the split outer `PAGE_NOACCESS` subpages.
        let wblock = VirtualBlock::new(PAGE, true);
        // SAFETY: seed physical page.
        unsafe { std::ptr::write_bytes(wblock.ptr, 0x6F, PAGE) };
        let wshadow = make_test_shadow(h, wblock.addr() as u64, PAGE as u64, true).expect("wshadow");
        pipe.dispatch_on(h, &list, &wshadow, 0, PAGE, 0x8383_0000, 1);
        // SAFETY: open `[+0x2000..+0x4000)` to `PAGE_READWRITE` and stamp sentinel.
        assert_ne!(
            unsafe { VirtualProtect((wblock.addr() + 0x2000) as *const _, 0x2000, PAGE_READWRITE, &mut old) },
            0
        );
        // SAFETY: `[+0x2000..+0x4000)` is `PAGE_READWRITE`.
        unsafe { std::ptr::write_volatile((wblock.addr() + 0x2004) as *mut u32, 0xBEEF_CAFE) };
        lazy_release_written_range(wblock.addr() + 0x2000, 0x2000);
        assert_eq!(wshadow.deferred_count.load(Ordering::Acquire), 0);
        assert_eq!(page_protect(wblock.addr() + 4), PAGE_READWRITE);
        assert_eq!(page_protect(wblock.addr() + 0xf804), PAGE_READWRITE);
        // SAFETY: all subpages are now `PAGE_READWRITE`.
        unsafe {
            assert_eq!(std::ptr::read_volatile((wblock.addr() + 4) as *const u32), 0x8383_0000 ^ 1);
            assert_eq!(std::ptr::read_volatile((wblock.addr() + 0x2004) as *const u32), 0xBEEF_CAFE);
            assert_eq!(
                std::ptr::read_volatile((wblock.addr() + 0xf804) as *const u32),
                0x8383_0000 ^ (0xf804 / 4) as u32
            );
        }
        drop(wshadow);

        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// Regression test 3 (v2 hot-page thrash cooldown & O(1) fast-reject cache):
    /// verifies strike escalation (2 flushes in 60 frames -> 300 frames -> 1800
    /// frames -> permanent `LAZY_INELIGIBLE`), preservation of `LAZY_RESYNC`
    /// coherence through every flush, immediate `LAZY_INELIGIBLE` caching for
    /// non-write-watch private allocations, and that Frame-1 virgin
    /// `MEM_WRITE_WATCH` pages (`count > 0`) remain `LAZY_EAGER` for Frame 2.
    #[test]
    fn lazy_copyback_v2_hot_page_cooldown_and_ineligible_fast_reject() {
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        // Part A: Hot-page strike escalation (frame 100 -> 110 -> 415 -> 2220).
        let _frame_guard = LazyFrameGuard::set(100);
        let block = VirtualBlock::new(PAGE, true);
        // SAFETY: seed physical page.
        unsafe { std::ptr::write_bytes(block.ptr, 0x10, PAGE) };
        let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");

        // Flush 1 at frame 100 (`strikes = 1` -> returns to `LAZY_EAGER`).
        pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x1000_0000, 1);
        assert_eq!(shadow.lazy_state[0].load(Ordering::Acquire), LAZY_DEFERRED);
        // SAFETY: VEH flush preserves exact GPU data and sets `LAZY_RESYNC` + strike 1.
        assert_eq!(unsafe { *block.ptr.cast::<u32>() }, 0x1000_0000);
        assert_eq!(shadow.lazy_state[0].load(Ordering::Acquire), LAZY_RESYNC);

        // Flush 2 at frame 110 (`strikes = 2`, within 60-frame window -> cooled until 410).
        LazyFrameGuard::update(110);
        pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x2000_0000, 1);
        assert_eq!(shadow.lazy_state[0].load(Ordering::Acquire), LAZY_DEFERRED);
        assert_eq!(unsafe { *block.ptr.cast::<u32>() }, 0x2000_0000);
        assert_eq!(shadow.lazy_state[0].load(Ordering::Acquire), LAZY_RESYNC);

        // Next update/refresh preserves `LAZY_RESYNC` coherence AND transitions to `LAZY_HOT`.
        // Write a distinct CPU word first and verify `update()` brings it into `shadow.data`.
        // SAFETY: page was flushed to `PAGE_READWRITE`.
        unsafe { *block.ptr.add(4).cast::<u32>() = 0x1122_3344 };
        let _ = shadow.update(block.addr(), PAGE);
        assert_eq!(shadow.lazy_state[0].load(Ordering::Acquire), LAZY_HOT);
        assert_eq!(shadow.resync_count.load(Ordering::Acquire), 0);
        assert!(
            !shadow.try_defer_piece(0, PAGE),
            "LAZY_HOT page must reject re-arm during 300-frame cooldown"
        );
        // SAFETY: verify CPU write synced into shadow staging/copy after `LAZY_RESYNC`.
        unsafe { assert_eq!(*shadow.data.add(4).cast::<u32>(), 0x1122_3344) };

        LazyFrameGuard::update(409);
        assert!(!shadow.try_defer_piece(0, PAGE), "still cooled at frame 409 < 410");

        // Flush 3 at frame 415 (>= 410 expiry -> re-arms, then 3rd strike cools 1800 until 2215).
        LazyFrameGuard::update(415);
        pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x3000_0000, 1);
        assert_eq!(shadow.lazy_state[0].load(Ordering::Acquire), LAZY_DEFERRED);
        assert_eq!(unsafe { *block.ptr.cast::<u32>() }, 0x3000_0000);
        let _ = shadow.update(block.addr(), PAGE);
        assert_eq!(shadow.lazy_state[0].load(Ordering::Acquire), LAZY_HOT);
        assert!(!shadow.try_defer_piece(0, PAGE));

        // Flush 4 at frame 2220 (>= 2215 -> re-arms, then 4th strike marks permanent `LAZY_INELIGIBLE`).
        LazyFrameGuard::update(2220);
        pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x4000_0000, 1);
        assert_eq!(shadow.lazy_state[0].load(Ordering::Acquire), LAZY_DEFERRED);
        assert_eq!(unsafe { *block.ptr.cast::<u32>() }, 0x4000_0000);
        let _ = shadow.update(block.addr(), PAGE);
        assert_eq!(shadow.lazy_state[0].load(Ordering::Acquire), LAZY_INELIGIBLE);
        LazyFrameGuard::update(1_000_000);
        assert!(
            !shadow.try_defer_piece(0, PAGE),
            "permanently LAZY_INELIGIBLE page must fast-reject even far in future"
        );
        drop(shadow);

        // Part B: Plain non-MEM_WRITE_WATCH private page is immediately cached `LAZY_INELIGIBLE`.
        LazyFrameGuard::update(10);
        let plain_block = VirtualBlock::new(PAGE, false);
        // SAFETY: seed non-write-watch private allocation.
        unsafe { std::ptr::write_bytes(plain_block.ptr, 0x55, PAGE) };
        let plain_shadow =
            make_test_shadow(h, plain_block.addr() as u64, PAGE as u64, true).expect("plain_shadow");
        plain_shadow.watched.store(1, Ordering::Relaxed);
        plain_shadow.copied.lock().unwrap_or_else(|e| e.into_inner()).set(0, 1);
        assert!(!plain_shadow.try_defer_piece(0, PAGE));
        assert_eq!(plain_shadow.lazy_state[0].load(Ordering::Acquire), LAZY_INELIGIBLE);
        assert!(!plain_shadow.try_defer_piece(0, PAGE));
        drop(plain_shadow);

        // Part C: Virgin demand-zero `MEM_WRITE_WATCH` page (`GetWriteWatch` reports
        // untouched 4 KiB subpages with `count > 0`) rejects Frame-1 arming while
        // remaining `LAZY_EAGER` so Frame 2 arms once committed/reset.
        LazyFrameGuard::update(1);
        let virgin_block = VirtualBlock::new(PAGE, true);
        let virgin_shadow =
            make_test_shadow(h, virgin_block.addr() as u64, PAGE as u64, false).expect("virgin_shadow");
        let _ = virgin_shadow.update(virgin_block.addr(), PAGE);
        assert!(!virgin_shadow.try_defer_piece(0, PAGE));
        assert_eq!(virgin_shadow.lazy_state[0].load(Ordering::Acquire), LAZY_EAGER);
        LazyFrameGuard::update(2);
        // SAFETY: commit all 64 KiB private subpages before resetting write-watch.
        unsafe { std::ptr::write_bytes(virgin_block.ptr, 0x77, PAGE) };
        let _ = virgin_shadow.update(virgin_block.addr(), PAGE);
        assert!(virgin_shadow.try_defer_piece(0, PAGE));
        assert_eq!(virgin_shadow.lazy_state[0].load(Ordering::Acquire), LAZY_DEFERRED);
        drop(virgin_shadow);

        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    #[test]
    fn lazy_other_owner_summary_fast_reject() {
        let _guard = LazyTestGuard::enable();
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
        assert!(!lazy_may_have_other_in_range(1, 0, PAGE * 100));
        assert!(!lazy_has_other_armed(1, 0, PAGE * 100));

        // Insert pages for owner 10 at pages 100 and 105.
        {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            lazy_table_insert_busy(&mut table, 100, 10, Arc::new(FlushWait::new()));
            lazy_table_insert_busy(&mut table, 105, 10, Arc::new(FlushWait::new()));
        }
        // Owner 10 sees NO other owner across its own entire span (lock-free).
        assert!(!lazy_may_have_other_in_range(10, 0, PAGE * 1000));
        assert!(!lazy_has_other_armed(10, 0, PAGE * 1000));
        // Another owner (20) rejects outside bounding box [100, 106) and inside gaps [101, 105).
        assert!(!lazy_may_have_other_in_range(20, 0, PAGE * 100));
        assert!(!lazy_may_have_other_in_range(20, PAGE * 106, PAGE * 200));
        assert!(!lazy_may_have_other_in_range(20, PAGE * 101, PAGE * 105));
        // Owner 20 detects overlap at page 100 and page 105.
        assert!(lazy_may_have_other_in_range(20, PAGE * 100, PAGE * 101));
        assert!(lazy_has_other_armed(20, PAGE * 100, PAGE * 101));
        assert!(lazy_has_other_armed(20, PAGE * 105, PAGE * 106));

        // Add 9 more owners to test overflow (> LAZY_OWNER_SLOTS) fallback.
        {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            for k in 1..=9usize {
                lazy_table_insert_busy(&mut table, 200 + k * 2, 100 + k as u64, Arc::new(FlushWait::new()));
            }
        }
        assert!(lazy_has_other_armed(10, PAGE * 202, PAGE * 203));
        assert!(!lazy_may_have_other_in_range(10, PAGE * 203, PAGE * 204));
        assert!(!lazy_has_other_armed(101, PAGE * 202, PAGE * 203));

        // Clean up all inserted pages and verify summary resets cleanly.
        {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            lazy_table_remove_page(&mut table, 100);
            lazy_table_remove_page(&mut table, 105);
            for k in 1..=9usize {
                lazy_table_remove_page(&mut table, 200 + k * 2);
            }
        }
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
        assert_eq!(LAZY_SUMMARY_LEN.load(Ordering::Acquire), 0);
        assert!(!lazy_may_have_other_in_range(20, PAGE * 100, PAGE * 300));
    }
}
