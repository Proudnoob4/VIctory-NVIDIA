//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and
//! the title's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//!
//! The default budget is half a second per title frame. A pass the title
//! runs only once and whose pipeline takes longer is lost. Passes over the
//! screen (no vertex input, no depth) have workers and a longer budget of
//! their own ([`screen_budget`]): the title fills its lighting tables with
//! such draws once, at the world's first frame, and behind a world's worth of
//! pipelines they were dropped; that run had no lamps and no diffuse light to
//! its end (run textures-d2, the first after a new translation; Belobug
//! 18e57db8 found the same).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline, TargetFormats};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queue_state: AtomicU8,
    /// Taken by a worker from [`Queue::slow`].
    was_slow: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw or dispatch has requested it yet.
    key: String,
    used: AtomicBool,
    /// A pass over the screen ([`GraphicsState::screen_pass`]).
    screen: bool,
    /// A compute pipeline whose dispatches wait outside the graphics budget.
    compute: bool,
    compute_dropped: AtomicBool,
    compute_given_up: AtomicBool,
    /// When a worker began making it, excluding time in the queue.
    making_since: OnceLock<Instant>,
    /// Lets a scheduler test publish its result only once the caller waits.
    #[cfg(test)]
    wait_entered: Mutex<Option<std::sync::mpsc::Sender<()>>>,
    /// LOCAL: how long draws have waited for it so far, in microseconds
    /// ([`wait_once`]).
    waited_us: AtomicU64,
    /// LOCAL (fxceless a35c557393): the formats its state declares (zeros
    /// without a graphics state), and the same shaders and state for other
    /// formats bound at a draw ([`for_formats`]): a chain, one link per set.
    pub(crate) formats: TargetFormats,
    variant: OnceLock<Arc<Pending>>,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(graphics.as_ref().map(|g| g.describe()).unwrap_or_default().as_bytes());
        Pending {
            formats: graphics.as_ref().map_or([0; 9], |g| g.formats()),
            variant: OnceLock::new(),
            screen: graphics.as_ref().is_some_and(|g| g.screen_pass()),
            compute: graphics.is_none()
                && shaders.first().is_some_and(|s| s.stage == "compute")
                && compute_waits(),
            compute_dropped: AtomicBool::new(false),
            compute_given_up: AtomicBool::new(false),
            making_since: OnceLock::new(),
            #[cfg(test)]
            wait_entered: Mutex::new(None),
            shaders,
            graphics,
            queue_state: AtomicU8::new(NOT_QUEUED),
            was_slow: AtomicBool::new(false),
            made: OnceLock::new(),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
            waited_us: AtomicU64::new(0),
        }
    }

    pub(super) fn key(&self) -> &str {
        &self.key
    }

    fn make(&self) {
        self.making_since.get_or_init(Instant::now);
        self.made.get_or_init(|| {
            let began = Instant::now();
            let made = HostPipeline::new(&self.shaders, self.graphics.as_deref());
            super::library::note_time(&self.key, began.elapsed().as_millis() as u64);
            made
        });
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some("being made".into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// The hashes of the shaders `p` is made of.
pub(crate) fn hashes(p: &Pending) -> String {
    let hashes: Vec<&str> = p.shaders.iter().map(|s| s.hash.as_str()).collect();
    hashes.join("+")
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads.
fn wait_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_PIPELINE_WAIT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(500);
        tracing::info!(millis, "host pipeline wait budget per frame");
        Duration::from_millis(millis)
    })
}

/// As [`wait_budget`], for passes over the screen (`VICTORY_SCREEN_WAIT_MS`,
/// 10 s; 0: like the others). Below the title's 30 s watchdog.
fn screen_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_SCREEN_WAIT_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(10_000);
        Duration::from_millis(millis)
    })
}

/// Whether passes over the screen go before the others.
fn screens_first() -> bool {
    !screen_budget().is_zero()
}

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// Pipelines made again for the formats bound at a draw, since the start.
static VARIANTS: AtomicU64 = AtomicU64::new(0);

/// Whether a draw into views of other formats than its pipeline declares
/// gets a host pipeline for those formats ([`for_formats`]); on unless
/// `VICTORY_TARGET_VARIANTS=0`.
fn variants_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TARGET_VARIANTS").is_none_or(|v| v != "0"))
}

/// The host pipeline for a draw with `p` into views of the formats `bound`
/// that `p` does not declare: that of the same shaders and state for those
/// formats, queued and waited for like the title's own.
#[cold]
pub(crate) fn for_formats<'a>(p: &'a Arc<Pending>, bound: &TargetFormats) -> Option<&'a HostPipeline> {
    match p.graphics.as_deref() {
        Some(g) if variants_on() => ready(variant(p, g, &super::host::formats_for(&p.formats, bound))),
        _ => ready(p),
    }
}

/// The pipeline of `p`'s shaders and state `g` for the formats `bound`:
/// found in the chain from `p`, or added to it (logged once per set).
fn variant<'a>(p: &'a Arc<Pending>, g: &GraphicsState, bound: &TargetFormats) -> &'a Arc<Pending> {
    let mut at = p;
    for _ in 0..64 {
        let next = at.variant.get_or_init(|| {
            let so_far = VARIANTS.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::info!(
                pipeline = %hashes(p),
                declared = ?p.formats,
                bound = ?bound,
                so_far,
                "a pipeline is made again for the render target formats bound at its draw (VICTORY_TARGET_VARIANTS=0: the draw goes out with the title's pipeline)"
            );
            Arc::new(Pending::new(p.shaders.clone(), Some(Box::new(g.with_formats(bound)))))
        });
        if next.formats == *bound {
            return next;
        }
        at = next;
    }
    p
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    ready_with(p, &WAIT_STATE)
}

fn ready_with<'a>(p: &'a Arc<Pending>, waits: &WaitState) -> Option<&'a HostPipeline> {
    // A still-pending pipeline was requested by the title too. Prewarm it
    // on the next launch, even if this run never got to use it.
    if !p.used.load(Ordering::Relaxed) && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    if p.made.get().is_none() {
        start(p, REQUESTED);
        wait(p, crate::frame::current(), waits);
    }
    p.made.get()?.as_ref()
}

/// Makes `p` on this thread if it is not made yet (waiting for a worker
/// that is making it); whether it has a host pipeline.
pub(crate) fn now(p: &Arc<Pending>) -> bool {
    p.make();
    p.made.get().is_some_and(|made| made.is_some())
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, EARLY);
    }
}

/// Whether the frame's budget is one for all threads together: a thread
/// that waits takes what is left of it and gives back what it did not
/// use. Counted only when a wait had ended, the budget was one per thread:
/// each of the title's five recording threads began its first wait of a
/// frame with nothing counted yet and waited the whole half second, 2.5 s
/// of waiting in every frame in which pipelines were being made (runs
/// 34-49: "2504 ms waiting for pipelines" in each such frame).
/// On by default; `VICTORY_PIPELINE_WAIT_SHARED=0` counts as before.
fn shared_budget() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_SHARED").is_none_or(|v| v != "0"))
}

/// LOCAL (the problem of fxceless issue #1181): whether one pipeline is
/// waited for at most one budget over the whole run, not one budget every
/// frame. A pipeline whose make takes longer than the budget (a driver
/// compile of tens of seconds, or one that never ends) otherwise cost the
/// whole budget in every frame in which a draw asked for it: with
/// `VICTORY_PIPELINE_WAIT_MS=10000` the picture stood still while the thing
/// drawn with it was in view, and moved again when the camera turned away.
/// After its budget its draws are dropped until it is made, as with any
/// pipeline past the frame's budget. On by default;
/// `VICTORY_PIPELINE_WAIT_ONCE=0` waits as before.
fn wait_once() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_ONCE").is_none_or(|v| v != "0"))
}

/// A once-only compute dispatch loses its output for the whole session if
/// its pipeline is still queued after the graphics budget is spent. Under
/// Wine, wait separately and promote it ahead of requested graphics. This
/// is not Wine-specific, but other hosts retain their defaults until tested.
/// `VICTORY_COMPUTE_WAIT=1` enables it anywhere; `0` disables it.
/// Ported from Beqren's f63017a (avatarmanz/VIctory-7900XTX#166).
fn compute_waits() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = crate::wine::on_under_wine("VICTORY_COMPUTE_WAIT");
        if on {
            tracing::info!(secs = COMPUTE_BUDGET.as_secs(), "compute dispatches wait for their pipelines");
        }
        on
    })
}

/// One wall-clock deadline for all compute waits in a frame, also the
/// maximum age of a worker's unfinished compute job. Queuing is separate:
/// a dispatch promotes its pipeline before waiting. The default graphics
/// and screen budgets plus this bound stay below the title's watchdog.
const COMPUTE_BUDGET: Duration = Duration::from_secs(10);

#[derive(Default)]
struct ComputeDeadline {
    first: Option<(u64, Instant)>,
}

impl ComputeDeadline {
    fn left(&mut self, frame: u64, now: Instant, making: Duration) -> Duration {
        // An older recording thread must not reset a newer frame's budget.
        if self.first.is_none_or(|(first_frame, _)| frame > first_frame) {
            self.first = Some((frame, now));
        }
        let (_, began) = self.first.unwrap();
        COMPUTE_BUDGET
            .saturating_sub(now.saturating_duration_since(began))
            .min(COMPUTE_BUDGET.saturating_sub(making))
    }
}

/// Distinct accounting: compute callers share a deadline, while ordinary
/// and screen callers reserve portions of their accumulated frame budgets.
struct WaitState {
    compute_first: Mutex<ComputeDeadline>,
    spent: [Mutex<(u64, Duration)>; 2],
}

impl WaitState {
    const fn new() -> Self {
        Self {
            compute_first: Mutex::new(ComputeDeadline { first: None }),
            spent: [const { Mutex::new((u64::MAX, Duration::ZERO)) }; 2],
        }
    }
}

static WAIT_STATE: WaitState = WaitState::new();

/// Waits for `p` while frame `frame` has budget left.
fn wait(p: &Pending, frame: u64, waits: &WaitState) {
    let screen = p.screen && screens_first();
    let making = p.making_since.get().map_or(Duration::ZERO, Instant::elapsed);
    let compute = p.compute && making < COMPUTE_BUDGET;
    let budget = if screen {
        screen_budget()
    } else if compute {
        COMPUTE_BUDGET
    } else {
        wait_budget()
    };
    if budget.is_zero() {
        return;
    }
    // LOCAL: what is left of this pipeline's own budget ([`wait_once`]).
    let own = if !compute && wait_once() {
        let own = budget.saturating_sub(Duration::from_micros(p.waited_us.load(Ordering::Relaxed)));
        if own.is_zero() {
            return;
        }
        own
    } else {
        Duration::MAX
    };
    // The frame the budget belongs to, and what was waited of it (or is
    // taken by a thread waiting now); of the passes over the screen too.
    let spent_of = &waits.spent[usize::from(screen)];
    let shared = shared_budget();
    let left = if compute {
        waits.compute_first
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .left(frame, Instant::now(), making)
    } else {
        let mut spent = spent_of.lock().unwrap_or_else(|e| e.into_inner());
        if spent.0 != frame {
            *spent = (frame, Duration::ZERO);
        }
        let left = budget.saturating_sub(spent.1).min(own);
        if shared {
            spent.1 += left;
        }
        left
    };
    if left.is_zero() {
        if compute && p.made.get().is_none() && !p.compute_dropped.swap(true, Ordering::Relaxed) {
            tracing::info!(
                shaders = %hashes(p),
                frame,
                "a compute dispatch was dropped: this frame's compute wait deadline expired"
            );
        }
        return;
    }
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        #[cfg(test)]
        if let Some(entered) = p.wait_entered.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = entered.send(());
        }
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
    drop(held);
    let waited = started.elapsed().min(left);
    if !compute && wait_once() && p.made.get().is_none() {
        let before = p.waited_us.fetch_add(waited.as_micros() as u64, Ordering::Relaxed);
        if Duration::from_micros(before) + waited >= budget {
            tracing::info!(
                shaders = ?p.shaders,
                ms = budget.as_millis() as u64,
                "a pipeline not made within its wait: its draws are dropped until it is (VICTORY_PIPELINE_WAIT_ONCE=0: waited for every frame)"
            );
        }
    }
    if screen && waited > Duration::from_millis(500) {
        tracing::info!(
            ms = waited.as_millis() as u64,
            made = p.made.get().is_some(),
            shaders = ?p.shaders,
            "a pass over the screen was waited for (VICTORY_SCREEN_WAIT_MS=0: dropped past half a second a frame)"
        );
    }
    if !compute {
        let mut spent = spent_of.lock().unwrap_or_else(|e| e.into_inner());
        // A frame that began meanwhile has its own budget.
        if spent.0 == frame {
            if shared {
                spent.1 = spent.1.saturating_sub(left - waited);
            } else {
                spent.1 += waited;
            }
        }
    }
    if compute && p.made.get().is_none() && !p.compute_dropped.swap(true, Ordering::Relaxed) {
        tracing::info!(
            shaders = %hashes(p),
            frame,
            waited_ms = waited.as_millis() as u64,
            "a compute dispatch was dropped: its pipeline was not made within this frame's compute waits"
        );
    }
    if compute
        && p.made.get().is_none()
        && p.making_since.get().is_some_and(|at| at.elapsed() >= COMPUTE_BUDGET)
        && !p.compute_given_up.swap(true, Ordering::Relaxed)
    {
        tracing::warn!(
            shaders = %hashes(p),
            "a compute pipeline was not made in 10 s: its dispatches use the ordinary budget from now on"
        );
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        crate::copystat::C::PipelineWaitUs,
        started.elapsed().as_micros() as u64,
    );
}

const NOT_QUEUED: u8 = 0;
const EARLY: u8 = 1;
const REQUESTED: u8 = 2;
const MAKING: u8 = 3;

/// How many workers may make pipelines at once that are known to take long
/// and that no draw has asked for: a quarter of them. On a fresh cache (a
/// change of the translator or the driver) the early list begins with
/// programs that take tens of seconds each: all 16 workers were in them
/// through the whole of the title's loading, 97 pipelines made in 20 s, and
/// the first minute and a half of the world ran at 0.5 to 1.6 s a frame,
/// dropping up to 2,900 draws a frame (run shaders-1).
/// `VICTORY_PIPELINE_SLOW=0`: in the order the title creates them, as before.
fn slow_limit() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        if std::env::var_os("VICTORY_PIPELINE_SLOW").is_some_and(|v| v == "0") {
            usize::MAX
        } else {
            (worker_count(None, std::thread::available_parallelism().map_or(2, |n| n.get())) / 4).max(1)
        }
    })
}

struct Queue {
    /// Requested passes over the screen: before all others, and the only ones
    /// [`SCREEN_WORKERS`] take.
    screen: VecDeque<Arc<Pending>>,
    /// Requested compute pipelines: before requested graphics, after screen passes.
    compute: VecDeque<Arc<Pending>>,
    early: VecDeque<Arc<Pending>>,
    /// Those of the early ones known to take long, and how many of them are
    /// being made ([`slow_limit`]).
    slow: VecDeque<Arc<Pending>>,
    slow_now: usize,
    requested: VecDeque<Arc<Pending>>,
}

impl Queue {
    const fn new() -> Self {
        Self {
            screen: VecDeque::new(),
            compute: VecDeque::new(),
            early: VecDeque::new(),
            slow: VecDeque::new(),
            slow_now: 0,
            requested: VecDeque::new(),
        }
    }

    /// Queues an unrequested pipeline or promotes one queued at creation.
    /// Returns whether it is new to the queue, or `None` if nothing changed.
    fn push(&mut self, p: &Arc<Pending>, priority: u8) -> Option<bool> {
        self.push_as(p, priority, slow_limit() != usize::MAX && super::library::slow(&p.key))
    }

    fn push_as(&mut self, p: &Arc<Pending>, priority: u8, slow: bool) -> Option<bool> {
        let prev = p.queue_state.load(Ordering::Relaxed);
        if prev >= priority {
            return None;
        }
        p.queue_state.store(priority, Ordering::Relaxed);
        if priority == REQUESTED && p.screen && screens_first() {
            self.screen.push_back(p.clone());
        } else if priority == REQUESTED && p.compute {
            self.compute.push_back(p.clone());
        } else if priority == REQUESTED {
            self.requested.push_back(p.clone());
        } else if slow {
            self.slow.push_back(p.clone());
        } else {
            self.early.push_back(p.clone());
        }
        Some(prev == NOT_QUEUED)
    }

    fn pop(&mut self, screens_only: bool, slow_limit: usize) -> Option<Arc<Pending>> {
        // An early entry can have been promoted and claimed already.
        let unclaimed = |list: &mut VecDeque<Arc<Pending>>| {
            std::iter::from_fn(|| list.pop_front()).find(|p| p.queue_state.load(Ordering::Relaxed) == EARLY)
        };
        let mut p = self.screen.pop_front();
        if p.is_none() && !screens_only {
            p = self.compute.pop_front()
                .or_else(|| self.requested.pop_front())
                .or_else(|| unclaimed(&mut self.early));
            if p.is_none() && self.slow_now < slow_limit {
                p = unclaimed(&mut self.slow);
                if let Some(p) = &p {
                    p.was_slow.store(true, Ordering::Relaxed);
                    self.slow_now += 1;
                }
            }
        }
        let p = p?;
        p.queue_state.store(MAKING, Ordering::Relaxed);
        Some(p)
    }
}

static QUEUE: Mutex<Queue> = Mutex::new(Queue::new());
static AVAILABLE: Condvar = Condvar::new();

/// Start or promote work for a draw without waiting behind the entire early
/// list. Queue changes and worker claims happen under the same lock.
fn start(p: &Arc<Pending>, priority: u8) {
    if p.queue_state.load(Ordering::Relaxed) >= priority {
        return;
    }
    if !workers() {
        if p.queue_state.swap(MAKING, Ordering::Relaxed) == NOT_QUEUED {
            p.make();
        }
        return;
    }
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(new) = queue.push(p, priority) {
        if new {
            PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        }
        // (One woken could be a worker that takes passes over the screen
        // alone: it would leave the pipeline to nobody.)
        AVAILABLE.notify_all();
    }
}

/// Workers that make passes over the screen alone, besides the others.
const SCREEN_WORKERS: usize = 2;

fn next(screens_only: bool) -> Arc<Pending> {
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some(p) = queue.pop(screens_only, slow_limit()) {
            return p;
        }
        queue = AVAILABLE.wait(queue).unwrap_or_else(|e| e.into_inner());
    }
}

/// How many workers make pipelines: `asked` (`VICTORY_PIPELINE_WORKERS`, 1 to
/// 64), else two thirds of the logical processors, 2 to 16. They run below the
/// title's own threads, and a cold cache queues hundreds at once.
fn worker_count(asked: Option<usize>, processors: usize) -> usize {
    match asked {
        Some(n) => n.clamp(1, 64),
        None => (processors * 2 / 3).clamp(2, 16),
    }
}

/// Starts the workers ([`worker_count`]) on first use.
fn workers() -> bool {
    static RUNNING: OnceLock<bool> = OnceLock::new();
    *RUNNING.get_or_init(|| {
        let mut running = false;
        let workers = worker_count(
            std::env::var("VICTORY_PIPELINE_WORKERS")
                .ok()
                .and_then(|v| v.trim().parse().ok()),
            std::thread::available_parallelism().map_or(2, |n| n.get()),
        );
        tracing::info!(workers, "pipeline workers");
        let screens = if screens_first() { SCREEN_WORKERS } else { 0 };
        for i in 0..workers + screens {
            let screens_only = i >= workers;
            let spawned = std::thread::Builder::new()
                .name(format!("victory-pipelines-{i}"))
                .spawn(move || {
                    // Below the title's own threads (loading, recording),
                    // and the streaming one, which is below normal itself
                    // (`VICTORY_PIPELINE_LOWEST=0`: level with it, as before).
                    let lowest = !screens_only && std::env::var_os("VICTORY_PIPELINE_LOWEST").is_none_or(|v| v != "0");
                    // SAFETY: a plain call on our own thread.
                    unsafe {
                        use windows::Win32::System::Threading::*;
                        let priority = if lowest { THREAD_PRIORITY_LOWEST } else { THREAD_PRIORITY_BELOW_NORMAL };
                        let _ = SetThreadPriority(GetCurrentThread(), priority);
                    }
                    loop {
                        let p = next(screens_only);
                        p.make();
                        if p.was_slow.load(Ordering::Relaxed) {
                            let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
                            queue.slow_now -= 1;
                            AVAILABLE.notify_all();
                        }
                        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                            drained();
                        } else {
                            super::library::save(false);
                        }
                    }
                });
            if let Err(e) = spawned {
                tracing::warn!(error = %e, "no pipeline worker");
            } else {
                running = true;
            }
        }
        running
    })
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_count_follows_the_processors_and_the_switch() {
        assert_eq!(worker_count(None, 1), 2);
        assert_eq!(worker_count(None, 24), 16);
        assert_eq!(worker_count(None, 64), 16);
        assert_eq!(worker_count(Some(0), 24), 1);
        assert_eq!(worker_count(Some(20), 24), 20);
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert_ne!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    #[test]
    fn pending_pipeline_is_recorded_for_the_next_launch() {
        let mut p = Pending::new(Vec::new(), None);
        p.key = "pending-pipeline-test".to_owned();
        // Simulate a worker that has taken the job but cannot finish it yet.
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let p = Arc::new(p);

        assert!(!super::super::library::wanted_early(&p.key));
        assert!(ready(&p).is_none());
        assert!(p.made.get().is_none());
        assert!(super::super::library::wanted_early(&p.key));
    }

    #[test]
    fn draws_promote_early_pipelines_without_claiming_them_twice() {
        let earlier = Arc::new(Pending::new(Vec::new(), None));
        let promoted = Arc::new(Pending::new(Vec::new(), None));
        let requested = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();

        assert_eq!(queue.push(&earlier, EARLY), Some(true));
        assert_eq!(queue.push(&promoted, EARLY), Some(true));
        assert_eq!(queue.push(&requested, REQUESTED), Some(true));
        assert_eq!(queue.push(&promoted, REQUESTED), Some(false));
        assert_eq!(queue.push(&promoted, REQUESTED), None);

        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &requested));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &promoted));
        assert_eq!(queue.push(&promoted, REQUESTED), None);
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &earlier));
        assert!(queue.pop(false, 1).is_none());
    }

    #[test]
    fn pipelines_known_to_take_long_leave_workers_for_the_others() {
        let new = || Arc::new(Pending::new(Vec::new(), None));
        let (long, longer, asked, quick) = (new(), new(), new(), new());
        let mut queue = Queue::new();
        for p in [&long, &longer, &asked] {
            assert_eq!(queue.push_as(p, EARLY, true), Some(true));
        }
        assert_eq!(queue.push_as(&quick, EARLY, false), Some(true));
        // The quick one first, then one long one at a time; one a draw asks
        // for waits for nobody.
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &quick));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &long));
        assert!(queue.pop(false, 1).is_none());
        assert_eq!(queue.push_as(&asked, REQUESTED, true), Some(false));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &asked));
        assert!(!asked.was_slow.load(Ordering::Relaxed));
        assert!(queue.pop(false, 1).is_none());
        queue.slow_now -= 1;
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &longer));
        assert!(queue.pop(false, 1).is_none());
    }

    /// Passes over the screen go before all others, and workers of their own
    /// take nothing else.
    #[test]
    fn passes_over_the_screen_go_first() {
        let of = |screen| {
            let mut p = Pending::new(Vec::new(), None);
            p.screen = screen;
            Arc::new(p)
        };
        let (world, screen) = (of(false), of(true));
        let mut queue = Queue::new();
        assert_eq!(queue.push(&world, REQUESTED), Some(true));
        assert_eq!(queue.push(&screen, REQUESTED), Some(true));
        assert!(Arc::ptr_eq(&queue.pop(true, 1).unwrap(), &screen));
        assert!(queue.pop(true, 1).is_none());
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &world));
    }

    #[test]
    fn waiting_compute_dispatches_precede_graphics_and_are_claimed_once() {
        let of = |compute, screen| {
            let mut p = Pending::new(Vec::new(), None);
            p.compute = compute;
            p.screen = screen;
            Arc::new(p)
        };
        let (world, compute, screen, early) =
            (of(false, false), of(true, false), of(false, true), of(false, false));
        let mut queue = Queue::new();
        assert_eq!(queue.push(&compute, EARLY), Some(true));
        assert_eq!(queue.push(&early, EARLY), Some(true));
        assert_eq!(queue.push(&world, REQUESTED), Some(true));
        assert_eq!(queue.push(&compute, REQUESTED), Some(false));
        assert_eq!(queue.push(&screen, REQUESTED), Some(true));
        assert!(Arc::ptr_eq(&queue.pop(true, 1).unwrap(), &screen));
        assert!(queue.pop(true, 1).is_none());
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &compute));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &world));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &early));
        assert!(queue.pop(false, 1).is_none());
    }

    #[test]
    fn compute_deadline_is_shared_and_older_frames_cannot_renew_it() {
        let start = Instant::now();
        let seconds = Duration::from_secs;
        let mut deadline = ComputeDeadline::default();
        assert_eq!(deadline.left(20, start, Duration::ZERO), COMPUTE_BUDGET);
        // A second caller does not find a reservation that excludes it.
        assert_eq!(deadline.left(20, start + seconds(3), Duration::ZERO), seconds(7));
        // Time already spent compiling limits this caller more tightly.
        assert_eq!(deadline.left(20, start + seconds(3), seconds(8)), seconds(2));
        assert_eq!(deadline.left(20, start + seconds(10), Duration::ZERO), Duration::ZERO);
        assert_eq!(deadline.left(19, start + seconds(11), Duration::ZERO), Duration::ZERO);
        assert_eq!(deadline.left(21, start + seconds(12), Duration::ZERO), COMPUTE_BUDGET);
        assert_eq!(deadline.left(20, start + seconds(13), Duration::ZERO), seconds(9));
        assert_eq!(deadline.left(21, start + seconds(13), COMPUTE_BUDGET), Duration::ZERO);
    }

    /// Exercise the ready path itself after graphics and cumulative waits
    /// are exhausted. The result is published only once the caller enters
    /// the condition wait; no sleep or driver pipeline is needed.
    #[test]
    fn ready_compute_waits_for_publication_after_graphics_budget_is_spent() {
        let mut p = Pending::new(Vec::new(), None);
        p.compute = true;
        p.used.store(true, Ordering::Relaxed);
        p.queue_state.store(MAKING, Ordering::Relaxed);
        p.making_since.set(Instant::now()).unwrap();
        p.waited_us.store(u64::MAX, Ordering::Relaxed);
        let (entered, waiting) = std::sync::mpsc::channel();
        *p.wait_entered.lock().unwrap() = Some(entered);
        let p = Arc::new(p);
        let waits = WaitState::new();
        *waits.spent[0].lock().unwrap() = (crate::frame::current(), Duration::MAX);
        std::thread::scope(|scope| {
            let caller = scope.spawn(|| {
                assert!(ready_with(&p, &waits).is_none());
                p.made.get().is_some()
            });
            let entered = waiting.recv_timeout(Duration::from_secs(2));
            // Publish a terminal host refusal, just as a worker does. Holding
            // DONE also guarantees that the notification cannot be missed.
            let held = DONE.0.lock().unwrap();
            assert!(p.made.set(None).is_ok());
            DONE.1.notify_all();
            drop(held);
            assert!(caller.join().unwrap(), "ready returned before the worker result");
            assert!(entered.is_ok(), "the compute caller did not wait: {entered:?}");
        });
        assert!(!p.compute_dropped.load(Ordering::Relaxed));
        assert_eq!(waits.spent[0].lock().unwrap().1, Duration::MAX);
    }

    #[test]
    fn expired_compute_deadline_and_old_workers_do_not_restart_long_waits() {
        let waits = WaitState::new();
        let frame = crate::frame::current();
        *waits.spent[0].lock().unwrap() = (frame, Duration::MAX);
        waits.compute_first.lock().unwrap().first = Some((frame, Instant::now() - COMPUTE_BUDGET));
        let mut p = Pending::new(Vec::new(), None);
        p.compute = true;
        p.used.store(true, Ordering::Relaxed);
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let p = Arc::new(p);
        assert!(ready_with(&p, &waits).is_none());
        assert!(p.compute_dropped.load(Ordering::Relaxed));
        assert!(p.made.get().is_none());
        // A worker older than the compute bound uses the exhausted ordinary
        // budget, even when a newer frame has a fresh compute deadline.
        p.making_since.set(Instant::now() - COMPUTE_BUDGET).unwrap();
        waits.compute_first.lock().unwrap().first = None;
        assert!(ready_with(&p, &waits).is_none());
        assert!(waits.compute_first.lock().unwrap().first.is_none());
    }
}
