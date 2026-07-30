//! Fork-join thread pool with spin-then-park workers, tuned for microsecond-scale
//! per-layer barriers. Workers spin briefly after each job so sequential layer
//! dispatches within one request never pay a wakeup syscall.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::Arc;
use std::thread::{self, JoinHandle, Thread};

const SPIN_ITERS: u32 = 200_000;

type Job = *const (dyn Fn(usize) + Sync);

struct Shared {
    seq: AtomicU64,
    remaining: AtomicUsize,
    job: UnsafeCell<Option<Job>>,
    shutdown: AtomicBool,
}
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

struct Worker {
    thread: Thread,
    parked: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<Worker>,
    n: usize,
}

fn pin_to_cpu(cpu: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

impl Pool {
    /// `n` total participants (main thread counts as one). `pin`: optional CPU ids, one per participant.
    pub fn new(n: usize, pin: Option<&[usize]>) -> Pool {
        assert!(n >= 1);
        if let Some(cpus) = pin {
            pin_to_cpu(cpus[0]);
        }
        let shared = Arc::new(Shared {
            seq: AtomicU64::new(0),
            remaining: AtomicUsize::new(0),
            job: UnsafeCell::new(None),
            shutdown: AtomicBool::new(false),
        });
        let workers = (1..n)
            .map(|tid| {
                let shared = shared.clone();
                let parked = Arc::new(AtomicBool::new(false));
                let parked2 = parked.clone();
                let cpu = pin.map(|c| c[tid]);
                let handle = thread::Builder::new()
                    .name(format!("gemm-{tid}"))
                    .spawn(move || {
                        if let Some(cpu) = cpu {
                            pin_to_cpu(cpu);
                        }
                        let mut last_seq = 0u64;
                        loop {
                            let mut spins = 0u32;
                            let seq = loop {
                                let s = shared.seq.load(Acquire);
                                if s != last_seq {
                                    break s;
                                }
                                spins += 1;
                                if spins > SPIN_ITERS {
                                    parked2.store(true, Release);
                                    if shared.seq.load(Acquire) != last_seq {
                                        parked2.store(false, Release);
                                        continue;
                                    }
                                    thread::park();
                                    parked2.store(false, Release);
                                    spins = 0;
                                } else {
                                    std::hint::spin_loop();
                                }
                            };
                            last_seq = seq;
                            if shared.shutdown.load(Acquire) {
                                break;
                            }
                            let job = unsafe { (*shared.job.get()).unwrap() };
                            unsafe { (*job)(tid) };
                            shared.remaining.fetch_sub(1, AcqRel);
                        }
                    })
                    .unwrap();
                Worker { thread: handle.thread().clone(), parked, handle: Some(handle) }
            })
            .collect();
        Pool { shared, workers, n }
    }

    pub fn n_threads(&self) -> usize {
        self.n
    }

    pub fn run(&self, f: &(dyn Fn(usize) + Sync)) {
        if self.n == 1 {
            f(0);
            return;
        }
        let job: Job = unsafe { std::mem::transmute(f) };
        unsafe { *self.shared.job.get() = Some(job) };
        self.shared.remaining.store(self.n - 1, Relaxed);
        self.shared.seq.fetch_add(1, Release);
        for w in &self.workers {
            if w.parked.load(Acquire) {
                w.thread.unpark();
            }
        }
        f(0);
        while self.shared.remaining.load(Acquire) != 0 {
            std::hint::spin_loop();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Release);
        self.shared.seq.fetch_add(1, Release);
        for w in &mut self.workers {
            w.thread.unpark();
            if let Some(h) = w.handle.take() {
                let _ = h.join();
            }
        }
    }
}
