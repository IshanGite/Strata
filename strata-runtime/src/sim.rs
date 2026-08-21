use super::Environment;
use async_trait::async_trait;
use bytes::Bytes;
use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::io::{Error, ErrorKind, Result};
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::Duration;
use tokio::sync::mpsc;

#[derive(Clone)]
pub struct SimulatedEnvironment {
    pub state: Arc<Mutex<SimState>>,
}

pub struct SimState {
    pub clock: Duration,
    pub timers: BTreeMap<Duration, Vec<Waker>>,
    pub files: HashMap<String, Vec<u8>>,
    pub network: HashMap<SocketAddr, mpsc::UnboundedSender<(SocketAddr, Bytes)>>,
    // inject faults here
    pub drop_rate: f64,
}

impl SimulatedEnvironment {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(SimState {
                clock: Duration::ZERO,
                timers: BTreeMap::new(),
                files: HashMap::new(),
                network: HashMap::new(),
                drop_rate: 0.0,
            })),
        }
    }

    // The test runner calls this
    pub fn step(&self) {
        let mut state = self.state.lock().unwrap();
        if let Some((&time, _)) = state.timers.iter().next() {
            state.clock = time;
            let wakers = state.timers.remove(&time).unwrap();
            for waker in wakers {
                waker.wake();
            }
        }
    }

    pub fn set_drop_rate(&self, rate: f64) {
        self.state.lock().unwrap().drop_rate = rate;
    }
}

pub struct SleepFuture {
    env: SimulatedEnvironment,
    deadline: Duration,
    registered: bool,
}

impl Future for SleepFuture {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let clock = { self.env.state.lock().unwrap().clock };
        if clock >= self.deadline {
            Poll::Ready(())
        } else {
            if !self.registered {
                self.env
                    .state
                    .lock()
                    .unwrap()
                    .timers
                    .entry(self.deadline)
                    .or_default()
                    .push(cx.waker().clone());
                self.registered = true;
            }
            Poll::Pending
        }
    }
}

#[async_trait]
impl Environment for SimulatedEnvironment {
    async fn sleep(&self, duration: Duration) {
        let deadline = {
            let state = self.state.lock().unwrap();
            state.clock + duration
        };
        SleepFuture {
            env: self.clone(),
            deadline,
            registered: false,
        }
        .await
    }

    fn now(&self) -> Duration {
        self.state.lock().unwrap().clock
    }

    fn spawn(&self, future: Pin<Box<dyn Future<Output = ()> + Send + 'static>>) {
        tokio::spawn(future);
    }

    fn read_file_sync(&self, path: &str) -> Result<Vec<u8>> {
        let state = self.state.lock().unwrap();
        if let Some(data) = state.files.get(path) {
            Ok(data.clone())
        } else {
            Err(Error::new(ErrorKind::NotFound, "file not found"))
        }
    }

    fn write_file_sync(&self, path: &str, data: &[u8]) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        state.files.insert(path.to_string(), data.to_vec());
        Ok(())
    }

    fn append_file_sync(&self, path: &str, data: &[u8]) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        state
            .files
            .entry(path.to_string())
            .or_default()
            .extend_from_slice(data);
        Ok(())
    }

    fn remove_file_sync(&self, path: &str) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        state.files.remove(path);
        Ok(())
    }

    async fn send_msg(&self, to: SocketAddr, data: Bytes) -> Result<()> {
        let state = self.state.lock().unwrap();
        // apply drop rate fault injection
        use rand::Rng;
        let mut rng = rand::thread_rng();
        if rng.gen_bool(state.drop_rate) {
            return Ok(()); // Dropped silently
        }

        if let Some(tx) = state.network.get(&to) {
            let from = "0.0.0.0:0".parse().unwrap();
            let _ = tx.send((from, data));
        }
        Ok(())
    }

    async fn listen(
        &self,
        addr: SocketAddr,
    ) -> Result<mpsc::UnboundedReceiver<(SocketAddr, Bytes)>> {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut state = self.state.lock().unwrap();
        state.network.insert(addr, tx);
        Ok(rx)
    }
}
