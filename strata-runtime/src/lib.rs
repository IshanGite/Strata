use async_trait::async_trait;
use bytes::Bytes;
use std::future::Future;
use std::io::Result;
use std::net::SocketAddr;
use std::pin::Pin;
use std::time::Duration;

#[async_trait]
pub trait Environment: Clone + Send + Sync + 'static {
    // Time
    async fn sleep(&self, duration: Duration);
    fn now(&self) -> Duration;

    // Spawning
    fn spawn(&self, future: Pin<Box<dyn Future<Output = ()> + Send + 'static>>);

    // Disk
    fn read_file_sync(&self, path: &str) -> Result<Vec<u8>>;
    fn write_file_sync(&self, path: &str, data: &[u8]) -> Result<()>;
    fn append_file_sync(&self, path: &str, data: &[u8]) -> Result<()>;
    fn remove_file_sync(&self, path: &str) -> Result<()>;

    // Network
    async fn send_msg(&self, to: SocketAddr, data: Bytes) -> Result<()>;
    async fn listen(
        &self,
        addr: SocketAddr,
    ) -> Result<tokio::sync::mpsc::UnboundedReceiver<(SocketAddr, Bytes)>>;
}

#[derive(Clone)]
pub struct TokioEnvironment;

#[async_trait]
impl Environment for TokioEnvironment {
    async fn sleep(&self, duration: Duration) {
        tokio::time::sleep(duration).await;
    }

    fn now(&self) -> Duration {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
    }

    fn spawn(&self, future: Pin<Box<dyn Future<Output = ()> + Send + 'static>>) {
        tokio::spawn(future);
    }

    fn read_file_sync(&self, path: &str) -> Result<Vec<u8>> {
        std::fs::read(path)
    }

    fn write_file_sync(&self, path: &str, data: &[u8]) -> Result<()> {
        std::fs::write(path, data)
    }

    fn append_file_sync(&self, path: &str, data: &[u8]) -> Result<()> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        file.write_all(data)
    }

    fn remove_file_sync(&self, path: &str) -> Result<()> {
        std::fs::remove_file(path)
    }

    async fn send_msg(&self, _to: SocketAddr, _data: Bytes) -> Result<()> {
        // Implementation left minimal for abstraction
        Ok(())
    }

    async fn listen(
        &self,
        _addr: SocketAddr,
    ) -> Result<tokio::sync::mpsc::UnboundedReceiver<(SocketAddr, Bytes)>> {
        let (_tx, rx) = tokio::sync::mpsc::unbounded_channel();
        Ok(rx)
    }
}

pub mod sim;
