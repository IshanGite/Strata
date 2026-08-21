use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use strata_net::{GrpcStrataClient, StrataNetworkClientTrait};
use strata_sharding::RoutingTable;
use strata_storage::HlcTimestamp;
use strata_txn::Mutation;

pub struct StrataClient {
    pub seed_addrs: Vec<String>,
    pub node_clients: Arc<Mutex<HashMap<String, GrpcStrataClient>>>,
    pub table: Arc<Mutex<RoutingTable>>,
}

impl StrataClient {
    pub async fn connect(seed_addrs: Vec<String>) -> Result<Self, String> {
        let mut clients = HashMap::new();
        let mut fetched_table = None;

        for addr in &seed_addrs {
            if let Ok(client) = GrpcStrataClient::connect(addr.clone()).await {
                if fetched_table.is_none() {
                    if let Ok(t) = client.get_routing_table().await {
                        fetched_table = Some(t);
                    }
                }
                clients.insert(addr.clone(), client);
            }
        }

        if clients.is_empty() {
            return Err("Failed to connect to any seed server".to_string());
        }

        let table = fetched_table.unwrap_or_default();

        Ok(Self {
            seed_addrs,
            node_clients: Arc::new(Mutex::new(clients)),
            table: Arc::new(Mutex::new(table)),
        })
    }

    pub async fn refresh_routing_table(&self) -> Result<(), String> {
        let clients = self.node_clients.lock().clone();
        for (_, client) in clients {
            if let Ok(table) = client.get_routing_table().await {
                *self.table.lock() = table;
                return Ok(());
            }
        }
        Err("Failed to refresh routing table".to_string())
    }

    async fn execute_with_failover<F, R, T>(&self, f: F) -> Result<T, String>
    where
        F: Fn(GrpcStrataClient) -> R,
        R: std::future::Future<Output = Result<T, String>>,
    {
        let clients = self.node_clients.lock().clone();
        let mut last_err = "No client available".to_string();

        for (_addr, client) in clients {
            match f(client).await {
                Ok(res) => return Ok(res),
                Err(e) => {
                    last_err = e;
                }
            }
        }

        for addr in &self.seed_addrs {
            if let Ok(client) = GrpcStrataClient::connect(addr.clone()).await {
                self.node_clients
                    .lock()
                    .insert(addr.clone(), client.clone());
                match f(client).await {
                    Ok(res) => return Ok(res),
                    Err(e) => last_err = e,
                }
            }
        }

        Err(last_err)
    }

    pub async fn put(&self, key: Vec<u8>, value: Vec<u8>) -> Result<(), String> {
        self.execute_with_failover(|client| {
            let k = key.clone();
            let v = value.clone();
            async move { client.put(k, v).await }
        })
        .await
    }

    pub async fn put_vector(&self, id: u64, vector: Vec<f32>) -> Result<(), String> {
        let key = format!("vec:{}", id).into_bytes();
        let value = bincode::serialize(&vector).map_err(|e| e.to_string())?;
        self.put(key, value).await
    }

    pub async fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, String> {
        let k = key.to_vec();
        self.execute_with_failover(|client| {
            let k = k.clone();
            async move { client.get(k, None).await }
        })
        .await
    }

    pub async fn delete(&self, key: &[u8]) -> Result<(), String> {
        let k = key.to_vec();
        self.execute_with_failover(|client| {
            let k = k.clone();
            async move { client.delete(k).await }
        })
        .await
    }

    pub async fn begin_txn(&self) -> Result<HlcTimestamp, String> {
        self.execute_with_failover(|client| async move { client.begin_txn().await })
            .await
    }

    pub async fn prewrite_txn(
        &self,
        start_ts: HlcTimestamp,
        mutations: Vec<Mutation>,
    ) -> Result<(), String> {
        self.execute_with_failover(|client| {
            let muts = mutations.clone();
            async move { client.prewrite_txn(start_ts, muts).await }
        })
        .await
    }

    pub async fn commit_txn(
        &self,
        start_ts: HlcTimestamp,
        commit_ts: HlcTimestamp,
        mutations: Vec<Mutation>,
    ) -> Result<(), String> {
        self.execute_with_failover(|client| {
            let muts = mutations.clone();
            async move { client.commit_txn(start_ts, commit_ts, muts).await }
        })
        .await
    }

    pub async fn abort_txn(
        &self,
        start_ts: HlcTimestamp,
        mutations: Vec<Mutation>,
    ) -> Result<(), String> {
        self.execute_with_failover(|client| {
            let muts = mutations.clone();
            async move { client.abort_txn(start_ts, muts).await }
        })
        .await
    }

    pub async fn search_knn(&self, vector: Vec<f32>, k: usize) -> Result<Vec<(u64, f32)>, String> {
        self.execute_with_failover(|client| {
            let vec = vector.clone();
            async move { client.search_knn(vec, k, None, None).await }
        })
        .await
    }

    pub async fn search_knn_advanced(
        &self,
        vector: Vec<f32>,
        k: usize,
        radius: Option<f32>,
        filter: Option<roaring::RoaringBitmap>,
    ) -> Result<Vec<(u64, f32)>, String> {
        self.execute_with_failover(|client| {
            let vec = vector.clone();
            let filt = filter.clone();
            async move { client.search_knn(vec, k, radius, filt).await }
        })
        .await
    }

    pub async fn search_range(
        &self,
        start_key: &[u8],
        end_key: &[u8],
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>, String> {
        let s_key = start_key.to_vec();
        let e_key = end_key.to_vec();
        self.execute_with_failover(|client| {
            let sk = s_key.clone();
            let ek = e_key.clone();
            async move { client.search_range(sk, ek).await }
        })
        .await
    }
}
