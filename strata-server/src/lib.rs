use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tonic::transport::Server;

use strata_consensus::NodeId;
use strata_index::hnsw::{HnswConfig, HnswIndex};
use strata_index::AnnIndex;
use strata_net::proto::{
    raft_service_server::{RaftService, RaftServiceServer},
    strata_service_server::{StrataService, StrataServiceServer},
    *,
};
use strata_net::{GrpcRaftTransport, GrpcStrataClient, StrataNetworkClientTrait};
use strata_planner::{QueryPlanner, VectorQuery};
use strata_sharding::{MultiRaftNode, RoutingTable, ShardCommand, ShardId, ShardRouter};
use strata_storage::HlcTimestamp;
use strata_txn::{DistributedTxnCoordinator, Hlc, Mutation, TransactionCoordinator};

pub struct StrataServerDaemon {
    pub node_id: NodeId,
    pub addr: String,
    pub db_dir: PathBuf,
    pub table: Arc<Mutex<RoutingTable>>,
    pub multi_raft: Arc<MultiRaftNode<GrpcRaftTransport>>,
    pub raft_transport: Arc<GrpcRaftTransport>,
    pub coordinator: Arc<DistributedTxnCoordinator<GrpcRaftTransport>>,
    pub planner: Arc<QueryPlanner>,
    pub hlc: Arc<Hlc>,
    pub indexes: Arc<Mutex<HashMap<ShardId, Arc<Mutex<HnswIndex>>>>>,
    pub node_addrs: Arc<Mutex<HashMap<NodeId, String>>>,
    pub node_servers:
        Arc<parking_lot::Mutex<HashMap<NodeId, Arc<MultiRaftNode<GrpcRaftTransport>>>>>,
    pub shutdown_tx: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
}

impl StrataServerDaemon {
    pub fn new(
        node_id: NodeId,
        addr: String,
        db_dir: PathBuf,
        node_addrs: HashMap<NodeId, String>,
    ) -> Self {
        let raft_transport = Arc::new(GrpcRaftTransport::new(node_id, node_addrs.clone()));
        let multi_raft = Arc::new(MultiRaftNode::new(
            node_id,
            db_dir.clone(),
            raft_transport.clone(),
        ));
        let table = multi_raft.table.clone();
        let hlc = Arc::new(Hlc::new(100, 0));

        let node_servers = Arc::new(parking_lot::Mutex::new(HashMap::new()));
        node_servers.lock().insert(node_id, multi_raft.clone());

        let coordinator = Arc::new(DistributedTxnCoordinator::new(
            hlc.clone(),
            node_servers.clone(),
            table.clone(),
        ));

        let planner = Arc::new(QueryPlanner::new());

        Self {
            node_id,
            addr,
            db_dir,
            table,
            multi_raft,
            raft_transport,
            coordinator,
            planner,
            hlc,
            indexes: Arc::new(Mutex::new(HashMap::new())),
            node_addrs: Arc::new(Mutex::new(node_addrs)),
            node_servers,
            shutdown_tx: Mutex::new(None),
        }
    }

    pub fn start_shard(&self, shard_id: ShardId, peers: Vec<NodeId>) {
        self.multi_raft.start_shard(shard_id, peers);
        let config = HnswConfig::default();
        let index = Arc::new(Mutex::new(HnswIndex::new(128, config)));
        self.indexes.lock().insert(shard_id, index);
    }

    pub fn stop_shard(&self, shard_id: ShardId) {
        self.multi_raft.stop_shard(shard_id);
        self.indexes.lock().remove(&shard_id);
    }

    pub fn set_node_server(&self, node_id: NodeId, server: Arc<MultiRaftNode<GrpcRaftTransport>>) {
        self.node_servers.lock().insert(node_id, server);
    }

    pub async fn run(self: Arc<Self>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let socket_addr = self.addr.parse()?;
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        *self.shutdown_tx.lock() = Some(tx);

        let multi_raft_ticker = self.multi_raft.clone();
        let ticker_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_millis(10));
            loop {
                interval.tick().await;
                multi_raft_ticker.tick();
            }
        });

        let raft_service = RaftServiceImpl {
            multi_raft: self.multi_raft.clone(),
        };
        let strata_service = StrataServiceImpl {
            daemon: self.clone(),
        };

        Server::builder()
            .add_service(RaftServiceServer::new(raft_service))
            .add_service(StrataServiceServer::new(strata_service))
            .serve_with_shutdown(socket_addr, async move {
                let _ = rx.await;
            })
            .await?;

        ticker_task.abort();
        Ok(())
    }

    pub fn shutdown(&self) {
        if let Some(tx) = self.shutdown_tx.lock().take() {
            let _ = tx.send(());
        }
    }

    // ── Handlers ──────────────────────────────────────────────────────────────

    pub async fn handle_put(&self, key: Vec<u8>, value: Vec<u8>) -> Result<(), String> {
        let shard_id = self.table.lock().shard_for_key(&key);
        let ts = self.hlc.local_event();

        // Check vector insert format: key starts with b"vec:"
        if key.starts_prefix(b"vec:") {
            if let Ok(id_str) = std::str::from_utf8(&key[4..]) {
                if let Ok(id) = id_str.parse::<u64>() {
                    if let Ok(vec) = bincode::deserialize::<Vec<f32>>(&value) {
                        let indexes = self.indexes.lock();
                        if let Some(idx_arc) = indexes.get(&shard_id) {
                            let _ = idx_arc.lock().insert(id, &vec);
                        }
                    }
                }
            }
        }

        let cmd = ShardCommand::Put {
            key: key.clone(),
            value,
            ts,
        };

        self.coordinator
            .propose_command(&key, cmd)
            .await
            .map_err(|e| e.to_string())
    }

    pub async fn handle_get(
        &self,
        key: Vec<u8>,
        read_ts: Option<HlcTimestamp>,
    ) -> Result<Option<Vec<u8>>, String> {
        let ts = read_ts.unwrap_or(HlcTimestamp {
            physical: u64::MAX,
            logical: u32::MAX,
        });

        // 1. Single-shard fast path direct storage lookup
        if let Ok(storage_arc) = self.coordinator.get_storage(&key) {
            let guard = storage_arc.lock();
            if let Some(ref storage) = *guard {
                use strata_storage::Storage;
                if let Ok(Some(val)) = storage.get(&key, ts) {
                    return Ok(Some(val));
                }
            }
        }

        // 2. 2PC Transactional resolution (Percolator write_key / data_key)
        self.coordinator
            .get(&key, ts)
            .await
            .map_err(|e| e.to_string())
    }

    pub async fn handle_delete(&self, key: Vec<u8>) -> Result<(), String> {
        let shard_id = self.table.lock().shard_for_key(&key);
        let ts = self.hlc.local_event();

        if key.starts_prefix(b"vec:") {
            if let Ok(id_str) = std::str::from_utf8(&key[4..]) {
                if let Ok(id) = id_str.parse::<u64>() {
                    let indexes = self.indexes.lock();
                    if let Some(idx_arc) = indexes.get(&shard_id) {
                        let _ = idx_arc.lock().delete(id);
                    }
                }
            }
        }

        let cmd = ShardCommand::Delete {
            key: key.clone(),
            ts,
        };
        self.coordinator
            .propose_command(&key, cmd)
            .await
            .map_err(|e| e.to_string())
    }

    pub async fn handle_search_knn(
        &self,
        vector: Vec<f32>,
        k: usize,
        radius: Option<f32>,
        filter: Option<roaring::RoaringBitmap>,
    ) -> Result<Vec<(u64, f32)>, String> {
        let routes = self.table.lock().routes.clone();
        let query = VectorQuery::new(vector.clone(), k);
        let query = if let Some(r) = radius {
            query.with_radius(r)
        } else {
            query
        };
        let query = if let Some(f) = filter.clone() {
            query.with_filter(f)
        } else {
            query
        };

        let mut all_results: Vec<(u64, f32)> = Vec::new();

        for route in routes {
            let shard_id = route.shard_id;

            // Local shard execution
            let local_idx = {
                let indexes = self.indexes.lock();
                indexes.get(&shard_id).cloned()
            };

            if let Some(idx_arc) = local_idx {
                let idx = idx_arc.lock();
                let res = if let Some(ref f) = query.filter {
                    idx.search_knn_filtered(&query.vector, query.k, f)
                } else {
                    idx.search_knn(&query.vector, query.k)
                };
                if let Ok(mut hits) = res {
                    if let Some(r) = query.radius {
                        hits.retain(|(_, d)| *d <= r);
                    }
                    all_results.extend(hits);
                }
            } else {
                // Remote shard scatter via gRPC client
                let leader_node = route.raft_group.first().copied().unwrap_or(1);
                let addr_opt = self.node_addrs.lock().get(&leader_node).cloned();
                if let Some(addr) = addr_opt {
                    if let Ok(client) = GrpcStrataClient::connect(addr).await {
                        if let Ok(hits) = client
                            .search_knn(vector.clone(), k, radius, filter.clone())
                            .await
                        {
                            all_results.extend(hits);
                        }
                    }
                }
            }
        }

        // Scatter-gather merge top-k by distance ascending
        all_results.sort_by(|a, b| a.1.total_cmp(&b.1));
        all_results.dedup_by_key(|item| item.0);
        all_results.truncate(k);

        Ok(all_results)
    }

    pub async fn handle_search_range(
        &self,
        start_key: Vec<u8>,
        end_key: Vec<u8>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>, String> {
        let routes = self.table.lock().routes.clone();
        let mut results = Vec::new();

        for route in routes {
            if !route.end_key.is_empty() && route.end_key < start_key {
                continue;
            }
            if !end_key.is_empty() && route.start_key >= end_key {
                continue;
            }

            let max_ts = HlcTimestamp {
                physical: u64::MAX,
                logical: u32::MAX,
            };

            if let Ok(storage_arc) = self.coordinator.get_storage(&route.start_key) {
                let guard = storage_arc.lock();
                if let Some(ref storage) = *guard {
                    use strata_storage::Storage;
                    if let Ok(iter) = storage.scan(&start_key, &end_key, max_ts) {
                        for (k, v) in iter {
                            if !k.starts_prefix(b"__") {
                                results.push((k, v));
                            }
                        }
                    }
                }
            }
        }

        results.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(results)
    }
}

trait StartsPrefix {
    fn starts_prefix(&self, prefix: &[u8]) -> bool;
}

impl StartsPrefix for [u8] {
    fn starts_prefix(&self, prefix: &[u8]) -> bool {
        self.starts_with(prefix)
    }
}

// ── gRPC Service Implementation ──────────────────────────────────────────────

struct RaftServiceImpl {
    multi_raft: Arc<MultiRaftNode<GrpcRaftTransport>>,
}

#[tonic::async_trait]
impl RaftService for RaftServiceImpl {
    async fn request_vote(
        &self,
        request: tonic::Request<RequestVoteMessage>,
    ) -> Result<tonic::Response<RequestVoteResponseMessage>, tonic::Status> {
        let msg = request.into_inner();
        let req: strata_consensus::RequestVoteReq = bincode::deserialize(&msg.payload)
            .map_err(|e| tonic::Status::invalid_argument(e.to_string()))?;

        let shard_id = ShardId(req.shard_id);
        let shard_node = {
            let shards = self.multi_raft.shards.lock();
            shards.get(&shard_id).cloned().ok_or_else(|| {
                tonic::Status::not_found(format!("Shard {} not found", req.shard_id))
            })?
        };

        let resp = shard_node.handle_request_vote_rpc(req).await;
        let payload =
            bincode::serialize(&resp).map_err(|e| tonic::Status::internal(e.to_string()))?;
        Ok(tonic::Response::new(RequestVoteResponseMessage { payload }))
    }

    async fn append_entries(
        &self,
        request: tonic::Request<AppendEntriesMessage>,
    ) -> Result<tonic::Response<AppendEntriesResponseMessage>, tonic::Status> {
        let msg = request.into_inner();
        let req: strata_consensus::AppendEntriesReq = bincode::deserialize(&msg.payload)
            .map_err(|e| tonic::Status::invalid_argument(e.to_string()))?;

        let shard_id = ShardId(req.shard_id);
        let shard_node = {
            let shards = self.multi_raft.shards.lock();
            shards.get(&shard_id).cloned().ok_or_else(|| {
                tonic::Status::not_found(format!("Shard {} not found", req.shard_id))
            })?
        };

        let resp = shard_node.handle_append_entries_rpc(req).await;
        let payload =
            bincode::serialize(&resp).map_err(|e| tonic::Status::internal(e.to_string()))?;
        Ok(tonic::Response::new(AppendEntriesResponseMessage {
            payload,
        }))
    }

    async fn install_snapshot(
        &self,
        request: tonic::Request<InstallSnapshotMessage>,
    ) -> Result<tonic::Response<InstallSnapshotResponseMessage>, tonic::Status> {
        let msg = request.into_inner();
        let req: strata_consensus::InstallSnapshotReq = bincode::deserialize(&msg.payload)
            .map_err(|e| tonic::Status::invalid_argument(e.to_string()))?;

        let shard_id = ShardId(req.shard_id);
        let shard_node = {
            let shards = self.multi_raft.shards.lock();
            shards.get(&shard_id).cloned().ok_or_else(|| {
                tonic::Status::not_found(format!("Shard {} not found", req.shard_id))
            })?
        };

        let resp = shard_node.handle_install_snapshot_rpc(req).await;
        let payload =
            bincode::serialize(&resp).map_err(|e| tonic::Status::internal(e.to_string()))?;
        Ok(tonic::Response::new(InstallSnapshotResponseMessage {
            payload,
        }))
    }
}

struct StrataServiceImpl {
    daemon: Arc<StrataServerDaemon>,
}

#[tonic::async_trait]
impl StrataService for StrataServiceImpl {
    async fn put(
        &self,
        request: tonic::Request<PutRequest>,
    ) -> Result<tonic::Response<PutResponse>, tonic::Status> {
        let req = request.into_inner();
        match self.daemon.handle_put(req.key, req.value).await {
            Ok(_) => Ok(tonic::Response::new(PutResponse {
                success: true,
                error: String::new(),
            })),
            Err(e) => Ok(tonic::Response::new(PutResponse {
                success: false,
                error: e,
            })),
        }
    }

    async fn get(
        &self,
        request: tonic::Request<GetRequest>,
    ) -> Result<tonic::Response<GetResponse>, tonic::Status> {
        let req = request.into_inner();
        let read_ts = req.read_ts.map(|ts| ts.into());
        match self.daemon.handle_get(req.key, read_ts).await {
            Ok(val_opt) => match val_opt {
                Some(val) => Ok(tonic::Response::new(GetResponse {
                    found: true,
                    value: val,
                    error: String::new(),
                })),
                None => Ok(tonic::Response::new(GetResponse {
                    found: false,
                    value: Vec::new(),
                    error: String::new(),
                })),
            },
            Err(e) => Ok(tonic::Response::new(GetResponse {
                found: false,
                value: Vec::new(),
                error: e,
            })),
        }
    }

    async fn delete(
        &self,
        request: tonic::Request<DeleteRequest>,
    ) -> Result<tonic::Response<DeleteResponse>, tonic::Status> {
        let req = request.into_inner();
        match self.daemon.handle_delete(req.key).await {
            Ok(_) => Ok(tonic::Response::new(DeleteResponse {
                success: true,
                error: String::new(),
            })),
            Err(e) => Ok(tonic::Response::new(DeleteResponse {
                success: false,
                error: e,
            })),
        }
    }

    async fn begin_txn(
        &self,
        _request: tonic::Request<BeginTxnRequest>,
    ) -> Result<tonic::Response<BeginTxnResponse>, tonic::Status> {
        let ts = self.daemon.coordinator.begin();
        Ok(tonic::Response::new(BeginTxnResponse {
            start_ts: Some(ts.into()),
        }))
    }

    async fn prewrite_txn(
        &self,
        request: tonic::Request<PrewriteTxnRequest>,
    ) -> Result<tonic::Response<PrewriteTxnResponse>, tonic::Status> {
        let req = request.into_inner();
        let start_ts = req
            .start_ts
            .ok_or_else(|| tonic::Status::invalid_argument("Missing start_ts"))?
            .into();
        let mutations: Result<Vec<Mutation>, String> =
            req.mutations.into_iter().map(|m| m.try_into()).collect();
        let mutations = mutations.map_err(tonic::Status::invalid_argument)?;

        match self
            .daemon
            .coordinator
            .prewrite_async(start_ts, &mutations)
            .await
        {
            Ok(_) => Ok(tonic::Response::new(PrewriteTxnResponse {
                success: true,
                error: String::new(),
            })),
            Err(e) => Ok(tonic::Response::new(PrewriteTxnResponse {
                success: false,
                error: e.to_string(),
            })),
        }
    }

    async fn commit_txn(
        &self,
        request: tonic::Request<CommitTxnRequest>,
    ) -> Result<tonic::Response<CommitTxnResponse>, tonic::Status> {
        let req = request.into_inner();
        let start_ts = req
            .start_ts
            .ok_or_else(|| tonic::Status::invalid_argument("Missing start_ts"))?
            .into();
        let commit_ts = req
            .commit_ts
            .ok_or_else(|| tonic::Status::invalid_argument("Missing commit_ts"))?
            .into();

        match self
            .daemon
            .coordinator
            .commit_async(start_ts, commit_ts)
            .await
        {
            Ok(_) => Ok(tonic::Response::new(CommitTxnResponse {
                success: true,
                error: String::new(),
            })),
            Err(e) => Ok(tonic::Response::new(CommitTxnResponse {
                success: false,
                error: e.to_string(),
            })),
        }
    }

    async fn abort_txn(
        &self,
        request: tonic::Request<AbortTxnRequest>,
    ) -> Result<tonic::Response<AbortTxnResponse>, tonic::Status> {
        let req = request.into_inner();
        let start_ts = req
            .start_ts
            .ok_or_else(|| tonic::Status::invalid_argument("Missing start_ts"))?
            .into();
        let mutations: Result<Vec<Mutation>, String> =
            req.mutations.into_iter().map(|m| m.try_into()).collect();
        let mutations = mutations.unwrap_or_default();

        match self
            .daemon
            .coordinator
            .abort_async(start_ts, &mutations)
            .await
        {
            Ok(_) => Ok(tonic::Response::new(AbortTxnResponse {
                success: true,
                error: String::new(),
            })),
            Err(e) => Ok(tonic::Response::new(AbortTxnResponse {
                success: false,
                error: e.to_string(),
            })),
        }
    }

    async fn search_knn(
        &self,
        request: tonic::Request<SearchKnnRequest>,
    ) -> Result<tonic::Response<SearchKnnResponse>, tonic::Status> {
        let req = request.into_inner();
        let filter = if !req.filter_bitmap.is_empty() {
            roaring::RoaringBitmap::deserialize_from(&req.filter_bitmap[..]).ok()
        } else {
            None
        };

        match self
            .daemon
            .handle_search_knn(req.vector, req.k as usize, req.radius, filter)
            .await
        {
            Ok(results) => {
                let proto_results = results
                    .into_iter()
                    .map(|(id, distance)| KnnPairProto { id, distance })
                    .collect();
                Ok(tonic::Response::new(SearchKnnResponse {
                    results: proto_results,
                    error: String::new(),
                }))
            }
            Err(e) => Ok(tonic::Response::new(SearchKnnResponse {
                results: Vec::new(),
                error: e,
            })),
        }
    }

    async fn search_range(
        &self,
        request: tonic::Request<SearchRangeRequest>,
    ) -> Result<tonic::Response<SearchRangeResponse>, tonic::Status> {
        let req = request.into_inner();
        match self
            .daemon
            .handle_search_range(req.start_key, req.end_key)
            .await
        {
            Ok(pairs) => {
                let proto_pairs = pairs
                    .into_iter()
                    .map(|(key, value)| KeyValuePairProto { key, value })
                    .collect();
                Ok(tonic::Response::new(SearchRangeResponse {
                    pairs: proto_pairs,
                    error: String::new(),
                }))
            }
            Err(e) => Ok(tonic::Response::new(SearchRangeResponse {
                pairs: Vec::new(),
                error: e,
            })),
        }
    }

    async fn get_routing_table(
        &self,
        _request: tonic::Request<GetRoutingTableRequest>,
    ) -> Result<tonic::Response<GetRoutingTableResponse>, tonic::Status> {
        let table = self.daemon.table.lock().clone();
        let bytes =
            bincode::serialize(&table).map_err(|e| tonic::Status::internal(e.to_string()))?;
        Ok(tonic::Response::new(GetRoutingTableResponse {
            table_bytes: bytes,
        }))
    }
}
