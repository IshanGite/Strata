pub mod proto {
    tonic::include_proto!("strata");
}

use async_trait::async_trait;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::transport::Channel;

use strata_consensus::{
    AppendEntriesReq, AppendEntriesResp, InstallSnapshotReq, InstallSnapshotResp, NodeId,
    RaftTransport, RequestVoteReq, RequestVoteResp, TransportError,
};
use strata_sharding::RoutingTable;
use strata_storage::HlcTimestamp;
use strata_txn::Mutation;

use proto::raft_service_client::RaftServiceClient;
use proto::strata_service_client::StrataServiceClient;

// ── Trait Abstractions for Transport Swap (gRPC / QUIC) ──────────────────────

#[async_trait]
pub trait RaftNetworkClientTrait: Send + Sync {
    async fn request_vote(&self, req: RequestVoteReq) -> Result<RequestVoteResp, TransportError>;
    async fn append_entries(
        &self,
        req: AppendEntriesReq,
    ) -> Result<AppendEntriesResp, TransportError>;
    async fn install_snapshot(
        &self,
        req: InstallSnapshotReq,
    ) -> Result<InstallSnapshotResp, TransportError>;
}

#[async_trait]
pub trait StrataNetworkClientTrait: Send + Sync {
    async fn put(&self, key: Vec<u8>, value: Vec<u8>) -> Result<(), String>;
    async fn get(
        &self,
        key: Vec<u8>,
        read_ts: Option<HlcTimestamp>,
    ) -> Result<Option<Vec<u8>>, String>;
    async fn delete(&self, key: Vec<u8>) -> Result<(), String>;
    async fn begin_txn(&self) -> Result<HlcTimestamp, String>;
    async fn prewrite_txn(
        &self,
        start_ts: HlcTimestamp,
        mutations: Vec<Mutation>,
    ) -> Result<(), String>;
    async fn commit_txn(
        &self,
        start_ts: HlcTimestamp,
        commit_ts: HlcTimestamp,
        mutations: Vec<Mutation>,
    ) -> Result<(), String>;
    async fn abort_txn(
        &self,
        start_ts: HlcTimestamp,
        mutations: Vec<Mutation>,
    ) -> Result<(), String>;
    async fn search_knn(
        &self,
        vector: Vec<f32>,
        k: usize,
        radius: Option<f32>,
        filter: Option<roaring::RoaringBitmap>,
    ) -> Result<Vec<(u64, f32)>, String>;
    async fn search_range(
        &self,
        start_key: Vec<u8>,
        end_key: Vec<u8>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>, String>;
    async fn get_routing_table(&self) -> Result<RoutingTable, String>;
}

// ── gRPC Implementation of Raft Transport ─────────────────────────────────────

#[derive(Clone)]
pub struct GrpcRaftTransport {
    pub node_id: NodeId,
    pub peer_addrs: Arc<Mutex<HashMap<NodeId, String>>>,
    pub clients: Arc<Mutex<HashMap<NodeId, RaftServiceClient<Channel>>>>,
}

static RUNTIME: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();

fn block_on_async<F: std::future::Future>(f: F) -> F::Output {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| handle.block_on(f))
    } else {
        let rt = RUNTIME.get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("Failed to build global tokio runtime for Raft transport")
        });
        rt.block_on(f)
    }
}

impl GrpcRaftTransport {
    pub fn new(node_id: NodeId, peer_addrs: HashMap<NodeId, String>) -> Self {
        Self {
            node_id,
            peer_addrs: Arc::new(Mutex::new(peer_addrs)),
            clients: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn add_peer(&self, node_id: NodeId, addr: String) {
        self.peer_addrs.lock().insert(node_id, addr);
    }

    pub fn remove_client(&self, node_id: NodeId) {
        self.clients.lock().remove(&node_id);
    }

    async fn get_client(&self, to: NodeId) -> Result<RaftServiceClient<Channel>, TransportError> {
        let addr = {
            let addrs = self.peer_addrs.lock();
            addrs.get(&to).cloned().ok_or_else(|| {
                TransportError::Other(format!("No address registered for node {}", to))
            })?
        };

        {
            let clients = self.clients.lock();
            if let Some(client) = clients.get(&to) {
                return Ok(client.clone());
            }
        }

        let channel = Channel::from_shared(addr.clone())
            .map_err(|e| TransportError::Other(e.to_string()))?
            .connect()
            .await
            .map_err(|_| TransportError::ConnectionRefused)?;

        let client = RaftServiceClient::new(channel);
        self.clients.lock().insert(to, client.clone());
        Ok(client)
    }
}

impl RaftTransport for GrpcRaftTransport {
    fn send_request_vote(
        &self,
        to: NodeId,
        req: RequestVoteReq,
    ) -> Result<RequestVoteResp, TransportError> {
        block_on_async(async {
            let mut client = match self.get_client(to).await {
                Ok(c) => c,
                Err(e) => return Err(e),
            };
            let payload = bincode::serialize(&req)
                .map_err(|e| TransportError::Serialization(e.to_string()))?;
            let msg = proto::RequestVoteMessage { payload };
            let resp = match client.request_vote(tonic::Request::new(msg)).await {
                Ok(r) => r.into_inner(),
                Err(e) => {
                    self.remove_client(to);
                    return Err(TransportError::Other(e.to_string()));
                }
            };
            let result: RequestVoteResp = bincode::deserialize(&resp.payload)
                .map_err(|e| TransportError::Serialization(e.to_string()))?;
            Ok(result)
        })
    }

    fn send_append_entries(
        &self,
        to: NodeId,
        req: AppendEntriesReq,
    ) -> Result<AppendEntriesResp, TransportError> {
        block_on_async(async {
            let mut client = match self.get_client(to).await {
                Ok(c) => c,
                Err(e) => return Err(e),
            };
            let payload = bincode::serialize(&req)
                .map_err(|e| TransportError::Serialization(e.to_string()))?;
            let msg = proto::AppendEntriesMessage { payload };
            let resp = match client.append_entries(tonic::Request::new(msg)).await {
                Ok(r) => r.into_inner(),
                Err(e) => {
                    self.remove_client(to);
                    return Err(TransportError::Other(e.to_string()));
                }
            };
            let result: AppendEntriesResp = bincode::deserialize(&resp.payload)
                .map_err(|e| TransportError::Serialization(e.to_string()))?;
            Ok(result)
        })
    }

    fn send_install_snapshot(
        &self,
        to: NodeId,
        req: InstallSnapshotReq,
    ) -> Result<InstallSnapshotResp, TransportError> {
        block_on_async(async {
            let mut client = match self.get_client(to).await {
                Ok(c) => c,
                Err(e) => return Err(e),
            };
            let payload = bincode::serialize(&req)
                .map_err(|e| TransportError::Serialization(e.to_string()))?;
            let msg = proto::InstallSnapshotMessage { payload };
            let resp = match client.install_snapshot(tonic::Request::new(msg)).await {
                Ok(r) => r.into_inner(),
                Err(e) => {
                    self.remove_client(to);
                    return Err(TransportError::Other(e.to_string()));
                }
            };
            let result: InstallSnapshotResp = bincode::deserialize(&resp.payload)
                .map_err(|e| TransportError::Serialization(e.to_string()))?;
            Ok(result)
        })
    }
}

// ── gRPC Client Implementation of Database Operations ────────────────────────

#[derive(Clone)]
pub struct GrpcStrataClient {
    pub client: StrataServiceClient<Channel>,
}

impl GrpcStrataClient {
    pub async fn connect(addr: String) -> Result<Self, String> {
        let channel = Channel::from_shared(addr)
            .map_err(|e| e.to_string())?
            .connect()
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self {
            client: StrataServiceClient::new(channel),
        })
    }
}

#[async_trait]
impl StrataNetworkClientTrait for GrpcStrataClient {
    async fn put(&self, key: Vec<u8>, value: Vec<u8>) -> Result<(), String> {
        let mut client = self.client.clone();
        let req = proto::PutRequest { key, value };
        let resp = client
            .put(tonic::Request::new(req))
            .await
            .map_err(|e| e.to_string())?
            .into_inner();
        if resp.success {
            Ok(())
        } else {
            Err(resp.error)
        }
    }

    async fn get(
        &self,
        key: Vec<u8>,
        read_ts: Option<HlcTimestamp>,
    ) -> Result<Option<Vec<u8>>, String> {
        let mut client = self.client.clone();
        let req = proto::GetRequest {
            key,
            read_ts: read_ts.map(|ts| ts.into()),
        };
        let resp = client
            .get(tonic::Request::new(req))
            .await
            .map_err(|e| e.to_string())?
            .into_inner();
        if !resp.error.is_empty() {
            return Err(resp.error);
        }
        if resp.found {
            Ok(Some(resp.value))
        } else {
            Ok(None)
        }
    }

    async fn delete(&self, key: Vec<u8>) -> Result<(), String> {
        let mut client = self.client.clone();
        let req = proto::DeleteRequest { key };
        let resp = client
            .delete(tonic::Request::new(req))
            .await
            .map_err(|e| e.to_string())?
            .into_inner();
        if resp.success {
            Ok(())
        } else {
            Err(resp.error)
        }
    }

    async fn begin_txn(&self) -> Result<HlcTimestamp, String> {
        let mut client = self.client.clone();
        let req = proto::BeginTxnRequest {};
        let resp = client
            .begin_txn(tonic::Request::new(req))
            .await
            .map_err(|e| e.to_string())?
            .into_inner();
        if let Some(ts) = resp.start_ts {
            Ok(ts.into())
        } else {
            Err("Missing start timestamp in response".to_string())
        }
    }

    async fn prewrite_txn(
        &self,
        start_ts: HlcTimestamp,
        mutations: Vec<Mutation>,
    ) -> Result<(), String> {
        let mut client = self.client.clone();
        let req = proto::PrewriteTxnRequest {
            start_ts: Some(start_ts.into()),
            mutations: mutations.into_iter().map(|m| m.into()).collect(),
        };
        let resp = client
            .prewrite_txn(tonic::Request::new(req))
            .await
            .map_err(|e| e.to_string())?
            .into_inner();
        if resp.success {
            Ok(())
        } else {
            Err(resp.error)
        }
    }

    async fn commit_txn(
        &self,
        start_ts: HlcTimestamp,
        commit_ts: HlcTimestamp,
        mutations: Vec<Mutation>,
    ) -> Result<(), String> {
        let mut client = self.client.clone();
        let req = proto::CommitTxnRequest {
            start_ts: Some(start_ts.into()),
            commit_ts: Some(commit_ts.into()),
            mutations: mutations.into_iter().map(|m| m.into()).collect(),
        };
        let resp = client
            .commit_txn(tonic::Request::new(req))
            .await
            .map_err(|e| e.to_string())?
            .into_inner();
        if resp.success {
            Ok(())
        } else {
            Err(resp.error)
        }
    }

    async fn abort_txn(
        &self,
        start_ts: HlcTimestamp,
        mutations: Vec<Mutation>,
    ) -> Result<(), String> {
        let mut client = self.client.clone();
        let req = proto::AbortTxnRequest {
            start_ts: Some(start_ts.into()),
            mutations: mutations.into_iter().map(|m| m.into()).collect(),
        };
        let resp = client
            .abort_txn(tonic::Request::new(req))
            .await
            .map_err(|e| e.to_string())?
            .into_inner();
        if resp.success {
            Ok(())
        } else {
            Err(resp.error)
        }
    }

    async fn search_knn(
        &self,
        vector: Vec<f32>,
        k: usize,
        radius: Option<f32>,
        filter: Option<roaring::RoaringBitmap>,
    ) -> Result<Vec<(u64, f32)>, String> {
        let mut client = self.client.clone();
        let filter_bitmap = if let Some(f) = filter {
            let mut buf = Vec::new();
            f.serialize_into(&mut buf).map_err(|e| e.to_string())?;
            buf
        } else {
            Vec::new()
        };

        let req = proto::SearchKnnRequest {
            vector,
            k: k as u64,
            radius,
            filter_bitmap,
        };
        let resp = client
            .search_knn(tonic::Request::new(req))
            .await
            .map_err(|e| e.to_string())?
            .into_inner();

        if !resp.error.is_empty() {
            return Err(resp.error);
        }

        Ok(resp
            .results
            .into_iter()
            .map(|r| (r.id, r.distance))
            .collect())
    }

    async fn search_range(
        &self,
        start_key: Vec<u8>,
        end_key: Vec<u8>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>, String> {
        let mut client = self.client.clone();
        let req = proto::SearchRangeRequest { start_key, end_key };
        let resp = client
            .search_range(tonic::Request::new(req))
            .await
            .map_err(|e| e.to_string())?
            .into_inner();
        if !resp.error.is_empty() {
            return Err(resp.error);
        }
        Ok(resp.pairs.into_iter().map(|p| (p.key, p.value)).collect())
    }

    async fn get_routing_table(&self) -> Result<RoutingTable, String> {
        let mut client = self.client.clone();
        let req = proto::GetRoutingTableRequest {};
        let resp = client
            .get_routing_table(tonic::Request::new(req))
            .await
            .map_err(|e| e.to_string())?
            .into_inner();
        let table: RoutingTable =
            bincode::deserialize(&resp.table_bytes).map_err(|e| e.to_string())?;
        Ok(table)
    }
}

// ── Converters ────────────────────────────────────────────────────────────────

impl From<HlcTimestamp> for proto::HlcTimestampProto {
    fn from(ts: HlcTimestamp) -> Self {
        Self {
            physical: ts.physical,
            logical: ts.logical,
        }
    }
}

impl From<proto::HlcTimestampProto> for HlcTimestamp {
    fn from(ts: proto::HlcTimestampProto) -> Self {
        Self {
            physical: ts.physical,
            logical: ts.logical,
        }
    }
}

impl From<Mutation> for proto::MutationProto {
    fn from(m: Mutation) -> Self {
        match m {
            Mutation::Put(key, value) => Self {
                r#type: proto::mutation_proto::MutationType::Put as i32,
                key,
                value,
            },
            Mutation::Delete(key) => Self {
                r#type: proto::mutation_proto::MutationType::Delete as i32,
                key,
                value: Vec::new(),
            },
        }
    }
}

impl TryFrom<proto::MutationProto> for Mutation {
    type Error = String;

    fn try_from(m: proto::MutationProto) -> Result<Self, Self::Error> {
        match proto::mutation_proto::MutationType::try_from(m.r#type) {
            Ok(proto::mutation_proto::MutationType::Put) => Ok(Mutation::Put(m.key, m.value)),
            Ok(proto::mutation_proto::MutationType::Delete) => Ok(Mutation::Delete(m.key)),
            Err(_) => Err("Unknown mutation type".to_string()),
        }
    }
}
