use tonic_web_wasm_client::Client;
use wasm_bindgen::prelude::*;

pub mod proto {
    tonic::include_proto!("strata");
}

use proto::strata_service_client::StrataServiceClient;

#[wasm_bindgen]
pub struct WasmStrataClient {
    client: StrataServiceClient<Client>,
}

#[wasm_bindgen]
impl WasmStrataClient {
    #[wasm_bindgen(constructor)]
    pub fn new(url: String) -> Self {
        let wasm_client = Client::new(url);
        Self {
            client: StrataServiceClient::new(wasm_client),
        }
    }

    #[wasm_bindgen]
    pub async fn put(&mut self, key: &[u8], value: &[u8]) -> Result<JsValue, JsValue> {
        let req = tonic::Request::new(proto::PutRequest {
            key: key.to_vec(),
            value: value.to_vec(),
        });

        let response = self
            .client
            .put(req)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let inner = response.into_inner();

        let js_obj = js_sys::Object::new();
        js_sys::Reflect::set(
            &js_obj,
            &JsValue::from_str("success"),
            &JsValue::from_bool(inner.success),
        )?;
        if !inner.error.is_empty() {
            js_sys::Reflect::set(
                &js_obj,
                &JsValue::from_str("error"),
                &JsValue::from_str(&inner.error),
            )?;
        }

        Ok(js_obj.into())
    }

    #[wasm_bindgen]
    pub async fn get(&mut self, key: &[u8]) -> Result<JsValue, JsValue> {
        let req = tonic::Request::new(proto::GetRequest {
            key: key.to_vec(),
            read_ts: None,
        });

        let response = self
            .client
            .get(req)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let inner = response.into_inner();

        let js_obj = js_sys::Object::new();
        js_sys::Reflect::set(
            &js_obj,
            &JsValue::from_str("found"),
            &JsValue::from_bool(inner.found),
        )?;
        if inner.found {
            // Need to pass bytes back nicely, let's use Uint8Array
            let arr = js_sys::Uint8Array::from(inner.value.as_slice());
            js_sys::Reflect::set(&js_obj, &JsValue::from_str("value"), &arr)?;
        }
        if !inner.error.is_empty() {
            js_sys::Reflect::set(
                &js_obj,
                &JsValue::from_str("error"),
                &JsValue::from_str(&inner.error),
            )?;
        }

        Ok(js_obj.into())
    }

    #[wasm_bindgen(js_name = searchKnn)]
    pub async fn search_knn(&mut self, vector: &[f32], k: u32) -> Result<JsValue, JsValue> {
        let req = tonic::Request::new(proto::SearchKnnRequest {
            vector: vector.to_vec(),
            k: k as u64,
            radius: None,
            filter_bitmap: vec![],
        });

        let response = self
            .client
            .search_knn(req)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let inner = response.into_inner();

        let js_arr = js_sys::Array::new();
        for res in inner.results {
            let obj = js_sys::Object::new();
            js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("id"),
                &JsValue::from_f64(res.id as f64),
            )?;
            js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("distance"),
                &JsValue::from_f64(res.distance as f64),
            )?;
            js_arr.push(&obj);
        }

        let result_obj = js_sys::Object::new();
        js_sys::Reflect::set(&result_obj, &JsValue::from_str("results"), &js_arr)?;
        if !inner.error.is_empty() {
            js_sys::Reflect::set(
                &result_obj,
                &JsValue::from_str("error"),
                &JsValue::from_str(&inner.error),
            )?;
        }

        Ok(result_obj.into())
    }
}
