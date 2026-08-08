use std::collections::BTreeSet;
use std::env;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use axum::{Json, Router, routing::get};
use futures::StreamExt;
use locus_core::{
    EngineCapabilities, EngineInstance, EngineInstanceId, EngineInstanceRef, ExecutionRole,
    ExecutionTarget, ExecutionTargetId, GenerationSemanticIdentity, InputKind,
    InputSemanticIdentity, ModelExecutionIdentity, OperationContext, OutputSemanticIdentity,
    ParallelLayout, PreparedStateAttachment, RequestId, RuntimeIdentity, SemanticComponentIdentity,
    SemanticIdentity, StateImportSpec, StateImportTarget, TransferReceipt,
};
use locus_engine::{
    EngineAdapter, EngineError, EngineEventStream, EngineRegistry, FakeEngineAdapter,
    FakeEngineOutput,
};
use locus_openai::{ApiConfig, router_with_config};
use locus_runtime::{DefaultInferenceService, InferenceService};
use locus_semantics::{
    BasicModelSemantics, ByteDecoder, ByteTokenizer, ModelProfile, ModelRegistry,
    SimpleTemplateRenderer,
};
use locus_state::{NullStateProvider, StateProvider};
use serde_json::json;
use tokio::net::TcpListener;

fn component(kind: &str) -> SemanticComponentIdentity {
    SemanticComponentIdentity {
        kind: kind.to_owned(),
        revision: "e2e-v1".to_owned(),
        fingerprint: format!("{kind}-e2e-v1"),
    }
}

struct DelayedEngineAdapter {
    inner: Arc<FakeEngineAdapter>,
    delay: Duration,
}

#[async_trait]
impl EngineAdapter for DelayedEngineAdapter {
    fn instance(&self) -> EngineInstance {
        self.inner.instance()
    }

    async fn execution_targets(
        &self,
        context: &OperationContext,
    ) -> Result<Vec<ExecutionTarget>, EngineError> {
        self.inner.execution_targets(context).await
    }

    async fn capabilities(
        &self,
        target: &ExecutionTarget,
        context: &OperationContext,
    ) -> Result<EngineCapabilities, EngineError> {
        self.inner.capabilities(target, context).await
    }

    async fn snapshot(
        &self,
        target: &ExecutionTarget,
        context: &OperationContext,
    ) -> Result<locus_core::EngineSnapshot, EngineError> {
        self.inner.snapshot(target, context).await
    }

    async fn prepare_state_import(
        &self,
        target: &ExecutionTarget,
        spec: &StateImportSpec,
        context: &OperationContext,
    ) -> Result<StateImportTarget, EngineError> {
        self.inner.prepare_state_import(target, spec, context).await
    }

    async fn commit_state_import(
        &self,
        import: &StateImportTarget,
        receipt: &TransferReceipt,
        context: &OperationContext,
    ) -> Result<PreparedStateAttachment, EngineError> {
        self.inner
            .commit_state_import(import, receipt, context)
            .await
    }

    async fn abort_state_import(
        &self,
        import: &StateImportTarget,
        context: &OperationContext,
    ) -> Result<(), EngineError> {
        self.inner.abort_state_import(import, context).await
    }

    async fn execute(
        &self,
        target: &ExecutionTarget,
        request: locus_core::CanonicalRequest,
        state: Option<PreparedStateAttachment>,
        context: OperationContext,
    ) -> Result<EngineEventStream, EngineError> {
        let mut inner = self.inner.execute(target, request, state, context).await?;
        let delay = self.delay;
        let stream = async_stream::try_stream! {
            while let Some(event) = inner.next().await {
                tokio::time::sleep(delay).await;
                yield event?;
            }
        };
        Ok(Box::pin(stream))
    }

    async fn cancel(
        &self,
        request_id: &RequestId,
        context: &OperationContext,
    ) -> Result<(), EngineError> {
        self.inner.cancel(request_id, context).await
    }
}

#[tokio::main]
async fn main() {
    let listen = env::var("LOCUS_E2E_LISTEN")
        .unwrap_or_else(|_| "127.0.0.1:18080".to_owned())
        .parse::<SocketAddr>()
        .expect("LOCUS_E2E_LISTEN must be a socket address");
    let api_key = env::var("LOCUS_E2E_API_KEY").unwrap_or_else(|_| "locus-test-key".to_owned());
    let model = ModelExecutionIdentity {
        model_revision: "sdk-fixture-v1".to_owned(),
        adapter_revision: None,
        execution_profile: "default".to_owned(),
    };
    let semantic_identity = SemanticIdentity {
        input: InputSemanticIdentity {
            tokenizer: component("tokenizer"),
            template: component("template"),
            multimodal_preprocessing: None,
        },
        generation: GenerationSemanticIdentity {
            sampling_normalization: component("sampling"),
            stop_behavior: component("stop"),
            constrained_generation: Some(component("structured-output")),
        },
        output: OutputSemanticIdentity {
            detokenizer: component("detokenizer"),
            reasoning_parser: None,
            tool_parser: None,
        },
        umbrella_fingerprint: Some("sdk-fixture-e2e-v1".to_owned()),
    };
    let models = ModelRegistry::new();
    models
        .register(Arc::new(BasicModelSemantics::new(
            ModelProfile {
                public_aliases: vec!["locus-test".to_owned()],
                model: model.clone(),
                semantic_identity: semantic_identity.clone(),
            },
            Arc::new(ByteTokenizer::new(
                semantic_identity.input.tokenizer.clone(),
            )),
            Arc::new(SimpleTemplateRenderer::new(
                semantic_identity.input.template.clone(),
            )),
            Arc::new(ByteDecoder),
        )))
        .expect("register fixture model");
    let engine_ref = EngineInstanceRef {
        id: EngineInstanceId::new("sdk-fixture"),
        generation: 1,
    };
    let fake_adapter = Arc::new(
        FakeEngineAdapter::new(
            EngineInstance {
                reference: engine_ref.clone(),
                runtime: RuntimeIdentity {
                    kind: "fake".to_owned(),
                    runtime_version: "e2e-v1".to_owned(),
                    adapter_version: "e2e-v1".to_owned(),
                },
                topology: "local".to_owned(),
                hardware: "cpu".to_owned(),
                health_endpoint: None,
            },
            ExecutionTarget {
                id: ExecutionTargetId::new("sdk-fixture/locus-test"),
                engine: engine_ref,
                model,
                role: ExecutionRole::Combined,
                parallel_layout: ParallelLayout {
                    tensor_parallel: 1,
                    pipeline_parallel: 1,
                    expert_parallel: 1,
                    layout_revision: "e2e-v1".to_owned(),
                },
                residency: "resident".to_owned(),
                capability_revision: "e2e-v1".to_owned(),
            },
            EngineCapabilities {
                supported_input_kinds: BTreeSet::from([InputKind::TokenSequence]),
                emits_token_deltas: false,
                emits_text_deltas: true,
                emits_reasoning_deltas: false,
                emits_tool_calls: false,
                supports_structured_output: true,
                supported_state_kinds: BTreeSet::new(),
            },
        )
        .with_output(FakeEngineOutput {
            token_deltas: Vec::new(),
            text_deltas: vec!["{\"answer\":".to_owned(), "\"ok\"}".to_owned()],
            ..FakeEngineOutput::default()
        }),
    );
    let engines = EngineRegistry::new();
    let adapter: Arc<dyn EngineAdapter> = Arc::new(DelayedEngineAdapter {
        inner: Arc::clone(&fake_adapter),
        delay: Duration::from_millis(40),
    });
    engines.register(adapter).expect("register fixture engine");
    let state: Arc<dyn StateProvider> = Arc::new(NullStateProvider::default());
    let service: Arc<dyn InferenceService> =
        Arc::new(DefaultInferenceService::new(models, engines, state));
    let api = router_with_config(
        service,
        ApiConfig {
            bearer_token: Some(api_key),
            ..ApiConfig::default()
        },
    )
    .expect("build fixture API");
    let counts_adapter = fake_adapter;
    let fixture = Router::new().route(
        "/test/call-counts",
        get(move || {
            let adapter = Arc::clone(&counts_adapter);
            async move {
                let counts = adapter.call_counts();
                Json(json!({
                    "execute": counts.execute,
                    "cancel": counts.cancel,
                }))
            }
        }),
    );
    let listener = TcpListener::bind(listen)
        .await
        .expect("bind fixture server");
    println!(
        "LOCUS_E2E_READY={}",
        listener.local_addr().expect("local address")
    );
    axum::serve(listener, api.merge(fixture))
        .await
        .expect("serve fixture API");
}
