"""Typed environment-based settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_EMBEDDING_VERSION = "minilm-l6-v2-1110a243-template-v1-normalized-float32"


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FRAUD_",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Fraud and Anomaly Detection Network"
    app_version: str = "0.1.0"
    environment: Literal["local", "test", "development", "staging", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    cors_allowed_origins: tuple[str, ...] = ()
    investigator_default_limit: int = Field(default=25, ge=1, le=100)
    investigator_max_limit: int = Field(default=100, ge=1, le=500)
    auth_mode: Literal["local_bypass", "entra"] = "local_bypass"
    entra_tenant_id: str | None = None
    entra_audience: str | None = None
    entra_issuer: str | None = None
    entra_jwks_url: str | None = None
    entra_required_scope: str = "Fraud.Investigate"
    auth_jwks_cache_seconds: int = Field(default=3600, ge=60, le=86_400)
    auth_request_timeout_seconds: int = Field(default=5, gt=0, le=30)

    postgres_dsn: SecretStr | None = None
    test_postgres_dsn: SecretStr | None = None
    cosmos_nosql_endpoint: SecretStr | None = None
    cosmos_nosql_key: SecretStr | None = None
    cosmos_use_managed_identity: bool = False
    azure_managed_identity_client_id: str | None = None
    cosmos_nosql_database: str | None = None
    cosmos_raw_container: str = "transactions_raw"
    cosmos_embedding_container: str = "behavior_embeddings"
    cosmos_request_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    cosmos_batch_size: int = Field(default=500, ge=1, le=5_000)
    cosmos_page_size: int = Field(default=500, ge=1, le=5_000)
    cosmos_max_retries: int = Field(default=3, ge=0, le=10)
    embedding_batch_size: int = Field(default=256, ge=1, le=2_048)
    embedding_page_size: int = Field(default=500, ge=1, le=5_000)
    vector_search_limit: int = Field(default=10, ge=1, le=100)
    vector_similarity_metric: Literal["cosine"] = "cosine"
    behavioral_template_version: str = "behavior-template-v1"
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    embedding_version: str = DEFAULT_EMBEDDING_VERSION
    embedding_normalize: bool = True
    embedding_dtype: Literal["float32"] = "float32"
    embedding_model_cache: Path = Path("artifacts/models/sentence-transformers")
    behavioral_template_artifact: Path = Path("data/processed/behavior_templates.parquet")
    embedding_local_artifact: Path = Path("data/processed/behavior_embeddings.parquet")
    embedding_progress_interval: int = Field(default=5_000, ge=1)
    model_feature_artifact: Path = Path("data/processed/model_features_v2.parquet")
    model_artifact_root: Path = Path("artifacts/models/releases")
    incremental_chunk_steps: int = Field(default=24, ge=1, le=168)
    incremental_artifact_root: Path = Path("artifacts/incremental/phase8")
    incremental_data_root: Path = Path("data/processed/phase8")
    gremlin_endpoint: SecretStr | None = None
    gremlin_username: str | None = None
    gremlin_database: str | None = None
    gremlin_graph: str | None = None
    gremlin_key: SecretStr | None = None
    gremlin_request_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    gremlin_max_retries: int = Field(default=3, ge=0, le=10)
    gremlin_projection_batch_size: int = Field(default=250, ge=1, le=5_000)
    gremlin_partition_shards: int = Field(default=64, ge=1, le=4_096)
    louvain_seed: int = 73_751
    louvain_resolution: float = Field(default=1.0, gt=0, le=10)
    model_artifact_uri: SecretStr | None = None

    @model_validator(mode="after")
    def validate_authentication_boundary(self) -> Self:
        if self.environment not in {"local", "test"} and self.auth_mode != "entra":
            raise ValueError("non-local environments require Entra authentication")
        if self.auth_mode == "entra":
            required = (
                self.entra_tenant_id,
                self.entra_audience,
                self.entra_issuer,
                self.entra_jwks_url,
                self.entra_required_scope,
            )
            if not all(required):
                raise ValueError("Entra authentication configuration is incomplete")
        return self

    @property
    def postgres_configured(self) -> bool:
        return self.postgres_dsn is not None

    @property
    def document_vector_configured(self) -> bool:
        return (
            self.cosmos_nosql_endpoint is not None
            and (self.cosmos_nosql_key is not None or self.cosmos_use_managed_identity)
            and bool(self.cosmos_nosql_database)
            and bool(self.cosmos_raw_container)
        )

    @property
    def graph_configured(self) -> bool:
        return (
            self.gremlin_endpoint is not None
            and bool(self.gremlin_username)
            and bool(self.gremlin_database)
            and bool(self.gremlin_graph)
            and self.gremlin_key is not None
        )

    @property
    def embedding_configured(self) -> bool:
        return self.document_vector_configured and bool(self.cosmos_embedding_container)

    @property
    def model_artifact_configured(self) -> bool:
        return self.model_artifact_uri is not None


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance."""

    return Settings()
