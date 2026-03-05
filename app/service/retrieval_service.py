"""检索服务，封装 BaseVector-Core RetrieverService。"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from base_db import DocumentClient, DocumentChunkClient
from base_db.abstract.abstract_base_core import AbstractBaseCore
from milvus_service import (
    FulltextSearchRequest,
    HybridSearchRequest,
    KeywordSearchRequest,
    PhraseMatchSearchRequest,
    RetrieverService,
    SemanticSearchRequest,
    TextMatchSearchRequest,
)


def _configure_embedding_logging() -> None:
    """降低 embedding/transformers 加载噪音，避免每次请求刷屏。"""
    # 避免 tokenizers 并行导致的告警噪音
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # transformers 的日志与进度条
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        # 某些版本提供进度条开关
        if hasattr(hf_logging, "disable_progress_bar"):
            hf_logging.disable_progress_bar()
    except Exception:
        # 不强依赖 transformers 的具体版本
        pass

    # sentence-transformers / transformers 的 python logger
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)


class _EnvAbstractCore(AbstractBaseCore):
    """使用环境变量提供 BaseDB 认证信息的实现。"""

    @staticmethod
    def get_authorization() -> str:
        token = os.getenv("DB_AUTHORIZATION", "")
        if not token:
            raise RuntimeError("环境变量 DB_AUTHORIZATION 未配置，无法访问 BaseDB 服务")
        return token


def _load_db_env() -> None:
    """加载 .env 以便 BaseDB 读取 DB_SERVICE_URL 等配置。"""
    from dotenv import load_dotenv

    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)


_document_client: Optional[DocumentClient] = None
_chunk_client: Optional[DocumentChunkClient] = None


def _get_document_client() -> DocumentClient:
    global _document_client
    if _document_client is None:
        _load_db_env()
        _document_client = DocumentClient(abstract_impl=_EnvAbstractCore())
    return _document_client


def _get_chunk_client() -> DocumentChunkClient:
    global _chunk_client
    if _chunk_client is None:
        _load_db_env()
        _chunk_client = DocumentChunkClient(abstract_impl=_EnvAbstractCore())
    return _chunk_client


def _resolve_embedding_model_path() -> tuple[str, bool]:
    """
    解析 EMBEDDING_MODEL：若为本地路径则转为绝对路径并返回 (path, local_only=True)。
    返回 (model_path_or_id, use_local_files_only)。
    """
    model_name = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v5-text-small").strip()
    if not model_name:
        model_name = "jinaai/jina-embeddings-v5-text-small"

    # 视为本地路径：workspace/...、./、/、或包含 \（Windows）
    is_local_like = (
        model_name.startswith("workspace/")
        or model_name.startswith("workspace\\")
        or model_name.startswith("./")
        or model_name.startswith(".\\")
        or os.path.isabs(model_name)
        or (len(model_name) >= 2 and model_name[1] == ":")
        or "\\" in model_name
    )
    if is_local_like:
        if model_name.startswith("workspace/") or model_name.startswith("workspace\\"):
            # 相对项目根目录的 workspace/xxx
            root = Path(__file__).resolve().parents[2]
            path = (root / model_name.replace("\\", "/")).resolve()
        else:
            path = Path(model_name).resolve()
        # 统一按本地模型加载，避免把 workspace/xxx 当成 HuggingFace org/repo 请求网络
        return (str(path), True)
    return model_name, False


def _get_embedding_device() -> str:
    """获取 Embedding 模型运行设备：有 GPU 时用 cuda，否则用 cpu。可通过环境变量 EMBEDDING_DEVICE 覆盖。"""
    import torch

    env_device = os.getenv("EMBEDDING_DEVICE", "").strip().lower()
    if env_device in ("cuda", "cpu", "mps"):
        return env_device
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _get_embedding_model():
    """懒加载 embedding 模型，优先使用本地路径且不请求 Hugging Face，有 GPU 时使用 cuda。"""
    _configure_embedding_logging()
    from sentence_transformers import SentenceTransformer

    model_path_or_id, local_only = _resolve_embedding_model_path()
    device = _get_embedding_device()
    try:
        return SentenceTransformer(
            model_path_or_id,
            device=device,
            local_files_only=local_only,
            trust_remote_code=True,
        )
    except TypeError:
        return SentenceTransformer(
            model_path_or_id,
            device=device,
            local_files_only=local_only,
        )


def _encode_query(query: str) -> list[float]:
    """将查询文本编码为向量（已 L2 归一化）。"""
    model = _get_embedding_model()
    vec = model.encode(
        [query],
        task="retrieval",
        show_progress_bar=False,
        normalize_embeddings=True,
    )[0].tolist()
    return vec


def _get_collection_name() -> str:
    """从 .env 获取检索集合名称。"""
    return os.getenv("COLLECTION_NAME", "papers_chunks_collection")


async def get_original_text_by_doc_id(doc_id: int) -> str:
    """从文档表获取原文 markdown_content。"""
    try:
        client = _get_document_client()
        doc = await client.get_document(doc_id=doc_id)
    except Exception:
        return ""

    if isinstance(doc, dict):
        data = doc.get("data")
        return (data.get("markdown_content") if isinstance(data, dict) else None) or ""
    return getattr(doc, "markdown_content", "") or ""


async def get_parent_content_by_chunk_id(chunk_id: int) -> str:
    """根据 chunk_id 直接从文档切片表获取 parent_content。"""
    try:
        client = _get_chunk_client()
        chunk = await client.get_document_chunk(chunk_id=chunk_id)
    except Exception:
        return ""

    if isinstance(chunk, dict):
        data = chunk.get("data")
        return (data.get("parent_content") if isinstance(data, dict) else None) or ""
    return getattr(chunk, "parent_content", "") or ""


def _build_extra_params(**params: Any) -> dict[str, Any]:
    """组装 extra_params，过滤 None 值。"""
    return {k: v for k, v in params.items() if v is not None}


def _escape_like_value(v: str) -> str:
    """转义 LIKE 值中的单引号，防止注入。"""
    return str(v).replace("'", "''")


def _int_list_to_ids(val: Union[int, List[int], None]) -> None | list[int]:
    """将单值或数组规范化为整数列表，过滤非法值。"""
    if val is None:
        return None
    if isinstance(val, int):
        return [val] if val >= 0 else None
    if isinstance(val, (list, tuple)):
        ids = [int(x) for x in val if isinstance(x, (int, float)) and int(x) == x and int(x) >= 0]
        return ids if ids else None
    return None


def _ids_to_milvus_expr(field: str, ids: list[int]) -> str:
    """将 doc_id/kb_id/security_level 的 ID 列表转为 Milvus 表达式。"""
    if len(ids) == 1:
        return f"{field} == {ids[0]}"
    return f"{field} in [{', '.join(str(x) for x in ids)}]"


def _build_metadata_filter(
    keyword_text: str | None = None,
    author: str | None = None,
    paper_title: str | None = None,
    doc_id: Union[int, List[int], None] = None,
    kb_id: Union[int, List[int], None] = None,
    security_level: Union[int, List[int], None] = None,
) -> str | None:
    """将 keyword_text、author、paper_title、doc_id、kb_id、security_level 组装为 Milvus 过滤表达式。"""
    conditions: list[str] = []
    if keyword_text and str(keyword_text).strip():
        v = _escape_like_value(keyword_text.strip())
        conditions.append(f'keywords_text like "%{v}%"')
    if author and str(author).strip():
        v = _escape_like_value(author.strip())
        conditions.append(f'authors like "%{v}%"')
    if paper_title and str(paper_title).strip():
        v = _escape_like_value(paper_title.strip())
        conditions.append(f'title like "%{v}%"')
    doc_ids = _int_list_to_ids(doc_id)
    if doc_ids is not None:
        conditions.append(_ids_to_milvus_expr("doc_id", doc_ids))
    kb_ids = _int_list_to_ids(kb_id)
    if kb_ids is not None:
        conditions.append(_ids_to_milvus_expr("kb_id", kb_ids))
    sec_levels = _int_list_to_ids(security_level)
    if sec_levels is not None:
        conditions.append(_ids_to_milvus_expr("security_level", sec_levels))
    return " and ".join(conditions) if conditions else None


def semantic_search(
    query: str,
    top_k: int | None = None,
    rerank_enabled: bool | None = None,
    similarity_threshold: float | None = None,
    keyword_text: str | None = None,
    author: str | None = None,
    paper_title: str | None = None,
    doc_id: Union[int, List[int], None] = None,
    kb_id: Union[int, List[int], None] = None,
    security_level: Union[int, List[int], None] = None,
    **kwargs: Any,
):
    """语义检索，Query 由本地嵌入模型自动转为向量，集合名称从 .env 获取。"""
    query_vector = _encode_query(query)
    req_kwargs: dict[str, Any] = {
        "query": query,
        "query_vector": query_vector,
        "collection_name": _get_collection_name(),
        **kwargs,
    }
    if top_k is not None:
        req_kwargs["top_k"] = top_k
    if rerank_enabled is not None:
        req_kwargs["rerank_enabled"] = rerank_enabled
    if similarity_threshold is not None:
        req_kwargs["similarity_threshold"] = similarity_threshold
    metadata_filter = _build_metadata_filter(
        keyword_text=keyword_text,
        author=author,
        paper_title=paper_title,
        doc_id=doc_id,
        kb_id=kb_id,
        security_level=security_level,
    )
    if metadata_filter is not None:
        req_kwargs["milvus_expr"] = metadata_filter

    req = SemanticSearchRequest(**req_kwargs)
    return RetrieverService.semantic_search(req)


def keyword_search(
    query: str,
    top_k: int | None = None,
    min_match_count: int | None = None,
    keyword_text: str | None = None,
    author: str | None = None,
    paper_title: str | None = None,
    doc_id: Union[int, List[int], None] = None,
    kb_id: Union[int, List[int], None] = None,
    security_level: Union[int, List[int], None] = None,
    **kwargs: Any,
):
    """关键词检索，集合名称从 .env 获取。"""
    req_kwargs: dict[str, Any] = {
        "query": query,
        "collection_name": _get_collection_name(),
        **kwargs,
    }
    if top_k is not None:
        req_kwargs["top_k"] = top_k
    metadata_filter = _build_metadata_filter(
        keyword_text=keyword_text,
        author=author,
        paper_title=paper_title,
        doc_id=doc_id,
        kb_id=kb_id,
        security_level=security_level,
    )
    if metadata_filter is not None:
        req_kwargs["milvus_expr"] = metadata_filter
    extra = _build_extra_params(min_match_count=min_match_count)
    if extra:
        req_kwargs["extra_params"] = extra

    req = KeywordSearchRequest(**req_kwargs)
    return RetrieverService.keyword_search(req)


def hybrid_search(
    query: str,
    top_k: int | None = None,
    rerank_enabled: bool | None = None,
    similarity_threshold: float | None = None,
    semantic_weight: float | None = None,
    keyword_weight: float | None = None,
    keyword_text: str | None = None,
    author: str | None = None,
    paper_title: str | None = None,
    doc_id: Union[int, List[int], None] = None,
    kb_id: Union[int, List[int], None] = None,
    security_level: Union[int, List[int], None] = None,
    **kwargs: Any,
):
    """混合检索（语义 + 关键词），Query 由本地嵌入模型自动转为向量，集合名称从 .env 获取，支持重排和阈值过滤。"""
    query_vector = _encode_query(query)
    req_kwargs: dict[str, Any] = {
        "query": query,
        "query_vector": query_vector,
        "collection_name": _get_collection_name(),
        **kwargs,
    }
    if top_k is not None:
        req_kwargs["top_k"] = top_k
    if rerank_enabled is not None:
        req_kwargs["rerank_enabled"] = rerank_enabled
    if similarity_threshold is not None:
        req_kwargs["similarity_threshold"] = similarity_threshold
    metadata_filter = _build_metadata_filter(
        keyword_text=keyword_text,
        author=author,
        paper_title=paper_title,
        doc_id=doc_id,
        kb_id=kb_id,
        security_level=security_level,
    )
    if metadata_filter is not None:
        req_kwargs["milvus_expr"] = metadata_filter
    extra = _build_extra_params(semantic_weight=semantic_weight, keyword_weight=keyword_weight)
    if extra:
        req_kwargs["extra_params"] = extra

    req = HybridSearchRequest(**req_kwargs)
    return RetrieverService.hybrid_search(req)


def fulltext_search(
    query: str,
    top_k: int | None = None,
    min_match_count: int | None = None,
    match_mode: str | None = None,
    keyword_text: str | None = None,
    author: str | None = None,
    paper_title: str | None = None,
    doc_id: Union[int, List[int], None] = None,
    kb_id: Union[int, List[int], None] = None,
    security_level: Union[int, List[int], None] = None,
    **kwargs: Any,
):
    """全文检索，集合名称从 .env 获取。"""
    req_kwargs: dict[str, Any] = {
        "query": query,
        "collection_name": _get_collection_name(),
        **kwargs,
    }
    if top_k is not None:
        req_kwargs["top_k"] = top_k
    metadata_filter = _build_metadata_filter(
        keyword_text=keyword_text,
        author=author,
        paper_title=paper_title,
        doc_id=doc_id,
        kb_id=kb_id,
        security_level=security_level,
    )
    if metadata_filter is not None:
        req_kwargs["milvus_expr"] = metadata_filter
    extra = _build_extra_params(min_match_count=min_match_count, match_mode=match_mode)
    if extra:
        req_kwargs["extra_params"] = extra

    req = FulltextSearchRequest(**req_kwargs)
    return RetrieverService.fulltext_search(req)


def text_match_search(
    query: str,
    top_k: int | None = None,
    keyword_text: str | None = None,
    author: str | None = None,
    paper_title: str | None = None,
    match_type: str | None = None,
    case_sensitive: bool | None = None,
    doc_id: Union[int, List[int], None] = None,
    kb_id: Union[int, List[int], None] = None,
    security_level: Union[int, List[int], None] = None,
    **kwargs: Any,
):
    """文本匹配检索，集合名称从 .env 获取。"""
    req_kwargs: dict[str, Any] = {
        "query": query,
        "collection_name": _get_collection_name(),
        **kwargs,
    }
    if top_k is not None:
        req_kwargs["top_k"] = top_k
    metadata_filter = _build_metadata_filter(
        keyword_text=keyword_text,
        author=author,
        paper_title=paper_title,
        doc_id=doc_id,
        kb_id=kb_id,
        security_level=security_level,
    )
    if metadata_filter is not None:
        req_kwargs["milvus_expr"] = metadata_filter
    extra = _build_extra_params(match_type=match_type, case_sensitive=case_sensitive)
    if extra:
        req_kwargs["extra_params"] = extra

    req = TextMatchSearchRequest(**req_kwargs)
    return RetrieverService.text_match_search(req)


def phrase_match_search(
    query: str,
    top_k: int | None = None,
    keyword_text: str | None = None,
    author: str | None = None,
    paper_title: str | None = None,
    case_sensitive: bool | None = None,
    allow_partial: bool | None = None,
    doc_id: Union[int, List[int], None] = None,
    kb_id: Union[int, List[int], None] = None,
    security_level: Union[int, List[int], None] = None,
    **kwargs: Any,
):
    """短语匹配检索，集合名称从 .env 获取。"""
    req_kwargs: dict[str, Any] = {
        "query": query,
        "collection_name": _get_collection_name(),
        **kwargs,
    }
    if top_k is not None:
        req_kwargs["top_k"] = top_k
    metadata_filter = _build_metadata_filter(
        keyword_text=keyword_text,
        author=author,
        paper_title=paper_title,
        doc_id=doc_id,
        kb_id=kb_id,
        security_level=security_level,
    )
    if metadata_filter is not None:
        req_kwargs["milvus_expr"] = metadata_filter
    extra = _build_extra_params(case_sensitive=case_sensitive, allow_partial=allow_partial)
    if extra:
        req_kwargs["extra_params"] = extra

    req = PhraseMatchSearchRequest(**req_kwargs)
    return RetrieverService.phrase_match_search(req)
