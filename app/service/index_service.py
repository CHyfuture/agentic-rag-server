"""索引服务：将 JSON 论文结构 + original_text 通过父子切片 + 多向量写入 Milvus 集合。

参考 `parent_child_index_from_json.py`，但封装为可在应用内调用的 Service。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from dotenv import load_dotenv
from pymilvus import (
    Collection,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from sentence_transformers import SentenceTransformer

from base_db import DocumentClient, DocumentChunkClient
from base_db.abstract.abstract_base_core import AbstractBaseCore
from base_db.parameters.document_chunk_parameters import DocumentChunkModel
from base_db.parameters.document_parameters import DocumentModel
from milvus_service import (
    ChunkRequest,
    ChunkerService,
    CreateCollectionRequest,
    InsertRequest,
    StorageService,
)


# ===========================
# 环境与 Milvus 连接
# ===========================

logger = logging.getLogger(__name__)


def _load_milvus_env() -> None:
    """确保 MILVUS_* 相关环境变量已加载。

    - 优先读取项目根目录下的 `.env`
    - 然后将关键变量写入 os.environ，方便 pymilvus / BaseVector-Core 统一读取
    """
    # app/service/index_service.py -> app/service -> app -> project_root
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    milvus_host = os.getenv("MILVUS_HOST", "192.168.31.51")
    milvus_port = os.getenv("MILVUS_PORT", "19530")
    milvus_user = os.getenv("MILVUS_USER", "")
    milvus_password = os.getenv("MILVUS_PASSWORD", "")
    milvus_db_name = os.getenv("MILVUS_DB_NAME", "default")

    os.environ["MILVUS_HOST"] = milvus_host
    os.environ["MILVUS_PORT"] = milvus_port
    if milvus_user:
        os.environ["MILVUS_USER"] = milvus_user
    if milvus_password:
        os.environ["MILVUS_PASSWORD"] = milvus_password
    os.environ["MILVUS_DB_NAME"] = milvus_db_name


def _connect_milvus() -> None:
    """使用 pymilvus 建立连接，用于 has_collection / 额外索引操作。"""
    _load_milvus_env()

    milvus_host = os.getenv("MILVUS_HOST", "192.168.31.51")
    milvus_port = os.getenv("MILVUS_PORT", "19530")
    milvus_user = os.getenv("MILVUS_USER", "")
    milvus_password = os.getenv("MILVUS_PASSWORD", "")
    milvus_db_name = os.getenv("MILVUS_DB_NAME", "default")

    uri = f"http://{milvus_host}:{milvus_port}"

    # 若已连接会复用，不会重复创建
    connections.connect(
        alias="default",
        uri=uri,
        user=milvus_user or None,
        password=milvus_password or None,
        db_name=milvus_db_name,
    )


# ===========================
# 向量模型与工具函数
# ===========================


_embedding_model: Optional[SentenceTransformer] = None


def _is_truthy_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_embedding_model_path() -> tuple[str, bool]:
    """
    将 EMBEDDING_MODEL 解析为「本地绝对路径」或「Hugging Face repo id」。

    返回 (model_path_or_id, is_local_path_like)。
    - 当配置为 workspace/... 或 Windows 路径等时，确保传给 SentenceTransformer 的是本地路径，
      避免被当成 Hugging Face 仓库名触发网络请求（例如 https://huggingface.co/workspace/...）。
    """
    model_name = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v5-text-small").strip()
    if not model_name:
        model_name = "jinaai/jina-embeddings-v5-text-small"

    is_local_like = (
        model_name.startswith("workspace/")
        or model_name.startswith("workspace\\")
        or model_name.startswith("./")
        or model_name.startswith(".\\")
        or os.path.isabs(model_name)
        or (len(model_name) >= 2 and model_name[1] == ":")
        or "\\" in model_name
    )

    if not is_local_like:
        return model_name, False

    if model_name.startswith("workspace/") or model_name.startswith("workspace\\"):
        root = Path(__file__).resolve().parents[2]
        path = (root / model_name.replace("\\", "/")).resolve()
        return str(path), True

    return str(Path(model_name).resolve()), True


def _get_embedding_device() -> str:
    """获取 Embedding 模型运行设备：有 GPU 时用 cuda，否则用 cpu。可通过环境变量 EMBEDDING_DEVICE 覆盖。"""
    env_device = os.getenv("EMBEDDING_DEVICE", "").strip().lower()
    if env_device in ("cuda", "cpu", "mps"):
        return env_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_sentence_transformer(model_path_or_id: str, *, local_files_only: bool) -> SentenceTransformer:
    """
    兼容不同版本 sentence-transformers：
    - 新版本支持 trust_remote_code
    - 本地路径优先使用 local_files_only，避免任何网络请求
    - 优先使用 GPU（cuda），无 GPU 时回退到 CPU
    """
    device = _get_embedding_device()
    try:
        return SentenceTransformer(
            model_path_or_id,
            device=device,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
    except TypeError:
        return SentenceTransformer(
            model_path_or_id,
            device=device,
            local_files_only=local_files_only,
        )


def _get_embedding_model() -> SentenceTransformer:
    """懒加载向量模型，复用与检索相同的 EMBEDDING_MODEL 配置。"""
    global _embedding_model
    if _embedding_model is not None:
        # 保持原语义：进程内首次加载后永远复用，不因 EMBEDDING_MODEL 改变而重载
        return _embedding_model

    model_path_or_id, is_local_like = _resolve_embedding_model_path()

    # 你选择的是“优先本地，缺文件再尝试线上”
    # 通过环境变量允许关闭线上回退（更适合严格离线环境）
    allow_online_fallback = _is_truthy_env("EMBEDDING_ALLOW_ONLINE_FALLBACK", default=True)

    try:
        # 对于本地路径：强制 local_files_only，避免任何下载行为
        # 对于远程 repo id：允许正常下载（保持原有行为）
        _embedding_model = _load_sentence_transformer(
            model_path_or_id,
            local_files_only=is_local_like,
        )
        return _embedding_model
    except Exception:
        if not (is_local_like and allow_online_fallback):
            raise

    fallback_id = "jinaai/jina-embeddings-v5-text-small"
    _embedding_model = _load_sentence_transformer(fallback_id, local_files_only=False)
    return _embedding_model


def _hash_id(s: str) -> int:
    """将字符串哈成有符号 int64 范围内的正整数，作为 ID 使用。"""
    h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
    return h % (2**63 - 1)


def _zero_vector(dim: int) -> List[float]:
    return [0.0] * dim


def _get_default_collection_name() -> str:
    """从环境变量读取默认集合名，默认为 `papers_chunks`。"""
    return os.getenv("COLLECTION_NAME", "papers_chunks")


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


# ===========================
# 集合创建（父子切片 + 多向量）
# ===========================


def ensure_parent_child_collection(
    collection_name: str,
    dim: int,
) -> None:
    """若集合不存在则按父子切片 + 多向量 schema 创建；存在则补齐多向量索引。"""
    _connect_milvus()

    if utility.has_collection(collection_name, using="default"):
        # 集合已存在：校验向量维度是否一致，否则插入会报错
        coll = Collection(collection_name, using="default")
        for f in coll.schema.fields:
            if f.name == "vector_content" and f.dtype == DataType.FLOAT_VECTOR:
                if f.params.get("dim") != dim:
                    raise ValueError(
                        f"集合 {collection_name} 的向量维度为 {f.params.get('dim')}，"
                        f"与当前模型维度 {dim} 不一致。请删除该集合后重新运行，或设置正确的 EMBEDDING_DIM。"
                    )
                break
        return

    # 定义所有字段（包括主键、主向量、多向量和元数据）
    fields: List[FieldSchema] = []

    # 主键
    fields.append(
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=False,
            description="全局唯一块 ID",
        )
    )

    # 多向量字段
    fields.append(
        FieldSchema(
            name="vector_content",
            dtype=DataType.FLOAT_VECTOR,
            dim=dim,
            description="对 content 编码的主向量",
        )
    )

    # 文档标识与父子结构字段
    fields.extend(
        [
            FieldSchema(
                name="doc_id",
                dtype=DataType.INT64,
                description="文档唯一标识",
            ),
            FieldSchema(
                name="kb_id",
                dtype=DataType.INT64,
                description="文件分类标识",
            ),
            FieldSchema(
                name="security_level",
                dtype=DataType.INT64,
                description="文件访问级别",
            ),
            FieldSchema(
                name="owner_id",
                dtype=DataType.INT64,
                description="所有者 ID",
            ),
            FieldSchema(
                name="chunk_index",
                dtype=DataType.INT64,
                description="当前块在文档中的索引",
            ),
            FieldSchema(
                name="chunk_type",
                dtype=DataType.VARCHAR,
                max_length=16,
                description="块类型：parent / child",
            ),
            FieldSchema(
                name="position_start",
                dtype=DataType.INT64,
                description="在 original_text 中的起始字符位置",
            ),
            FieldSchema(
                name="position_end",
                dtype=DataType.INT64,
                description="在 original_text 中的结束字符位置",
            ),
            FieldSchema(
                name="section_title",
                dtype=DataType.VARCHAR,
                max_length=1024,
                description="最近的 Markdown 标题",
            ),
            FieldSchema(
                name="section_path",
                dtype=DataType.VARCHAR,
                max_length=2048,
                description="章节路径",
            ),
            FieldSchema(
                name="page",
                dtype=DataType.INT64,
                description="页码（如有）",
            ),
        ]
    )

    # 文本内容字段
    fields.extend(
        [
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="块正文内容",
            ),
            FieldSchema(
                name="abstract_text",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="论文摘要文本",
            ),
            FieldSchema(
                name="keywords_text",
                dtype=DataType.VARCHAR,
                max_length=4096,
                description="关键词拼接文本",
            ),
            FieldSchema(
                name="summary_text",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="总结/结论文本",
            ),
        ]
    )

    # 元信息字段
    fields.extend(
        [
            FieldSchema(
                name="title",
                dtype=DataType.VARCHAR,
                max_length=2048,
                description="论文标题",
            ),
            FieldSchema(
                name="authors",
                dtype=DataType.VARCHAR,
                max_length=4096,
                description="作者列表（JSON）",
            ),
            FieldSchema(
                name="institutions",
                dtype=DataType.VARCHAR,
                max_length=4096,
                description="机构列表（JSON）",
            ),
            FieldSchema(
                name="tags",
                dtype=DataType.VARCHAR,
                max_length=4096,
                description="标签/领域/任务等",
            ),
        ]
    )

    # 元数据字段列表：排除主键与主向量字段（vector_content）
    metadata_fields = [f for f in fields if f.name not in ("id", "vector_content")]

    # 通过 StorageService 封装创建集合
    req = CreateCollectionRequest(
        collection_name=collection_name,
        dimension=dim,
        description="父子切片 + 多向量 论文块集合",
        auto_id=False,
        primary_field="id",
        dense_vector_field="vector_content",
        metadata_fields=metadata_fields,
        dense_index_params={
            "metric_type": "IP",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        },
    )

    StorageService.create_collection(req)


# ===========================
# JSON 处理与入库
# ===========================


def _load_json(path: Path) -> Dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_id_from_item(it: Any) -> Optional[int]:
    """从单条返回值中提取 id。"""
    cid = None
    if isinstance(it, dict):
        cid = it.get("id") or it.get("chunk_id")
        if cid is None and "data" in it:
            inner = it.get("data")
            if isinstance(inner, dict):
                cid = inner.get("id") or inner.get("chunk_id")
    elif hasattr(it, "id"):
        cid = getattr(it, "id", None)
    elif hasattr(it, "chunk_id"):
        cid = getattr(it, "chunk_id", None)
    return int(cid) if isinstance(cid, (int, float)) and int(cid) == cid else None


def _extract_chunk_ids_from_batch_response(
    created: Any,
    records: List[Dict[str, Any]],
) -> List[int]:
    """
    从 create_document_chunk_batch 返回值中提取 chunk id，按 records 顺序一一对应。

    通过 (doc_id, chunk_index, content) 匹配每条 record 与返回值，保证 id 与 records 严格对应，
    不依赖 API 返回顺序。
    """
    if created is None or not records:
        logger.debug("[_extract_chunk_ids] created 为空或 records 为空")
        return []
    # dict 用 .get() 取键，避免 getattr(dict, "items") 误取到 dict.items 方法
    if isinstance(created, (list, tuple)):
        items = created
    elif isinstance(created, dict):
        items = created.get("data") or created.get("items") or []
    else:
        items = getattr(created, "data", None) or []
    if not isinstance(items, (list, tuple)):
        logger.warning(f"[_extract_chunk_ids] created 的 data/items 不是列表: type={type(items).__name__}")
        return []
    if len(items) != len(records):
        logger.warning(
            f"[_extract_chunk_ids] BaseDB 返回 items 数量({len(items)}) 与 records 数量({len(records)}) 不一致"
        )
        return []

    def _key(r: Dict[str, Any]) -> tuple:
        return (r.get("doc_id"), r.get("chunk_index"), r.get("content", ""))

    def _item_key(it: Any) -> tuple:
        if isinstance(it, dict):
            return (
                it.get("doc_id"),
                it.get("chunk_order") or it.get("chunk_index"),
                it.get("chunk_text") or it.get("content", ""),
            )
        return (
            getattr(it, "doc_id", None),
            getattr(it, "chunk_order", None) or getattr(it, "chunk_index", None),
            getattr(it, "chunk_text", "") or getattr(it, "content", ""),
        )

    # 按 key 分组，支持 (doc_id, chunk_index, content) 重复时按顺序消费
    item_lists_by_key: Dict[tuple, List[Any]] = defaultdict(list)
    for it in items:
        item_lists_by_key[_item_key(it)].append(it)

    ids: List[int] = []
    for r in records:
        k = _key(r)
        lst = item_lists_by_key.get(k)
        if not lst:
            logger.warning(f"[_extract_chunk_ids] 无法匹配 record: doc_id={k[0]}, chunk_index={k[1]}, content_preview={str(k[2])[:50]}...")
            return []
        it = lst.pop(0)
        cid = _get_id_from_item(it)
        if cid is None:
            logger.warning(f"[_extract_chunk_ids] 无法从 item 提取 id: item={it!r}")
            return []
        ids.append(cid)
    return ids


def _extract_paper_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    title: str = data.get("title", "") or ""
    authors_raw: List[Dict[str, Any]] = data.get("authors", []) or []
    abstract: str = data.get("abstract", "") or ""
    keywords_list: List[str] = data.get("keywords", []) or []
    conclusion: str = data.get("conclusion", "") or ""
    original_text: str = data.get("original_text", "") or ""

    authors_list = authors_raw
    institutions_list = list({a.get("school") for a in authors_raw if a.get("school")})
    authors_str = json.dumps(authors_list, ensure_ascii=False)
    institutions_str = json.dumps(institutions_list, ensure_ascii=False)
    tags: List[str] = keywords_list

    return {
        "title": title,
        "authors_raw": authors_raw,
        "abstract": abstract,
        "keywords_list": keywords_list,
        "conclusion": conclusion,
        "original_text": original_text,
        "authors_str": authors_str,
        "institutions_str": institutions_str,
        "tags": tags,
    }


def _extract_doc_id_from_create_response(created: Any) -> Optional[int]:
    """从 BaseDB create_document 返回值中提取 doc_id。"""
    doc_id = None
    if isinstance(created, dict):
        created_data = created.get("data")
        if isinstance(created_data, dict):
            doc_id = created_data.get("id")
        elif isinstance(created_data, (list, tuple)) and created_data:
            first = created_data[0]
            doc_id = first.get("id") if isinstance(first, dict) else None
    elif hasattr(created, "data"):
        data = getattr(created, "data", None)
        if isinstance(data, dict):
            doc_id = data.get("id")
        elif isinstance(data, (list, tuple)) and data:
            first = data[0]
            doc_id = first.get("id") if isinstance(first, dict) else None

    if isinstance(doc_id, bool):
        return None
    if isinstance(doc_id, (int, float)):
        return int(doc_id)
    return None


def _extract_markdown_title(markdown_text: str, fallback: str) -> str:
    """从 Markdown 提取一级标题；未命中时使用 fallback。"""
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            if title:
                return title
    return fallback


def _build_markdown_paper_data(markdown_text: str, filename: str) -> Dict[str, Any]:
    """将 Markdown 文本封装为 insert_single_paper_data 所需 data 结构。"""
    file_stem = Path(filename).stem if filename else "untitled"
    title = _extract_markdown_title(markdown_text, file_stem)
    return {
        "title": title,
        "authors": [],
        "abstract": "",
        "keywords": [],
        "conclusion": "",
        "original_text": markdown_text,
    }


def _build_plain_text_paper_data(text: str, filename: str, *, title: str = "") -> Dict[str, Any]:
    """将解析后的纯文本封装为 insert_single_paper_data 所需 data 结构。"""
    file_stem = Path(filename).stem if filename else "untitled"
    final_title = (title or "").strip() or file_stem
    return {
        "title": final_title,
        "authors": [],
        "abstract": "",
        "keywords": [],
        "conclusion": "",
        "original_text": text,
    }


def _build_records_and_chunks(
    *,
    data: Dict[str, Any],
    kb_id: int,
    doc_id: int,
    model: SentenceTransformer,
    dim: int,
    tenant_id: int = 1,
    security_level: int = 1,
    owner_id: int = 1,
) -> Tuple[List[Dict[str, Any]], List[DocumentChunkModel]]:
    """从 JSON 数据构建 Milvus 记录与 DocumentChunkModel."""
    meta = _extract_paper_metadata(data)
    title: str = meta["title"]
    authors_str: str = meta["authors_str"]
    institutions_str: str = meta["institutions_str"]
    abstract_text: str = meta["abstract"]
    keywords_list: List[str] = meta["keywords_list"]
    summary_text: str = meta["conclusion"]
    original_text: str = meta["original_text"]

    logger.error(f"[_build_records_and_chunks] 入参: original_text 长度={len(original_text)}, title={title[:50] if title else ''}...")
    if not original_text:
        logger.error("[_build_records_and_chunks] 原因: original_text 为空，直接返回 []")
        return [], []

    keywords_text = "；".join(keywords_list)

    chunk_req = ChunkRequest(
        text=original_text,
        strategy="parent_child",
    )
    chunks = ChunkerService.chunk(chunk_req)
    chunks_list = list(chunks) if chunks else []
    chunks_count = len(chunks_list)
    logger.error(f"[_build_records_and_chunks] ChunkerService 返回 chunks 数量: {chunks_count}")

    if not chunks_list:
        logger.error("[_build_records_and_chunks] 原因: ChunkerService 切片结果为空，返回 []")
        return [], []

    parents: Dict[int, Any] = {}
    children_by_parent_index: Dict[int, List[Any]] = {}
    chunks_without_parent: List[Any] = []  # child 但 parent_idx 为 None 的 chunk
    for i, c in enumerate(chunks_list):
        metadata: Dict[str, Any] = getattr(c, "metadata", {}) or {}
        chunk_type: str = (metadata.get("chunk_type") or "").strip().lower()
        chunk_index: int = int(getattr(c, "chunk_index", 0) or 0)

        # 优先从 metadata 或 parent_chunk_id 获取父块索引
        parent_idx: Optional[int] = None
        for key in ("parent_chunk_index", "parent_index", "parent_id"):
            if key in metadata and metadata[key] is not None:
                try:
                    parent_idx = int(metadata[key])
                    break
                except Exception:
                    parent_idx = None
        if parent_idx is None:
            top_parent = getattr(c, "parent_chunk_id", None)
            if top_parent is not None:
                try:
                    parent_idx = int(top_parent)
                except Exception:
                    parent_idx = None

        # 判定 parent/child：有 parent_idx 即为 child；或 metadata 明确为 child
        is_child = chunk_type == "child" or parent_idx is not None
        if is_child and parent_idx is not None:
            children_by_parent_index.setdefault(parent_idx, []).append(c)
        elif is_child and parent_idx is None:
            chunks_without_parent.append(c)
        else:
            parents[chunk_index] = c

        # 前 5 个 chunk 打印详细诊断
        if i < 5:
            logger.error(
                f"[_build_records_and_chunks] chunk[{i}] chunk_index={chunk_index} chunk_type={chunk_type!r} "
                f"parent_idx={parent_idx} metadata.keys={list(metadata.keys())} -> "
                f"{'child' if (is_child and parent_idx is not None) else 'parent' if not is_child else 'child(无parent_idx)'}"
            )

    flat_children_count = sum(len(v) for v in children_by_parent_index.values())
    logger.error(
        f"[_build_records_and_chunks] 解析结果: parent 块 {len(parents)} 个(parent_keys={list(parents.keys())[:20]}), "
        f"child 块 {flat_children_count} 个(child_parent_keys={list(children_by_parent_index.keys())[:20]}), "
        f"child 但无 parent_idx 的块 {len(chunks_without_parent)} 个"
    )

    # 诊断：child 引用的 parent 是否存在于 parents 中
    orphan_parent_indices = [k for k in children_by_parent_index if k not in parents]
    if orphan_parent_indices:
        logger.error(
            f"[_build_records_and_chunks] 诊断: 有 {len(orphan_parent_indices)} 个 child 引用的 parent_index 不在 parents 中: "
            f"{orphan_parent_indices[:10]}，这些 child 不会被写入"
        )

    # 先按顺序收集所有 child 的 (chunk, parent_content)，再批量 encode
    flat_children: List[Tuple[Any, str]] = []
    for parent_index in sorted(parents.keys()):
        parent_chunk = parents[parent_index]
        parent_content: str = getattr(parent_chunk, "content", "") or ""
        for c in children_by_parent_index.get(parent_index, []):
            flat_children.append((c, parent_content))

    # Fallback1：全为 child 且 parent_idx 为 None 时，将这些 chunk 视为 parent 以便后续 fallback2 使用
    if not parents and chunks_without_parent:
        logger.error(
            f"[_build_records_and_chunks] 无 parent 块，但有 {len(chunks_without_parent)} 个 child(无 parent_idx)，"
            "将其视为 parent 以便写入"
        )
        for i, c in enumerate(chunks_without_parent):
            parents[100000 + i] = c  # 合成索引避免与 chunk_index 冲突

    # Fallback2：若无 child 块但有 parent 块，将每个 parent 视为自引用（parent 即 child），确保有数据可写入
    if not flat_children and parents:
        c0 = next(iter(parents.values()))
        meta0 = getattr(c0, "metadata", {}) or {}
        logger.error(
            "[_build_records_and_chunks] 无 child 块，触发 fallback：将 parent 块作为可写入块。"
            f"首个 chunk 诊断: metadata.keys={list(meta0.keys())} chunk_type={meta0.get('chunk_type')!r} "
            f"parent_chunk_id={getattr(c0, 'parent_chunk_id', None)} content_len={len(getattr(c0, 'content', '') or '')}"
        )
        for parent_index in sorted(parents.keys()):
            parent_chunk = parents[parent_index]
            parent_content = getattr(parent_chunk, "content", "") or ""
            flat_children.append((parent_chunk, parent_content))

    records: List[Dict[str, Any]] = []
    chunk_models: List[DocumentChunkModel] = []

    embedding_model_name = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v5-text-small")

    contents = [getattr(c, "content", "") or "" for c, _ in flat_children]
    to_encode = [t for t in contents if t]
    logger.error(
        f"[_build_records_and_chunks] flat_children 数量={len(flat_children)}, 非空 content 数量={len(to_encode)}"
    )
    if to_encode:
        content_vecs = model.encode(
            to_encode,
            task="retrieval",
            show_progress_bar=False,
            batch_size=64,
        )
        vec_iter = iter(content_vecs)
    else:
        vec_iter = iter([])
        if flat_children:
            logger.error(
                "[_build_records_and_chunks] 诊断: flat_children 非空但所有 content 为空，将使用零向量"
            )

    if not flat_children:
        logger.error(
            "[_build_records_and_chunks] 原因: flat_children 为空 -> records 为空。"
            f"可能原因: 1) 全为 child 且 parent_idx 不在 parents 中 2) 全为 child 且 parent_idx 为 None 3) parents 为空"
        )
        return [], []

    for i, ((c, parent_content), content) in enumerate(zip(flat_children, contents), start=1):
        chunk_index: int = int(getattr(c, "chunk_index", 0) or 0)
        start_index: int = int(getattr(c, "start_index", 0) or 0)
        end_index: int = int(getattr(c, "end_index", 0) or 0)

        metadata: Dict[str, Any] = getattr(c, "metadata", {}) or {}
        chunk_type: str = metadata.get("chunk_type") or "child"
        section_title: str = metadata.get("section_title") or ""
        section_path: str = metadata.get("section_path") or ""
        page_val = metadata.get("page")
        page: Optional[int] = int(page_val) if page_val is not None else None

        vec_content = next(vec_iter).tolist() if content else _zero_vector(dim)

        id_val = _hash_id(f"{doc_id}_{i}_{content[:80]}" if content else f"{doc_id}_{i}")

        record: Dict[str, Any] = {
            "id": id_val,
            "doc_id": doc_id,
            "kb_id": kb_id,
            "security_level": security_level,
            "owner_id": owner_id,
            "chunk_index": chunk_index,
            "chunk_type": chunk_type,
            "position_start": start_index,
            "position_end": end_index,
            "section_title": section_title,
            "section_path": section_path,
            "page": page if page is not None else -1,
            "content": content,
            "abstract_text": abstract_text,
            "keywords_text": keywords_text,
            "summary_text": summary_text,
            "title": title,
            "authors": authors_str,
            "institutions": institutions_str,
            "tags": "",
            "vector_content": vec_content,
        }
        records.append(record)

        chunk = DocumentChunkModel()
        chunk.doc_id = doc_id
        chunk.tenant_id = tenant_id
        chunk.chunk_text = content
        chunk.chunk_order = chunk_index
        chunk.token_count = 0
        chunk.embedding_model = embedding_model_name
        chunk.parent_content = parent_content
        chunk.extra = {
            "section_title": section_title,
            "section_path": section_path,
            "page": page if page is not None else -1,
            "kb_id": kb_id,
            "vector_chunk_id": id_val,
        }
        chunk_models.append(chunk)

    logger.error(f"[_build_records_and_chunks] 构建完成: records={len(records)}, chunk_models={len(chunk_models)}")
    return records, chunk_models

async def build_index_from_json_contents(
    kb_id: int,
    items: Sequence[Tuple[str, str]],
    collection_name: Optional[str] = None,
    model_name: Optional[str] = None,
    dim: Optional[int] = None,
    skip_base_db: bool = False,
) -> Dict[str, Any]:
    """从上传的 JSON 内容批量构建索引，写入 Milvus；可选写入 BaseDB。

    - skip_base_db=True：仅写入 Milvus，不调用 BaseDB（适用于 BaseDB 不可用或本地脚本场景）
    - skip_base_db=False：先写入 BaseDB 获取 doc_id，再写入 Milvus 与切片表（与 API 行为一致）
    """
    logger.error(f"[build_index_from_json_contents] 开始: kb_id={kb_id}, 文件数={len(items)}, skip_base_db={skip_base_db}")

    if not items:
        logger.warning("[build_index_from_json_contents] 无待处理文件")
        return {
            "kb_id": kb_id,
            "total_documents": 0,
            "total_chunks": 0,
            "milvus_records": 0,
            "skipped_files": [],
        }

    _load_milvus_env()
    if not skip_base_db:
        _load_db_env()

    collection = collection_name or _get_default_collection_name()

    if model_name:
        os.environ["EMBEDDING_MODEL"] = model_name
        model = _get_embedding_model()
    else:
        model = _get_embedding_model()

    # 维度必须与模型输出一致：jina-embeddings-v5-text-small 为 1024，默认 768 会报错
    if dim is not None:
        vector_dim = dim
    else:
        env_dim = os.getenv("EMBEDDING_DIM")
        vector_dim = int(env_dim) if env_dim else model.get_sentence_embedding_dimension()

    ensure_parent_child_collection(collection, vector_dim)
    logger.error(f"[build_index_from_json_contents] 集合 {collection} 已就绪, vector_dim={vector_dim}")

    doc_client = _get_document_client() if not skip_base_db else None
    chunk_client = _get_chunk_client() if not skip_base_db else None

    tenant_id = int(os.getenv("DB_TENANT_ID", "1"))
    owner_id = int(os.getenv("DB_OWNER_ID", "1"))
    security_level = int(os.getenv("DB_SECURITY_LEVEL", "1"))
    status = os.getenv("DB_DOCUMENT_STATUS", "indexed")

    total_documents = 0
    total_chunks = 0
    milvus_records = 0
    skipped_files: List[str] = []

    for filename, content in items:
        logger.error(f"[build_index_from_json_contents] 处理文件: {filename}")
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"[build_index_from_json_contents] JSON 解析失败 {filename}: {e}")
            skipped_files.append(filename)
            continue

        meta = _extract_paper_metadata(data)
        original_text: str = meta["original_text"]
        if not original_text:
            logger.warning(f"[build_index_from_json_contents] {filename} original_text 为空，跳过")
            skipped_files.append(filename)
            continue
        logger.error(f"[build_index_from_json_contents] {filename} original_text 长度: {len(original_text)}")

        title: str = meta["title"]
        authors_raw: List[Dict[str, Any]] = meta["authors_raw"]
        institutions_list = list(
            {a.get("school") for a in authors_raw if a.get("school")}
        )
        tags: List[str] = meta["tags"]

        doc_extra: Dict[str, Any] = {
            "title": title,
            "tags": tags,
            "authors": authors_raw,
            "institutions": institutions_list,
            "source_file": filename,
        }

        if skip_base_db:
            doc_id = _hash_id(f"{filename}_{original_text[:100]}")
            logger.error(f"[build_index_from_json_contents] skip_base_db=True, 使用 hash doc_id={doc_id}")
        else:
            document = DocumentModel()
            document.kb_id = kb_id
            document.tenant_id = tenant_id
            document.owner_id = owner_id
            document.file_name = filename
            document.minio_path = f"pdf/{filename}"
            document.file_type = "pdf"
            document.file_size = len(original_text.encode("utf-8"))
            document.markdown_content = original_text
            document.status = status
            document.security_level = security_level
            document.extra = doc_extra

            try:
                created = await doc_client.create_document(document)
                logger.error(f"[build_index_from_json_contents] BaseDB create_document 返回类型: {type(created).__name__}")
            except Exception as exc:
                logger.error(f"[build_index_from_json_contents] BaseDB 创建文档失败 {filename}: {exc}", exc_info=True)
                skipped_files.append(filename)
                continue

            # 兼容多种返回结构: {"data": {"id": N}} / {"data": [{"id": N}]} / 对象.data.id
            doc_id = _extract_doc_id_from_create_response(created)
            if not isinstance(doc_id, int):
                logger.error(
                    f"[build_index_from_json_contents] 无法从 BaseDB 响应提取 doc_id, created={created!r}"
                )
                skipped_files.append(filename)
                continue
            logger.error(f"[build_index_from_json_contents] doc_id={doc_id}")
        records, chunk_models = _build_records_and_chunks(
            data=data,
            kb_id=kb_id,
            doc_id=doc_id,
            model=model,
            dim=vector_dim,
            tenant_id=tenant_id,
            security_level=security_level,
            owner_id=owner_id,
        )

        if not records:
            logger.warning(f"[build_index_from_json_contents] {filename} records 为空（切片无 child 块），跳过")
            skipped_files.append(filename)
            continue

        # 使用 BaseDB 时：先写入 DB 获取 chunk_id，再替换 records 的 id 后写入 Milvus，保证 id 一致
        if not skip_base_db:
            try:
                created = await chunk_client.create_document_chunk_batch(chunk_models)
                logger.error(
                    f"[build_index_from_json_contents] BaseDB create_document_chunk_batch 返回类型: {type(created).__name__}, "
                    f"records 数={len(records)}"
                )
            except Exception as exc:
                logger.error(f"[build_index_from_json_contents] BaseDB 创建 chunk 失败 {filename}: {exc}", exc_info=True)
                skipped_files.append(filename)
                continue
            chunk_ids = _extract_chunk_ids_from_batch_response(created, records)
            if not chunk_ids:
                logger.error(
                    f"[build_index_from_json_contents] 无法从 BaseDB 响应提取 chunk_ids, "
                    f"created 类型={type(created).__name__}, records 数={len(records)}。"
                    "请检查 BaseDB create_document_chunk_batch 返回格式与 _extract_chunk_ids_from_batch_response 匹配逻辑。"
                )
                skipped_files.append(filename)
                continue
            logger.error(f"[build_index_from_json_contents] 已获取 chunk_ids 数量: {len(chunk_ids)}")
            for rec, cid in zip(records, chunk_ids):
                rec["id"] = cid

        logger.error(f"[build_index_from_json_contents] 准备写入 Milvus: collection={collection}, records 数={len(records)}")
        try:
            insert_req = InsertRequest(
                collection_name=collection,
                records=records,
            )
            ids = StorageService.insert(insert_req)
            logger.error(f"[build_index_from_json_contents] Milvus 写入成功: 返回 ids 数量={len(ids) if ids else 0}")
        except Exception as exc:
            logger.error(f"[build_index_from_json_contents] Milvus 写入失败 {filename}: {exc}", exc_info=True)
            skipped_files.append(filename)
            continue
        inserted_count = len(ids) if ids else 0
        total_documents += 1
        total_chunks += len(chunk_models)
        milvus_records += inserted_count
        logger.error(f"[build_index_from_json_contents] {filename} 完成: doc_id={doc_id}, 写入 Milvus {inserted_count} 条")

    logger.error(
        f"[build_index_from_json_contents] 全部完成: total_documents={total_documents}, "
        f"total_chunks={total_chunks}, milvus_records={milvus_records}, skipped={len(skipped_files)}"
    )
    return {
        "kb_id": kb_id,
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        "milvus_records": milvus_records,
        "skipped_files": skipped_files,
    }


async def build_index_from_markdown_contents(
    kb_id: int,
    items: Sequence[Tuple[str, str]],
    *,
    skip_base_db: bool = False,
) -> Dict[str, Any]:
    """从上传的 Markdown 内容批量构建索引。"""
    logger.error(
        f"[build_index_from_markdown_contents] 开始: kb_id={kb_id}, 文件数={len(items)}, skip_base_db={skip_base_db}"
    )
    if not items:
        return {
            "kb_id": kb_id,
            "total_documents": 0,
            "total_chunks": 0,
            "milvus_records": 0,
            "skipped_files": [],
        }

    _load_db_env()
    doc_client = _get_document_client()

    tenant_id = int(os.getenv("DB_TENANT_ID", "1"))
    owner_id = int(os.getenv("DB_OWNER_ID", "1"))
    security_level = int(os.getenv("DB_SECURITY_LEVEL", "1"))
    status = os.getenv("DB_DOCUMENT_STATUS", "indexed")

    total_documents = 0
    total_chunks = 0
    milvus_records = 0
    skipped_files: List[str] = []

    for filename, markdown_text in items:
        source_name = filename or "unknown.md"
        if not markdown_text.strip():
            logger.warning(f"[build_index_from_markdown_contents] {source_name} 内容为空，跳过")
            skipped_files.append(source_name)
            continue

        paper_data = _build_markdown_paper_data(markdown_text, source_name)
        title = paper_data.get("title", "")
        doc_extra: Dict[str, Any] = {
            "title": title,
            "tags": [],
            "authors": [],
            "institutions": [],
            "source_file": source_name,
        }

        document = DocumentModel()
        document.kb_id = kb_id
        document.tenant_id = tenant_id
        document.owner_id = owner_id
        document.file_name = source_name
        document.minio_path = f"markdown/{source_name}"
        document.file_type = "md"
        document.file_size = len(markdown_text.encode("utf-8"))
        document.markdown_content = markdown_text
        document.status = status
        document.security_level = security_level
        document.extra = doc_extra

        try:
            created = await doc_client.create_document(document)
        except Exception as exc:
            logger.error(
                f"[build_index_from_markdown_contents] BaseDB 创建文档失败 {source_name}: {exc}",
                exc_info=True,
            )
            skipped_files.append(source_name)
            continue

        doc_id = _extract_doc_id_from_create_response(created)
        if not isinstance(doc_id, int):
            logger.error(
                f"[build_index_from_markdown_contents] 无法提取 doc_id: source={source_name}, created={created!r}"
            )
            skipped_files.append(source_name)
            continue

        try:
            result = await insert_single_paper_data(
                kb_id=kb_id,
                doc_id=doc_id,
                data=paper_data,
                tenant_id=tenant_id,
                security_level=security_level,
                owner_id=owner_id,
                filename=source_name,
                skip_base_db=skip_base_db,
            )
        except Exception as exc:
            logger.error(
                f"[build_index_from_markdown_contents] insert_single_paper_data 异常 {source_name}: {exc}",
                exc_info=True,
            )
            skipped_files.append(source_name)
            continue

        if not result.get("success"):
            logger.warning(
                f"[build_index_from_markdown_contents] 写入失败 {source_name}: {result.get('error', 'unknown')}"
            )
            skipped_files.append(source_name)
            continue

        total_documents += 1
        chunk_count = int(result.get("chunk_count", 0) or 0)
        total_chunks += chunk_count
        milvus_ids = result.get("milvus_ids", []) or []
        milvus_records += len(milvus_ids)

    return {
        "kb_id": kb_id,
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        "milvus_records": milvus_records,
        "skipped_files": skipped_files,
    }


async def build_index_from_parsed_document_contents(
    kb_id: int,
    items: Sequence[Dict[str, Any]],
    *,
    skip_base_db: bool = False,
) -> Dict[str, Any]:
    """从解析后的文档内容批量构建索引。"""
    logger.error(
        f"[build_index_from_parsed_document_contents] 开始: kb_id={kb_id}, 文件数={len(items)}, skip_base_db={skip_base_db}"
    )
    if not items:
        return {
            "kb_id": kb_id,
            "total_documents": 0,
            "total_chunks": 0,
            "milvus_records": 0,
            "skipped_files": [],
        }

    _load_db_env()
    doc_client = _get_document_client()

    tenant_id = int(os.getenv("DB_TENANT_ID", "1"))
    owner_id = int(os.getenv("DB_OWNER_ID", "1"))
    security_level = int(os.getenv("DB_SECURITY_LEVEL", "1"))
    status = os.getenv("DB_DOCUMENT_STATUS", "indexed")

    total_documents = 0
    total_chunks = 0
    milvus_records = 0
    skipped_files: List[str] = []

    for item in items:
        source_name = str(item.get("filename") or "unknown.bin")
        parsed_text = str(item.get("content") or "")
        parsed_title = str(item.get("title") or "")
        file_type = str(item.get("file_type") or "bin")

        if not parsed_text.strip():
            logger.warning(f"[build_index_from_parsed_document_contents] {source_name} 解析内容为空，跳过")
            skipped_files.append(source_name)
            continue

        paper_data = _build_plain_text_paper_data(parsed_text, source_name, title=parsed_title)
        title = paper_data.get("title", "")
        doc_extra: Dict[str, Any] = {
            "title": title,
            "tags": [],
            "authors": [],
            "institutions": [],
            "source_file": source_name,
            "parsed": True,
        }

        document = DocumentModel()
        document.kb_id = kb_id
        document.tenant_id = tenant_id
        document.owner_id = owner_id
        document.file_name = source_name
        document.minio_path = f"documents/{source_name}"
        document.file_type = file_type
        document.file_size = len(parsed_text.encode("utf-8"))
        document.markdown_content = parsed_text
        document.status = status
        document.security_level = security_level
        document.extra = doc_extra

        try:
            created = await doc_client.create_document(document)
        except Exception as exc:
            logger.error(
                f"[build_index_from_parsed_document_contents] BaseDB 创建文档失败 {source_name}: {exc}",
                exc_info=True,
            )
            skipped_files.append(source_name)
            continue

        doc_id = _extract_doc_id_from_create_response(created)
        if not isinstance(doc_id, int):
            logger.error(
                f"[build_index_from_parsed_document_contents] 无法提取 doc_id: source={source_name}, created={created!r}"
            )
            skipped_files.append(source_name)
            continue

        try:
            result = await insert_single_paper_data(
                kb_id=kb_id,
                doc_id=doc_id,
                data=paper_data,
                tenant_id=tenant_id,
                security_level=security_level,
                owner_id=owner_id,
                filename=source_name,
                skip_base_db=skip_base_db,
            )
        except Exception as exc:
            logger.error(
                f"[build_index_from_parsed_document_contents] insert_single_paper_data 异常 {source_name}: {exc}",
                exc_info=True,
            )
            skipped_files.append(source_name)
            continue

        if not result.get("success"):
            logger.warning(
                f"[build_index_from_parsed_document_contents] 写入失败 {source_name}: {result.get('error', 'unknown')}"
            )
            skipped_files.append(source_name)
            continue

        total_documents += 1
        chunk_count = int(result.get("chunk_count", 0) or 0)
        total_chunks += chunk_count
        milvus_ids = result.get("milvus_ids", []) or []
        milvus_records += len(milvus_ids)

    return {
        "kb_id": kb_id,
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        "milvus_records": milvus_records,
        "skipped_files": skipped_files,
    }


async def insert_single_paper_data(
    kb_id: int,
    doc_id: int,
    data: Dict[str, Any],
    *,
    tenant_id: int = 1,
    security_level: int = 1,
    owner_id: int = 1,
    filename: str = "paper.json",
    skip_base_db: bool = False,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """将单条论文数据插入 Milvus（含切片、向量化、可选 BaseDB 切片写入）。

    仅保留 chunk 信息的保存和处理逻辑，不保存 DocumentModel。
    doc_id、tenant_id、security_level 等 chunk 保存所需参数均由请求传入。

    入参:
        kb_id: 知识库 ID
        doc_id: 文档 ID（由调用方提供，不创建文档）
        data: 论文 JSON 对象，需包含 original_text 及 title、authors、abstract、keywords、conclusion 等
        tenant_id: 租户 ID，默认 1
        security_level: 密级，默认 1
        owner_id: 所有者 ID，默认 1
        filename: 源文件名，用于记录标识，默认 "paper.json"
        skip_base_db: 是否跳过 BaseDB 切片写入，True 时仅写入 Milvus
        collection_name: 集合名称，默认从环境变量读取

    返回:
        {
            "doc_id": int,
            "chunk_count": int,
            "milvus_ids": List[int],
            "success": bool,
            "error": str | None,  # 失败时
        }
    """
    _load_milvus_env()
    if not skip_base_db:
        _load_db_env()

    collection = collection_name or _get_default_collection_name()
    model = _get_embedding_model()
    env_dim = os.getenv("EMBEDDING_DIM")
    vector_dim = int(env_dim) if env_dim else model.get_sentence_embedding_dimension()
    ensure_parent_child_collection(collection, vector_dim)

    meta = _extract_paper_metadata(data)
    original_text: str = meta["original_text"]
    if not original_text:
        return {"success": False, "error": "original_text 为空", "doc_id": doc_id, "chunk_count": 0, "milvus_ids": []}

    records, chunk_models = _build_records_and_chunks(
        data=data,
        kb_id=kb_id,
        doc_id=doc_id,
        model=model,
        dim=vector_dim,
        tenant_id=tenant_id,
        security_level=security_level,
        owner_id=owner_id,
    )

    if not records:
        return {"success": False, "error": "切片结果为空", "doc_id": doc_id, "chunk_count": 0, "milvus_ids": []}

    if not skip_base_db:
        chunk_client = _get_chunk_client()
        try:
            created = await chunk_client.create_document_chunk_batch(chunk_models)
        except Exception as exc:
            return {"success": False, "error": str(exc), "doc_id": doc_id, "chunk_count": 0, "milvus_ids": []}
        chunk_ids = _extract_chunk_ids_from_batch_response(created, records)
        if not chunk_ids:
            return {"success": False, "error": "无法获取 chunk_id", "doc_id": doc_id, "chunk_count": 0, "milvus_ids": []}
        for rec, cid in zip(records, chunk_ids):
            rec["id"] = cid

    insert_req = InsertRequest(collection_name=collection, records=records)
    logger.error(f"[insert_single_paper_data] 写入 Milvus: collection={collection}, records 数={len(records)}")
    ids = StorageService.insert(insert_req)

    return {
        "success": True,
        "doc_id": doc_id,
        "chunk_count": len(records),
        "milvus_ids": [int(x) for x in ids] if ids else [],
    }


async def delete_document_by_doc_id(
    doc_id: int,
    *,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """按 doc_id 删除文档：同步删除 Milvus 向量与 BaseDB chunk 数据。"""
    _load_milvus_env()
    _load_db_env()

    collection = collection_name or _get_default_collection_name()

    # 1. 删除 Milvus 中该 doc_id 的向量记录
    _connect_milvus()
    coll = Collection(collection, using="default")
    coll.delete(f"doc_id == {doc_id}")
    coll.flush()

    # 2. 删除 BaseDB 中该 doc_id 的 chunk
    chunk_client = _get_chunk_client()
    try:
        await chunk_client.remove_document_chunks_by_doc_id(doc_id=doc_id)
    except Exception as exc:
        return {"success": False, "doc_id": doc_id, "error": f"删除 BaseDB 切片失败: {exc}"}

    return {"success": True, "doc_id": doc_id}


async def update_single_paper_data(
    kb_id: int,
    doc_id: int,
    data: Dict[str, Any],
    *,
    tenant_id: int = 1,
    security_level: int = 1,
    owner_id: int = 1,
    filename: str = "paper.json",
    skip_base_db: bool = False,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """更新单条论文数据：同步删除 Milvus 与 BaseDB 中该 doc_id 的旧记录，再按 insert 逻辑重新切片、向量化、入库。

    更新操作始终同步 BaseDB：删除旧 chunk 并重新新增，与 Milvus 保持一致。
    """
    _load_milvus_env()
    _load_db_env()

    collection = collection_name or _get_default_collection_name()
    model = _get_embedding_model()
    env_dim = os.getenv("EMBEDDING_DIM")
    vector_dim = int(env_dim) if env_dim else model.get_sentence_embedding_dimension()
    ensure_parent_child_collection(collection, vector_dim)

    meta = _extract_paper_metadata(data)
    original_text: str = meta["original_text"]
    if not original_text:
        return {"success": False, "error": "original_text 为空", "doc_id": doc_id, "chunk_count": 0, "milvus_ids": []}

    # 1. 删除 Milvus 中该 doc_id 的旧记录
    _connect_milvus()
    coll = Collection(collection, using="default")
    coll.delete(f"doc_id == {doc_id}")
    coll.flush()

    # 2. 同步删除 BaseDB 中该 doc_id 的旧 chunk
    chunk_client = _get_chunk_client()
    try:
        await chunk_client.remove_document_chunks_by_doc_id(doc_id=doc_id)
    except Exception as exc:
        return {"success": False, "error": f"删除 BaseDB 切片失败: {exc}", "doc_id": doc_id, "chunk_count": 0, "milvus_ids": []}

    # 3. 构建新记录（与 insert 相同逻辑）
    records, chunk_models = _build_records_and_chunks(
        data=data,
        kb_id=kb_id,
        doc_id=doc_id,
        model=model,
        dim=vector_dim,
        tenant_id=tenant_id,
        security_level=security_level,
        owner_id=owner_id,
    )

    if not records:
        return {"success": False, "error": "切片结果为空", "doc_id": doc_id, "chunk_count": 0, "milvus_ids": []}

    # 4. 同步重新新增 BaseDB chunk
    try:
        created = await chunk_client.create_document_chunk_batch(chunk_models)
    except Exception as exc:
        return {"success": False, "error": str(exc), "doc_id": doc_id, "chunk_count": 0, "milvus_ids": []}
    chunk_ids = _extract_chunk_ids_from_batch_response(created, records)
    if not chunk_ids:
        return {"success": False, "error": "无法获取 chunk_id", "doc_id": doc_id, "chunk_count": 0, "milvus_ids": []}
    for rec, cid in zip(records, chunk_ids):
        rec["id"] = cid

    # 5. 插入 Milvus
    insert_req = InsertRequest(collection_name=collection, records=records)
    ids = StorageService.insert(insert_req)

    return {
        "success": True,
        "doc_id": doc_id,
        "chunk_count": len(records),
        "milvus_ids": [int(x) for x in ids] if ids else [],
    }

