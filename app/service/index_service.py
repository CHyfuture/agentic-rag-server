"""索引服务：将 JSON 论文结构 + original_text 通过父子切片 + 多向量写入 Milvus 集合。

参考 `parent_child_index_from_json.py`，但封装为可在应用内调用的 Service。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from pymilvus import (
    Collection,
    CollectionSchema,
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


def _load_sentence_transformer(model_path_or_id: str, *, local_files_only: bool) -> SentenceTransformer:
    """
    兼容不同版本 sentence-transformers：
    - 新版本支持 trust_remote_code
    - 本地路径优先使用 local_files_only，避免任何网络请求
    """
    try:
        return SentenceTransformer(
            model_path_or_id,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
    except TypeError:
        return SentenceTransformer(
            model_path_or_id,
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
        _create_vector_indexes_if_needed(collection_name)
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
    fields.append(
        FieldSchema(
            name="vector_abstract",
            dtype=DataType.FLOAT_VECTOR,
            dim=dim,
            description="对 abstract_text 编码的向量",
        )
    )
    fields.append(
        FieldSchema(
            name="vector_keywords",
            dtype=DataType.FLOAT_VECTOR,
            dim=dim,
            description="对 keywords_text 编码的向量",
        )
    )
    fields.append(
        FieldSchema(
            name="vector_summary",
            dtype=DataType.FLOAT_VECTOR,
            dim=dim,
            description="对 summary_text 编码的向量",
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
                name="chunk_index",
                dtype=DataType.INT64,
                description="当前块在文档中的索引",
            ),
            FieldSchema(
                name="parent_chunk_id",
                dtype=DataType.INT64,
                description="父块对应的文件名（从 1 开始），无父块时为 -1",
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

    # 为其他向量字段创建索引
    _create_vector_indexes_if_needed(collection_name)


def _create_vector_indexes_if_needed(collection_name: str) -> None:
    """为除主向量外的多向量字段创建索引（如未创建）。"""
    coll = Collection(collection_name)
    existing_idx_fields = {idx.field_name for idx in coll.indexes}

    index_params = {
        "metric_type": "IP",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200},
    }

    for vec_field in ("vector_abstract", "vector_keywords", "vector_summary"):
        if vec_field in [f.name for f in coll.schema.fields]:
            if vec_field not in existing_idx_fields:
                coll.create_index(field_name=vec_field, index_params=index_params)


# ===========================
# JSON 处理与入库
# ===========================


def _load_json(path: Path) -> Dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def _build_records_and_chunks(
    *,
    data: Dict[str, Any],
    kb_id: int,
    doc_id: int,
    model: SentenceTransformer,
    dim: int,
    start_child_id: int = 1,
    start_parent_file_id: int = 1,
    tenant_id: int = 1,
    security_level: int = 1,
) -> Tuple[List[Dict[str, Any]], List[DocumentChunkModel], int]:
    """从 JSON 数据构建 Milvus 记录与 DocumentChunkModel."""
    meta = _extract_paper_metadata(data)
    title: str = meta["title"]
    authors_str: str = meta["authors_str"]
    institutions_str: str = meta["institutions_str"]
    abstract_text: str = meta["abstract"]
    keywords_list: List[str] = meta["keywords_list"]
    summary_text: str = meta["conclusion"]
    original_text: str = meta["original_text"]

    if not original_text:
        return [], [], 0

    keywords_text = "；".join(keywords_list)

    vec_abstract = (
        model.encode([abstract_text], task="retrieval")[0].tolist()
        if abstract_text
        else _zero_vector(dim)
    )
    vec_keywords = (
        model.encode([keywords_text], task="retrieval")[0].tolist()
        if keywords_text
        else _zero_vector(dim)
    )
    vec_summary = (
        model.encode([summary_text], task="retrieval")[0].tolist()
        if summary_text
        else _zero_vector(dim)
    )

    chunk_req = ChunkRequest(
        text=original_text,
        strategy="parent_child",
    )
    chunks = ChunkerService.chunk(chunk_req)
    if not chunks:
        return [], [], 0

    parents: Dict[int, Any] = {}
    children_by_parent_index: Dict[int, List[Any]] = {}
    for c in chunks:
        metadata: Dict[str, Any] = getattr(c, "metadata", {}) or {}
        chunk_type: str = metadata.get("chunk_type") or ""
        chunk_index: int = int(getattr(c, "chunk_index", 0) or 0)

        parent_idx: Optional[int] = None
        if chunk_type == "child":
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

        if chunk_type == "parent" or not chunk_type:
            parents[chunk_index] = c
        elif chunk_type == "child" and parent_idx is not None:
            children_by_parent_index.setdefault(parent_idx, []).append(c)

    next_parent_file_id = start_parent_file_id
    next_child_id = start_child_id

    records: List[Dict[str, Any]] = []
    chunk_models: List[DocumentChunkModel] = []

    embedding_model_name = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v5-text-small")

    for parent_index in sorted(parents.keys()):
        parent_chunk = parents[parent_index]
        parent_content: str = getattr(parent_chunk, "content", "") or ""

        file_id = next_parent_file_id
        next_parent_file_id += 1

        for c in children_by_parent_index.get(parent_index, []):
            content: str = getattr(c, "content", "") or ""
            chunk_index: int = int(getattr(c, "chunk_index", 0) or 0)
            start_index: int = int(getattr(c, "start_index", 0) or 0)
            end_index: int = int(getattr(c, "end_index", 0) or 0)

            metadata: Dict[str, Any] = getattr(c, "metadata", {}) or {}
            chunk_type: str = metadata.get("chunk_type") or "child"
            section_title: str = metadata.get("section_title") or ""
            section_path: str = metadata.get("section_path") or ""
            page_val = metadata.get("page")
            page: Optional[int] = int(page_val) if page_val is not None else None

            if content:
                vec_content = model.encode([content], task="retrieval")[0].tolist()
            else:
                vec_content = _zero_vector(dim)

            id_val = next_child_id
            next_child_id += 1

            record: Dict[str, Any] = {
                "id": id_val,
                "doc_id": doc_id,
                "kb_id": kb_id,
                "security_level": security_level,
                "chunk_index": chunk_index,
                "parent_chunk_id": file_id,
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
                "vector_abstract": vec_abstract,
                "vector_keywords": vec_keywords,
                "vector_summary": vec_summary,
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

    used_parent_files = next_parent_file_id - start_parent_file_id
    return records, chunk_models, used_parent_files


def process_one_json(
    json_path: Path,
    collection_name: str,
    model: SentenceTransformer,
    dim: int,
    doc_id: int,
    start_id: int = 1,
    start_parent_file_id: int = 1,
) -> tuple[int, int]:
    """处理单个 JSON 文件：切片 + 向量化 + 插入集合。"""
    data = _load_json(json_path)
    records, _, used_parent_files = _build_records_and_chunks(
        data=data,
        kb_id=1,
        doc_id=doc_id,
        model=model,
        dim=dim,
        start_child_id=start_id,
        start_parent_file_id=start_parent_file_id,
        tenant_id=1,
        security_level=1,
    )
    if not records:
        return 0, 0

    insert_req = InsertRequest(
        collection_name=collection_name,
        records=records,
    )
    ids = StorageService.insert(insert_req)
    inserted_count = len(ids)
    return inserted_count, used_parent_files


def index_json_file(
    file_path: str | Path,
    collection_name: Optional[str] = None,
    model_name: Optional[str] = None,
    dim: Optional[int] = None,
) -> int:
    """索引单个 JSON 文件，返回插入记录数。

    - collection_name 为空时使用环境变量 COLLECTION_NAME（默认为 `papers_chunks`）
    - model_name 为空时使用环境变量 EMBEDDING_MODEL
    - dim 为空时默认 768，可按模型维度覆盖
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {path}")

    collection = collection_name or _get_default_collection_name()
    vector_dim = dim or int(os.getenv("EMBEDDING_DIM", "768"))

    # 准备 Milvus 集合
    ensure_parent_child_collection(collection, vector_dim)

    # 加载/构建模型
    if model_name:
        os.environ["EMBEDDING_MODEL"] = model_name
        model = _get_embedding_model()
    else:
        model = _get_embedding_model()

    # 这里简单用 1 作为 doc_id 起始值，若有需要可将 doc_id / start_id 从外部传入
    inserted, _ = process_one_json(
        json_path=path,
        collection_name=collection,
        model=model,
        dim=vector_dim,
        doc_id=1,
        start_id=1,
        start_parent_file_id=1,
    )
    return inserted


def index_json_dir(
    input_dir: str | Path,
    collection_name: Optional[str] = None,
    model_name: Optional[str] = None,
    dim: Optional[int] = None,
) -> int:
    """索引目录下所有 `.json` 文件，返回总插入记录数。"""
    dir_path = Path(input_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {dir_path}")

    json_files = sorted(dir_path.glob("*.json"))
    if not json_files:
        return 0

    collection = collection_name or _get_default_collection_name()
    vector_dim = dim or int(os.getenv("EMBEDDING_DIM", "768"))

    ensure_parent_child_collection(collection, vector_dim)

    if model_name:
        os.environ["EMBEDDING_MODEL"] = model_name
        model = _get_embedding_model()
    else:
        model = _get_embedding_model()

    total = 0
    next_id = 1
    next_parent_file_id = 1
    for idx, json_path in enumerate(json_files, start=1):
        inserted, used_parent_files = process_one_json(
            json_path=json_path,
            collection_name=collection,
            model=model,
            dim=vector_dim,
            doc_id=idx,
            start_id=next_id,
            start_parent_file_id=next_parent_file_id,
        )
        total += inserted
        next_id += inserted
        next_parent_file_id += used_parent_files
    return total


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
    if not items:
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

    next_child_id = 1
    next_parent_file_id = 1
    next_doc_id = 1

    for filename, content in items:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            skipped_files.append(filename)
            continue

        meta = _extract_paper_metadata(data)
        original_text: str = meta["original_text"]
        if not original_text:
            skipped_files.append(filename)
            continue

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
            doc_id = next_doc_id
            next_doc_id += 1
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
            except Exception as exc:
                skipped_files.append(filename)
                continue

            if isinstance(created, dict):
                doc_id = created.get("data")["id"]
            else:
                doc_id = None

            if not isinstance(doc_id, int):
                skipped_files.append(filename)
                continue

        records, chunk_models, used_parent_files = _build_records_and_chunks(
            data=data,
            kb_id=kb_id,
            doc_id=doc_id,
            model=model,
            dim=vector_dim,
            start_child_id=next_child_id,
            start_parent_file_id=next_parent_file_id,
            tenant_id=tenant_id,
            security_level=security_level,
        )

        if not records:
            skipped_files.append(filename)
            continue

        insert_req = InsertRequest(
            collection_name=collection,
            records=records,
        )
        ids = StorageService.insert(insert_req)

        if not skip_base_db:
            await chunk_client.create_document_chunk_batch(chunk_models)

        inserted_count = len(ids)
        total_documents += 1
        total_chunks += len(chunk_models)
        milvus_records += inserted_count

        next_child_id += inserted_count
        next_parent_file_id += used_parent_files

    return {
        "kb_id": kb_id,
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        "milvus_records": milvus_records,
        "skipped_files": skipped_files,
    }

