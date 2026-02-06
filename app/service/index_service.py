"""索引服务：将 JSON 论文结构 + original_text 通过父子切片 + 多向量写入 Milvus 集合。

参考 `parent_child_index_from_json.py`，但封装为可在应用内调用的 Service。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _get_embedding_model() -> SentenceTransformer:
    """懒加载向量模型，复用与检索相同的 EMBEDDING_MODEL 配置。"""
    global _embedding_model
    if _embedding_model is None:
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
        _embedding_model = SentenceTransformer(model_name)
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
        # 集合已存在，仅尝试为多向量字段创建索引
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


def process_one_json(
    json_path: Path,
    collection_name: str,
    model: SentenceTransformer,
    dim: int,
    doc_id: int,
    start_id: int = 1,
    start_parent_file_id: int = 1,
) -> tuple[int, int]:
    """处理单个 JSON 文件：切片 + 向量化 + 插入集合。

    返回:
        inserted_count: 本文件插入的子块条数
        used_parent_files: 本文件新创建的父块文件数量
    """
    data = _load_json(json_path)

    title: str = data.get("title", "") or ""
    authors_raw: List[Dict[str, Any]] = data.get("authors", []) or []
    abstract: str = data.get("abstract", "") or ""
    keywords_list: List[str] = data.get("keywords", []) or []
    conclusion: str = data.get("conclusion", "") or ""
    original_text: str = data.get("original_text", "") or ""

    if not original_text:
        # 缺少正文直接跳过
        return 0

    # 将原始文本按 doc_id 写入 workspace/doc/{doc_id}.txt，便于后续关联
    project_root = Path(__file__).resolve().parents[2]
    doc_dir = project_root / "workspace" / "doc"
    doc_dir.mkdir(parents=True, exist_ok=True)
    doc_path = doc_dir / f"{doc_id}.txt"
    doc_path.write_text(original_text, encoding="utf-8")

    # authors / institutions 结构化后转为字符串存储（与 schema 中 VARCHAR 类型一致）
    authors_list = authors_raw
    institutions_list = list(
        {a.get("school") for a in authors_raw if a.get("school")}
    )
    authors_str = json.dumps(authors_list, ensure_ascii=False)
    institutions_str = json.dumps(institutions_list, ensure_ascii=False)

    # 文本字段
    abstract_text = abstract
    keywords_text = "；".join(keywords_list)
    summary_text = conclusion

    # 文档级多向量（一次编码，写入所有块）
    vec_abstract = (
        model.encode([abstract_text])[0].tolist()
        if abstract_text
        else _zero_vector(dim)
    )
    vec_keywords = (
        model.encode([keywords_text])[0].tolist()
        if keywords_text
        else _zero_vector(dim)
    )
    vec_summary = (
        model.encode([summary_text])[0].tolist()
        if summary_text
        else _zero_vector(dim)
    )

    # 父子切片
    chunk_req = ChunkRequest(
        text=original_text,
        strategy="parent_child",
    )
    chunks = ChunkerService.chunk(chunk_req)
    if not chunks:
        return 0

    # 先按父子关系整理结构：父块 + children 数组
    parents: Dict[int, Any] = {}
    children_by_parent_index: Dict[int, List[Any]] = {}
    for c in chunks:
        metadata: Dict[str, Any] = getattr(c, "metadata", {}) or {}
        chunk_type: str = metadata.get("chunk_type") or ""
        chunk_index: int = int(getattr(c, "chunk_index", 0) or 0)

        # 子块的父索引：优先 metadata.parent_chunk_index，其次顶层 parent_chunk_id
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

    # 父块内容输出目录：项目根目录下 workspace/parent
    parent_dir = project_root / "workspace" / "parent"
    parent_dir.mkdir(parents=True, exist_ok=True)

    # 文件 ID（从 start_parent_file_id 开始）以及子块主键 ID（可由外部通过 start_id 控制起点）
    next_parent_file_id = start_parent_file_id
    next_child_id = start_id

    records: List[Dict[str, Any]] = []

    # 按 chunk_index 顺序遍历父块，每个父块写入一个文件，文件 ID 作为其所有子块的 parent_chunk_id
    for parent_index in sorted(parents.keys()):
        parent_chunk = parents[parent_index]
        parent_meta: Dict[str, Any] = getattr(parent_chunk, "metadata", {}) or {}
        parent_content: str = getattr(parent_chunk, "content", "") or ""

        # 写入父块文件
        file_id = next_parent_file_id
        next_parent_file_id += 1
        if parent_content:
            file_path = parent_dir / f"{file_id}.txt"
            file_path.write_text(parent_content, encoding="utf-8")

        # 处理该父块下的所有子块
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

            # 对当前子块正文内容编码向量
            if content:
                vec_content = model.encode([content])[0].tolist()
            else:
                vec_content = _zero_vector(dim)

            id_val = next_child_id
            next_child_id += 1

            record: Dict[str, Any] = {
                "id": id_val,
                "doc_id": doc_id,
                "kb_id": 1,
                "security_level": 1,
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

    if not records:
        return 0, 0

    insert_req = InsertRequest(
        collection_name=collection_name,
        records=records,
    )
    ids = StorageService.insert(insert_req)
    inserted_count = len(ids)
    used_parent_files = next_parent_file_id - start_parent_file_id
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
    # 简单地使用从 1 开始递增的 doc_id；
    # 子块主键 ID（start_id）与父块文件 ID（start_parent_file_id）需要在所有文件之间全局递增，避免重复。
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

