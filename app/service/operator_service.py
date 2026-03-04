"""算子服务：封装 ParserService、ChunkerService、StorageService。"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List

from milvus_service import (
    ChunkRequest as MilvusChunkRequest,
    ChunkerService,
    CreateCollectionRequest,
    InsertRequest,
    ParseRequest,
    ParserService,
    StorageService,
)
from pymilvus import Collection, DataType, FieldSchema, connections, utility


def _load_milvus_env() -> None:
    """加载 Milvus 环境变量。"""
    from dotenv import load_dotenv

    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)


def _connect_milvus() -> None:
    """连接 Milvus。"""
    import os

    _load_milvus_env()
    host = os.getenv("MILVUS_HOST", "192.168.31.51")
    port = os.getenv("MILVUS_PORT", "19530")
    user = os.getenv("MILVUS_USER", "")
    password = os.getenv("MILVUS_PASSWORD", "")
    db_name = os.getenv("MILVUS_DB_NAME", "default")
    uri = f"http://{host}:{port}"
    connections.connect(
        alias="default",
        uri=uri,
        user=user or None,
        password=password or None,
        db_name=db_name,
    )


# ========== Parser ==========


def parse_file(file_path: str | Path, parser_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """解析文档文件，返回 content、metadata、structure。"""
    req = ParseRequest(file_path=str(file_path), parser_config=parser_config or {})
    result = ParserService.parse(req)
    return {
        "content": getattr(result, "content", "") or "",
        "metadata": getattr(result, "metadata", None) or {},
        "structure": getattr(result, "structure", None) or {},
    }


def parse_uploaded_file(content: bytes, filename: str, parser_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """解析上传的文件内容（先写入临时文件再解析）。"""
    suffix = Path(filename).suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(content)
        tmp_path = f.name
    try:
        return parse_file(tmp_path, parser_config)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ========== Chunker ==========


def chunk_text(
    text: str,
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    config: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """对文本进行切片，返回 chunks 列表。"""
    req_kwargs: Dict[str, Any] = {"text": text}
    if strategy is not None:
        req_kwargs["strategy"] = strategy
    if chunk_size is not None:
        req_kwargs["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        req_kwargs["chunk_overlap"] = chunk_overlap

    cfg = config or {}
    if chunk_size is not None and "chunk_size" not in cfg:
        cfg["chunk_size"] = chunk_size
    if chunk_overlap is not None and "chunk_overlap" not in cfg:
        cfg["chunk_overlap"] = chunk_overlap
    if cfg:
        req_kwargs["config"] = cfg

    req = MilvusChunkRequest(**req_kwargs)
    chunks = ChunkerService.chunk(req)

    result: List[Dict[str, Any]] = []
    for c in chunks:
        result.append({
            "chunk_index": getattr(c, "chunk_index", 0),
            "content": getattr(c, "content", "") or "",
            "start_index": getattr(c, "start_index", 0),
            "end_index": getattr(c, "end_index", 0),
            "metadata": getattr(c, "metadata", None) or {},
            "parent_chunk_id": getattr(c, "parent_chunk_id", None),
        })
    return result


# ========== Storage ==========


def create_collection(
    collection_name: str,
    dimension: int,
    description: str = "",
    dense_vector_field: str = "vector_content",
    auto_id: bool = False,
) -> None:
    """创建 Milvus 集合（使用简化 schema：id, vector, content）。"""
    metadata_fields: List[FieldSchema] = [
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    ]
    req = CreateCollectionRequest(
        collection_name=collection_name,
        dimension=dimension,
        description=description or "",
        auto_id=auto_id,
        primary_field="id",
        dense_vector_field=dense_vector_field,
        metadata_fields=metadata_fields,
    )
    StorageService.create_collection(req)


def list_collections() -> List[str]:
    """列出所有集合。"""
    _connect_milvus()
    return utility.list_collections(using="default")


def collection_exists(collection_name: str) -> bool:
    """检查集合是否存在。"""
    _connect_milvus()
    return utility.has_collection(collection_name, using="default")


def delete_collection(collection_name: str) -> None:
    """删除集合。"""
    _connect_milvus()
    utility.drop_collection(collection_name, using="default")


def insert_records(collection_name: str, records: List[Dict[str, Any]]) -> List[int]:
    """插入数据到集合。"""
    req = InsertRequest(collection_name=collection_name, records=records)
    result = StorageService.insert(req)
    return [int(x) for x in result] if result else []


def update_records(
    collection_name: str,
    expr: str,
    new_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """更新数据（delete + insert）。"""
    _connect_milvus()
    coll = Collection(collection_name, using="default")
    coll.delete(expr)
    coll.flush()
    ids = insert_records(collection_name, new_data)
    return {
        "deleted_count": len(new_data),  # 估算
        "inserted_count": len(ids),
        "inserted_ids": ids,
    }


def delete_records(
    collection_name: str,
    expr: str | None = None,
    ids: List[int] | None = None,
) -> Dict[str, Any]:
    """删除数据。"""
    if ids is not None:
        expr = f"id in {ids}"
    if expr is None:
        raise ValueError("必须提供 expr 或 ids")
    _connect_milvus()
    coll = Collection(collection_name, using="default")
    coll.delete(expr)
    coll.flush()
    return {"deleted_count": len(ids) if ids else -1, "expr": expr}
