"""算子 API 请求/响应模型。"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ========== Parser ==========


class ParseResponse(BaseModel):
    """文档解析响应。"""

    content: str = Field(..., description="解析后的文本内容（Markdown 格式）")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")
    structure: Dict[str, Any] = Field(default_factory=dict, description="文档结构信息")


# ========== Chunker ==========


class ChunkRequest(BaseModel):
    """文档切片请求。"""

    text: str = Field(..., description="待切片的文本内容")
    strategy: str | None = Field(
        None,
        description="切片策略：fixed / semantic / title / parent_child",
    )
    chunk_size: int | None = Field(None, ge=1, le=4096, description="块大小（字符数）")
    chunk_overlap: int | None = Field(None, ge=0, le=1024, description="块重叠大小")
    config: Dict[str, Any] | None = Field(None, description="切片器额外配置")


class ChunkItem(BaseModel):
    """单个文档块。"""

    chunk_index: int = Field(..., description="块索引")
    content: str = Field(..., description="块内容")
    start_index: int = Field(0, description="在原文中的起始位置")
    end_index: int = Field(0, description="在原文中的结束位置")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="块元数据")
    parent_chunk_id: int | None = Field(None, description="父块 ID（父子切片时）")


class ChunkResponse(BaseModel):
    """文档切片响应。"""

    chunks: List[ChunkItem] = Field(..., description="切片结果列表")


# ========== Storage - Collection ==========


class CreateCollectionRequest(BaseModel):
    """创建集合请求。"""

    collection_name: str = Field(..., description="集合名称")
    dimension: int = Field(..., ge=1, description="向量维度")
    description: str = Field("", description="集合描述")
    dense_vector_field: str = Field("vector_content", description="稠密向量字段名")
    auto_id: bool = Field(False, description="是否自动生成主键 ID")


class CollectionListResponse(BaseModel):
    """集合列表响应。"""

    collections: List[str] = Field(..., description="集合名称列表")


class CollectionExistsResponse(BaseModel):
    """集合存在检查响应。"""

    exists: bool = Field(..., description="集合是否存在")


class DeleteCollectionRequest(BaseModel):
    """删除集合请求。"""

    collection_name: str = Field(..., description="要删除的集合名称")


# ========== Storage - Insert / Update / Delete ==========


class InsertPaperRequest(BaseModel):
    """插入论文数据请求：仅保留 chunk 保存与处理逻辑，不保存 DocumentModel，chunk 相关参数均由请求传入。"""

    kb_id: int = Field(..., description="知识库 ID")
    doc_id: int = Field(..., description="文档 ID（由调用方提供，不创建文档）")
    data: Dict[str, Any] = Field(
        ...,
        description="论文数据，需包含 original_text，以及 title、authors、abstract、keywords、conclusion 等",
    )
    tenant_id: int = Field(1, description="租户 ID")
    security_level: int = Field(1, description="密级")
    owner_id: int = Field(1, description="所有者 ID")
    filename: str = Field("paper.json", description="源文件名，用于记录标识")
    skip_base_db: bool = Field(False, description="是否跳过 BaseDB 切片写入，True 时仅写入 Milvus")


class InsertPaperResponse(BaseModel):
    """插入论文响应。"""

    success: bool = Field(..., description="是否成功")
    doc_id: int | None = Field(None, description="文档 ID（BaseDB 或哈希生成）")
    chunk_count: int = Field(0, description="插入的 chunk 数量")
    milvus_ids: List[int] = Field(default_factory=list, description="Milvus 中的 ID 列表")
    error: str | None = Field(None, description="失败时的错误信息")


class UpdatePaperRequest(BaseModel):
    """更新论文数据请求：入参与 InsertPaperRequest 相同，先删除该 doc_id 的旧记录再重新插入。"""

    kb_id: int = Field(..., description="知识库 ID")
    doc_id: int = Field(..., description="文档 ID")
    data: Dict[str, Any] = Field(
        ...,
        description="论文数据，需包含 original_text，以及 title、authors、abstract、keywords、conclusion 等",
    )
    tenant_id: int = Field(1, description="租户 ID")
    security_level: int = Field(1, description="密级")
    owner_id: int = Field(1, description="所有者 ID")
    filename: str = Field("paper.json", description="源文件名")
    skip_base_db: bool = Field(False, description="是否跳过 BaseDB 切片写入")


class DeletePaperRequest(BaseModel):
    """按 doc_id 删除文档请求：同步删除 Milvus 向量与 BaseDB chunk。"""

    doc_id: int = Field(..., description="文档 ID")
    collection_name: str | None = Field(None, description="集合名称，默认从环境变量读取")


class DeletePaperResponse(BaseModel):
    """删除文档响应。"""

    success: bool = Field(..., description="是否成功")
    doc_id: int = Field(..., description="文档 ID")
    error: str | None = Field(None, description="失败时的错误信息")


class InsertRequest(BaseModel):
    """插入数据请求。"""

    collection_name: str = Field(..., description="集合名称")
    records: List[Dict[str, Any]] = Field(..., description="要插入的数据列表")


class InsertResponse(BaseModel):
    """插入响应。"""

    ids: List[int] = Field(..., description="插入记录的 ID 列表")
