"""算子 API 路由：Parser、Chunker、Storage。"""

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.api.schemas.operator import (
    ChunkRequest,
    ChunkResponse,
    ChunkItem,
    CollectionExistsResponse,
    CollectionListResponse,
    CreateCollectionRequest,
    DeleteCollectionRequest,
    DeletePaperRequest,
    DeletePaperResponse,
    InsertPaperRequest,
    InsertPaperResponse,
    InsertRequest,
    InsertResponse,
    ParseResponse,
    UpdatePaperRequest,
)
from app.service import index_service, operator_service

router = APIRouter()


# ========== Parser ==========


@router.post("/parser/parse", response_model=ParseResponse)
async def parse_document(
    file: UploadFile = File(..., description="待解析的文档文件（PDF/Word/PPT/HTML/MD/TXT）"),
):
    """文档解析：上传文件并解析为 Markdown 文本及元数据。"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件名")
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"读取文件失败: {e}") from e
    try:
        result = operator_service.parse_uploaded_file(content, file.filename)
        return ParseResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ========== Chunker ==========


@router.post("/chunker/chunk", response_model=ChunkResponse)
async def chunk_text(req: ChunkRequest):
    """文档切片：对文本按指定策略切分为 chunks。"""
    try:
        chunks = operator_service.chunk_text(
            text=req.text,
            strategy=req.strategy,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            config=req.config,
        )
        items = [ChunkItem(**c) for c in chunks]
        return ChunkResponse(chunks=items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ========== Storage - Collection ==========


@router.post("/storage/collection/create")
async def create_collection(req: CreateCollectionRequest):
    """创建 Milvus 集合。"""
    try:
        operator_service.create_collection(
            collection_name=req.collection_name,
            dimension=req.dimension,
            description=req.description,
            dense_vector_field=req.dense_vector_field,
            auto_id=req.auto_id,
        )
        return {"message": "集合创建成功", "collection_name": req.collection_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/storage/collection/list", response_model=CollectionListResponse)
async def list_collections():
    """列出所有 Milvus 集合。"""
    try:
        collections = operator_service.list_collections()
        return CollectionListResponse(collections=collections)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/storage/collection/exists", response_model=CollectionExistsResponse)
async def collection_exists(collection_name: str = Query(..., description="集合名称")):
    """检查集合是否存在。"""
    try:
        exists = operator_service.collection_exists(collection_name)
        return CollectionExistsResponse(exists=exists)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/storage/collection/delete")
async def delete_collection(req: DeleteCollectionRequest):
    """删除集合。"""
    try:
        operator_service.delete_collection(req.collection_name)
        return {"message": "集合已删除", "collection_name": req.collection_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ========== Storage - Insert / Update / Delete ==========


@router.post("/storage/insert_paper", response_model=InsertPaperResponse)
async def insert_paper_data(req: InsertPaperRequest):
    """插入论文数据：仅提供论文 JSON，服务端完成切片、向量化、BaseDB 写入（可选）、Milvus 插入。"""
    try:
        result = await index_service.insert_single_paper_data(
            kb_id=req.kb_id,
            doc_id=req.doc_id,
            data=req.data,
            tenant_id=req.tenant_id,
            security_level=req.security_level,
            owner_id=req.owner_id,
            filename=req.filename,
            skip_base_db=req.skip_base_db,
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "插入失败"))
        return InsertPaperResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/storage/update_paper", response_model=InsertPaperResponse)
async def update_paper_data(req: UpdatePaperRequest):
    """更新论文数据：先删除该 doc_id 的 Milvus/BaseDB 旧记录，再按 insert 逻辑重新切片、向量化、入库。"""
    try:
        result = await index_service.update_single_paper_data(
            kb_id=req.kb_id,
            doc_id=req.doc_id,
            data=req.data,
            tenant_id=req.tenant_id,
            security_level=req.security_level,
            owner_id=req.owner_id,
            filename=req.filename,
            skip_base_db=req.skip_base_db,
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "更新失败"))
        return InsertPaperResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/storage/delete_paper", response_model=DeletePaperResponse)
async def delete_paper_data(req: DeletePaperRequest):
    """按 doc_id 删除文档：同步删除 Milvus 向量与 BaseDB chunk 数据。"""
    try:
        result = await index_service.delete_document_by_doc_id(
            doc_id=req.doc_id,
            collection_name=req.collection_name,
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "删除失败"))
        return DeletePaperResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/storage/insert", response_model=InsertResponse)
async def insert_data(req: InsertRequest):
    """向集合插入数据。"""
    try:
        ids = operator_service.insert_records(
            collection_name=req.collection_name,
            records=req.records,
        )
        return InsertResponse(ids=ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


