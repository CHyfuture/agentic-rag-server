"""检索 API 路由。"""

from fastapi import APIRouter, HTTPException

from app.api.schemas.retrieval import (
    FulltextSearchRequest,
    HybridSearchRequest,
    KeywordSearchRequest,
    PhraseMatchSearchRequest,
    SemanticSearchRequest,
    TextMatchSearchRequest,
)
from app.service import retrieval_service

router = APIRouter()


async def _result_to_dict(
    r,
    *,
    return_original_text: bool = False,
    return_parent_chunk: bool = False,
) -> dict:
    """将 RetrievalResultDTO 转为字典，并按需从 DB 获取原文与父块内容。"""
    metadata = getattr(r, "metadata", {}) or {}
    doc_id = getattr(r, "document_id", None)
    chunk_id = getattr(r, "chunk_id", None)

    original_text = ""
    if return_original_text and isinstance(doc_id, int):
        original_text = await retrieval_service.get_original_text_by_doc_id(doc_id)

    parent_chunk = ""
    if return_parent_chunk and isinstance(chunk_id, int):
        parent_chunk = await retrieval_service.get_parent_content_by_chunk_id(
            chunk_id=chunk_id
        )

    return {
        "chunk_id": getattr(r, "chunk_id", None),
        "doc_id": doc_id,
        "content": getattr(r, "content", ""),
        "score": getattr(r, "score", 0.0),
        "metadata": metadata,
        "parent_chunk": parent_chunk,
        "original_text": original_text,
    }


@router.post("/semantic")
async def semantic_search(req: SemanticSearchRequest):
    """语义检索。"""
    try:
        results = retrieval_service.semantic_search(
            query=req.query,
            top_k=req.top_k,
            rerank_enabled=req.rerank_enabled,
            similarity_threshold=req.similarity_threshold,
            keyword_text=req.keyword_text,
            author=req.author,
            paper_title=req.paper_title,
            doc_id=req.doc_id,
            kb_id=req.kb_id,
            security_level=req.security_level,
        )
        result_dicts = []
        for r in results:
            d = await _result_to_dict(
                r,
                return_original_text=req.return_original_text is True,
                return_parent_chunk=req.return_parent_chunk is True,
            )
            result_dicts.append(d)
        return {"results": result_dicts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/keyword")
async def keyword_search(req: KeywordSearchRequest):
    """关键词检索。"""
    try:
        results = retrieval_service.keyword_search(
            query=req.query,
            top_k=req.top_k,
            min_match_count=req.min_match_count,
            keyword_text=req.keyword_text,
            author=req.author,
            paper_title=req.paper_title,
            doc_id=req.doc_id,
            kb_id=req.kb_id,
            security_level=req.security_level,
        )
        result_dicts = []
        for r in results:
            d = await _result_to_dict(
                r,
                return_original_text=req.return_original_text is True,
                return_parent_chunk=req.return_parent_chunk is True,
            )
            result_dicts.append(d)
        return {"results": result_dicts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hybrid")
async def hybrid_search(req: HybridSearchRequest):
    """混合检索（语义 + 关键词）。"""
    try:
        results = retrieval_service.hybrid_search(
            query=req.query,
            top_k=req.top_k,
            rerank_enabled=req.rerank_enabled,
            similarity_threshold=req.similarity_threshold,
            semantic_weight=req.semantic_weight,
            keyword_weight=req.keyword_weight,
            keyword_text=req.keyword_text,
            author=req.author,
            paper_title=req.paper_title,
            doc_id=req.doc_id,
            kb_id=req.kb_id,
            security_level=req.security_level,
        )
        result_dicts = []
        for r in results:
            d = await _result_to_dict(
                r,
                return_original_text=req.return_original_text is True,
                return_parent_chunk=req.return_parent_chunk is True,
            )
            result_dicts.append(d)
        return {"results": result_dicts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fulltext")
async def fulltext_search(req: FulltextSearchRequest):
    """全文检索。"""
    try:
        results = retrieval_service.fulltext_search(
            query=req.query,
            top_k=req.top_k,
            min_match_count=req.min_match_count,
            match_mode=req.match_mode,
            keyword_text=req.keyword_text,
            author=req.author,
            paper_title=req.paper_title,
            doc_id=req.doc_id,
            kb_id=req.kb_id,
            security_level=req.security_level,
        )
        result_dicts = []
        for r in results:
            d = await _result_to_dict(
                r,
                return_original_text=req.return_original_text is True,
                return_parent_chunk=req.return_parent_chunk is True,
            )
            result_dicts.append(d)
        return {"results": result_dicts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text_match")
async def text_match_search(req: TextMatchSearchRequest):
    """文本匹配检索。"""
    try:
        results = retrieval_service.text_match_search(
            query=req.query,
            top_k=req.top_k,
            match_type=req.match_type,
            case_sensitive=req.case_sensitive,
            keyword_text=req.keyword_text,
            author=req.author,
            paper_title=req.paper_title,
            doc_id=req.doc_id,
            kb_id=req.kb_id,
            security_level=req.security_level,
        )
        result_dicts = []
        for r in results:
            d = await _result_to_dict(
                r,
                return_original_text=req.return_original_text is True,
                return_parent_chunk=req.return_parent_chunk is True,
            )
            result_dicts.append(d)
        return {"results": result_dicts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/phrase_match")
async def phrase_match_search(req: PhraseMatchSearchRequest):
    """短语匹配检索。"""
    try:
        results = retrieval_service.phrase_match_search(
            query=req.query,
            top_k=req.top_k,
            case_sensitive=req.case_sensitive,
            allow_partial=req.allow_partial,
            keyword_text=req.keyword_text,
            author=req.author,
            paper_title=req.paper_title,
            doc_id=req.doc_id,
            kb_id=req.kb_id,
            security_level=req.security_level,
        )
        result_dicts = []
        for r in results:
            d = await _result_to_dict(
                r,
                return_original_text=req.return_original_text is True,
                return_parent_chunk=req.return_parent_chunk is True,
            )
            result_dicts.append(d)
        return {"results": result_dicts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
