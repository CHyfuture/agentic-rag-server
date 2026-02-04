"""检索 API 路由。"""

from fastapi import APIRouter, HTTPException

from app.api.schemas.retrieval import (
    SemanticSearchRequest,
    KeywordSearchRequest,
    HybridSearchRequest,
    FulltextSearchRequest,
    TextMatchSearchRequest,
    PhraseMatchSearchRequest,
)
from app.service import retrieval_service

router = APIRouter()


def _result_to_dict(r) -> dict:
    """将 RetrievalResultDTO 转为字典。"""
    return {
        "chunk_id": getattr(r, "chunk_id", None),
        "doc_id": getattr(r, "document_id", None),
        "content": getattr(r, "content", ""),
        "score": getattr(r, "score", 0.0),
        "metadata": getattr(r, "metadata", {}) or {},
        "parent_chunk": "",
        "original_text": "",
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
            milvus_expr=req.milvus_expr,
            keyword_text=req.keyword_text,
            author=req.author,
            paper_title=req.paper_title,
        )
        return {"results": [_result_to_dict(r) for r in results]}
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
        )
        return {"results": [_result_to_dict(r) for r in results]}
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
        )
        return {"results": [_result_to_dict(r) for r in results]}
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
        )
        return {"results": [_result_to_dict(r) for r in results]}
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
        )
        return {"results": [_result_to_dict(r) for r in results]}
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
        )
        return {"results": [_result_to_dict(r) for r in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
