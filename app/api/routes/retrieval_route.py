"""检索 API 路由。"""

from pathlib import Path

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

# 项目根目录下的 workspace（retrieval_route 在 app/api/routes/，需 parents[3] 到项目根）
_WORKSPACE_ROOT = Path(__file__).resolve().parents[3] / "workspace"


def _read_txt_file(dir_name: str, file_id: int | None) -> str:
    """从 workspace 下指定目录读取 {file_id}.txt 内容，不存在或异常时返回空字符串。"""
    if file_id is None:
        return ""
    path = _WORKSPACE_ROOT / dir_name / f"{file_id}.txt"
    try:
        print(f"文件地址：{path}")
        if path.is_file():
            return path.read_text(encoding="utf-8")
    except OSError:
        pass
    return ""


def _result_to_dict(
    r,
    *,
    return_original_text: bool = False,
    return_parent_chunk: bool = False,
) -> dict:
    """将 RetrievalResultDTO 转为字典。"""
    metadata = getattr(r, "metadata", {}) or {}
    doc_id = getattr(r, "document_id", None)
    parent_chunk_id = metadata.get("parent_chunk_id") if isinstance(metadata, dict) else None

    original_text = ""
    if return_original_text and doc_id is not None:
        original_text = _read_txt_file("doc", doc_id)

    parent_chunk = ""
    if return_parent_chunk and parent_chunk_id is not None:
        parent_chunk = _read_txt_file("parent", parent_chunk_id)

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
        )
        return {
            "results": [
                _result_to_dict(
                    r,
                    return_original_text=req.return_original_text is True,
                    return_parent_chunk=req.return_parent_chunk is True,
                )
                for r in results
            ]
        }
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
        return {
            "results": [
                _result_to_dict(
                    r,
                    return_original_text=req.return_original_text is True,
                    return_parent_chunk=req.return_parent_chunk is True,
                )
                for r in results
            ]
        }
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
        return {
            "results": [
                _result_to_dict(
                    r,
                    return_original_text=req.return_original_text is True,
                    return_parent_chunk=req.return_parent_chunk is True,
                )
                for r in results
            ]
        }
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
        return {
            "results": [
                _result_to_dict(
                    r,
                    return_original_text=req.return_original_text is True,
                    return_parent_chunk=req.return_parent_chunk is True,
                )
                for r in results
            ]
        }
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
        return {
            "results": [
                _result_to_dict(
                    r,
                    return_original_text=req.return_original_text is True,
                    return_parent_chunk=req.return_parent_chunk is True,
                )
                for r in results
            ]
        }
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
        return {
            "results": [
                _result_to_dict(
                    r,
                    return_original_text=req.return_original_text is True,
                    return_parent_chunk=req.return_parent_chunk is True,
                )
                for r in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
