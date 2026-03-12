"""索引构建 API 路由。"""

from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.service import index_service, operator_service

router = APIRouter()


@router.post("/index/build_json")
async def build_index(
    kb_id: int = Form(..., description="知识库 ID"),
    files: List[UploadFile] = File(..., description="包含论文 JSON 的文件列表"),
):
    """批量上传 JSON 文件并为指定知识库构建索引。"""
    if not files:
        raise HTTPException(status_code=400, detail="至少需要上传一个 JSON 文件")

    items: List[tuple[str, str]] = []
    for f in files:
        try:
            raw = await f.read()
            text = raw.decode("utf-8")
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"读取文件失败: {f.filename}: {exc}",
            ) from exc
        items.append((f.filename or "unknown.json", text))

    try:
        result = await index_service.build_index_from_json_contents(
            kb_id=kb_id,
            items=items,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


@router.post("/index/build_markdown")
async def build_markdown_index(
    kb_id: int = Form(..., description="知识库 ID"),
    files: List[UploadFile] = File(..., description="Markdown 文件列表"),
):
    """批量上传 Markdown 文件并为指定知识库构建索引。"""
    if not files:
        raise HTTPException(status_code=400, detail="至少需要上传一个 Markdown 文件")

    items: List[tuple[str, str]] = []
    skipped_files: List[str] = []
    for f in files:
        filename = f.filename or "unknown.md"
        lower_name = filename.lower()
        if not (lower_name.endswith(".md") or lower_name.endswith(".markdown")):
            skipped_files.append(filename)
            continue
        try:
            raw = await f.read()
            text = raw.decode("utf-8")
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"读取文件失败: {filename}: {exc}",
            ) from exc
        items.append((filename, text))

    if not items:
        return {
            "kb_id": kb_id,
            "total_documents": 0,
            "total_chunks": 0,
            "milvus_records": 0,
            "skipped_files": skipped_files,
        }

    try:
        result = await index_service.build_index_from_markdown_contents(
            kb_id=kb_id,
            items=items,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if skipped_files:
        result["skipped_files"] = list(result.get("skipped_files", [])) + skipped_files
    return result


@router.post("/index/build_documents")
async def build_documents_index(
    kb_id: int = Form(..., description="知识库 ID"),
    files: List[UploadFile] = File(..., description="文档文件列表（PDF/Word/PPT/HTML/TXT 等）"),
):
    """批量上传文档文件，先解析内容，再为指定知识库构建索引。"""
    if not files:
        raise HTTPException(status_code=400, detail="至少需要上传一个文档文件")

    allowed_suffixes = {
        ".pdf",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".txt",
        ".html",
        ".htm",
        ".md",
        ".markdown",
    }

    parsed_items: List[Dict[str, Any]] = []
    skipped_files: List[str] = []

    for f in files:
        filename = f.filename or "unknown.bin"
        suffix = Path(filename).suffix.lower()
        if suffix not in allowed_suffixes:
            skipped_files.append(filename)
            continue

        try:
            raw = await f.read()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"读取文件失败: {filename}: {exc}",
            ) from exc

        try:
            parsed = operator_service.parse_uploaded_file(raw, filename)
        except Exception as exc:
            skipped_files.append(filename)
            continue

        metadata = parsed.get("metadata", {}) if isinstance(parsed, dict) else {}
        title = metadata.get("title", "") if isinstance(metadata, dict) else ""
        parsed_items.append(
            {
                "filename": filename,
                "file_type": suffix.lstrip(".") or "bin",
                "title": title,
                "content": parsed.get("content", "") if isinstance(parsed, dict) else "",
            }
        )

    if not parsed_items:
        return {
            "kb_id": kb_id,
            "total_documents": 0,
            "total_chunks": 0,
            "milvus_records": 0,
            "skipped_files": skipped_files,
        }

    try:
        result = await index_service.build_index_from_parsed_document_contents(
            kb_id=kb_id,
            items=parsed_items,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if skipped_files:
        result["skipped_files"] = list(result.get("skipped_files", [])) + skipped_files
    return result

