"""索引构建 API 路由。"""

from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.service import index_service

router = APIRouter()


@router.post("/index/build")
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

