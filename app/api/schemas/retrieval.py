"""检索 API 请求/响应模型。"""

from typing import Any

from pydantic import BaseModel, Field


class SemanticSearchRequest(BaseModel):
    """语义检索请求。"""

    query: str = Field(..., description="查询文本，服务端自动编码为向量")
    top_k: int | None = Field(None, ge=1, le=100, description="返回条数")
    rerank_enabled: bool | None = Field(None, description="是否启用重排序")
    similarity_threshold: float | None = Field(None, ge=0, le=1, description="相似度阈值")
    keyword_text: str | None = Field(None, description="论文关键词模糊匹配过滤条件")
    author: str | None = Field(None, description="论文作者模糊匹配过滤条件")
    paper_title: str | None = Field(None, description="论文标题模糊匹配过滤条件")
    return_original_text: bool | None = Field(None, description="是否返回全文")
    return_parent_chunk: bool | None = Field(None, description="是否返回父 chunk")


class KeywordSearchRequest(BaseModel):
    """关键词检索请求。"""

    query: str = Field(..., description="查询关键词")
    top_k: int | None = Field(None, ge=1, le=100, description="返回条数")
    keyword_text: str | None = Field(None, description="论文关键词模糊匹配过滤条件")
    author: str | None = Field(None, description="论文作者模糊匹配过滤条件")
    paper_title: str | None = Field(None, description="论文标题模糊匹配过滤条件")
    min_match_count: int | None = Field(None, ge=0, description="最少匹配关键词数量，少于此值返回分数0")
    return_original_text: bool | None = Field(None, description="是否返回全文")
    return_parent_chunk: bool | None = Field(None, description="是否返回父 chunk")


class HybridSearchRequest(BaseModel):
    """混合检索请求。"""

    query: str = Field(..., description="查询文本，服务端自动编码为向量")
    top_k: int | None = Field(None, ge=1, le=100, description="返回条数")
    rerank_enabled: bool | None = Field(None, description="是否启用重排序")
    similarity_threshold: float | None = Field(None, ge=0, le=1, description="相似度阈值")
    keyword_text: str | None = Field(None, description="论文关键词模糊匹配过滤条件")
    author: str | None = Field(None, description="论文作者模糊匹配过滤条件")
    paper_title: str | None = Field(None, description="论文标题模糊匹配过滤条件")
    semantic_weight: float | None = Field(None, ge=0, le=1, description="语义检索权重，建议与keyword_weight和为1.0")
    keyword_weight: float | None = Field(None, ge=0, le=1, description="关键词检索权重")
    return_original_text: bool | None = Field(None, description="是否返回全文")
    return_parent_chunk: bool | None = Field(None, description="是否返回父 chunk")


class FulltextSearchRequest(BaseModel):
    """全文检索请求。"""

    query: str = Field(..., description="全文查询文本")
    top_k: int | None = Field(None, ge=1, le=100, description="返回条数")
    keyword_text: str | None = Field(None, description="论文关键词模糊匹配过滤条件")
    author: str | None = Field(None, description="论文作者模糊匹配过滤条件")
    paper_title: str | None = Field(None, description="论文标题模糊匹配过滤条件")
    min_match_count: int | None = Field(None, ge=0, description="最少匹配关键词数量")
    match_mode: str | None = Field(None, description="匹配模式：or(任一匹配)/and(全部匹配)")
    return_original_text: bool | None = Field(None, description="是否返回全文")
    return_parent_chunk: bool | None = Field(None, description="是否返回父 chunk")


class TextMatchSearchRequest(BaseModel):
    """文本匹配检索请求。"""

    query: str = Field(..., description="匹配文本")
    top_k: int | None = Field(None, ge=1, le=100, description="返回条数")
    keyword_text: str | None = Field(None, description="论文关键词模糊匹配过滤条件")
    author: str | None = Field(None, description="论文作者模糊匹配过滤条件")
    paper_title: str | None = Field(None, description="论文标题模糊匹配过滤条件")
    match_type: str | None = Field(None, description="匹配类型：exact(精确)/fuzzy(模糊)")
    case_sensitive: bool | None = Field(None, description="是否区分大小写")
    return_original_text: bool | None = Field(None, description="是否返回全文")
    return_parent_chunk: bool | None = Field(None, description="是否返回父 chunk")


class PhraseMatchSearchRequest(BaseModel):
    """短语匹配检索请求。"""

    query: str = Field(..., description="短语匹配文本")
    top_k: int | None = Field(None, ge=1, le=100, description="返回条数")
    keyword_text: str | None = Field(None, description="论文关键词模糊匹配过滤条件")
    author: str | None = Field(None, description="论文作者模糊匹配过滤条件")
    paper_title: str | None = Field(None, description="论文标题模糊匹配过滤条件")
    case_sensitive: bool | None = Field(None, description="是否区分大小写")
    allow_partial: bool | None = Field(None, description="是否允许部分匹配")
    return_original_text: bool | None = Field(None, description="是否返回全文")
    return_parent_chunk: bool | None = Field(None, description="是否返回父 chunk")


class QaRequest(BaseModel):
    """知识问答。"""
    query: str = Field(..., description="查询文本")
    rag_type: int = Field(..., description="RAG实现方式，1旧版本，2新版本")
