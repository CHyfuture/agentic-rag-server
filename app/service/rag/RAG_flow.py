#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG流程实现
根据提供的流程图，实现完整的检索增强生成流程
"""

import logging
import json
import os
import time
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import concurrent.futures
import requests

# 导入项目的retrieval_service模块
from app.service import retrieval_service

# 终端颜色支持（重要信息高亮）
try:
    from colorama import Fore, Style, init as colorama_init

    colorama_init(autoreset=True)
    _COLOR_ENABLED = True
except Exception:
    # 如果环境中没有 colorama，则退化为普通输出
    _COLOR_ENABLED = False


def _print_info(text: str) -> None:
    """普通信息输出（默认白色）。"""
    if _COLOR_ENABLED:
        print(Fore.WHITE + str(text) + Style.RESET_ALL)
    else:
        print(str(text))


def _print_warning(text: str) -> None:
    """重要/告警信息输出（红色高亮）。"""
    if _COLOR_ENABLED:
        print(Fore.RED + str(text) + Style.RESET_ALL)
    else:
        print(str(text))


# 日志文件路径（统一放在当前 test 目录）
_CURRENT_DIR = Path(__file__).resolve().parent
_CURRENT_DIR.mkdir(parents=True, exist_ok=True)
RAG_FLOW_LOG_PATH = _CURRENT_DIR / "rag_flow.log"

# chunk summary 最大并发数，降低 API 限流/失败，缓解多跳多论文指标下降
_SUMMARY_MAX_WORKERS = 4
# 回退为原文时进入融合的最大字符数，避免长原文主导融合导致 Judge 误判不足
_FALLBACK_CONTENT_MAX_CHARS = 800

# 设置日志（仅写入 test 目录下的 rag_flow.log）
logger = logging.getLogger("RAGFlow")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(RAG_FLOW_LOG_PATH, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - RAGFlow - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def _append_trace_log(trace: Dict[str, Any]) -> None:
    """
    将本次 RAG 流程的完整 trace 以结构化形式写入统一的 rag_flow.log 中。
    使用一条 TRACE 级别的JSON记录，便于后续程序和人工同时分析。
    """
    try:
        logger.info("TRACE %s", json.dumps(trace, ensure_ascii=False))
    except Exception as e:
        # 不影响主流程
        logger.error(f"写入RAG流程trace日志失败: {str(e)}")


def _truncate_for_fusion(text: str, max_chars: int = _FALLBACK_CONTENT_MAX_CHARS) -> str:
    """回退为原文时截断后再送入融合，避免超长原文主导上下文。"""
    s = str(text).strip() if text is not None else ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars]


# DeepSeek API配置
API_KEY = "sk-9e728482e15f4cecba9ead88ff7e9cc8"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

class DeepSeekClient:
    """DeepSeek API客户端"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def chat_completion(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        调用DeepSeek API进行对话完成
        """
        # 通过环境变量提供可控采样参数，默认尽量确定性以提升复现性
        def _get_env_float(name: str, default: float) -> float:
            try:
                v = os.getenv(name, "").strip()
                return float(v) if v != "" else float(default)
            except Exception:
                return float(default)

        def _get_env_int(name: str) -> Optional[int]:
            try:
                v = os.getenv(name, "").strip()
                if v == "":
                    return None
                return int(v)
            except Exception:
                return None

        temperature = _get_env_float("DEEPSEEK_TEMPERATURE", 0.0)
        top_p = _get_env_float("DEEPSEEK_TOP_P", 1.0)
        seed = _get_env_int("DEEPSEEK_SEED")
        timeout_s = _get_env_float("DEEPSEEK_TIMEOUT_SECONDS", 30.0)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        # seed 兼容性不保证：仅在显式配置时发送
        if seed is not None:
            payload["seed"] = seed

        try:
            logger.info("调用DeepSeek API进行对话完成")
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout_s
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"DeepSeek API调用失败: {str(e)}")
            print(f"DeepSeek API调用失败: {str(e)}")
            return None


class RAGFlow:
    """RAG流程实现类"""

    def __init__(self):
        # 初始化DeepSeek客户端
        self.deepseek = DeepSeekClient(API_KEY, BASE_URL, MODEL)

        # 最大迭代次数（完整RAG轮数）
        self.max_iterations = 3
        # 最多允许的 query 改写轮数
        self.max_rewrite_rounds = 2

        # 记录上一轮 evidence 抽取结果，供 Judge 规则与 trace 使用
        self._last_evidence_items: List[Dict[str, Any]] = []

    def _safe_json_loads(self, text: str) -> Optional[Dict[str, Any]]:
        """尽力解析 LLM 输出的 JSON。"""
        if not text:
            return None
        s = str(text).strip()
        if not (s.startswith("{") and s.endswith("}")):
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                s = s[start : end + 1]
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _extract_anchors(self, query: str) -> List[str]:
        """
        从 query 中提取“锚点”用于防漂移：方法名/缩写/符号等（不依赖题型标签）。
        仅提取拉丁字母开头的 token（如 ACAttack、SAM2.1、L_st）。
        """
        if not query:
            return []
        anchors = re.findall(r"[A-Za-z][A-Za-z0-9_.\-]{1,}", query)
        seen = set()
        out: List[str] = []
        for a in anchors:
            if a not in seen:
                seen.add(a)
                out.append(a)
        return out[:12]

    def _enforce_anchors(self, rewritten_query: str, anchors: List[str], original_query: str) -> str:
        """若改写丢失关键锚点，则回退或补齐，避免扩题导致检索漂移。"""
        rq = (rewritten_query or "").strip()
        if not rq:
            return original_query
        if not anchors:
            return rq

        missing = [a for a in anchors if a not in rq]
        if not missing:
            return rq

        if len(rq) > max(80, len(original_query) * 2):
            return original_query
        return (rq + " " + " ".join(missing)).strip()

    def _estimate_rewrite_difference(self, original_query: str, rewritten_query: str) -> Dict[str, float]:
        """
        粗略估计改写前后差异度：
        - overlap_ratio: 字符级 Jaccard 重叠度
        - length_ratio: 长度比（改写长度 / 原始长度）
        """
        try:
            orig_chars = set(original_query)
            rew_chars = set(rewritten_query)
            if not orig_chars or not rew_chars:
                overlap = 0.0
            else:
                overlap = len(orig_chars & rew_chars) / max(1, len(orig_chars | rew_chars))
            length_ratio = (len(rewritten_query) / max(1, len(original_query)))
            return {"overlap_ratio": overlap, "length_ratio": length_ratio}
        except Exception as e:
            logger.warning(f"估计查询改写差异失败: {str(e)}")
            return {"overlap_ratio": 0.0, "length_ratio": 1.0}

    def understand_and_rewrite_query(self, original_query: str, context: Optional[str] = None) -> str:
        """
        理解并改写用户查询
        Args:
            original_query: 用户原始查询
            context: 上下文信息（用于多轮）
        Returns:
            改写后的查询
        """
        logger.info(f"开始理解并改写查询，原始查询: {original_query}")
        _print_info("\n=== Query理解和改写 ===")
        _print_info(f"原始查询: {original_query}")

        messages = [
            {
                "role": "system",
                "content": "你是一个专业的学术查询理解和改写助手，专注于提高学术论文检索的效果。请严格遵循以下要求：\n1. 深入分析用户的学术查询意图，识别核心实体、数值和关键术语\n2. 如果有上下文，结合上下文理解用户的真实研究需求\n3. 在严格保留原始问题语义和关键实体不变的前提下，对查询进行结构化改写，使其更适合检索\n4. 仅做“检索友好化”的改写：不要引入新的子问题/新任务（禁止扩题），不要要求额外解释（如“具体设计如何”）\n5. 必须保留原问题中的关键锚点（方法名/缩写/符号），尤其是英文缩写与专有名词\n6. 改写输出尽量短，偏关键词/短语，避免长句冗余\n7. 仅输出改写后的查询文本，不要添加任何其他说明或解释"
            },
            {
                "role": "user",
                "content": (
                    f"原始查询: {original_query}\n上下文: {context}\n\n改写后的查询:"
                    if context
                    else f"原始查询: {original_query}\n\n改写后的查询:"
                )
            }
        ]

        rewritten_query = self.deepseek.chat_completion(messages)

        if not rewritten_query:
            logger.warning("查询改写失败，使用原始查询")
            _print_warning("查询改写失败，使用原始查询")
            return original_query

        rewritten_query = rewritten_query.strip()
        anchors = self._extract_anchors(original_query)
        rewritten_query = self._enforce_anchors(rewritten_query, anchors, original_query)
        diff_stats = self._estimate_rewrite_difference(original_query, rewritten_query)
        overlap = diff_stats.get("overlap_ratio", 0.0)
        length_ratio = diff_stats.get("length_ratio", 1.0)

        logger.info(
            f"查询改写成功，改写后的查询: {rewritten_query} "
            f"(overlap={overlap:.3f}, length_ratio={length_ratio:.3f})"
        )
        _print_info(f"改写后的查询: {rewritten_query}")

        # 如果改写几乎完全脱离原查询，则认为改写过激，回退到原始查询
        if overlap < 0.1 and (length_ratio > 2.0 or length_ratio < 0.5):
            logger.warning(
                f"查询改写幅度过大，可能偏离原始语义，回退到原始查询。overlap={overlap:.3f}, length_ratio={length_ratio:.3f}"
            )
            _print_warning("查询改写幅度过大，回退到原始查询")
            return original_query

        return rewritten_query

    def _extract_doc_info_from_result(self, result: Any) -> Dict[str, Optional[str]]:
        """
        从检索结果对象中尽可能提取文献信息（doc_id、doc_title）。
        兼容多种字段命名方式。
        """
        doc_id = getattr(result, "doc_id", None)
        doc_title = getattr(result, "doc_title", None) or getattr(result, "title", None)

        metadata = getattr(result, "metadata", None) or getattr(result, "meta", None)
        if isinstance(metadata, dict):
            doc_id = doc_id or metadata.get("doc_id") or metadata.get("paper_id")
            doc_title = doc_title or metadata.get("doc_title") or metadata.get("title")

        return {"doc_id": doc_id, "doc_title": doc_title}

    def _analyze_chunk_structure(self, content: str) -> Dict[str, float]:
        """
        对 chunk 文本做简单结构分析，识别表格 / 引用 / 公式等特征。
        返回 0~1 之间的启发式得分，用于轻微调整排序与证据权重。
        """
        text = (content or "").strip()
        if not text:
            return {"table_score": 0.0, "citation_score": 0.0, "formula_score": 0.0}

        # 表格特征：多行、包含 | 或 , 或制表符，且数字密集
        lines = text.splitlines()
        num_lines = len(lines)
        table_like_lines = 0
        digit_lines = 0
        for line in lines:
            line_strip = line.strip()
            if not line_strip:
                continue
            if "|" in line_strip or "\t" in line_strip or ("," in line_strip and any(ch.isdigit() for ch in line_strip)):
                table_like_lines += 1
            if len(re.findall(r"\d", line_strip)) >= 3:
                digit_lines += 1

        table_score = 0.0
        if num_lines >= 2:
            ratio_table = table_like_lines / num_lines
            ratio_digit = digit_lines / num_lines
            table_score = min(1.0, 0.6 * ratio_table + 0.4 * ratio_digit)

        # 引用特征：出现 [1]、[12]、[3,4] 等模式
        citation_matches = re.findall(r"\[[0-9,\s\-]{1,6}\]", text)
        citation_score = 0.0
        if citation_matches:
            citation_score = min(1.0, len(citation_matches) / 4.0)

        # 公式特征：等号、^、_、LaTeX 命令等
        formula_score = 0.0
        if any(sym in text for sym in ["=", "+", "-", "*", "/", "^", "_"]):
            formula_score += 0.3
        if any(kw in text for kw in ["\\frac", "\\sum", "\\int", "\\log", "\\exp"]):
            formula_score += 0.4
        if re.search(r"\b\d+(\.\d+)?\s*(×|x)\s*\d+(\.\d+)?\b", text):
            formula_score += 0.3
        formula_score = min(1.0, formula_score)

        return {
            "table_score": round(table_score, 3),
            "citation_score": round(citation_score, 3),
            "formula_score": round(formula_score, 3),
        }

    def retrieve_documents(self, query: str, collect_trace: bool = False, doc_id: Union[int, List[int], None] = None,
        kb_id: Union[int, List[int], None] = None,
        security_level: Union[int, List[int], None] = None,) -> Any:
        """
        执行多路召回检索
        Args:
            query: 查询文本
            collect_trace:
            doc_id: 文档ID
            kb_id: 知识库分类
            security_level: 访问级别
        Returns:
            检索结果列表
        """
        logger.info(f"开始执行多路召回检索，查询: {query}")
        _print_info("\n=== 执行检索（多路召回） ===")
        _print_info(f"查询文本: {query}")

        try:
            # 尝试多种检索方式，确保至少有一种能返回结果
            all_results = []
            seen_chunk_ids = set()
            retrieval_record: Dict[str, Any] = {
                "query": query,
                "hybrid": {"count": 0, "chunks": []},
                "semantic": {"count": 0, "chunks": []},
                "keyword": {"count": 0, "chunks": []},
                "merged_results": [],
            }
            collection_name = retrieval_service._get_collection_name()
            logger.info(f"使用集合: {collection_name}")
            _print_info(f"使用集合: {collection_name}")

            # 1. 优先尝试混合检索（推荐方式）
            logger.info("尝试混合检索...")
            _print_info("\n1. 尝试混合检索...")
            try:
                hybrid_results = retrieval_service.hybrid_search(
                    query=query,
                    top_k=15,  # 减少返回结果数量，提高性能
                    doc_id=doc_id,
                    kb_id=kb_id,
                    security_level=security_level
                )
                logger.info(f"混合检索结果数: {len(hybrid_results)}")
                _print_info(f"混合检索结果数: {len(hybrid_results)}")
                retrieval_record["hybrid"]["count"] = len(hybrid_results)

                for result in hybrid_results:
                    chunk_id = getattr(result, "chunk_id", None)
                    if chunk_id is not None and chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        all_results.append(result)
                    info = self._extract_doc_info_from_result(result)
                    retrieval_record["hybrid"]["chunks"].append(
                        {
                            "chunk_id": chunk_id,
                            "score": getattr(result, "score", None),
                            "content_preview": getattr(result, "content", "")[:400] if hasattr(result, "content") else "",
                            "doc_id": info["doc_id"],
                            "doc_title": info["doc_title"],
                        }
                    )
            except Exception as e:
                logger.error(f"混合检索失败: {str(e)}")
                _print_warning(f"混合检索失败: {str(e)}")

            # 2. 如果混合检索失败或结果不足，尝试语义检索
            if len(all_results) < 5:
                logger.info("混合检索结果不足，尝试语义检索...")
                _print_info("\n2. 尝试语义检索...")
                try:
                    semantic_results = retrieval_service.semantic_search(
                        query=query,
                        top_k=12,  # 减少返回结果数量，提高性能
                        doc_id=doc_id,
                        kb_id=kb_id,
                        security_level=security_level
                    )
                    logger.info(f"语义检索结果数: {len(semantic_results)}")
                    _print_info(f"语义检索结果数: {len(semantic_results)}")
                    retrieval_record["semantic"]["count"] = len(semantic_results)

                    for result in semantic_results:
                        chunk_id = getattr(result, "chunk_id", None)
                        if chunk_id is not None and chunk_id not in seen_chunk_ids:
                            seen_chunk_ids.add(chunk_id)
                            all_results.append(result)
                        info = self._extract_doc_info_from_result(result)
                        retrieval_record["semantic"]["chunks"].append(
                            {
                                "chunk_id": chunk_id,
                                "score": getattr(result, "score", None),
                                "content_preview": getattr(result, "content", "")[:400] if hasattr(result, "content") else "",
                                "doc_id": info["doc_id"],
                                "doc_title": info["doc_title"],
                            }
                        )
                except Exception as e:
                    logger.error(f"语义检索失败: {str(e)}")
                    _print_warning(f"语义检索失败: {str(e)}")

            # 3. 如果仍然不足，尝试关键词检索
            if len(all_results) < 5:
                logger.info("语义检索结果不足，尝试关键词检索...")
                _print_info("\n3. 尝试关键词检索...")
                try:
                    keyword_results = retrieval_service.keyword_search(
                        query=query,
                        top_k=12,  # 减少返回结果数量，提高性能
                        doc_id=doc_id,
                        kb_id=kb_id,
                        security_level=security_level
                    )
                    logger.info(f"关键词检索结果数: {len(keyword_results)}")
                    _print_info(f"关键词检索结果数: {len(keyword_results)}")
                    retrieval_record["keyword"]["count"] = len(keyword_results)

                    for result in keyword_results:
                        chunk_id = getattr(result, "chunk_id", None)
                        if chunk_id is not None and chunk_id not in seen_chunk_ids:
                            seen_chunk_ids.add(chunk_id)
                            all_results.append(result)
                        info = self._extract_doc_info_from_result(result)
                        retrieval_record["keyword"]["chunks"].append(
                            {
                                "chunk_id": chunk_id,
                                "score": getattr(result, "score", None),
                                "content_preview": getattr(result, "content", "")[:400] if hasattr(result, "content") else "",
                                "doc_id": info["doc_id"],
                                "doc_title": info["doc_title"],
                            }
                        )
                except Exception as e:
                    logger.error(f"关键词检索失败: {str(e)}")
                    _print_warning(f"关键词检索失败: {str(e)}")

            # 4. 对合并结果做轻量结构感知排序（不修改原始 score，仅调整 rank）
            scored_results: List[Dict[str, Any]] = []
            for result in all_results:
                base_score = getattr(result, "score", 0.0) or 0.0
                content = getattr(result, "content", "") if hasattr(result, "content") else ""
                struct_scores = self._analyze_chunk_structure(content)
                # 轻微加权：表格 > 引用 > 公式
                boost = (
                    0.25 * struct_scores.get("table_score", 0.0)
                    + 0.2 * struct_scores.get("citation_score", 0.0)
                    + 0.15 * struct_scores.get("formula_score", 0.0)
                )
                adjusted_score = float(base_score) + float(boost)
                scored_results.append(
                    {
                        "result": result,
                        "base_score": base_score,
                        "adjusted_score": adjusted_score,
                        "struct_scores": struct_scores,
                    }
                )

            scored_results.sort(key=lambda x: x["adjusted_score"], reverse=True)

            logger.info(f"合并后去重结果数: {len(scored_results)}")
            _print_info(f"\n合并后去重结果数: {len(scored_results)}")

            # 记录合并后的结果（保留预览和完整内容，便于后续评估）
            for idx, item in enumerate(scored_results):
                result = item["result"]
                struct_scores = item["struct_scores"]
                info = self._extract_doc_info_from_result(result)
                retrieval_record["merged_results"].append(
                    {
                        "rank": idx + 1,
                        "chunk_id": getattr(result, "chunk_id", None),
                        "score": getattr(result, "score", None),
                        "content_preview": getattr(result, "content", "")[:400] if hasattr(result, "content") else "",
                        "content_full": getattr(result, "content", "") if hasattr(result, "content") else "",
                        "doc_id": info["doc_id"],
                        "doc_title": info["doc_title"],
                        "structure_scores": struct_scores,
                    }
                )

            # 如果有结果，显示前3个结果的内容预览
            if all_results:
                logger.info("检索到结果，显示前3个结果预览")
                _print_info("\n检索结果预览:")
                for i, result in enumerate(all_results[:3]):
                    if hasattr(result, 'score') and hasattr(result, 'content'):
                        _print_info(f"结果 {i + 1} (分数: {result.score:.3f}): {result.content[:100]}...")

            if collect_trace:
                return all_results, retrieval_record
            return all_results

        except Exception as e:
            logger.error(f"检索失败: {str(e)}")
            _print_warning(f"检索失败: {str(e)}")
            # 移除详细的错误堆栈，减少日志噪音
            if collect_trace:
                return [], retrieval_record
            return []

    def summarize_single_chunk(self, chunk_content: str) -> str:
        """
        兼容旧接口：保留该方法名，但内部改为“证据抽取”（仅返回可用于融合的短文本）。
        """
        try:
            # 无 query 时只能退化为截断原文，避免丢失数字/符号
            return _truncate_for_fusion(chunk_content.strip())
        except Exception as e:
            logger.error(f"单个chunk summary失败: {str(e)}")
            return _truncate_for_fusion(chunk_content.strip())

    def extract_evidence_single_chunk(self, query: str, chunk_content: str) -> Dict[str, Any]:
        """
        对单个 chunk 做“证据抽取 + 关键事实提取”，优先保留原文数字/符号，降低多次调用导致的信息丢失。
        返回结构化 dict，失败时回退为截断原文作为 evidence。
        """
        content = (chunk_content or "").strip()
        if not content:
            return {"evidence_sentences": [], "key_facts": [], "relevance": "", "raw": ""}

        messages = [
            {
                "role": "system",
                "content": "你是一个严谨的学术证据抽取助手。请从给定的论文片段中抽取能直接回答问题的证据。\n"
                "要求：\n"
                "1) 优先原文拷贝，必须保留所有数字、符号、等号与变量（例如 s = 5、L_st 等）\n"
                "2) 只输出严格 JSON（不要代码块、不要多余文本）\n"
                "3) evidence_sentences 给出 1-3 句最相关原文（尽量短）\n"
                "4) key_facts 从证据句中提取关键事实点（短字符串列表），例如 \"s=5\"、\"T_obs=9\"、\"kernel=3x3\" 等\n"
                "5) relevance 用一句话说明证据与问题的对应关系",
            },
            {
                "role": "user",
                "content": (
                    f"问题: {query}\n\n论文片段:\n{content}\n\n"
                    "请输出 JSON，格式如下：\n"
                    "{\n"
                    "  \"evidence_sentences\": [\"...\"],\n"
                    "  \"key_facts\": [\"...\"],\n"
                    "  \"relevance\": \"...\"\n"
                    "}\n"
                ),
            },
        ]

        raw = self.deepseek.chat_completion(messages) or ""
        obj = self._safe_json_loads(raw)
        if obj is None:
            evidence = _truncate_for_fusion(content)
            # 尽力抽取常见等式/数字作为 key_facts
            key_facts: List[str] = []
            try:
                # 例如 s=5 / T_obs=9 / K=10
                key_facts.extend(re.findall(r"\b[A-Za-z]\w*\s*=\s*\d+(\.\d+)?\b", evidence))
            except Exception:
                pass
            if not key_facts:
                nums = re.findall(r"\b\d+(\.\d+)?\b", evidence)
                flat_nums: List[str] = []
                for n in nums[:5]:
                    if isinstance(n, tuple):
                        flat_nums.append(str(n[0]))
                    else:
                        flat_nums.append(str(n))
                key_facts = flat_nums
            return {"evidence_sentences": [evidence], "key_facts": key_facts, "relevance": "", "raw": raw.strip()}

        evidence_sentences = obj.get("evidence_sentences") if isinstance(obj.get("evidence_sentences"), list) else []
        key_facts = obj.get("key_facts") if isinstance(obj.get("key_facts"), list) else []
        relevance = obj.get("relevance") if isinstance(obj.get("relevance"), str) else ""

        ev_clean: List[str] = []
        for s in evidence_sentences[:3]:
            if isinstance(s, str) and s.strip():
                ev_clean.append(_truncate_for_fusion(s.strip(), max_chars=400))

        kf_clean: List[str] = []
        for k in key_facts[:8]:
            if isinstance(k, str) and k.strip():
                kf_clean.append(k.strip())

        if not ev_clean:
            ev_clean = [_truncate_for_fusion(content)]

        return {"evidence_sentences": ev_clean, "key_facts": kf_clean, "relevance": relevance.strip(), "raw": raw.strip()}

    def _rule_based_sufficiency(self, query: str, fused_info: str) -> tuple[bool, str]:
        """
        不依赖题型标签的规则优先 sufficiency 判定：
        - 若 evidence 已抽出 key_facts 或命中锚点+数值/等式，则直接判足够，避免 LLM Judge 误判触发改写。
        返回 (sufficient, reason)。
        """
        anchors = self._extract_anchors(query)
        evidence_items = getattr(self, "_last_evidence_items", []) or []

        # 1) 只要抽取到了 key_facts（如 s=5），通常就足够尝试回答
        for it in evidence_items:
            kf = it.get("key_facts")
            if isinstance(kf, list) and any(isinstance(x, str) and x.strip() for x in kf):
                return True, "rule:key_facts_present"

        # 2) 融合信息中出现锚点 + 数值/等式 → 足够
        text = (fused_info or "").strip()
        if text:
            has_anchor = any(a in text for a in anchors) if anchors else False
            has_number_or_eq = bool(re.search(r"(\b\d+(\.\d+)?\b|=|×|x|帧|epoch|mAP|EAO|IoU)", text))
            if has_anchor and has_number_or_eq:
                return True, "rule:anchor_and_number"

        # 3) evidence 句子中出现锚点 + 数值/等式 → 足够
        for it in evidence_items:
            ev = it.get("evidence_sentences")
            if not isinstance(ev, list):
                continue
            joined = "\n".join([s for s in ev if isinstance(s, str)])
            if not joined.strip():
                continue
            has_anchor = any(a in joined for a in anchors) if anchors else False
            has_number_or_eq = bool(re.search(r"(\b\d+(\.\d+)?\b|=|×|x|帧|epoch|mAP|EAO|IoU)", joined))
            if has_anchor and has_number_or_eq:
                return True, "rule:evidence_anchor_and_number"

        return False, "rule:no_strong_signal"

    def summarize_and_aggregate(self, query: str, results: List[Any]) -> str:
        """
        对检索结果进行summary/排序/过滤/聚合
        Args:
            query: 查询文本
            results: 检索结果列表
        Returns:
            聚合后的信息
        """
        logger.info(f"开始对检索结果进行summary/排序/过滤/聚合，查询: {query}")
        _print_info("\n=== summary/排序/过滤/聚合 ===")

        if not results:
            logger.warning("未检索到相关信息")
            _print_warning("警告: 未检索到相关信息")
            return "未检索到相关信息"

        # 1. 排序（按分数降序）
        try:
            # 更安全的排序逻辑
            def get_score(result):
                try:
                    return getattr(result, 'score', 0.0)
                except:
                    return 0.0

            sorted_results = sorted(results, key=get_score, reverse=True)
            logger.info(f"排序后结果数: {len(sorted_results)}")
            _print_info(f"排序后结果数: {len(sorted_results)}")
        except Exception as e:
            logger.error(f"排序失败: {str(e)}")
            _print_warning(f"排序失败: {str(e)}")
            sorted_results = results

        # 2. 过滤（限制处理的chunk数量以提高性能，取前10个最相关的chunk）
        #    在原始 score 基础上加入结构感知加权，让表格/引用chunk略微优先。
        scored_for_summary: List[Dict[str, Any]] = []
        for result in sorted_results:
            try:
                base_score = getattr(result, "score", 0.0) or 0.0
                content = getattr(result, "content", "") if hasattr(result, "content") else ""
                struct_scores = self._analyze_chunk_structure(content)
                boost = (
                    0.3 * struct_scores.get("table_score", 0.0)
                    + 0.2 * struct_scores.get("citation_score", 0.0)
                    + 0.1 * struct_scores.get("formula_score", 0.0)
                )
                adjusted_score = float(base_score) + float(boost)
                meta = self._extract_doc_info_from_result(result)
                scored_for_summary.append(
                    {
                        "result": result,
                        "base_score": base_score,
                        "adjusted_score": adjusted_score,
                        "struct_scores": struct_scores,
                        "doc_id": meta.get("doc_id"),
                        "doc_title": meta.get("doc_title"),
                        "chunk_id": getattr(result, "chunk_id", None),
                    }
                )
            except Exception as e:
                logger.warning(f"为summary构造结构感知得分时出错: {str(e)}")

        scored_for_summary.sort(key=lambda x: x["adjusted_score"], reverse=True)
        filtered_items = scored_for_summary[:10]  # 减少处理的数量，提高性能
        logger.info(f"过滤后结果数: {len(filtered_items)}")
        _print_info(f"过滤后结果数: {len(filtered_items)}")

        # 3. 对每个chunk进行证据抽取，避免上下文过长（并发调用以提升性能）
        chunk_evidences: List[str] = []

        # 预先收集需要处理的 chunk 信息
        indexed_chunks: List[Dict[str, Any]] = []
        for i, item in enumerate(filtered_items):
            try:
                result = item["result"]
                score = float(item.get("base_score", getattr(result, "score", 0.0) or 0.0))
                content = getattr(result, "content", "")
                chunk_id = item.get("chunk_id", getattr(result, "chunk_id", None))
                doc_id = item.get("doc_id")
                doc_title = item.get("doc_title")
                struct_scores = item.get("struct_scores", {})
                if content:
                    indexed_chunks.append(
                        {
                            "index": i,
                            "score": score,
                            "content": content,
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "doc_title": doc_title,
                            "structure_scores": struct_scores,
                        }
                    )
            except Exception as e:
                logger.warning(f"预处理结果{i + 1}时出错: {str(e)}")
                continue

        if not indexed_chunks:
            logger.warning("所有检索结果在预处理阶段即为空")
            _print_warning("警告: 所有检索结果在预处理阶段即为空")
            return "检索结果处理后为空"

        # 计算合适的并发线程数，避免过多并发压垮外部API
        max_workers = min(len(indexed_chunks), _SUMMARY_MAX_WORKERS)
        logger.info(
            f"开始并发summary处理，共 {len(indexed_chunks)} 个chunk，使用线程数: {max_workers}"
        )
        _print_info(
            f"\n开始并发summary处理，共 {len(indexed_chunks)} 个chunk，使用线程数: {max_workers}"
        )

        index_to_evidence: Dict[int, Dict[str, Any]] = {}
        index_to_score: Dict[int, float] = {}
        index_to_meta: Dict[int, Dict[str, Any]] = {}
        index_to_struct_scores: Dict[int, Dict[str, float]] = {}

        # 使用线程池并发调用 extract_evidence_single_chunk
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index: Dict[concurrent.futures.Future, int] = {}

                for item in indexed_chunks:
                    idx = item["index"]
                    score = item["score"]
                    content = item["content"]
                    index_to_score[idx] = score
                    index_to_meta[idx] = {
                        "chunk_id": item.get("chunk_id"),
                        "doc_id": item.get("doc_id"),
                        "doc_title": item.get("doc_title"),
                    }
                    index_to_struct_scores[idx] = item.get("structure_scores", {})

                    logger.info(f"提交第{idx + 1}个chunk进行证据抽取，分数: {score:.3f}")
                    _print_info(f"提交第{idx + 1}个chunk进行证据抽取，分数: {score:.3f}...")

                    future = executor.submit(self.extract_evidence_single_chunk, query, content)
                    future_to_index[future] = idx

                for future in concurrent.futures.as_completed(future_to_index):
                    idx = future_to_index[future]
                    score = index_to_score.get(idx, 0.0)
                    try:
                        ev_obj = future.result()
                        if isinstance(ev_obj, dict) and ev_obj.get("evidence_sentences"):
                            index_to_evidence[idx] = ev_obj
                        else:
                            original_content = next((it["content"] for it in indexed_chunks if it["index"] == idx), "")
                            index_to_evidence[idx] = {
                                "evidence_sentences": [_truncate_for_fusion(original_content)],
                                "key_facts": [],
                                "relevance": "",
                                "raw": "",
                            }
                            logger.warning(f"第{idx + 1}个chunk证据抽取为空，使用原始内容回退")
                    except Exception as e:
                        # 并发层异常，记录日志并回退为原文（截断后进融合）
                        original_content = next(
                            (item["content"] for item in indexed_chunks if item["index"] == idx),
                            "",
                        )
                        index_to_evidence[idx] = {
                            "evidence_sentences": [_truncate_for_fusion(original_content)],
                            "key_facts": [],
                            "relevance": "",
                            "raw": "",
                        }
                        logger.warning(
                            f"第{idx + 1}个chunk并发证据抽取任务失败，使用原始内容回退: {str(e)}"
                        )
        except Exception as e:
            # 线程池整体异常，记录日志并全部回退为原文（截断后进融合）
            logger.error(f"并发证据抽取整体失败，将全部使用原始内容回退: {str(e)}")
            _print_warning("并发证据抽取整体失败，将使用原始内容回退")
            for item in indexed_chunks:
                idx = item["index"]
                index_to_evidence[idx] = {
                    "evidence_sentences": [_truncate_for_fusion(item.get("content", ""))],
                    "key_facts": [],
                    "relevance": "",
                    "raw": "",
                }
                index_to_score[idx] = float(item.get("score", 0.0))
                index_to_meta[idx] = {
                    "chunk_id": item.get("chunk_id"),
                    "doc_id": item.get("doc_id"),
                    "doc_title": item.get("doc_title"),
                }
                index_to_struct_scores[idx] = item.get("structure_scores", {})

        # 按原始顺序构造 chunk_evidences，保持算法行为一致
        evidence_items_for_judge: List[Dict[str, Any]] = []
        for i in range(len(filtered_items)):
            if i in index_to_evidence and i in index_to_score:
                score = index_to_score.get(i, 0.0)
                ev_obj = index_to_evidence[i]
                meta = index_to_meta.get(i, {})
                doc_title = meta.get("doc_title") or ""
                chunk_id = meta.get("chunk_id")
                doc_id = meta.get("doc_id")
                struct_scores = index_to_struct_scores.get(i, {}) or {}
                key_facts = ev_obj.get("key_facts") if isinstance(ev_obj.get("key_facts"), list) else []
                ev_sents = ev_obj.get("evidence_sentences") if isinstance(ev_obj.get("evidence_sentences"), list) else []
                key_facts_str = ", ".join([x for x in key_facts if isinstance(x, str)])[:200]
                ev_text_lines: List[str] = [s for s in ev_sents if isinstance(s, str)]

                # 邻接 chunk 证据扩展：同一 doc_id 内、chunk_id 相邻的证据一并纳入
                if doc_id is not None and chunk_id is not None:
                    try:
                        neighbor_texts: List[str] = []
                        for j, ev_j in index_to_evidence.items():
                            if j == i:
                                continue
                            meta_j = index_to_meta.get(j, {})
                            doc_id_j = meta_j.get("doc_id")
                            chunk_id_j = meta_j.get("chunk_id")
                            if doc_id_j != doc_id or chunk_id_j is None:
                                continue
                            if abs(int(chunk_id_j) - int(chunk_id)) <= 2:
                                ev_j_sents = ev_j.get("evidence_sentences")
                                if isinstance(ev_j_sents, list):
                                    joined = "\n".join(
                                        [s for s in ev_j_sents if isinstance(s, str)]
                                    )
                                    if joined.strip():
                                        neighbor_texts.append(
                                            f"[邻接chunk {chunk_id_j} 证据]\n{joined}"
                                        )
                        if neighbor_texts:
                            ev_text_lines.append("\n".join(neighbor_texts))
                    except Exception as e:
                        logger.warning(f"邻接chunk证据扩展失败: {str(e)}")

                ev_text = "\n".join(ev_text_lines)

                tags: List[str] = []
                if struct_scores.get("table_score", 0.0) > 0.0:
                    tags.append("table")
                if struct_scores.get("citation_score", 0.0) > 0.0:
                    tags.append("citation")
                if struct_scores.get("formula_score", 0.0) > 0.0:
                    tags.append("formula")
                tags_str = ", ".join(tags)

                evidence_items_for_judge.append(
                    {
                        "rank": i + 1,
                        "chunk_id": chunk_id,
                        "doc_title": doc_title,
                        "score": score,
                        "key_facts": key_facts,
                        "evidence_sentences": ev_sents,
                        "structure_scores": struct_scores,
                        "doc_id": doc_id,
                    }
                )

                chunk_evidences.append(
                    f"[结果{i + 1} 相关性分数: {score:.3f}]\n"
                    f"doc_title: {doc_title}\n"
                    f"chunk_id: {chunk_id}\n"
                    f"tags: {tags_str}\n"
                    f"key_facts: {key_facts_str}\n"
                    f"evidence:\n{ev_text}\n"
                )

        # 存储上一轮 evidence，供 Judge 规则与 trace 使用
        self._last_evidence_items = evidence_items_for_judge

        if not chunk_evidences:
            logger.warning("所有检索结果处理后为空")
            _print_warning("警告: 所有检索结果处理后为空")
            return "检索结果处理后为空"

        # 4. 整合所有chunk的summary
        combined_summaries = "\n".join(chunk_evidences)
        logger.info(f"整合后的证据长度: {len(combined_summaries)} 字符")
        _print_info(f"\n整合后的证据长度: {len(combined_summaries)} 字符")

        # 5. 使用DeepSeek进行最终信息融合
        logger.info("使用DeepSeek API进行最终信息融合...")
        _print_info("\n使用DeepSeek API进行最终信息融合...")

        messages = [
            {
                "role": "system",
                "content": "你是一个专业的学术信息整合助手。请严格遵循以下要求：\n1. 仔细阅读用户的学术查询和提供的证据片段（证据中可能包含关键数字/符号）\n2. 只基于证据回答，必须保留关键数字、符号、等号与变量（例如 s=5、L_st 等）\n3. 忽略与查询无关的内容\n4. 将相关信息整合成一段连贯、逻辑清晰的文本\n5. 保持信息的准确性，不要添加任何检索结果中没有的内容\n6. 如果没有相关信息，直接返回'未找到相关信息'\n7. 输出应尽量简洁，优先给出可直接回答问题的事实"
            },
            {
                "role": "user",
                "content": f"学术查询: {query}\n\n以下是各段论文内容的证据与关键事实，请基于这些信息整合出完整回答：\n{combined_summaries}\n\n整合后的回答："
            }
        ]
        fused_info = self.deepseek.chat_completion(messages)
        if fused_info and fused_info.strip() and fused_info.strip() != "未找到相关信息":
            logger.info(f"信息融合成功，融合后信息长度: {len(fused_info)} 字符")
            _print_info("信息融合成功:")
            _print_info(f"融合后信息长度: {len(fused_info)} 字符")
            _print_info(f"融合后信息: {fused_info[:200]}...")
            return fused_info
        else:
            logger.warning("信息融合失败，使用备用方案")
            _print_warning("信息融合失败，使用备用方案")

            # 改进的备用方案
            backup_info = "\n".join(chunk_evidences[:5])  # 只使用前5个证据片段

            # 如果备用方案仍然为空，提供更明确的错误信息
            if not backup_info.strip():
                backup_info = "检索结果内容为空，可能原因：集合为空、数据格式不匹配、检索服务配置问题"

            logger.info(f"备用方案信息长度: {len(backup_info)} 字符")
            _print_info(f"备用方案信息长度: {len(backup_info)} 字符")
            return backup_info

    def judge_information_sufficiency(self, query: str, fused_info: str) -> bool:
        """
        判断信息是否足够回答查询
        Args:
            query: 查询文本
            fused_info: 融合后的信息
        Returns:
            True: 信息足够；False: 信息不足
        """
        logger.info(f"开始判断信息是否足够回答查询，查询: {query}")
        _print_info("\n=== Judge环节 ===")
        _print_info(f"查询: {query}")
        _print_info(f"可用信息长度: {len(fused_info)} 字符")

        # 默认清空本轮 Judge 原始输出记录
        try:
            self._last_judge_raw = None  # type: ignore[attr-defined]
        except Exception:
            # 如果属性设置失败，不影响主流程
            pass

        # 只对完全空的信息进行过滤，允许短答案
        if not fused_info or fused_info.strip() == "":
            logger.warning("判断结果: 信息为空，不足")
            _print_warning("判断结果: 信息为空，不足")
            try:
                self._last_judge_raw = "信息为空，被直接判定为不足"  # type: ignore[attr-defined]
            except Exception:
                pass
            return False

        # 特殊情况：如果信息非常短但可能是有效答案
        # 例如："是"、"否"、"2023年"等
        if len(fused_info) < 50:
            logger.info(f"注意：信息较短 ({len(fused_info)} 字符)，但仍进行语义判断")
            _print_info(f"注意：信息较短 ({len(fused_info)} 字符)，但仍进行语义判断")

        # 规则优先判定：减少 LLM Judge 误判触发改写
        try:
            sufficient_rule, rule_reason = self._rule_based_sufficiency(query, fused_info)
            try:
                self._last_judge_rule = rule_reason  # type: ignore[attr-defined]
            except Exception:
                pass
            if sufficient_rule:
                logger.info(f"Judge规则判定为足够: {rule_reason}")
                _print_info(f"Judge规则判定为足够: {rule_reason}")
                try:
                    self._last_judge_raw = f"规则判定: 足够 ({rule_reason})"  # type: ignore[attr-defined]
                except Exception:
                    pass
                return True
        except Exception as e:
            logger.warning(f"Judge规则判定异常，将回退到LLM Judge: {str(e)}")

        # 使用LLM进行智能语义判断（兜底）
        try:
            messages = [
                {
                    "role": "system",
                    "content": "你是一个智能的信息评估专家。请仔细分析用户的查询和提供的信息：\n1. 不管信息长度如何，如果包含了回答用户查询所需的核心内容，能够形成一个完整、准确的答案，返回'足够'\n2. 即使信息较长，但缺少关键内容，无法回答用户问题，返回'不足'\n3. 特别注意：有些问题的答案本身就很短（如'是/否'、具体年份等），这些情况也应该返回'足够'\n4. 请只返回'足够'或'不足'，不要添加任何其他解释或说明"
                },
                {
                    "role": "user",
                    "content": f"查询: {query}\n\n可用信息:\n{fused_info}\n\n请严格按照要求判断并输出结果："
                }
            ]

            judgment = self.deepseek.chat_completion(messages)

            if judgment:
                # 保留原始输出，便于后续在trace中展示
                raw_judgment = str(judgment).strip()
                try:
                    self._last_judge_raw = raw_judgment  # type: ignore[attr-defined]
                except Exception:
                    pass

                # 清理和标准化返回结果用于逻辑分支
                normalized = raw_judgment.strip().lower()
                logger.info(f"判断结果: {normalized}")
                _print_info(f"判断结果: {normalized}")

                # 更宽松的结果处理
                if "足够" in normalized or "sufficient" in normalized:
                    return True
                elif "不足" in normalized or "insufficient" in normalized:
                    return False
                else:
                    # 如果返回格式不符合要求，默认认为信息足够，尝试生成响应
                    logger.warning("返回格式异常，默认认为信息足够")
                    _print_warning("返回格式异常，默认认为信息足够")
                    return True
            else:
                logger.warning("LLM判断失败，默认认为信息足够，尝试生成响应")
                _print_warning("LLM判断失败，默认认为信息足够，尝试生成响应")
                try:
                    self._last_judge_raw = "LLM调用失败，按默认策略判定为足够"  # type: ignore[attr-defined]
                except Exception:
                    pass
                # API调用失败时，默认认为信息足够，不中断流程
                return True
        except Exception as e:
            logger.error(f"判断环节出错: {str(e)}")
            _print_warning(f"判断环节出错: {str(e)}")
            # 出错时默认认为信息足够，不中断流程
            try:
                self._last_judge_raw = f"Judge环节异常: {str(e)}"  # type: ignore[attr-defined]
            except Exception:
                pass
            return True

    def generate_response(self, query: str, fused_info: str) -> str:
        """
        生成最终响应
        Args:
            query: 查询文本
            fused_info: 融合后的信息
        Returns:
            最终响应
        """
        logger.info(f"开始生成最终响应，查询: {query}")
        _print_info("\n=== 生成响应 ===")

        system_prompt = """
        你是一位顶级知识库引擎，擅长将复杂知识以清晰、严谨、结构化的方式呈现。
            任务：整合、提炼、结构化知识库信息。
            行动：逻辑重组原文要点，用专业语言归纳总结，形成连贯完整的回答。
            要求：
                1. 专业、严谨、客观、精炼。
                2. 使用Markdown格式化。
                3. 数学公式使用LaTeX格式。
        """
        messages = [
            {
                "role": "system",
                "content": system_prompt

            },
            {
                "role": "user",
                "content": f"查询: {query}\n\n可用信息:\n{fused_info}\n\n请基于上述信息较为简洁完整地回答用户的问题："
            }
        ]

        response = self.deepseek.chat_completion(messages)

        if response:
            logger.info(f"最终响应生成成功，响应长度: {len(response)} 字符")
            _print_info("最终响应:")
            _print_info(response)
            return response
        else:
            logger.error("响应生成失败")
            _print_warning("响应生成失败")
            return "抱歉，无法生成响应。"

    def run(
        self,
        original_query: str,
        sample_id: Optional[str] = None,
        return_trace: bool = False,
        doc_id: Union[int, List[int], None] = None,
        kb_id: Union[int, List[int], None] = None,
        security_level: Union[int, List[int], None] = None,
    ):
        """
        执行完整的RAG流程
        Args:
            original_query: 用户原始查询
            sample_id: 可选的样本ID（用于评估时关联 QA.json）
            doc_id: 文档ID
            kb_id: 知识库分类
            security_level: 访问级别
            return_trace: 是否返回本次流程的完整trace结构
        """
        start_time = datetime.now()
        logger.info(f"RAG流程启动，原始查询: {original_query}")
        _print_info("\n" + "=" * 60)
        _print_info("RAG流程启动")
        _print_info("=" * 60)

        trace: Dict[str, Any] = {
            "sample_id": sample_id,
            "original_query": original_query,
            "start_time": start_time.isoformat(),
            "iterations": [],
            "final_response_preview": None,
            "status": "running",
        }

        iteration = 0
        context: Optional[str] = None
        final_response: Optional[str] = None
        current_query = original_query
        rewrite_rounds = 0
        rewrite_history: List[Dict[str, Any]] = []

        # 实现真正的多轮迭代，支持最多 max_iterations 轮
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"开始第 {iteration} 轮迭代，当前查询: {current_query}")
            _print_info(f"\n--- 第 {iteration} 轮迭代 ---")

            iteration_record: Dict[str, Any] = {
                "iteration_index": iteration,
                "query_used": current_query,
                "retrieval": None,
                "summary_preview": None,
                "judge_sufficient": None,
            }

            # 1. 检索文档（第一轮使用原始 query，后续轮次可能使用改写后的 query）
            results, retrieval_record = self.retrieve_documents(current_query, collect_trace=True, doc_id=doc_id, kb_id=kb_id, security_level=security_level)
            iteration_record["retrieval"] = retrieval_record

            if not results:
                logger.warning("未检索到任何文档")
                _print_warning("\n未检索到任何文档，可能的原因：")
                _print_warning("1. 可能集合中没有数据或连接失败")

                if rewrite_rounds < self.max_rewrite_rounds:
                    # 尝试基于当前查询和上下文进行适度改写
                    rewrite_before = current_query
                    context = (
                        f"当前查询: {current_query}\n"
                        f"检索结果: 未找到任何相关文档，请在不改变问题核心语义的前提下，"
                        f"尝试从不同表述方式、补充关键信息等角度改写查询，以提高检索召回。"
                    )
                    new_query = self.understand_and_rewrite_query(original_query, context)
                    rewrite_rounds += 1
                    rewrite_history.append(
                        {
                            "round": rewrite_rounds,
                            "before": rewrite_before,
                            "after": new_query,
                        }
                    )
                    current_query = new_query
                    logger.info(
                        f"第 {iteration} 轮未检索到文档，尝试第 {rewrite_rounds} 次查询改写，"
                        f"下一轮查询: {current_query}"
                    )
                    _print_warning(
                        f"\n第 {iteration} 轮信息不足，进行第 {rewrite_rounds} 次查询改写后继续下一轮迭代"
                    )
                    trace["iterations"].append(iteration_record)
                    continue
                else:
                    # 达到最大改写轮数，直接给出失败提示
                    final_response = "抱歉，无法连接到文档数据库或数据库中没有相关数据。"
                    logger.warning("达到最大查询改写次数，仍未检索到任何文档")
                    _print_warning("达到最大查询改写次数，仍未检索到任何文档")
                    break

            # 2. summary/排序/过滤/聚合
            fused_info = self.summarize_and_aggregate(current_query, results)
            iteration_record["summary_preview"] = {
                "length": len(fused_info),
                "preview": fused_info[:200],
            }
            # 记录完整的summary内容，便于测试报告中回溯
            iteration_record["summary_full"] = fused_info
            # 记录 evidence 抽取结果（预览），便于评估稳定性
            try:
                iteration_record["evidence"] = [
                    {
                        "rank": it.get("rank"),
                        "chunk_id": it.get("chunk_id"),
                        "doc_title": it.get("doc_title"),
                        "score": it.get("score"),
                        "key_facts": it.get("key_facts"),
                        "evidence_preview": (
                            "\n".join(it.get("evidence_sentences", []))[:200]
                            if isinstance(it.get("evidence_sentences"), list)
                            else ""
                        ),
                    }
                    for it in (getattr(self, "_last_evidence_items", []) or [])[:10]
                ]
            except Exception:
                iteration_record["evidence"] = None

            # 3. Judge环节 - 使用智能语义判断
            logger.info("开始Judge环节")
            _print_info("\nJudge环节：")
            sufficient = self.judge_information_sufficiency(current_query, fused_info)
            iteration_record["judge_sufficient"] = sufficient
            # 记录Judge原始输出文本（如果有）
            try:
                iteration_record["judge_raw"] = getattr(self, "_last_judge_raw", None)
            except Exception:
                iteration_record["judge_raw"] = None
            # 记录Judge规则判定原因（如果有）
            try:
                iteration_record["judge_rule"] = getattr(self, "_last_judge_rule", None)
            except Exception:
                iteration_record["judge_rule"] = None

            # 记录本轮迭代信息
            trace["iterations"].append(iteration_record)

            if sufficient:
                logger.info("信息足够，生成最终响应")
                _print_info("信息足够，生成最终响应")
                # 4. 生成响应
                final_response = self.generate_response(current_query, fused_info)
                break  # 信息足够，退出循环
            else:
                # 信息不足，准备下一轮迭代
                logger.info(f"第 {iteration} 轮信息不足")
                _print_warning(f"第 {iteration} 轮信息不足")

                if rewrite_rounds < self.max_rewrite_rounds and iteration < self.max_iterations:
                    # 更新上下文，包含当前轮的信息，并尝试新的改写
                    rewrite_before = current_query
                    context = (
                        f"当前查询: {current_query}\n"
                        f"已检索到的信息片段（截断）: {fused_info[:200]}...\n"
                        f"当前信息不足以完全回答问题，请在不改变问题核心含义的前提下，"
                        f"适度调整或扩展查询表达，以获取更相关的文献片段。"
                    )
                    new_query = self.understand_and_rewrite_query(original_query, context)
                    rewrite_rounds += 1
                    rewrite_history.append(
                        {
                            "round": rewrite_rounds,
                            "before": rewrite_before,
                            "after": new_query,
                        }
                    )
                    current_query = new_query
                    logger.info(
                        f"第 {iteration} 轮信息不足，尝试第 {rewrite_rounds} 次查询改写，"
                        f"下一轮查询: {current_query}"
                    )
                    _print_warning(
                        f"更新上下文并进行第 {rewrite_rounds} 次查询改写，准备第 {iteration + 1} 轮迭代"
                    )
                    continue  # 继续下一轮完整的RAG流程
                else:
                    # 达到最大改写轮数或最大迭代次数，使用当前信息生成尽力回答
                    logger.info("已达到最大迭代次数或查询改写次数，基于当前信息生成最终响应")
                    _print_warning("已达到最大迭代次数或查询改写次数，基于当前信息生成最终响应")
                    final_response = self.generate_response(current_query, fused_info)
                    break

        # 确保无论如何都有最终响应
        if not final_response:
            final_response = "抱歉，经过多轮检索，仍无法获取足够的信息来回答您的查询。"

        end_time = datetime.now()
        trace["end_time"] = end_time.isoformat()
        trace["elapsed_seconds"] = (end_time - start_time).total_seconds()
        trace["rewrite_rounds"] = rewrite_rounds
        trace["rewrite_history"] = rewrite_history
        trace["final_response_preview"] = final_response[:200] if isinstance(final_response, str) else None
        trace["final_query"] = current_query
        trace["status"] = "success"

        logger.info(f"RAG流程结束，最终响应: {final_response[:100]}...")
        _print_info("\n" + "=" * 60)
        _print_info("RAG流程结束")
        _print_info("=" * 60)

        # 追加结构化 trace 日志
        _append_trace_log(trace)

        if return_trace:
            return final_response, trace
        return final_response


if __name__ == "__main__":
    print("欢迎使用RAG问答系统！")

    # 获取用户输入
    while True:
        query = input("\n请输入您的问题（输入'退出'结束）：")

        if query.strip() == "退出":
            print("感谢使用，再见！")
            break

        if not query.strip():
            print("查询不能为空，请重新输入")
            continue

        # 执行RAG流程
        rag = RAGFlow()
        response = rag.run(query)

        print("\n" + "=" * 80)
        print("最终回答：")
        print(response)
        print("=" * 80)