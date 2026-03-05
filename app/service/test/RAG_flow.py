#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG流程实现
根据提供的流程图，实现完整的检索增强生成流程
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages
        }

        try:
            logger.info("调用DeepSeek API进行对话完成")
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
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
                "content": "你是一个专业的学术查询理解和改写助手，专注于提高学术论文检索的效果。请严格遵循以下要求：\n1. 深入分析用户的学术查询意图，识别核心实体、数值和关键术语\n2. 如果有上下文，结合上下文理解用户的真实研究需求\n3. 在严格保留原始问题语义和关键实体不变的前提下，对查询进行适度的结构化和扩展，使其更适合检索\n4. 改写应避免只是做同义词替换（改动过小），也不要将问题改成完全不同的场景或任务（改动过大）\n5. 适当增加能够帮助检索的限定词（如方法名称、指标、数据集名称等），使查询更加明确、具体、全面\n6. 仅输出改写后的查询文本，不要添加任何其他说明或解释\n7. 保持查询简洁，避免冗余信息"
            },
            {
                "role": "user",
                "content": f"原始查询: {original_query}{f'\n上下文: {context}' if context else ''}\n\n改写后的查询:"
            }
        ]

        rewritten_query = self.deepseek.chat_completion(messages)

        if not rewritten_query:
            logger.warning("查询改写失败，使用原始查询")
            _print_warning("查询改写失败，使用原始查询")
            return original_query

        rewritten_query = rewritten_query.strip()
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

    def retrieve_documents(self, query: str, collect_trace: bool = False) -> Any:
        """
        执行多路召回检索
        Args:
            query: 查询文本
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
                    top_k=15  # 减少返回结果数量，提高性能
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
                        top_k=12  # 减少返回结果数量，提高性能
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
                        top_k=12  # 减少返回结果数量，提高性能
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

            # 4. 显示合并后的结果
            logger.info(f"合并后去重结果数: {len(all_results)}")
            _print_info(f"\n合并后去重结果数: {len(all_results)}")

            # 记录合并后的结果（保留预览和完整内容，便于后续评估）
            for idx, result in enumerate(all_results):
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
        对单个chunk进行summary，特别优化表格处理
        """
        try:
            # 检测是否包含表格内容
            if '<table>' in chunk_content.lower() or '|' in chunk_content:
                # 表格内容使用专门的prompt
                messages = [
                    {
                        "role": "system",
                        "content": "你是一个专业的学术内容总结助手，特别擅长处理表格数据。请严格遵循以下要求：\n1. 如果内容包含表格，优先提取表格中的关键数据和趋势\n2. 用简洁的文字描述表格的主要发现和数据模式\n3. 突出重要的数值对比和结论\n4. 保持总结准确、简洁，避免冗长描述\n5. 如果是实验结果表格，重点说明方法间的性能对比"
                    },
                    {
                        "role": "user",
                        "content": f"请总结以下包含表格的学术内容：\n{chunk_content}\n\n表格总结："
                    }
                ]
            else:
                # 普通内容使用原有prompt
                messages = [
                    {
                        "role": "system",
                        "content": "你是一个专业的学术内容总结助手。请严格遵循以下要求：\n1. 阅读以下学术论文片段，提取核心信息和关键论点\n2. 总结必须准确、简洁，保留所有重要的学术观点和事实\n3. 不添加任何原文中没有的信息或个人解读\n4. 使用清晰的语言结构，突出重点内容\n5. 保持总结长度适中，避免过于冗长或过于简略\n6. 如果内容是论文摘要，保留其主要结构和核心结论"
                    },
                    {
                        "role": "user",
                        "content": f"请总结以下学术内容：\n{chunk_content}\n\n总结："
                    }
                ]
            summary = self.deepseek.chat_completion(messages)
            if summary and summary.strip():
                return summary.strip()
            return chunk_content.strip()
        except Exception as e:
            logger.error(f"单个chunk summary失败: {str(e)}")
            return chunk_content.strip()

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
        filtered_results = sorted_results[:10]  # 减少处理的数量，提高性能
        logger.info(f"过滤后结果数: {len(filtered_results)}")
        _print_info(f"过滤后结果数: {len(filtered_results)}")

        # 3. 对每个chunk单独进行summary，避免上下文过长
        chunk_summaries = []
        for i, result in enumerate(filtered_results):
            try:
                score = getattr(result, 'score', 0.0)
                content = getattr(result, 'content', '')
                if content:
                    logger.info(f"正在处理第{i + 1}个chunk，分数: {score:.3f}")
                    _print_info(f"处理第{i + 1}个chunk，分数: {score:.3f}...")
                    # 对单个chunk进行summary
                    chunk_summary = self.summarize_single_chunk(content)
                    if chunk_summary:
                        chunk_summaries.append(f"[结果{i + 1} 相关性分数: {score:.3f}]\n{chunk_summary}\n")
            except Exception as e:
                logger.warning(f"处理结果{i + 1}时出错: {str(e)}")
                continue

        if not chunk_summaries:
            logger.warning("所有检索结果处理后为空")
            _print_warning("警告: 所有检索结果处理后为空")
            return "检索结果处理后为空"

        # 4. 整合所有chunk的summary
        combined_summaries = "\n".join(chunk_summaries)
        logger.info(f"整合后的summary长度: {len(combined_summaries)} 字符")
        _print_info(f"\n整合后的summary长度: {len(combined_summaries)} 字符")

        # 5. 使用DeepSeek进行最终信息融合
        logger.info("使用DeepSeek API进行最终信息融合...")
        _print_info("\n使用DeepSeek API进行最终信息融合...")

        messages = [
            {
                "role": "system",
                "content": "你是一个专业的学术信息整合助手。请严格遵循以下要求：\n1. 仔细阅读用户的学术查询和提供的各段论文摘要\n2. 提取所有与查询直接相关的关键学术信息\n3. 忽略与查询无关的内容\n4. 将相关信息整合成一段连贯、逻辑清晰的文本\n5. 保持信息的准确性，不要添加任何检索结果中没有的内容\n6. 如果有多个相关观点，按重要性排序\n7. 如果没有相关信息，直接返回'未找到相关信息'\n8. 确保回答专业、准确，适合学术研究场景"
            },
            {
                "role": "user",
                "content": f"学术查询: {query}\n\n以下是各段论文内容的摘要，请基于这些信息整合出完整回答：\n{combined_summaries}\n\n整合后的回答："
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
            backup_info = "\n".join(chunk_summaries[:5])  # 只使用前5个summary

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

        # 只对完全空的信息进行过滤，允许短答案
        if not fused_info or fused_info.strip() == "":
            logger.warning("判断结果: 信息为空，不足")
            _print_warning("判断结果: 信息为空，不足")
            return False

        # 特殊情况：如果信息非常短但可能是有效答案
        # 例如："是"、"否"、"2023年"等
        if len(fused_info) < 50:
            logger.info(f"注意：信息较短 ({len(fused_info)} 字符)，但仍进行语义判断")
            _print_info(f"注意：信息较短 ({len(fused_info)} 字符)，但仍进行语义判断")

        # 使用LLM进行智能语义判断
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
                # 清理和标准化返回结果
                judgment = judgment.strip().lower()
                logger.info(f"判断结果: {judgment}")
                _print_info(f"判断结果: {judgment}")

                # 更宽松的结果处理
                if "足够" in judgment or "sufficient" in judgment:
                    return True
                elif "不足" in judgment or "insufficient" in judgment:
                    return False
                else:
                    # 如果返回格式不符合要求，默认认为信息足够，尝试生成响应
                    logger.warning("返回格式异常，默认认为信息足够")
                    _print_warning("返回格式异常，默认认为信息足够")
                    return True
            else:
                logger.warning("LLM判断失败，默认认为信息足够，尝试生成响应")
                _print_warning("LLM判断失败，默认认为信息足够，尝试生成响应")
                # API调用失败时，默认认为信息足够，不中断流程
                return True
        except Exception as e:
            logger.error(f"判断环节出错: {str(e)}")
            _print_warning(f"判断环节出错: {str(e)}")
            # 出错时默认认为信息足够，不中断流程
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

        messages = [
            {
                "role": "system",
                "content": "你是一个专业的论文问答助手。请根据提供的论文中的信息，简洁明了地回答用户的问题。要求：\n1. 回答要准确、重点突出，基于提供的信息\n2. 保持回答简洁，避免冗长的表述和不必要的细节，但重要内容不能缺失\n3. 如果信息不足，请直接说明无法回答\n4. 不要添加额外的推测或内容\n5. 使用清晰、专业的语言风格\n6. 如果是实验方法型的问题，需要准确概况出论文所使用的核心实验方法"

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
    ):
        """
        执行完整的RAG流程
        Args:
            original_query: 用户原始查询
            sample_id: 可选的样本ID（用于评估时关联 QA.json）
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
            results, retrieval_record = self.retrieve_documents(current_query, collect_trace=True)
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

            # 3. Judge环节 - 使用智能语义判断
            logger.info("开始Judge环节")
            _print_info("\nJudge环节：")
            sufficient = self.judge_information_sufficiency(current_query, fused_info)
            iteration_record["judge_sufficient"] = sufficient

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