#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG流程实现
根据提供的流程图，实现完整的检索增强生成流程
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional
import requests

# 设置日志
logging.basicConfig(
    filename='rag_flow.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RAGFlow')

# 导入项目的retrieval_service模块
from app.service import retrieval_service

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

        # 最大迭代次数
        self.max_iterations = 3

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
        print("\n=== Query理解和改写 ===")
        print(f"原始查询: {original_query}")

        messages = [
            {
                "role": "system",
                "content": "你是一个专业的查询理解和改写助手。请分析用户的查询意图，将其改写成更适合检索的形式。如果有上下文，请结合上下文理解用户的真实需求。改写后的查询应该保留原始意图，但更加明确和具体。"
            },
            {
                "role": "user",
                "content": f"原始查询: {original_query}{f'\n上下文: {context}' if context else ''}\n请输出改写后的查询："
            }
        ]

        rewritten_query = self.deepseek.chat_completion(messages)

        if rewritten_query:
            logger.info(f"查询改写成功，改写后的查询: {rewritten_query}")
            print(f"改写后的查询: {rewritten_query}")
            return rewritten_query
        else:
            logger.warning("查询改写失败，使用原始查询")
            print("查询改写失败，使用原始查询")
            return original_query

    def retrieve_documents(self, query: str) -> List[Any]:
        """
        执行多路召回检索
        Args:
            query: 查询文本
        Returns:
            检索结果列表
        """
        logger.info(f"开始执行多路召回检索，查询: {query}")
        print("\n=== 执行检索（多路召回） ===")
        print(f"查询文本: {query}")

        try:
            # 尝试多种检索方式，确保至少有一种能返回结果
            all_results = []
            seen_chunk_ids = set()
            collection_name = retrieval_service._get_collection_name()
            logger.info(f"使用集合: {collection_name}")
            print(f"使用集合: {collection_name}")

            # 1. 个人认为应该优先尝试混合检索（推荐方式）
            logger.info("尝试混合检索...")
            print("\n1. 尝试混合检索...")
            try:
                hybrid_results = retrieval_service.hybrid_search(
                    query=query,
                    top_k=20
                )
                logger.info(f"混合检索结果数: {len(hybrid_results)}")
                print(f"混合检索结果数: {len(hybrid_results)}")

                for result in hybrid_results:
                    if hasattr(result, 'chunk_id') and result.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(result.chunk_id)
                        all_results.append(result)
            except Exception as e:
                logger.error(f"混合检索失败: {str(e)}")
                print(f"混合检索失败: {str(e)}")

            # 2. 如果混合检索失败或结果不足，尝试语义检索
            if len(all_results) < 5:
                logger.info("混合检索结果不足，尝试语义检索...")
                print("\n2. 尝试语义检索...")
                try:
                    semantic_results = retrieval_service.semantic_search(
                        query=query,
                        top_k=15
                    )
                    logger.info(f"语义检索结果数: {len(semantic_results)}")
                    print(f"语义检索结果数: {len(semantic_results)}")

                    for result in semantic_results:
                        if hasattr(result, 'chunk_id') and result.chunk_id not in seen_chunk_ids:
                            seen_chunk_ids.add(result.chunk_id)
                            all_results.append(result)
                except Exception as e:
                    logger.error(f"语义检索失败: {str(e)}")
                    print(f"语义检索失败: {str(e)}")

            # 3. 如果仍然不足，尝试关键词检索
            if len(all_results) < 5:
                logger.info("语义检索结果不足，尝试关键词检索...")
                print("\n3. 尝试关键词检索...")
                try:
                    keyword_results = retrieval_service.keyword_search(
                        query=query,
                        top_k=15
                    )
                    logger.info(f"关键词检索结果数: {len(keyword_results)}")
                    print(f"关键词检索结果数: {len(keyword_results)}")

                    for result in keyword_results:
                        if hasattr(result, 'chunk_id') and result.chunk_id not in seen_chunk_ids:
                            seen_chunk_ids.add(result.chunk_id)
                            all_results.append(result)
                except Exception as e:
                    logger.error(f"关键词检索失败: {str(e)}")
                    print(f"关键词检索失败: {str(e)}")

            # 4. 显示合并后的结果
            logger.info(f"合并后去重结果数: {len(all_results)}")
            print(f"\n合并后去重结果数: {len(all_results)}")

            # 如果有结果，显示前3个结果的内容预览
            if all_results:
                logger.info("检索到结果，显示前3个结果预览")
                print("\n检索结果预览:")
                for i, result in enumerate(all_results[:3]):
                    if hasattr(result, 'score') and hasattr(result, 'content'):
                        print(f"结果 {i+1} (分数: {result.score:.3f}): {result.content[:100]}...")

            return all_results

        except Exception as e:
            logger.error(f"检索失败: {str(e)}")
            print(f"检索失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

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
        print("\n=== summary/排序/过滤/聚合 ===")

        if not results:
            logger.warning("未检索到相关信息")
            print("警告: 未检索到相关信息")
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
            print(f"排序后结果数: {len(sorted_results)}")
        except Exception as e:
            logger.error(f"排序失败: {str(e)}")
            print(f"排序失败: {str(e)}")
            sorted_results = results

        # 2. 过滤（取前20个结果，增加容错性）
        filtered_results = sorted_results[:20]
        logger.info(f"过滤后结果数: {len(filtered_results)}")
        print(f"过滤后结果数: {len(filtered_results)}")

        # 3. 准备用于信息融合的内容
        # 更安全的内容提取
        retrieval_context_parts = []
        for i, result in enumerate(filtered_results):
            try:
                score = getattr(result, 'score', 0.0)
                content = getattr(result, 'content', '')
                if content:
                    retrieval_context_parts.append(f"[结果{i+1} 相关性分数: {score:.3f}]\n{content}\n")
            except Exception as e:
                logger.warning(f"处理结果{i+1}时出错: {str(e)}")
                continue

        retrieval_context = "\n".join(retrieval_context_parts)

        if not retrieval_context.strip():
            logger.warning("所有检索结果内容为空")
            print("警告: 所有检索结果内容为空")
            return "检索结果内容为空"

        logger.info(f"准备的检索上下文长度: {len(retrieval_context)} 字符")
        print(f"\n准备的检索上下文长度: {len(retrieval_context)} 字符")

        # 4. 使用DeepSeek进行信息融合
        logger.info("使用DeepSeek API进行信息融合...")
        print("\n使用DeepSeek API进行信息融合...")
        
        # 如果上下文太长，进行截断
        if len(retrieval_context) > 8000:
            logger.warning(f"检索上下文过长 ({len(retrieval_context)} 字符)，进行截断")
            retrieval_context = retrieval_context[:8000] + "...\n[内容已截断]"
            print(f"警告: 检索上下文过长，已截断至8000字符")

        messages = [
            {
                "role": "system",
                "content": "你是一个专业的信息分析助手。请仔细阅读用户的查询和提供的检索结果，然后：\n1. 提取所有与查询直接相关的关键信息\n2. 忽略不相关的内容\n3. 将相关信息整合成一段连贯的文本\n4. 不要添加任何检索结果中没有的信息\n5. 如果没有相关信息，直接返回'未找到相关信息'"
            },
            {
                "role": "user",
                "content": f"用户查询: {query}\n\n以下是检索到的相关内容，请进行整合：\n{retrieval_context}\n\n请输出整合后的信息："
            }
        ]
        fused_info = self.deepseek.chat_completion(messages)
        if fused_info and fused_info.strip() and fused_info.strip() != "未找到相关信息":
            logger.info(f"信息融合成功，融合后信息长度: {len(fused_info)} 字符")
            print("信息融合成功:")
            print(f"融合后信息长度: {len(fused_info)} 字符")
            print(f"融合后信息: {fused_info[:200]}...")
            return fused_info
        else:
            logger.warning("信息融合失败，使用备用方案")
            print("信息融合失败，使用备用方案")
            
            # 改进的备用方案
            backup_parts = []
            for i, result in enumerate(filtered_results[:5]):
                try:
                    content = getattr(result, 'content', '')
                    if content:
                        backup_parts.append(f"{i+1}. {content}")
                except:
                    continue
            
            backup_info = "\n".join(backup_parts)

            # 如果备用方案仍然为空，提供更明确的错误信息
            if not backup_info.strip():
                backup_info = "检索结果内容为空，可能原因：集合为空、数据格式不匹配、检索服务配置问题"

            logger.info(f"备用方案信息长度: {len(backup_info)} 字符")
            print(f"备用方案信息长度: {len(backup_info)} 字符")
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
        print("\n=== Judge环节 ===")
        print(f"查询: {query}")
        print(f"可用信息长度: {len(fused_info)} 字符")

        # 只对完全空的信息进行过滤，允许短答案
        if not fused_info or fused_info.strip() == "":
            logger.warning("判断结果: 信息为空，不足")
            print("判断结果: 信息为空，不足")
            return False

        # 特殊情况：如果信息非常短但可能是有效答案
        # 例如："是"、"否"、"2023年"等
        if len(fused_info) < 50:
            logger.info(f"注意：信息较短 ({len(fused_info)} 字符)，但仍进行语义判断")
            print(f"注意：信息较短 ({len(fused_info)} 字符)，但仍进行语义判断")

        # 使用LLM进行智能语义判断
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
            print(f"判断结果: {judgment}")

            # 更宽松的结果处理
            if "足够" in judgment or "sufficient" in judgment:
                return True
            elif "不足" in judgment or "insufficient" in judgment:
                return False
            else:
                # 如果返回格式不符合要求，默认认为信息足够，尝试生成响应
                logger.warning("返回格式异常，默认认为信息足够")
                print("返回格式异常，默认认为信息足够")
                return True
        else:
            logger.warning("LLM判断失败，默认认为信息足够，尝试生成响应")
            print("LLM判断失败，默认认为信息足够，尝试生成响应")
            # API调用失败时，默认认为信息足够，不中断流程
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
        print("\n=== 生成响应 ===")

        messages = [
            {
                "role": "system",
                "content": "你是一个专业的问答助手。请根据提供的信息，回答用户的问题。回答要准确、全面、清晰，基于提供的信息，不要添加额外内容。如果信息不足，请说明无法回答。"
            },
            {
                "role": "user",
                "content": f"查询: {query}\n\n可用信息:\n{fused_info}\n\n请基于上述信息回答用户的问题："
            }
        ]

        response = self.deepseek.chat_completion(messages)

        if response:
            logger.info(f"最终响应生成成功，响应长度: {len(response)} 字符")
            print("最终响应:")
            print(response)
            return response
        else:
            logger.error("响应生成失败")
            print("响应生成失败")
            return "抱歉，无法生成响应。"

    def run(self, original_query: str):
        """
        执行完整的RAG流程
        Args:
            original_query: 用户原始查询
        """
        logger.info(f"RAG流程启动，原始查询: {original_query}")
        print("\n" + "="*60)
        print("RAG流程启动")
        print("="*60)

        iteration = 0
        context = None
        final_response = None

        # 实现真正的多轮迭代，支持最多max_iterations轮
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"开始第 {iteration} 轮迭代")
            print(f"\n--- 第 {iteration} 轮迭代 ---")

            # 1. Query理解和改写（使用上下文优化查询）
            rewritten_query = self.understand_and_rewrite_query(original_query, context)

            # 2. 检索文档
            results = self.retrieve_documents(rewritten_query)

            if not results:
                logger.warning("未检索到任何文档")
                print("\n未检索到任何文档，可能的原因：")
                print("1. 可能集合中没有数据或连接失败")
                # 如果是最后一轮迭代，生成最终响应
                if iteration == self.max_iterations:
                    final_response = "抱歉，无法连接到文档数据库或数据库中没有相关数据。"
                else:
                    # 更新上下文，继续下一轮
                    context = f"当前查询: {rewritten_query}\n检索结果: 未找到任何相关文档，请尝试更通用的查询或检查数据库连接。"
                    logger.info(f"第 {iteration} 轮信息不足，更新上下文进行第 {iteration+1} 轮迭代")
                    print(f"\n第 {iteration} 轮信息不足，更新上下文进行第 {iteration+1} 轮迭代")
                    continue  # 继续下一轮迭代

            # 3. summary/排序/过滤/聚合
            fused_info = self.summarize_and_aggregate(rewritten_query, results)

            # 4. Judge环节 - 使用智能语义判断
            logger.info(f"开始Judge环节")
            print("\nJudge环节：")
            if self.judge_information_sufficiency(rewritten_query, fused_info):
                logger.info("信息足够，生成最终响应")
                print("信息足够，生成最终响应")
                # 5. 生成响应
                final_response = self.generate_response(rewritten_query, fused_info)
                break  # 信息足够，退出循环
            else:
                # 信息不足，准备下一轮迭代
                logger.info(f"第 {iteration} 轮信息不足")
                print(f"第 {iteration} 轮信息不足")

                # 如果是最后一轮迭代，生成响应
                if iteration == self.max_iterations:
                    logger.info("已达到最大迭代次数，生成最终响应")
                    print("已达到最大迭代次数，生成最终响应")
                    final_response = self.generate_response(rewritten_query, fused_info)
                    break  # 达到最大迭代次数，退出循环
                else:
                    # 更新上下文，包含当前轮的信息
                    context = f"当前查询: {rewritten_query}\n已检索到的信息: {fused_info[:100]}...\n当前信息不足以完全回答问题，请优化查询以获取更相关的信息。"
                    logger.info(f"更新上下文，进行第 {iteration+1} 轮迭代")
                    print(f"更新上下文，进行第 {iteration+1} 轮迭代")
                    continue  # 继续下一轮完整的RAG流程

        # 确保无论如何都有最终响应
        if not final_response:
            final_response = "抱歉，经过多轮检索，仍无法获取足够的信息来回答您的查询。"

        logger.info(f"RAG流程结束，最终响应: {final_response[:100]}...")
        print("\n" + "="*60)
        print("RAG流程结束")
        print("="*60)

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

        print("\n" + "="*80)
        print("最终回答：")
        print(response)
        print("="*80)