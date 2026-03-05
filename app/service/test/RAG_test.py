#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG系统综合测试脚本 - 修正版
确保按问题类型分析中使用正确的字段名称
"""

import sys
import os
import json
import time
import logging
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

# 获取项目根目录
def get_project_root():
    """获取项目根目录"""
    current_dir = Path(__file__).parent.absolute()

    # 向上查找直到找到包含app目录的根目录
    root = current_dir
    while root.parent != root:
        if (root / "app").exists():
            return root
        root = root.parent

    return current_dir

project_root = get_project_root()
print(f"[INFO] 检测到项目根目录: {project_root}")

# 测试与日志目录（统一放在 app/service/test 下）
test_dir = project_root / "app" / "service" / "test"
test_dir.mkdir(parents=True, exist_ok=True)

# 添加项目根目录到Python路径
sys.path.insert(0, str(project_root))

# 导入RAG流程
try:
    from app.service.test.RAG_flow import RAGFlow
    print("[INFO] 成功导入 RAG_flow")
except ImportError as e:
    # 避免在默认 GBK 控制台下因 emoji 导致编码错误
    print(f"[ERROR] 导入RAG_flow失败: {e}")
    sys.exit(1)

# 配置日志（写入 test 目录）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(test_dir / 'rag_evaluation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CorrectedRAGEvaluator:
    """修正版RAG评估器"""

    def __init__(self):
        # 注意：本评估器仅在离线评测阶段使用 QA.json 中的标准答案和标准 chunk，
        # 这些数据不会作为检索语料或提示内容传入 RAGFlow，只用于计算评估指标。
        self.rag_flow = RAGFlow()

        # 构建正确的测试数据路径（使用 QA.json）
        self.test_data_path = test_dir / "QA.json"
        # 报告与日志保存路径
        self.report_dir = test_dir

        print(f"📁 测试数据路径: {self.test_data_path}")
        print(f"📁 报告保存目录: {self.report_dir}")

        # 验证文件是否存在
        if not self.test_data_path.exists():
            raise FileNotFoundError(f"测试数据文件不存在: {self.test_data_path}")

        if not self.report_dir.exists():
            self.report_dir.mkdir(parents=True, exist_ok=True)

        print("✅ 路径验证通过")
        self.results = []

    def load_test_cases(self) -> List[Dict]:
        """加载测试用例"""
        try:
            logger.info(f"尝试加载测试数据文件: {self.test_data_path}")
            with open(self.test_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"成功加载 {len(data)} 个测试用例")
            return data
        except Exception as e:
            logger.error(f"加载测试数据失败: {e}")
            raise

    def run_evaluation(self):
        """运行完整评估"""
        print("\n🚀 RAG系统综合评估开始")
        print("=" * 60)
        print(f"📁 测试数据路径: {self.test_data_path}")
        print(f"📁 报告保存目录: {self.report_dir}")

        # 加载测试数据
        try:
            test_cases = self.load_test_cases()
        except Exception as e:
            print(f"❌ 无法加载测试数据: {e}")
            return

        print(f"📊 加载了 {len(test_cases)} 个测试用例")

        # 逐个测试
        successful_tests = 0
        failed_tests = 0

        for i, case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] 测试用例 {case['sample_id']}")
            try:
                result = self.evaluate_case(case)
                self.results.append(result)
                successful_tests += 1
                print(f"   ✅ 测试成功")
            except Exception as e:
                logger.error(f"测试用例 {case['sample_id']} 执行失败: {e}")
                failed_tests += 1
                # 添加错误结果（字段名中文化）
                error_result = {
                    'sample_id': case['sample_id'],
                    'query': case.get('query'),
                    '错误信息': str(e),
                    '耗时': 0
                }
                self.results.append(error_result)
                print(f"   ❌ 测试失败: {e}")
                continue

        print(f"\n✅ 成功执行: {successful_tests} 个用例")
        print(f"❌ 执行失败: {failed_tests} 个用例")

        if successful_tests > 0:
            # 生成报告
            self.generate_comprehensive_report()

        print("\n✅ 评估完成！")

    def evaluate_case(self, case: Dict) -> Dict:
        """评估单个测试用例"""
        start_time = time.time()

        # 执行RAG流程，获取完整 trace
        rag_response, trace = self.rag_flow.run(
            case['query'],
            sample_id=case.get('sample_id'),
            return_trace=True
        )
        execution_time = time.time() - start_time

        # 计算评估指标
        metrics = self.calculate_evaluation_metrics(case, rag_response, execution_time, trace)

        # 构建详细结果（字段名中文化，保留指定英文键）
        result: Dict[str, Any] = {
            'sample_id': case['sample_id'],
            'query': case['query'],
            'query_type': case['query_type'],
            'level': case['level'],
            '标准答案': case['ground_truth_answer'],
            '系统回答': rag_response,
            '耗时': execution_time,
            '评估指标': metrics,
            'RAG流程详情': trace,
        }

        # 补充标准chunk和检索到的chunk信息，便于对比
        standard_chunks = []
        for doc in case.get('relevant_documents', []):
            for chunk_text in doc.get('relevant_chunks', []):
                standard_chunks.append({
                    'doc_id': doc.get('doc_id'),
                    'doc_title': doc.get('doc_title'),
                    'chunk文本': chunk_text,
                })

        # 取最终一轮迭代的检索结果
        iterations = trace.get('iterations', []) if isinstance(trace, dict) else []
        final_iter = iterations[-1] if iterations else {}
        retrieval = final_iter.get('retrieval', {}) if isinstance(final_iter, dict) else {}
        retrieved_chunks = retrieval.get('merged_results', []) if isinstance(retrieval, dict) else []

        result['标准chunk信息'] = standard_chunks
        result['检索到的chunk信息'] = retrieved_chunks

        print(f"   ⏱️  耗时: {execution_time:.2f}s")
        auto_correct = metrics.get('答案是否正确', False)
        print(f"   📊 自动评估: {'✅ 正确' if auto_correct else '❌ 错误'}")

        return result

    def _build_gt_chunks(self, case: Dict) -> List[Dict[str, Any]]:
        """从 QA.json 中构建标准 chunk 列表。"""
        gt_chunks: List[Dict[str, Any]] = []
        for doc in case.get('relevant_documents', []):
            doc_id = doc.get('doc_id')
            doc_title = doc.get('doc_title')
            for text in doc.get('relevant_chunks', []):
                gt_chunks.append(
                    {
                        'doc_id': doc_id,
                        'doc_title': doc_title,
                        'text': str(text),
                    }
                )
        return gt_chunks

    def _build_retrieved_chunks_from_trace(self, trace: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        从 RAG 流程 trace 中提取最终一轮的检索结果（chunk 级信息）。
        仅取前 top_k 个，以控制评估成本。
        """
        if not isinstance(trace, dict):
            return []
        iterations = trace.get('iterations', [])
        if not iterations:
            return []
        final_iter = iterations[-1]
        retrieval = final_iter.get('retrieval', {}) if isinstance(final_iter, dict) else {}
        merged_results = retrieval.get('merged_results', []) if isinstance(retrieval, dict) else []
        return merged_results[:top_k]

    def _judge_chunk_match_with_llm(self, gt_chunk: str, retrieved_chunk: str) -> Dict[str, Any]:
        """
        利用 DeepSeek 判断两个 chunk 文本是否指向原论文中的同一片段。
        返回 {is_hit: bool, raw_decision: str}
        """
        system_prompt = (
            "你是一个学术论文内容对齐助手。现在给出两段学术文本片段，"
            "请判断它们是否在描述同一篇论文中的同一个具体内容片段（同一事实/同一结论/同一表格或图像的描述）。\n"
            "判断时请做到：\n"
            "1. 只要两段文本在描述的对象、数值（允许极小舍入误差）、关系和结论一致，即使标点符号不同、"
            "   中文/英文混用、语序略有调整、前后多了少量上下文，也应判定为“命中”。\n"
            "2. 如果两段文本虽然来自同一篇论文，但描述的是明显不同的位置或完全不同的结论，才判定为“未命中”。\n"
            "3. 不要因为括号、顿号、单位写法等细枝末节差异而判定为“未命中”。\n"
            "4. 不要进行主观臆测，仍需确保核心事实一致时才视为命中。\n"
            "5. 最终只输出“命中”或“未命中”四个字，不要添加任何解释或其他文字。"
        )
        user_prompt = (
            "标准chunk文本：\n"
            f"{gt_chunk}\n\n"
            "检索到的chunk文本（可能被截断）：\n"
            f"{retrieved_chunk}\n\n"
            "请根据上述要求判断这两个片段是否在描述同一个具体内容片段，只输出“命中”或“未命中”："
        )

        try:
            decision = self.rag_flow.deepseek.chat_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except Exception as e:
            logger.error(f"调用DeepSeek进行chunk对齐判断失败: {e}")
            return {"is_hit": False, "raw_decision": f"调用失败: {e}"}

        if not decision:
            return {"is_hit": False, "raw_decision": ""}

        decision_str = str(decision).strip()
        normalized = decision_str.replace(" ", "")
        is_hit = "命中" in normalized and "未命中" not in normalized
        return {"is_hit": is_hit, "raw_decision": decision_str}

    def _calculate_chunk_level_metrics(
        self,
        gt_chunks: List[Dict[str, Any]],
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        基于 chunk 级别计算召回率 / 精确率 / F1，并记录详细对齐结果。
        """
        gt_total = len(gt_chunks)
        if not gt_chunks or not retrieved_chunks:
            return {
                "chunk召回率": 0.0,
                "chunk精确率": 0.0,
                "chunkF1": 0.0,
                "chunk级对齐详情": [],
                "命中标准chunk数量": 0,
                "标准chunk总数": gt_total,
                "Hit@K": 0.0,
                "MRR": 0.0,
            }

        align_details: List[Dict[str, Any]] = []
        hit_gt_indices = set()
        hit_retrieved_indices = set()
        hit_ranks = []

        for gt_idx, gt in enumerate(gt_chunks):
            gt_text = gt["text"]
            for r_idx, r in enumerate(retrieved_chunks):
                # 评估阶段优先使用完整chunk文本；若不存在则回退到预览文本
                retrieved_text = str(r.get("content_full") or r.get("content_preview", ""))
                judge_result = self._judge_chunk_match_with_llm(gt_text, retrieved_text)
                is_hit = judge_result["is_hit"]
                detail = {
                    "标准chunk编号": gt_idx,
                    "doc_id": gt.get("doc_id"),
                    "doc_title": gt.get("doc_title"),
                    "标准chunk文本": gt_text,
                    "检索chunk排序": r_idx + 1,
                    "检索chunk_doc_id": r.get("doc_id"),
                    "检索chunk_doc_title": r.get("doc_title"),
                    "检索chunk预览": retrieved_text,
                    "LLM判定结果": judge_result["raw_decision"],
                    "是否命中": is_hit,
                }
                align_details.append(detail)
                if is_hit:
                    hit_gt_indices.add(gt_idx)
                    hit_retrieved_indices.add(r_idx)
                    hit_ranks.append(r_idx + 1)
                    # 对于每个标准chunk，一旦找到一个命中即可停止继续判断，避免额外开销
                    break

        retrieved_total = len(retrieved_chunks)
        hit_gt_count = len(hit_gt_indices)
        hit_retrieved_count = len(hit_retrieved_indices)

        recall = hit_gt_count / gt_total if gt_total > 0 else 0.0
        precision = hit_retrieved_count / retrieved_total if retrieved_total > 0 else 0.0
        if recall + precision > 0:
            f1 = 2 * recall * precision / (recall + precision)
        else:
            f1 = 0.0

        # Hit@K 与 MRR（以参与评估的检索条数 K 为上界）
        if hit_ranks:
            hit_at_k = 1.0
            first_hit_rank = min(hit_ranks)
            mrr = 1.0 / first_hit_rank
        else:
            hit_at_k = 0.0
            first_hit_rank = 0
            mrr = 0.0

        return {
            "chunk召回率": round(recall, 3),
            "chunk精确率": round(precision, 3),
            "chunkF1": round(f1, 3),
            "chunk级对齐详情": align_details,
            "命中标准chunk数量": hit_gt_count,
            "标准chunk总数": gt_total,
            "Hit@K": round(hit_at_k, 3),
            "MRR": round(mrr, 3),
            "首命中rank": first_hit_rank,
        }

    def calculate_evaluation_metrics(self, case: Dict, response: str, exec_time: float, trace: Dict) -> Dict:
        """计算完整的评估指标（对齐图片中的指标体系）"""
        ground_truth_raw = str(case['ground_truth_answer'])
        response_raw = str(response)
        ground_truth = ground_truth_raw.lower().strip()
        rag_answer = response_raw.lower().strip()

        # 1. 基础准确率与 EM
        auto_correct = self.check_accuracy(ground_truth, rag_answer)
        em_score = self._compute_em(ground_truth_raw, response_raw)

        # 2. chunk 级检索指标（召回率 / 精确率 / F1 / Hit@K / MRR）
        gt_chunks = self._build_gt_chunks(case)
        retrieved_chunks = self._build_retrieved_chunks_from_trace(trace, top_k=10)
        chunk_metrics = self._calculate_chunk_level_metrics(gt_chunks, retrieved_chunks)

        # 3. 多跳推理得分 MultiHopScore（仅对“多跳推理型”计算）
        qtype = case.get('query_type', '')
        align_details = chunk_metrics.get("chunk级对齐详情", [])
        # 统计有效证据 chunk：命中的标准 chunk 数量
        effective_evidence_count = chunk_metrics.get("命中标准chunk数量", 0)
        multihop_score = 0.0
        multihop_R = 0.0
        multihop_E = 0.0
        multihop_A = 0.0
        if qtype == "多跳推理型" and effective_evidence_count >= 2:
            evidence_texts = []
            for d in align_details:
                if d.get("是否命中"):
                    std_text = str(d.get("标准chunk文本", ""))
                    ret_text = str(d.get("检索chunk预览", ""))
                    evidence_texts.append(
                        f"[标准chunk] {std_text}\n[检索chunk] {ret_text}"
                    )
            scores = self._llm_score_multihop(
                case, response_raw, "\n\n".join(evidence_texts)
            )
            multihop_R = scores.get("R", 0.0)
            multihop_E = scores.get("E", 0.0)
            multihop_A = scores.get("A", 0.0)
            multihop_score = 0.3 * multihop_R + 0.3 * multihop_E + 0.4 * multihop_A

        # 4. 多论文推理得分 MultiDocScore（仅对“多论文推理型”计算）
        multidoc_score = 0.0
        multidoc_R = 0.0
        multidoc_E = 0.0
        multidoc_A = 0.0
        if qtype == "多论文推理型":
            used_doc_ids = set()
            for d in align_details:
                if d.get("是否命中"):
                    doc_id = d.get("doc_id") or d.get("检索chunk_doc_id")
                    if doc_id is not None:
                        used_doc_ids.add(str(doc_id))
            if len(used_doc_ids) >= 2:
                # 基于命中文档构造证据摘要
                doc_to_snippets: Dict[str, List[str]] = {}
                for d in align_details:
                    if not d.get("是否命中"):
                        continue
                    doc_id = str(d.get("doc_id") or d.get("检索chunk_doc_id") or "")
                    if not doc_id:
                        continue
                    text = str(d.get("标准chunk文本") or d.get("检索chunk预览") or "")
                    doc_to_snippets.setdefault(doc_id, []).append(text)
                doc_summaries = []
                for doc_id, snippets in doc_to_snippets.items():
                    joined = "\n".join(snippets[:3])
                    doc_summaries.append(f"[文档 {doc_id}] 关键证据片段:\n{joined}")
                scores = self._llm_score_multi_doc(
                    case, response_raw, "\n\n".join(doc_summaries)
                )
                multidoc_R = scores.get("R", 0.0)
                multidoc_E = scores.get("E", 0.0)
                multidoc_A = scores.get("A", 0.0)
                multidoc_score = 0.3 * multidoc_R + 0.3 * multidoc_E + 0.4 * multidoc_A

        metrics: Dict[str, Any] = {}
        metrics.update(chunk_metrics)
        metrics.update(
            {
                "答案是否正确": auto_correct,
                "EM": em_score,
                "MultiHopScore": round(multihop_score, 3),
                "MultiHop_R": round(multihop_R, 3),
                "MultiHop_E": round(multihop_E, 3),
                "MultiHop_A": round(multihop_A, 3),
                "MultiDocScore": round(multidoc_score, 3),
                "MultiDoc_R": round(multidoc_R, 3),
                "MultiDoc_E": round(multidoc_E, 3),
                "MultiDoc_A": round(multidoc_A, 3),
                "耗时": exec_time,
            }
        )
        return metrics

    def _compute_em(self, truth: str, response: str) -> int:
        """
        严格的 EM（Exact Match）指标：
        - 预处理后完全一致记为 1，否则 0。
        """
        def _normalize(s: str) -> str:
            import re

            s = s.strip().lower()
            # 全角转半角的简单处理
            s = s.replace("％", "%").replace("×", "x")
            # 去掉常见标点和空白
            s = re.sub(r"[，,。．\.！？!?\s]", "", s)
            return s

        if not truth or not response:
            return 0
        return 1 if _normalize(truth) == _normalize(response) else 0

    def _llm_score_multihop(self, case: Dict, response: str, evidence: str) -> Dict[str, float]:
        """
        让 LLM 对多跳推理题进行 R/E/A 打分，返回形如 {\"R\":0-1, \"E\":0-1, \"A\":0-1}。
        """
        system_prompt = (
            "你是一个学术多跳推理评估助手。现在需要根据问题、标准答案、系统回答以及若干证据片段，"
            "从三个维度对系统回答进行 0~1 区间的打分：\\n"
            "R：推理链条是否完整、逻辑是否正确；\\n"
            "E：是否充分利用并正确整合了提供的证据；\\n"
            "A：最终答案是否与标准答案在关键结论上一致。\\n"
            "请只输出一个 JSON 对象，例如 {\"R\":0.8,\"E\":0.9,\"A\":1.0}。"
        )
        user_prompt = (
            f"问题：{case.get('query', '')}\\n"
            f"标准答案：{case.get('ground_truth_answer', '')}\\n"
            f"系统回答：{response}\\n"
            f"证据片段：\\n{evidence}\\n\\n"
            "请给出 R/E/A 三个 0~1 的分数，并只输出 JSON："
        )

        try:
            decision = self.rag_flow.deepseek.chat_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            text = str(decision).strip()
            import json

            scores = json.loads(text)
        except Exception as e:
            logger.error(f"调用DeepSeek进行多跳推理评分失败: {e}")
            return {"R": 0.0, "E": 0.0, "A": 0.0}

        def _clamp(v: Any) -> float:
            try:
                x = float(v)
            except Exception:
                return 0.0
            if x < 0.0:
                return 0.0
            if x > 1.0:
                return 1.0
            return x

        return {
            "R": _clamp(scores.get("R", 0.0)),
            "E": _clamp(scores.get("E", 0.0)),
            "A": _clamp(scores.get("A", 0.0)),
        }

    def _llm_score_multi_doc(self, case: Dict, response: str, evidence_by_doc: str) -> Dict[str, float]:
        """
        让 LLM 对多论文推理题进行 R/E/A 打分，返回形如 {\"R\":0-1, \"E\":0-1, \"A\":0-1}。
        """
        system_prompt = (
            "你是一个多论文推理评估助手。现在给出问题、标准答案、系统回答，以及按文档聚合的证据摘要。"
            "请从三个维度在 0~1 区间为系统回答打分：\\n"
            "R：是否检索并利用了足够多篇相关论文；\\n"
            "E：是否正确整合了不同论文中的证据；\\n"
            "A：最终结论是否与标准答案以及多篇论文的证据一致。\\n"
            "请只输出一个 JSON 对象，例如 {\"R\":0.7,\"E\":0.8,\"A\":0.9}。"
        )
        user_prompt = (
            f"问题：{case.get('query', '')}\\n"
            f"标准答案：{case.get('ground_truth_answer', '')}\\n"
            f"系统回答：{response}\\n"
            f"多篇论文证据摘要：\\n{evidence_by_doc}\\n\\n"
            "请给出 R/E/A 三个 0~1 的分数，并只输出 JSON："
        )

        try:
            decision = self.rag_flow.deepseek.chat_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            text = str(decision).strip()
            import json

            scores = json.loads(text)
        except Exception as e:
            logger.error(f"调用DeepSeek进行多论文推理评分失败: {e}")
            return {"R": 0.0, "E": 0.0, "A": 0.0}

        def _clamp(v: Any) -> float:
            try:
                x = float(v)
            except Exception:
                return 0.0
            if x < 0.0:
                return 0.0
            if x > 1.0:
                return 1.0
            return x

        return {
            "R": _clamp(scores.get("R", 0.0)),
            "E": _clamp(scores.get("E", 0.0)),
            "A": _clamp(scores.get("A", 0.0)),
        }

    def check_accuracy(self, truth: str, response: str) -> bool:
        """检查回答准确性"""
        if not truth or not response:
            return False

        # 完全匹配
        if truth in response or response in truth:
            return True

        # 数值匹配
        import re
        truth_nums = re.findall(r'\d+\.?\d*', truth)
        response_nums = re.findall(r'\d+\.?\d*', response)

        if truth_nums and response_nums:
            common_numbers = set(truth_nums) & set(response_nums)
            if common_numbers:
                return True

        return False

    def assess_completeness(self, query: str, truth: str, response: str) -> float:
        """
        评估答案在“文本层面”的完整性 (0-1)，只考虑数值与单位覆盖度。
        最终答案完整性会在 calculate_evaluation_metrics 中叠加 chunk 召回率。
        """
        import re

        # 基础分：回答非空
        if not response.strip():
            return 0.0
        score = 0.3

        # 数值覆盖：标准答案与回答中至少有一个共同数字
        truth_nums = re.findall(r'\d+\.?\d*', truth)
        resp_nums = re.findall(r'\d+\.?\d*', response)
        common_nums = set(truth_nums) & set(resp_nums)
        if truth_nums and resp_nums and common_nums:
            score += 0.4

        # 单位覆盖：至少一个共同单位
        units = ['倍', '百分点', '%', '张', '次', '层', 'epoch', 'MB', 'kb', 'k']
        if any(u in truth and u in response for u in units):
            score += 0.3

        return min(1.0, score)

    def check_consistency(self, truth: str, response: str) -> float:
        """
        检查数值层面的事实一致性 Sn (0-1)：
        - 无数字 → 视为 1.0
        - 有数字 → 统计标准答案中的数字，在回答中能找到相对误差≤20%的匹配比例。
        """
        import re
        truth_nums = re.findall(r'\d+\.?\d*', truth)
        response_nums = re.findall(r'\d+\.?\d*', response)

        if not truth_nums:
            return 1.0
        if not response_nums:
            return 0.0

        matched = 0
        for t in truth_nums:
            t_val = float(t)
            local_match = False
            for r in response_nums:
                r_val = float(r)
                if max(t_val, r_val) == 0:
                    if t_val == r_val:
                        local_match = True
                        break
                else:
                    if abs(t_val - r_val) / max(t_val, r_val) <= 0.2:
                        local_match = True
                        break
            if local_match:
                matched += 1

        return matched / len(truth_nums) if truth_nums else 1.0

    def generate_comprehensive_report(self):
        """生成综合评估报告"""
        # 统计汇总
        valid_results = [r for r in self.results if '错误信息' not in r]
        total_cases = len(valid_results)

        if total_cases == 0:
            print("没有有效的测试结果")
            return

        auto_correct = len([r for r in valid_results if r.get('评估指标', {}).get('答案是否正确')])

        # 计算平均指标
        def _avg(key: str) -> float:
            values = [
                r['评估指标'][key]
                for r in valid_results
                if '评估指标' in r and key in r['评估指标']
            ]
            if not values:
                return 0.0
            return round(sum(values) / len(values), 3)

        avg_metrics = {
            '平均chunk召回率': _avg('chunk召回率'),
            '平均chunk精确率': _avg('chunk精确率'),
            '平均chunkF1': _avg('chunkF1'),
            '平均Hit@K': _avg('Hit@K'),
            '平均MRR': _avg('MRR'),
            '平均EM': _avg('EM'),
            '平均MultiHopScore': _avg('MultiHopScore'),
            '平均MultiDocScore': _avg('MultiDocScore'),
            '平均响应时间': round(
                sum(r['评估指标'].get('耗时', 0.0) for r in valid_results) / total_cases, 2
            ),
        }

        # 按问题类型统计（使用正确的字段名称）
        type_stats = self.analyze_by_query_type_with_correct_names(valid_results)

        # 构建完整报告
        report = {
            '评估基本信息': {
                '评估时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                '测试数据路径': str(self.test_data_path),
                '报告保存目录': str(self.report_dir),
                '测试用例总数': total_cases,
                '自动判定的问答匹配率': f"{auto_correct / total_cases:.1%}"
            },
            '详细指标说明': self.get_structured_metric_explanations(),
            '平均性能指标': avg_metrics,
            '按问题类型分析': type_stats,
            '详细测试结果': self.results
        }

        # 保存JSON报告到指定目录
        report_path = self.report_dir / 'rag_evaluation_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 生成静态HTML报告
        html_path = self.report_dir / 'rag_evaluation_report.html'
        html_content = self._build_html_report(report)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"\n💾 评估JSON报告已保存到: {report_path}")
        print(f"💾 评估HTML报告已保存到: {html_path}")

        # 打印摘要
        self.print_evaluation_summary(report)

    def _build_html_report(self, report: Dict[str, Any]) -> str:
        """基于JSON报告生成适合人眼阅读的详细HTML报告。"""
        eval_info = report.get('评估基本信息', {})
        avg_metrics = report.get('平均性能指标', {})
        type_stats = report.get('按问题类型分析', {})
        details = report.get('详细测试结果', [])

        def esc(s: Any) -> str:
            return (
                str(s)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

        html_parts = []
        html_parts.append("<!DOCTYPE html>")
        html_parts.append("<html lang='zh-CN'>")
        html_parts.append("<head>")
        html_parts.append("<meta charset='utf-8' />")
        html_parts.append("<title>RAG 评估报告</title>")
        html_parts.append(
            "<style>"
            "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:16px;}"
            "h1,h2,h3{margin-top:16px;}"
            "table{border-collapse:collapse;width:100%;margin:8px 0;}"
            "th,td{border:1px solid #ddd;padding:4px 8px;font-size:13px;vertical-align:top;}"
            "th{background:#f5f5f5;text-align:left;}"
            "details{margin:8px 0;border:1px solid #ddd;border-radius:4px;padding:4px 8px;}"
            "summary{font-weight:bold;cursor:pointer;}"
            ".hit{background:#e6ffed;}"
            ".miss{background:#ffecec;}"
            ".metric-badge{display:inline-block;padding:2px 6px;border-radius:4px;background:#f5f5f5;margin-right:4px;font-size:12px;}"
            ".section-header{margin-top:24px;border-bottom:2px solid #ddd;padding-bottom:4px;}"
            "</style>"
        )
        html_parts.append("</head>")
        html_parts.append("<body>")

        # 概览
        html_parts.append("<h1>RAG 系统评估报告</h1>")
        html_parts.append("<div>")
        html_parts.append(f"<div class='metric-badge'>评估时间：{esc(eval_info.get('评估时间', ''))}</div>")
        html_parts.append(f"<div class='metric-badge'>测试用例总数：{esc(eval_info.get('测试用例总数', ''))}</div>")
        html_parts.append(
            f"<div class='metric-badge'>自动判定的问答匹配率：{esc(eval_info.get('自动判定的问答匹配率', ''))}</div>"
        )
        html_parts.append("</div>")

        # 平均指标
        html_parts.append("<h2 class='section-header'>整体平均指标</h2>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>指标</th><th>数值</th></tr>")
        for k in [
            "平均chunk召回率",
            "平均chunk精确率",
            "平均chunkF1",
            "平均Hit@K",
            "平均MRR",
            "平均EM",
            "平均MultiHopScore",
            "平均MultiDocScore",
            "平均响应时间",
        ]:
            if k in avg_metrics:
                html_parts.append(f"<tr><td>{esc(k)}</td><td>{esc(avg_metrics[k])}</td></tr>")
        html_parts.append("</table>")

        # 按问题类型分析
        if type_stats:
            html_parts.append("<h2 class='section-header'>按问题类型分析</h2>")
            html_parts.append("<table>")
            html_parts.append(
                "<tr><th>问题类型</th><th>数量</th><th>自动判定的问答匹配率</th>"
                "<th>平均chunk召回率</th><th>平均chunk精确率</th><th>平均chunkF1</th><th>平均耗时</th></tr>"
            )
            for qtype, stats in type_stats.items():
                html_parts.append(
                    "<tr>"
                    f"<td>{esc(qtype)}</td>"
                    f"<td>{esc(stats.get('数量', ''))}</td>"
                    f"<td>{esc(stats.get('自动判定的问答匹配率', ''))}</td>"
                    f"<td>{esc(stats.get('平均chunk召回率', ''))}</td>"
                    f"<td>{esc(stats.get('平均chunk精确率', ''))}</td>"
                    f"<td>{esc(stats.get('平均chunkF1', ''))}</td>"
                    f"<td>{esc(stats.get('平均耗时', ''))}</td>"
                    "</tr>"
                )
            html_parts.append("</table>")

        # 详细用例
        html_parts.append("<h2 class='section-header'>详细测试结果</h2>")
        for item in details:
            sample_id = item.get("sample_id", "")
            query = item.get("query", "")
            query_type = item.get("query_type", "")
            level = item.get("level", "")
            metrics = item.get("评估指标", {})

            summary_title = f"sample_id={sample_id} | level={level} | type={query_type}"
            html_parts.append("<details>")
            html_parts.append(f"<summary>{esc(summary_title)}</summary>")

            # 基本信息
            html_parts.append("<h3>基本信息</h3>")
            html_parts.append("<table>")
            html_parts.append("<tr><th>字段</th><th>内容</th></tr>")
            html_parts.append(f"<tr><td>sample_id</td><td>{esc(sample_id)}</td></tr>")
            html_parts.append(f"<tr><td>query</td><td>{esc(query)}</td></tr>")
            html_parts.append(f"<tr><td>query_type</td><td>{esc(query_type)}</td></tr>")
            html_parts.append(f"<tr><td>level</td><td>{esc(level)}</td></tr>")
            html_parts.append(f"<tr><td>标准答案</td><td>{esc(item.get('标准答案', ''))}</td></tr>")
            html_parts.append(f"<tr><td>系统回答</td><td>{esc(item.get('系统回答', ''))}</td></tr>")
            html_parts.append(f"<tr><td>耗时(秒)</td><td>{esc(round(item.get('耗时', 0.0), 3))}</td></tr>")
            html_parts.append("</table>")

            # 指标信息
            html_parts.append("<h3>评估指标</h3>")
            html_parts.append("<table>")
            html_parts.append("<tr><th>指标</th><th>数值</th></tr>")
            for k, v in metrics.items():
                if k == "chunk级对齐详情":
                    continue
                html_parts.append(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>")
            html_parts.append("</table>")

            # 标准chunk信息
            std_chunks = item.get("标准chunk信息", [])
            if std_chunks:
                html_parts.append("<h3>标准chunk信息</h3>")
                html_parts.append("<table>")
                html_parts.append("<tr><th>doc_id</th><th>doc_title</th><th>标准chunk文本</th></tr>")
                for c in std_chunks:
                    html_parts.append(
                        "<tr>"
                        f"<td>{esc(c.get('doc_id', ''))}</td>"
                        f"<td>{esc(c.get('doc_title', ''))}</td>"
                        f"<td>{esc(c.get('chunk文本', ''))}</td>"
                        "</tr>"
                    )
                html_parts.append("</table>")

            # 检索到的chunk信息
            retrieved_chunks = item.get("检索到的chunk信息", [])
            if retrieved_chunks:
                html_parts.append("<h3>检索到的chunk信息（来自RAG流程trace的最终一轮）</h3>")
                html_parts.append("<table>")
                html_parts.append("<tr><th>rank</th><th>doc_id</th><th>doc_title</th><th>score</th><th>chunk预览</th></tr>")
                for c in retrieved_chunks:
                    html_parts.append(
                        "<tr>"
                        f"<td>{esc(c.get('rank', ''))}</td>"
                        f"<td>{esc(c.get('doc_id', ''))}</td>"
                        f"<td>{esc(c.get('doc_title', ''))}</td>"
                        f"<td>{esc(c.get('score', ''))}</td>"
                        f"<td>{esc(c.get('content_preview', ''))}</td>"
                        "</tr>"
                    )
                html_parts.append("</table>")

            # chunk级对齐详情
            align_details = metrics.get("chunk级对齐详情", [])
            if align_details:
                html_parts.append("<h3>chunk级对齐详情</h3>")
                html_parts.append("<table>")
                html_parts.append(
                    "<tr>"
                    "<th>标准chunk编号</th><th>doc_id</th><th>doc_title</th><th>标准chunk文本</th>"
                    "<th>检索chunk排序</th><th>检索chunk_doc_id</th><th>检索chunk_doc_title</th>"
                    "<th>检索chunk预览</th><th>LLM判定结果</th><th>是否命中</th>"
                    "</tr>"
                )
                for d in align_details:
                    hit = d.get("是否命中", False)
                    row_class = "hit" if hit else "miss"
                    html_parts.append(
                        f"<tr class='{row_class}'>"
                        f"<td>{esc(d.get('标准chunk编号', ''))}</td>"
                        f"<td>{esc(d.get('doc_id', ''))}</td>"
                        f"<td>{esc(d.get('doc_title', ''))}</td>"
                        f"<td>{esc(d.get('标准chunk文本', ''))}</td>"
                        f"<td>{esc(d.get('检索chunk排序', ''))}</td>"
                        f"<td>{esc(d.get('检索chunk_doc_id', ''))}</td>"
                        f"<td>{esc(d.get('检索chunk_doc_title', ''))}</td>"
                        f"<td>{esc(d.get('检索chunk预览', ''))}</td>"
                        f"<td>{esc(d.get('LLM判定结果', ''))}</td>"
                        f"<td>{esc('命中' if hit else '未命中')}</td>"
                        "</tr>"
                    )
                html_parts.append("</table>")

            # 可选：原始JSON
            html_parts.append("<details>")
            html_parts.append("<summary>查看此用例的原始JSON</summary>")
            html_parts.append("<pre style='white-space:pre-wrap;font-size:12px;'>")
            html_parts.append(esc(json.dumps(item, ensure_ascii=False, indent=2)))
            html_parts.append("</pre>")
            html_parts.append("</details>")

            html_parts.append("</details>")

        html_parts.append("</body></html>")
        return "".join(html_parts)

    def get_structured_metric_explanations(self) -> Dict[str, Any]:
        """获取结构化的指标说明"""
        return {
            '检索阶段指标': {
                'chunk召回率': {
                    '定义': '在标准答案给出的所有标准chunk中，有多少比例被检索结果成功覆盖。',
                    '计算公式': '被判定为命中的标准chunk数量 / 标准chunk总数量',
                    '理想值': '1.0 (100%)',
                    '当前实现': '基于DeepSeek对齐判断的chunk级召回率',
                    '说明': '衡量检索是否找到了应当包含答案的那些具体文本片段。'
                },
                'chunk精确率': {
                    '定义': '在检索返回的chunk中，有多少比例真正对应标准答案中的chunk。',
                    '计算公式': '被判定为命中的检索chunk数量 / 参与评估的检索chunk总数量',
                    '理想值': '1.0 (100%)',
                    '当前实现': '基于DeepSeek对齐判断的chunk级精确率',
                    '说明': '反映检索结果的“干净程度”，无关chunk越少越好。'
                },
                'chunkF1': {
                    '定义': 'chunk召回率和chunk精确率的调和平均数。',
                    '计算公式': 'F1 = 2 * 召回率 * 精确率 / (召回率 + 精确率)',
                    '理想值': '1.0',
                    '当前实现': '基于上述chunk召回率与chunk精确率计算得到的综合指标',
                    '说明': '综合考虑检索的“找全”和“找准”。'
                },
                'Hit@K': {
                    '定义': '在 Top-K 检索结果中是否至少出现一个命中标准chunk。',
                    '计算公式': '若存在命中标准chunk则为1，否则为0；当前实现中 K=10。',
                    '理想值': '1.0',
                    '当前实现': '基于最终一轮检索结果（前10个chunk）的对齐情况计算 Hit@10',
                    '说明': '衡量检索在前若干条结果中是否已经覆盖到至少一个关键证据。'
                },
                'MRR': {
                    '定义': '基于首个命中标准chunk排序位置的平均倒数排名 (Mean Reciprocal Rank)。',
                    '计算公式': '若首个命中标准chunk的rank为 r，则 MRR = 1/r；若无命中则为0。',
                    '理想值': '1.0',
                    '当前实现': '基于最终一轮检索结果中首个命中标准chunk的rank计算',
                    '说明': '反映相关证据在检索结果列表中的靠前程度。'
                }
            },
            '生成阶段指标': {
                '答案是否正确': {
                    '定义': '基于较宽松的规则自动判断生成答案是否与标准答案一致。',
                    '计算公式': '布尔值判断 (正确=1, 错误=0)',
                    '判断方法': [
                        '文本包含关系',
                        '数字是否有交集'
                    ],
                    '理想值': '1.0 (100%)',
                    '当前实现': '用于快速自动打标，仍建议人工抽查复核'
                },
                'EM': {
                    '定义': 'Exact Match，严格意义上的答案完全匹配度。',
                    '计算公式': '对标准答案与系统回答做归一化（大小写、空格、常见标点），若字符串完全一致则记为1，否则为0。',
                    '理想值': '1.0',
                    '当前实现': '作为严格的自动 EM 指标，可视作 EM@1'
                },
                'MultiHopScore': {
                    '定义': '针对“多跳推理型”问题，评估系统在多步推理任务上的整体表现。',
                    '计算公式': 'MultiHopScore = 0.3*R + 0.3*E + 0.4*A',
                    '评分细则': {
                        'R': '推理链条的完整性与逻辑正确性（0~1，由 LLM 依据证据给出）',
                        'E': '是否充分、正确地整合了多段证据（0~1，由 LLM 给出）',
                        'A': '最终答案是否与标准答案在关键结论上一致（0~1，由 LLM 给出）',
                        '有效性条件': '仅当命中的标准chunk数量不少于2且问题类型为“多跳推理型”时才计算该指标，否则为0'
                    },
                    '理想值': '1.0',
                    '说明': '反映系统在需要跨多个证据片段进行推理时的整体能力。'
                },
                'MultiDocScore': {
                    '定义': '针对“多论文推理型”问题，评估系统在整合多篇论文证据时的推理与回答质量。',
                    '计算公式': 'MultiDocScore = 0.3*R + 0.3*E + 0.4*A',
                    '评分细则': {
                        'R': '是否检索并利用了足够多篇相关论文（0~1，由 LLM 给出）',
                        'E': '是否正确整合了不同论文中的关键信息与证据（0~1，由 LLM 给出）',
                        'A': '最终结论是否与标准答案及多篇论文的证据一致（0~1，由 LLM 给出）',
                        '有效性条件': '仅当命中证据涉及不少于2个不同文档时才计算该指标，否则为0'
                    },
                    '理想值': '1.0',
                    '说明': '衡量系统在跨论文整合与比较分析场景下的表现。'
                },
            },
            '整体性能指标': {
                '平均响应时间': {
                    '定义': '系统处理单个查询的平均耗时',
                    '计算公式': '所有测试用例执行时间总和 / 测试用例数量',
                    '单位': '秒',
                    '说明': '衡量系统的响应效率和性能'
                }
            }
        }

    def analyze_by_query_type_with_correct_names(self, results: List[Dict]) -> Dict:
        """按问题类型分析性能 - 使用正确的字段名称"""
        type_stats = {}

        for result in results:
            query_type = result['query_type']
            if query_type not in type_stats:
                type_stats[query_type] = {
                    '数量': 0,
                    '自动判定正确数量': 0,
                    '总耗时': 0.0,
                    'chunk召回率汇总': 0.0,
                    'chunk精确率汇总': 0.0,
                    'chunkF1汇总': 0.0,
                }

            stats = type_stats[query_type]
            stats['数量'] += 1
            metrics = result.get('评估指标', {})
            if metrics.get('答案是否正确'):
                stats['自动判定正确数量'] += 1
            stats['总耗时'] += metrics.get('耗时', 0.0)
            stats['chunk召回率汇总'] += metrics.get('chunk召回率', 0.0)
            stats['chunk精确率汇总'] += metrics.get('chunk精确率', 0.0)
            stats['chunkF1汇总'] += metrics.get('chunkF1', 0.0)
            # 新增指标汇总
            stats.setdefault('Hit@K汇总', 0.0)
            stats.setdefault('MRR汇总', 0.0)
            stats.setdefault('EM汇总', 0.0)
            stats.setdefault('MultiHopScore汇总', 0.0)
            stats.setdefault('MultiDocScore汇总', 0.0)
            stats['Hit@K汇总'] += metrics.get('Hit@K', 0.0)
            stats['MRR汇总'] += metrics.get('MRR', 0.0)
            stats['EM汇总'] += metrics.get('EM', 0.0)
            stats['MultiHopScore汇总'] += metrics.get('MultiHopScore', 0.0)
            stats['MultiDocScore汇总'] += metrics.get('MultiDocScore', 0.0)

        # 计算比率，使用正确的字段名
        for query_type, stats in type_stats.items():
            count = stats['数量']
            if count > 0:
                stats['自动判定的问答匹配率'] = f"{stats['自动判定正确数量'] / count:.1%}"
                stats['平均chunk召回率'] = round(stats['chunk召回率汇总'] / count, 3)
                stats['平均chunk精确率'] = round(stats['chunk精确率汇总'] / count, 3)
                stats['平均chunkF1'] = round(stats['chunkF1汇总'] / count, 3)
                stats['平均Hit@K'] = round(stats.get('Hit@K汇总', 0.0) / count, 3)
                stats['平均MRR'] = round(stats.get('MRR汇总', 0.0) / count, 3)
                stats['平均EM'] = round(stats.get('EM汇总', 0.0) / count, 3)
                stats['平均MultiHopScore'] = round(stats.get('MultiHopScore汇总', 0.0) / count, 3)
                stats['平均MultiDocScore'] = round(stats.get('MultiDocScore汇总', 0.0) / count, 3)
                stats['平均耗时'] = f"{stats['总耗时'] / count:.2f}秒"

            # 清理中间累加字段
            stats.pop('chunk召回率汇总', None)
            stats.pop('chunk精确率汇总', None)
            stats.pop('chunkF1汇总', None)
            stats.pop('Hit@K汇总', None)
            stats.pop('MRR汇总', None)
            stats.pop('EM汇总', None)
            stats.pop('MultiHopScore汇总', None)
            stats.pop('MultiDocScore汇总', None)

        return type_stats

    def print_evaluation_summary(self, report: Dict):
        """打印评估摘要"""
        eval_info = report['评估基本信息']
        metrics = report['平均性能指标']

        print("\n" + "=" * 60)
        print("📈 RAG系统评估结果摘要")
        print("=" * 60)
        print(f"📅 评估时间: {eval_info['评估时间']}")
        print(f"📊 测试用例总数: {eval_info['测试用例总数']}")
        print(f"🎯 自动判定的问答匹配率: {eval_info['自动判定的问答匹配率']}")

        print("\n📏 核心评估指标:")
        print(f"   🔍 平均chunk召回率: {metrics['平均chunk召回率']}")
        print(f"   🎯 平均chunk精确率: {metrics['平均chunk精确率']}")
        print(f"   🔁 平均chunkF1: {metrics['平均chunkF1']}")
        print(f"   🎯 平均Hit@K: {metrics['平均Hit@K']}")
        print(f"   📐 平均MRR: {metrics['平均MRR']}")
        print(f"   ✅ 平均EM: {metrics['平均EM']}")
        print(f"   💡 平均MultiHopScore: {metrics['平均MultiHopScore']}")
        print(f"   📚 平均MultiDocScore: {metrics['平均MultiDocScore']}")
        print(f"   ⏱️  平均响应时间: {metrics['平均响应时间']}秒")

        print("\n📋 按问题类型分析:")
        for query_type, stats in report['按问题类型分析'].items():
            print(f"   {query_type}:")
            print(f"     数量: {stats['数量']}")
            print(f"     自动判定的问答匹配率: {stats['自动判定的问答匹配率']}")
            print(f"     平均耗时: {stats['平均耗时']}")

def main():
    """主函数"""
    print("RAG系统综合评估工具 - 字段名称修正版")
    print("=" * 60)
    print("功能特点:")
    print("1. 自动执行所有测试用例")
    print("2. 生成详细结构化评估报告")
    print("3. 报告保存在 app/service/test 目录下")
    print("4. 包含完整的指标计算公式和说明")
    print("5. 按问题类型分析，正确显示问答匹配率字段")
    print("=" * 60)

    try:
        evaluator = CorrectedRAGEvaluator()
        evaluator.run_evaluation()
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print("\n💡 解决方案:")
        print("1. 确保在项目根目录下运行此脚本")
        print("2. 确保 app/service/test/QA.json 文件存在")
    except KeyboardInterrupt:
        print("\n\n评估被用户中断")
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        logging.error(f"评估错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
