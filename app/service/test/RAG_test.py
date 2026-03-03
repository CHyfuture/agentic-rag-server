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
print(f"🔍 检测到项目根目录: {project_root}")

# 测试与日志目录（统一放在 app/service/test 下）
test_dir = project_root / "app" / "service" / "test"
test_dir.mkdir(parents=True, exist_ok=True)

# 添加项目根目录到Python路径
sys.path.insert(0, str(project_root))

# 导入RAG流程
try:
    from app.service.test.RAG_flow import RAGFlow
    print("✅ 成功导入 RAG_flow")
except ImportError as e:
    print(f"❌ 导入RAG_flow失败: {e}")
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
            "你是一个学术论文内容对齐助手。现在给出两段中文学术文本片段，"
            "请判断它们是否在描述同一篇论文中的同一个具体内容片段（同一事实/同一结论/同一表格或图像的描述）。\n"
            "要求：\n"
            "1. 如果两段文本只是措辞略有不同，但核心数值、对象、结论完全对应，则认为是“命中”。\n"
            "2. 如果两段文本只是在同一篇论文中但描述的是不同位置的内容，或核心结论/数值明显不同，则认为是“未命中”。\n"
            "3. 不要进行主观推测，严格基于给定文本判断。\n"
            "4. 最终只输出“命中”或“未命中”四个字，不要添加任何解释。"
        )
        user_prompt = f"标准chunk文本：\n{gt_chunk}\n\n检索到的chunk文本（可能被截断）：\n{retrieved_chunk}\n\n请根据要求判断："

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
        if not gt_chunks or not retrieved_chunks:
            return {
                "chunk召回率": 0.0,
                "chunk精确率": 0.0,
                "chunkF1": 0.0,
                "chunk级对齐详情": [],
            }

        align_details: List[Dict[str, Any]] = []
        hit_gt_indices = set()
        hit_retrieved_indices = set()

        for gt_idx, gt in enumerate(gt_chunks):
            gt_text = gt["text"]
            for r_idx, r in enumerate(retrieved_chunks):
                retrieved_text = str(r.get("content_preview", ""))
                judge_result = self._judge_chunk_match_with_llm(gt_text, retrieved_text)
                is_hit = judge_result["is_hit"]
                detail = {
                    "gt_index": gt_idx,
                    "gt_doc_id": gt.get("doc_id"),
                    "gt_doc_title": gt.get("doc_title"),
                    "gt_chunk文本": gt_text,
                    "retrieved_rank": r_idx + 1,
                    "retrieved_doc_id": r.get("doc_id"),
                    "retrieved_doc_title": r.get("doc_title"),
                    "retrieved_chunk预览": retrieved_text,
                    "LLM判定结果": judge_result["raw_decision"],
                    "是否命中": is_hit,
                }
                align_details.append(detail)
                if is_hit:
                    hit_gt_indices.add(gt_idx)
                    hit_retrieved_indices.add(r_idx)
                    # 对于每个标准chunk，一旦找到一个命中即可停止继续判断，避免额外开销
                    break

        gt_total = len(gt_chunks)
        retrieved_total = len(retrieved_chunks)
        hit_gt_count = len(hit_gt_indices)
        hit_retrieved_count = len(hit_retrieved_indices)

        recall = hit_gt_count / gt_total if gt_total > 0 else 0.0
        precision = hit_retrieved_count / retrieved_total if retrieved_total > 0 else 0.0
        if recall + precision > 0:
            f1 = 2 * recall * precision / (recall + precision)
        else:
            f1 = 0.0

        return {
            "chunk召回率": round(recall, 3),
            "chunk精确率": round(precision, 3),
            "chunkF1": round(f1, 3),
            "chunk级对齐详情": align_details,
        }

    def calculate_evaluation_metrics(self, case: Dict, response: str, exec_time: float, trace: Dict) -> Dict:
        """计算完整的评估指标（包含chunk级检索指标，字段名使用中文）"""
        ground_truth = str(case['ground_truth_answer']).lower().strip()
        rag_answer = str(response).lower().strip()

        # 基础准确率检查
        auto_correct = self.check_accuracy(ground_truth, rag_answer)

        # 完整性评估
        completeness = self.assess_completeness(case['query'], ground_truth, rag_answer)

        # 一致性检查
        consistency = self.check_consistency(ground_truth, rag_answer)

        # 根据问题类型设定推理能力
        reasoning_ability = 0.8 if case['query_type'] == '多跳推理型' else 0.5

        # chunk 级检索指标
        gt_chunks = self._build_gt_chunks(case)
        retrieved_chunks = self._build_retrieved_chunks_from_trace(trace, top_k=10)
        chunk_metrics = self._calculate_chunk_level_metrics(gt_chunks, retrieved_chunks)

        metrics: Dict[str, Any] = {
            '答案是否正确': auto_correct,
            '答案完整性': completeness,
            '事实一致性': consistency,
            '推理能力': reasoning_ability,
            '耗时': exec_time,
        }
        # 合并 chunk 级指标
        metrics.update(chunk_metrics)
        return metrics

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
        """评估答案完整性 (0-1)"""
        score = 0.5  # 基础分

        import re
        # 检查数值信息
        if re.search(r'\d+\.?\d*', truth) and re.search(r'\d+\.?\d*', response):
            score += 0.3

        # 检查单位信息
        units = ['倍', '百分点', '%', '张', '次', '层']
        for unit in units:
            if unit in truth and unit in response:
                score += 0.2
                break

        return min(1.0, score)

    def check_consistency(self, truth: str, response: str) -> float:
        """检查事实一致性 (0-1)"""
        score = 1.0

        # 检查数值矛盾
        import re
        truth_nums = re.findall(r'\d+\.?\d*', truth)
        response_nums = re.findall(r'\d+\.?\d*', response)

        if truth_nums and response_nums:
            for t_num in truth_nums[:2]:
                t_val = float(t_num)
                for r_num in response_nums[:2]:
                    r_val = float(r_num)
                    if abs(t_val - r_val) / max(t_val, r_val) > 0.8:
                        score -= 0.5
                        break

        return max(0.0, score)

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
            '平均答案完整性': _avg('答案完整性'),
            '平均事实一致性': _avg('事实一致性'),
            '平均推理能力': _avg('推理能力'),
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

        # 保存报告到指定目录
        report_path = self.report_dir / 'rag_evaluation_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n💾 评估报告已保存到: {report_path}")

        # 打印摘要
        self.print_evaluation_summary(report)

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
                }
            },
            '生成阶段指标': {
                '答案准确性': {
                    '定义': 'RAG生成的答案是否正确',
                    '计算公式': '布尔值判断 (正确=1, 错误=0)',
                    '判断方法': [
                        '完全文本匹配',
                        '数值匹配（提取数字进行比较）',
                        '关键词重叠度匹配 (>40%)'
                    ],
                    '理想值': '1.0 (100%)',
                    '当前实现': '自动判断，需人工复核'
                },
                '答案完整性': {
                    '定义': '回答是否包含问题所需的完整信息',
                    '计算公式': '基础分0.5 + 数值信息0.3 + 单位信息0.2',
                    '评分细则': {
                        '基础分 (0.5)': '回答非空且有一定相关性',
                        '数值信息 (0.3)': '包含与标准答案相关的数值',
                        '单位信息 (0.2)': '包含正确的计量单位'
                    },
                    '理想值': '1.0',
                    '说明': '评估回答的全面性和详尽程度'
                },
                '事实一致性': {
                    '定义': '回答中是否存在与标准答案相矛盾的事实',
                    '计算公式': '基础分1.0 - 矛盾扣分',
                    '扣分规则': {
                        '数值矛盾 (扣0.5)': '相同概念的数值差异超过80%',
                        '逻辑矛盾 (扣0.3)': '否定词使用不一致'
                    },
                    '理想值': '1.0',
                    '说明': '确保生成内容的事实准确性'
                }
            },
            '整体性能指标': {
                '推理能力': {
                    '定义': '处理复杂推理问题的能力',
                    '计算方法': {
                        '多跳推理型问题': '0.8',
                        '方法/实验型问题': '0.7',
                        '事实型问题': '0.5'
                    },
                    '说明': '根据不同问题类型的复杂程度设定基准值'
                },
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

        # 计算比率，使用正确的字段名
        for query_type, stats in type_stats.items():
            count = stats['数量']
            if count > 0:
                stats['自动判定的问答匹配率'] = f"{stats['自动判定正确数量'] / count:.1%}"
                stats['平均chunk召回率'] = round(stats['chunk召回率汇总'] / count, 3)
                stats['平均chunk精确率'] = round(stats['chunk精确率汇总'] / count, 3)
                stats['平均chunkF1'] = round(stats['chunkF1汇总'] / count, 3)
                stats['平均耗时'] = f"{stats['总耗时'] / count:.2f}秒"

            # 清理中间累加字段
            stats.pop('chunk召回率汇总', None)
            stats.pop('chunk精确率汇总', None)
            stats.pop('chunkF1汇总', None)

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
        print(f"   ✅ 平均答案完整性: {metrics['平均答案完整性']}")
        print(f"   🔄 平均事实一致性: {metrics['平均事实一致性']}")
        print(f"   💡 平均推理能力: {metrics['平均推理能力']}")
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
