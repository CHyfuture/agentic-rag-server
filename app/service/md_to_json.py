import os
import json
import requests
from pathlib import Path
import logging
from typing import List, Dict, Any
import re
import concurrent.futures
import time
from collections import Counter

# 路径配置，MD_DIR 为需要翻译的MD文件存放目录，JSON_DIR为生成的JSON文件存放的目录
MD_DIR = Path(r"E:\My_Project\Python\RAG\md")
JSON_DIR = Path(r"E:\My_Project\Python\RAG\json")
JSON_DIR.mkdir(parents=True, exist_ok=True)

#deepseek硬编码，可考虑移动到环境配置中
API_KEY = "sk-9e728482e15f4cecba9ead88ff7e9cc8"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API调用并发限制
MAX_WORKERS = 5  # 根据API速率限制调整，暂时设置为5，相对稳定

# 扩展停用词表（基础+学术通用停用词）
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'shall', 'should', 'may', 'might', 'can', 'could', 'must', 'it', 'this', 'that', 'these', 'those', 'i', 'we',
    'you', 'he', 'she', 'they', 'me', 'us', 'him', 'her', 'them', 'my', 'our', 'your', 'his', 'her', 'their',
    'which', 'who', 'whom', 'whose', 'what', 'where', 'when', 'why', 'how', 'all', 'any', 'some', 'no', 'not',
    'only', 'very', 'too', 'so', 'just', 'now', 'then', 'here', 'there', 'up', 'down', 'in', 'out', 'off', 'on',
    'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around', 'before', 'behind', 'below',
    'beneath', 'beside', 'between', 'beyond', 'during', 'inside', 'outside', 'over', 'through', 'under', 'within',
    'without', 'also', 'however', 'therefore', 'thus', 'hence', 'yet', 'still', 'even', 'indeed', 'furthermore',
    'moreover', 'nevertheless', 'nonetheless', 'additionally', 'consequently', 'accordingly', 'otherwise',
    'research', 'study', 'paper', 'article', 'analysis', 'investigation', 'exploration', 'examination', 'discussion'
}

# 专业词汇特征（匹配形容词+名词/名词+名词组合）
MULTI_WORD_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:[ ][A-Z][a-z]+){1,4})\b')  # 匹配2-5个首字母大写的单词组合（不跨行）
ADJ_NOUN_PATTERN = re.compile(r'\b([a-zA-Z]+(?:ed|ing|ive|ful|less|ous|ary|ical|ic)\s+[a-zA-Z]+)\b', re.IGNORECASE)  # 匹配形容词+名词


def detect_language(text: str) -> str:
    """
    轻量语言判定：返回 "zh" 或 "en"。
    规则：优先看中文论文标记词；否则按汉字占比；最后默认英文。
    """
    if not text:
        return "en"

    sample = text[:4000]
    if re.search(r'(?:摘要|关键词|结论|引言)', sample):
        return "zh"
    if re.search(r'(?:\bAbstract\b|\bKeywords?\b|\bConclusion(?:s)?\b|\bIntroduction\b)', sample, re.IGNORECASE):
        return "en"

    han = len(re.findall(r'[\u4e00-\u9fff]', sample))
    letters = len(re.findall(r'[A-Za-z]', sample))
    if han >= 30 and han > letters:
        return "zh"
    return "en"


def _extract_block(content: str, start_patterns: List[str], end_patterns: List[str]) -> str:
    """
    从 content 中按 start_patterns 定位起点，再按 end_patterns 找到最近终点，返回中间文本。
    - start_patterns: 匹配起点行/标题，起点为 match.end()
    - end_patterns: 在起点之后匹配终止边界，终点为 match.start()
    """
    if not content:
        return ""

    start_idx = None
    for sp in start_patterns:
        m = re.search(sp, content, re.IGNORECASE | re.MULTILINE)
        if m:
            start_idx = m.end()
            break
    if start_idx is None:
        return ""

    tail = content[start_idx:]
    end_idx_in_tail = None
    for ep in end_patterns:
        m2 = re.search(ep, tail, re.IGNORECASE | re.MULTILINE)
        if m2:
            cand = m2.start()
            end_idx_in_tail = cand if end_idx_in_tail is None else min(end_idx_in_tail, cand)

    block = tail[:end_idx_in_tail] if end_idx_in_tail is not None else tail
    return block.strip()


def _parse_first_json_object(text: str) -> Dict[str, Any]:
    """尽量从模型输出中提取第一段 JSON 对象。支持 ```json 代码块。"""
    if not text:
        return {}

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except Exception:
            return {}
    return {}


class FastPaperProcessor:
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        # 创建会话以重用连接
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def extract_with_regex(self, content: str) -> Dict[str, Any]:
        """使用正则表达式快速提取基本信息（优化关键词提取逻辑）"""
        result = {
            "title": "",
            "authors": [],
            "abstract": "",
            "keywords": [],
            "conclusion": ""  # 字段名改为英文"conclusion"
        }

        lang = detect_language(content)

        # 提取标题（通常在第一行或前几行）
        lines = content.split('\n')
        for line in lines[:30]:
            m = re.match(r'^\s*#{1,6}\s+(.+?)\s*$', line)
            if m:
                title = m.group(1).strip()
                if title:
                    result["title"] = title
                    break
        if not result["title"]:
            for line in lines[:10]:
                if line.strip() and len(line.strip()) > 10:
                    result["title"] = line.strip()
                    break

        # 提取作者信息和摘要
        if lang == "en":
            result["abstract"] = _extract_block(
                content,
                start_patterns=[
                    r'(?:^|\n)\s*(?:#+\s*)?Abstract\s*:\s*',
                    r'(?:^|\n)\s*(?:#+\s*)?Abstract\s*[:.\]]?\s*\n+',
                    r'(?:^|\n)\s*ABSTRACT\s*[:.\]]?\s*\n+',
                    r'(?:^|\n)\s*Abstract\.\s*',
                ],
                end_patterns=[
                    r'(?:^|\n)\s*(?:Keywords?|Key\s+Words?|Index\s+Terms?)\s*:',
                    r'(?:^|\n)\s*(?:关键词)\s*[:：]',
                    r'(?:^|\n)\s*(?:#+\s*)?(?:Introduction|1\s+Introduction)\b',
                    r'(?:^|\n)\s*(?:#+\s*)?(?:Conclusion|Conclusions)\b',
                    r'(?:^|\n)\s*(?:#+\s*)?(?:References|Bibliography)\b',
                ],
            )
        else:
            result["abstract"] = _extract_block(
                content,
                start_patterns=[
                    r'(?:^|\n)\s*(?:#+\s*)?(?:摘\s*要|摘要)\s*[:：]\s*',
                    r'(?:^|\n)\s*(?:#+\s*)?(?:摘\s*要|摘要)\s*[:：\]]?\s*\n+',
                    r'(?:^|\n)\s*[【\[]\s*(?:摘\s*要|摘要)\s*[】\]]\s*\n*',
                ],
                end_patterns=[
                    r'(?:^|\n)\s*(?:关键词)\s*[:：]',
                    r'(?:^|\n)\s*(?:Keywords?|Key\s+Words?|Index\s+Terms?)\s*:',
                    r'(?:^|\n)\s*(?:目录)\s*[:：]?',
                    r'(?:^|\n)\s*(?:引言|前言)\s*[:：]?',
                    r'(?:^|\n)\s*(?:#+\s*)?(?:Introduction|1\s*引言|一、引言)\b',
                    r'(?:^|\n)\s*(?:#+\s*)?(?:结论)\b',
                    r'(?:^|\n)\s*(?:#+\s*)?(?:参考文献|References)\b',
                ],
            )

        # ========== 优化关键词提取逻辑 ==========
        # 1. 优先提取原文标注的关键词（兼容多格式）
        keywords_patterns = [
            r'Keywords?\s*:\s*(.*?)(?:\n\n|\n\w)',  # Keywords: 或 Keyword:
            r'Key\s+Words?\s*:\s*(.*?)(?:\n\n|\n\w)',  # Key Words: 或 Key Word:
            r'关键词\s*：?\s*(.*?)(?:\n\n|\n\w)',  # 中文关键词：
            r'Index\s+Terms?\s*:\s*(.*?)(?:\n\n|\n\w)'  # Index Terms: （IEEE格式）
        ]

        original_keywords = ""
        for pattern in keywords_patterns:
            keywords_match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if keywords_match:
                original_keywords = keywords_match.group(1).strip()
                break

        # 处理原文标注的关键词（支持多词组合）
        if original_keywords:
            result["_has_original_keywords"] = True
            # 处理分隔符：中文、英文逗号/分号、空格
            original_keywords = (
                original_keywords.replace('、', ';')
                .replace('，', ';')
                .replace(',', ';')
                .replace('；', ';')
                .replace('  ', ';')
            )
            # 拆分并清洗
            raw_keywords = [k.strip() for k in re.split(r'[;,\n]', original_keywords) if k.strip()]
            # 过滤空值和停用词占比过高的条目
            for kw in raw_keywords:
                # 保留多词组合（2-5个单词）
                if len(kw.split()) >= 2 and len(kw.split()) <= 5:
                    result["keywords"].append(kw)
                # 单字词仅保留非停用词
                elif kw.lower() not in STOP_WORDS and len(kw) > 2:
                    result["keywords"].append(kw)
        else:
            result["_has_original_keywords"] = False

        # 2. 无原文标注时，从标题+摘要提取核心关键词
        if not result["keywords"]:
            # 合并标题+摘要作为提取源
            extract_source = f"{result['title']}\n{result['abstract']}"
            if not extract_source.strip():
                result["keywords"] = []
            else:
                # 提取多词组合（优先首字母大写的专业术语）
                multi_word_terms = MULTI_WORD_PATTERN.findall(extract_source)
                # 提取形容词+名词组合
                adj_noun_terms = ADJ_NOUN_PATTERN.findall(extract_source)
                # 合并所有候选术语
                candidate_terms = multi_word_terms + adj_noun_terms

                # 清洗候选术语：过滤停用词、去重、筛选长度
                cleaned_terms = []
                for term in candidate_terms:
                    # 转小写后拆分单词
                    words = term.lower().split()
                    # 过滤：单词数2-5个，且核心词非停用词
                    if 2 <= len(words) <= 5:
                        # 保留至少一个非停用词的组合
                        non_stop_words = [w for w in words if w not in STOP_WORDS]
                        if len(non_stop_words) >= 1:
                            cleaned_terms.append(term.strip())

                # 统计词频，取Top8核心术语（去重）
                term_counter = Counter(cleaned_terms)
                top_terms = [term for term, _ in term_counter.most_common(8)]
                # 最终去重并确保唯一性
                result["keywords"] = list(dict.fromkeys(top_terms))

                # 兜底：若仍无结果，提取单字核心词（排除停用词）
                if not result["keywords"]:
                    # 拆分所有单词并过滤
                    all_words = re.findall(r'\b[A-Za-z]+\b', extract_source.lower())
                    word_counter = Counter([w for w in all_words if w not in STOP_WORDS and len(w) > 2])
                    # 取Top5单字词，转为首字母大写
                    top_single_words = [w.capitalize() for w, _ in word_counter.most_common(5)]
                    result["keywords"] = top_single_words

        # ========== 关键词提取逻辑结束 ==========

        # 直接提取结论部分作为结论，增强支持多种Conclusion变体
        conclusion_patterns = [
            r'#\s*[0-9]*\.?\s*[Cc]onclusion(?:s)?(?:\s+and\s+[A-Za-z]+)*(?:\s+[A-Za-z]+)*\s*\n\n(.*?)(?:\n\nAcknowledgement|\n\n# References|\Z)',  # # Conclusion等标题后有两个换行符的情况
            r'#\s*[0-9]*\.?\s*[Cc]onclusion(?:s)?(?:\s+and\s+[A-Za-z]+)*(?:\s+[A-Za-z]+)*\s*\n(.*?)(?:\n\nAcknowledgement|\n\n# References|\Z)',   # # Conclusion等标题后有一个换行符的情况
            r'#\s*[0-9]*\.?\s*结论\s*\n\n(.*?)(?:\n\nAcknowledgement|\n\n# References|\Z)',  # 中文结论后有两个换行符的情况
            r'#\s*[0-9]*\.?\s*结论\s*\n(.*?)(?:\n\nAcknowledgement|\n\n# References|\Z)'   # 中文结论后有一个换行符的情况
        ]

        conclusion = ""
        for pattern in conclusion_patterns:
            conclusion_match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if conclusion_match:
                conclusion = conclusion_match.group(1).strip()
                break

        result["conclusion"] = conclusion

        return result

    def call_deepseek_sync(self, prompt: str, *, max_tokens: int = 1500) -> Dict[str, Any]:
        """同步调用DeepSeek API"""
        url = f"{BASE_URL}/chat/completions"

        data = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "你是一位专业的论文信息提取助手，请严格按照要求提取论文信息并返回JSON格式。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens
        }

        try:
            # 使用会话发送请求，重用连接
            response = self.session.post(url, json=data, timeout=60)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return _parse_first_json_object(content)
            else:
                logger.error(f"API返回错误状态码: {response.status_code}, 内容: {response.text}")
        except Exception as e:
            logger.error(f"API调用错误: {e}")

        return {}

    def extract_keywords_with_llm(self, *, title: str, abstract: str, conclusion: str, lang: str) -> List[str]:
        """原文无关键词时，用 LLM 从标题/摘要/结论提取同语种关键词。"""
        title = title or ""
        abstract = abstract or ""
        conclusion = conclusion or ""

        if lang == "zh":
            prompt = f"""请仅从下面的“标题、摘要、结论”中提取论文核心关键词。\n\n要求：\n- 仅输出中文关键词（不要夹杂英文，除非原文核心术语本身是英文缩写且必要）\n- 返回严格 JSON：{{\"keywords\": [\"词1\", \"词2\", ...]}}\n- 关键词数量 5-10 个，优先领域术语/方法/任务/数据/关键概念\n- 不要输出任何额外解释文字\n\n标题：{title}\n\n摘要：{abstract}\n\n结论：{conclusion}\n"""
        else:
            prompt = f"""Extract the paper's core keywords ONLY from the given title, abstract, and conclusion.\n\nRequirements:\n- Output English keywords only\n- Return strict JSON: {{\"keywords\": [\"kw1\", \"kw2\", ...]}}\n- 5-10 keywords, prefer technical terms, methods, tasks, datasets, key concepts\n- No extra text\n\nTitle: {title}\n\nAbstract: {abstract}\n\nConclusion: {conclusion}\n"""

        data = self.call_deepseek_sync(prompt, max_tokens=300)
        kws = data.get("keywords", [])
        if not isinstance(kws, list):
            return []

        cleaned: List[str] = []
        for k in kws:
            if not isinstance(k, str):
                continue
            kk = k.strip()
            if not kk:
                continue
            cleaned.append(kk)

        # 去重 + 控制长度
        uniq = list(dict.fromkeys(cleaned))
        return uniq[:12]

    def process_paper(self, md_file: Path) -> bool:
        """处理单个论文"""
        try:
            logger.info(f"处理: {md_file.name}")

            # 读取内容
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 先用正则提取基本信息（包括优化后的关键词）
            basic_info = self.extract_with_regex(content)
            has_original_keywords = bool(basic_info.pop("_has_original_keywords", False))
            lang = detect_language(f"{basic_info.get('title','')}\n{basic_info.get('abstract','')}\n{basic_info.get('conclusion','')}\n{content[:1500]}")

            # 构建更详细的提示词，明确要求使用"conclusion"字段
            prompt = f"""请从论文中提取以下信息，严格按照JSON格式返回，不要添加任何额外文本：

1. 标题 (title)：论文的完整标题
2. 作者信息 (authors)：数组格式，每个作者对象包含：
   - name：作者姓名
   - school：作者所属学校/机构
3. 摘要 (abstract)：Abstract.之后的所有内容，不包含作者信息
4. 关键词 (keywords)：数组格式，包含所有关键词；要求：
   - 优先提取原文标注的关键词，无标注时从标题+摘要提取核心专业词汇
   - 支持2-5个单词组成的英文关键词（如 "Machine Learning", "Convolutional Neural Network"）
   - 排除基础停用词（如 research/study/paper 等），保留体现论文核心的专业术语
5. 结论 (conclusion)：直接使用论文中的Conclusion部分内容，不要自行生成

当前已初步提取的信息：{basic_info}

论文内容：
前3000字符：
{content[:3000]}

请确保返回的JSON包含所有要求的字段，格式正确，特别是使用"conclusion"作为结论字段的名称。"""

            # 调用API
            api_result = self.call_deepseek_sync(prompt)

            # 合并结果（API结果优先，但保留基本信息作为后备）
            result = {**basic_info, **api_result}

            # 关键词合并策略：原文有关键词则强制保留，避免被 API 覆盖
            if has_original_keywords:
                result["keywords"] = basic_info.get("keywords", [])

            # 确保结论使用直接提取的结论（如果API没有返回或返回错误）
            if not result.get("conclusion") or result["conclusion"] == "":
                # 再次尝试提取结论，确保准确性
                conclusion_patterns = [
                    r'#\s*[0-9]*\.?\s*[Cc]onclusion(?:s)?(?:\s+and\s+[A-Za-z]+)*(?:\s+[A-Za-z]+)*\s*(.*?)(?:\n\nAcknowledgement|\n\n# References|\Z)',  # # Conclusion, # 5 Conclusion, # 5. Conclusion, # 4.5 Conclusion and Limitations等
                    r'#\s*[0-9]*\.?\s*结论\s*(.*?)(?:\n\nAcknowledgement|\n\n# References|\Z)'  # # 结论，# 5 结论，# 5. 结论
                ]

                for pattern in conclusion_patterns:
                    conclusion_match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                    if conclusion_match:
                        result["conclusion"] = conclusion_match.group(1).strip()
                        break

            # 确保所有必要字段都存在
            required_fields = ["title", "authors", "abstract", "keywords", "conclusion"]
            for field in required_fields:
                if field not in result:
                    result[field] = ""
                elif field == "authors" and not isinstance(result[field], list):
                    result[field] = []
                elif field == "keywords" and not isinstance(result[field], list):
                    result[field] = []

            # 原文无关键词时：优先用 LLM 从标题/摘要/结论抽取；失败再回退到正则/模型返回
            if not has_original_keywords:
                llm_kws = self.extract_keywords_with_llm(
                    title=result.get("title", basic_info.get("title", "")),
                    abstract=result.get("abstract", basic_info.get("abstract", "")),
                    conclusion=result.get("conclusion", basic_info.get("conclusion", "")),
                    lang=lang,
                )
                if llm_kws:
                    result["keywords"] = llm_kws
                elif not result["keywords"] or len(result["keywords"]) < 2:
                    result["keywords"] = basic_info.get("keywords", [])

            # 添加原文所有信息
            result["original_text"] = content

            # 保存
            json_file = JSON_DIR / f"{md_file.stem}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            return True

        except Exception as e:
            logger.error(f"处理失败 {md_file.name}: {e}")
            return False

    def process_all(self):
        """处理所有文件"""
        md_files = list(MD_DIR.glob("*.md"))
        logger.info(f"找到 {len(md_files)} 个文件")

        start_time = time.time()
        success = 0

        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务
            future_to_file = {executor.submit(self.process_paper, md_file): md_file for md_file in md_files}

            # 获取结果
            for future in concurrent.futures.as_completed(future_to_file):
                md_file = future_to_file[future]
                try:
                    if future.result():
                        success += 1
                except Exception as e:
                    logger.error(f"处理 {md_file.name} 时发生异常: {e}")

        end_time = time.time()
        logger.info(f"完成！成功处理 {success}/{len(md_files)} 个文件，耗时 {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    processor = FastPaperProcessor()
    processor.process_all()