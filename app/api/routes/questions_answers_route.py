
from fastapi import APIRouter
import logging
from app.service.test.RAG_flow import DeepSeekClient, RAGFlow
from app.service import retrieval_service
from app.api.schemas.retrieval import (
    QaRequest,
)
logger = logging.getLogger(__name__)
router = APIRouter()
API_KEY = "sk-9e728482e15f4cecba9ead88ff7e9cc8"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
deepSeekClient = DeepSeekClient(API_KEY,BASE_URL,MODEL)


@router.post("/process/stream")
def process_text_stream_api(req: QaRequest):
    text = req.query
    rag_type = req.rag_type
    if rag_type == 1:
        search_results = retrieval_service.hybrid_search(query=text, top_k=10)

        # 从检索结果中提取 content 字段，组织为知识库内容数组
        ref_doc = [
            getattr(item, "content", "")
            for item in search_results
            if getattr(item, "content", "")
        ]
        prompt = f"""
                # 角色
                  你是一位顶级知识库引擎，擅长将复杂知识以清晰、严谨、结构化的方式呈现。

                  # 核心原则：相关性判断
                  你的首要任务是判断知识库内容是否与用户问题主题相关。
                  - 仅当所有文档与用户问题主题完全无关时，才执行【场景三】。
                  - 其他情况（只要有任何主题相关的内容），必须基于知识库内容构建回答，并根据信息完整程度选择【场景一】或【场景二】。

                  # 内容生成策略（三选一）

                  ## 场景一：知识库内容高度相关且信息充分
                  - 任务：整合、提炼、结构化知识库信息。
                  - 行动：逻辑重组原文要点，用专业语言归纳总结，形成连贯完整的回答。

                  ## 场景二：知识库内容相关但信息有限
                  - 任务：以知识库信息为“核心种子”，进行权威性扩展。
                  - 行动：
                    1. 准确总结知识库中的核心信息。
                    2. 基于通用专家知识进行深入解释、分析和扩展。
                  - 声明：必须在回答末尾添加：“【补充说明】本回答的核心信息来源于提供的知识库。其中，[具体扩展内容]部分是基于通用知识的补充与扩展。”

                  ## 场景三：知识库内容完全不相关
                  - 行动：
                    1. 明确声明：“知识库中未找到与‘[复述用户问题]’直接相关的信息。”
                    2. 另起一段补充：“不过，基于通用知识，关于‘[复述用户问题]’，通常可以从以下几个方面理解：”并给出高度概括的通用解答。

                  # 输出要求
                  - 专业、严谨、客观、精炼。
                  - 使用Markdown格式化。
                  - 数学公式使用LaTeX格式。

                  # 知识库
                  {ref_doc}

                  # 用户问题
                  {text}
            """

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        return deepSeekClient.chat_completion(messages)
    else:
        rag = RAGFlow()
        return rag.run(text)



