## 概述

BaseVector-Core 是一个**纯 Python SDK**，围绕 Milvus 向量数据库，提供四大核心能力：

- **文档解析（Parsers）**：将 PDF / Word / PPT / HTML / Markdown / TXT 等文档解析为结构化文本与元数据。
- **文档切片（Chunkers）**：将长文本按多种策略切分为 Chunk（固定窗口 / 语义 / 标题 / 父子）。
- **文档存储（Storage）**：基于 Milvus 的集合管理与向量数据写入封装。
- **文档检索（Retrievers）**：提供语义检索、关键词检索、混合检索等能力。

并且：

- 支持通过 **`pip install git+...`** 的方式，在第三方项目中直接 `import milvus_service` 使用。
- 所有核心能力通过 **Service 层**（`ParserService` / `ChunkerService` / `StorageService` / `RetrieverService`）统一暴露，调用方式**简单稳定**。
- 支持通过 **插件机制**（`RegistryService` + `PluginRegistry`）扩展自定义解析器、切片器、检索器、存储算子。

> 本 README 面向“第三方调用方”，假设你只想把这些能力当作一个 SDK 来用。

---

## 能力总览

- **文档解析能力**
  - 支持格式：`PDF / DOCX / PPTX / HTML / Markdown / TXT`
  - 输出：纯文本内容 + 元数据（标题、作者、页码、结构信息等）
  - 入口：`ParserService.parse(ParseRequest)`

- **文档切片能力**
  - 策略：
    - `fixed`：固定窗口切片
    - `semantic`：语义切片
    - `title`：基于标题的切片
    - `parent_child`：父子块切片
  - 输出：`ChunkDTO` 列表（content / chunk_index / start_index / end_index / metadata）
  - 入口：`ChunkerService.chunk(ChunkRequest)`

- **文档存储能力（Milvus）**
  - 集合管理：创建 / 删除 / 获取 / 判断是否存在 / 列出集合
  - 数据插入：按指定 schema 写入向量 + 文本 + 元数据
  - 入口：`StorageService.*`

- **文档检索能力**
  - 模式：
    - `semantic_search`：语义向量检索
    - `keyword_search`：关键词 / BM25 风格检索
    - `hybrid_search`：语义 + 关键词混合检索
    - 以及 `fulltext_search` / `text_match_search` / `phrase_match_search`
  - 输出：`RetrievalResultDTO` 列表（chunk_id / document_id / content / score / metadata）
  - 入口：`RetrieverService.*_search`

更完整的参数与字段说明，请参考仓库根目录下的 `API_DOCUMENTATION.md`。

---

## 安装（第三方项目）

### 环境要求

- **Python**：3.10+
- **Milvus**：推荐 2.3+（建议使用官方 docker-compose 部署）

### 安装 SDK

在你的业务项目中直接从 Git 安装：

```bash
pip install "git+ssh://git@github.com/CHyfuture/BaseVector-Core.git"
```

> 如需指定分支或 tag，将 `@main` 替换为对应名称即可。

安装完成后，你只需要从 `milvus_service` 导入需要的接口。

---

## 项目结构（对调用方有用的部分）

```text
BaseVector-Core/
├── ability/                 # 内部能力实现（通常不需要直接 import）
│   ├── config.py            # Settings 配置管理
│   ├── operators/           # 各种算子（parsers / chunkers / retrievers / storage）
│   ├── storage/milvus_client.py
│   └── utils/...
│
├── milvus_service/          # ⭐ 第三方主要 import 的包
│   ├── __init__.py          # 导出 Settings / 工厂 / Service / Registry 等
│   └── service/
│       ├── parser_service.py
│       ├── chunker_service.py
│       ├── storage_service.py
│       ├── retriever_service.py
│       ├── registry_service.py
│       └── ability_descriptor.py
│
├── API_DOCUMENTATION.md     # 详细 API 文档（参数/字段说明）
├── pyproject.toml           # 打包配置（pip install git+ 使用）
├── requirements.txt         # 本地运行依赖
└── README.md                # 当前文件
```

**作为调用方，你只需要记住：**

- **只从 `milvus_service` 导入**
- `ability.*` 是内部实现，无需直接使用

---

## 配置方式（调用方怎么配）

库内部使用 `Settings`（通过 `milvus_service` 导出）管理配置，包括：

- Milvus 连接信息：`MILVUS_HOST / MILVUS_PORT / MILVUS_USER / MILVUS_PASSWORD / MILVUS_DB_NAME`
- Milvus 索引/搜索参数：`MILVUS_INDEX_TYPE / MILVUS_METRIC_TYPE / MILVUS_NLIST / MILVUS_NPROBE`
- 检索参数：`TOP_K / RERANK_ENABLED / RERANK_MODEL_NAME / SIMILARITY_THRESHOLD / RETRIEVAL_CANDIDATE_MULTIPLIER`
- 切片参数：`CHUNK_SIZE / CHUNK_OVERLAP / CHUNK_STRATEGY`
- 多租户：`ENABLE_MULTI_TENANT / DEFAULT_TENANT_ID`

### 1. 最推荐的方式：环境变量 + `.env` + `config/config.yaml`

默认优先级：**环境变量 > `.env` > `config/config.yaml` > 代码默认值**。

在你的业务项目根目录创建 `.env`（或设置系统环境变量）：

```env
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=
MILVUS_PASSWORD=
MILVUS_DB_NAME=default

TOP_K=10
RERANK_ENABLED=false
RERANK_MODEL_NAME=workspace/jina-reranker-v3
SIMILARITY_THRESHOLD=0.0
RETRIEVAL_CANDIDATE_MULTIPLIER=2

MILVUS_METRIC_TYPE=IP
MILVUS_NPROBE=10
KEYWORD_FTS_LANGUAGE=simple
```

然后在代码中**不需要做任何额外操作**，所有 Service 会自动使用这套默认配置。

#### 检索模块典型配置说明（对应上面的字段）

- **Milvus 连接相关**
  - **`MILVUS_HOST` / `MILVUS_PORT`**：
    - Milvus 实例地址与端口，例如 `localhost:19530` 或云上地址。
    - 所有基于 Milvus 的存储与检索都会使用这组连接信息，除非你在请求里用 `milvus_connection` 单独覆盖。
  - **`MILVUS_USER` / `MILVUS_PASSWORD` / `MILVUS_DB_NAME`**：
    - 启用了鉴权或多 DB 时使用；如果你是默认 `root / default`，可以留空或使用默认值。

- **检索底层搜索相关**
  - **`MILVUS_METRIC_TYPE`**：
    - 距离度量，常见：
      - `L2`：欧氏距离；
      - `IP`：内积，相当于相似度，通常用于归一化向量；
      - `COSINE`：余弦相似度。
    - 需要与你的向量模型使用方式匹配（例如 BGE 模型一般配合 `IP`）。
  - **`MILVUS_NPROBE`**：
    - 查询时要探查的桶数量，越大查询越慢但召回更充分。
    - 配合集合创建时的 `NLIST` 使用（`NLIST` 在 SDK 内部或你自己的建库流程中配置）。

- **检索策略相关**
  - **`TOP_K`**：
    - 全局默认的返回条数。
    - 所有 `*SearchRequest` 如果不显式指定 `top_k`，就用这个值。
  - **`RERANK_ENABLED`**：
    - 是否启用重排序（Rerank）。
    - `True`：检索阶段先取更多候选（见 `RETRIEVAL_CANDIDATE_MULTIPLIER`），再用重排模型对候选结果重新打分排序。
    - `False`：只使用 Milvus 原始相似度排序。
  - **`RERANK_MODEL_NAME`**：
    - 重排模型名称或路径，例如：
      - `BAAI/bge-reranker-large`（中文、精度高）；
      - `BAAI/bge-reranker-base`（速度更快）。
    - 内部会用 `sentence-transformers` 加载为 `CrossEncoder`。
  - **`SIMILARITY_THRESHOLD`**：
    - 相似度阈值（0–1），用于过滤得分较低的结果：
      - `0.0`：不做过滤；
      - `0.5 ~ 0.7`：过滤掉明显不相关的结果；
      - `0.7 ~ 0.9`：只保留非常相关的结果。
    - 注意：不同检索模式（语义 / 关键词 / 混合）分数分布可能不同，需要结合业务调试。
  - **`milvus_expr`（请求级参数）**：
    - **不是环境变量**，而是各检索请求（`SemanticSearchRequest` / `KeywordSearchRequest` / `HybridSearchRequest` / `FulltextSearchRequest` / `TextMatchSearchRequest` / `PhraseMatchSearchRequest`）上的字段。
    - 用于传入 Milvus 的过滤表达式，例如：
      - 只检索子块：`milvus_expr="chunk_type == 'child'"`
      - 只看部分文档：`milvus_expr="document_id > 100"`
      - 组合条件：`milvus_expr="chunk_type == 'child' && page <= 3"`
    - 内部会先通过 `validate_milvus_expr()` 做安全校验，然后在所有检索模式中统一生效。
  - **`RETRIEVAL_CANDIDATE_MULTIPLIER`**：
    - 启用重排时，先从 Milvus 拿 `TOP_K * RETRIEVAL_CANDIDATE_MULTIPLIER` 条候选，再做重排。
    - 例如：`TOP_K=10` 且 `RETRIEVAL_CANDIDATE_MULTIPLIER=2`，则先拉 20 条，重排后返回前 10 条。
    - 越大理论上效果越好，但计算开销也越大。
  - **`KEYWORD_FTS_LANGUAGE`**：
    - 关键词 / 全文检索使用的分词语言，典型值：
      - `simple`：简单的通用分词；
      - `english` / `chinese`：按语言优化的分词策略。
    - 主要影响 `keyword_search` / `fulltext_search` 等与关键词相关的模式。

> 小结：  
> - 不想管细节：保留默认值即可，直接用 `SemanticSearchRequest` 做语义检索。  
> - 想精细调优：重点关注 `MILVUS_METRIC_TYPE`、`MILVUS_NPROBE`、`TOP_K`、`RERANK_ENABLED`、`RERANK_MODEL_NAME`、`SIMILARITY_THRESHOLD` 和 `RETRIEVAL_CANDIDATE_MULTIPLIER`。

### 2. 如需在代码里显式指定一套“环境”（可选）

```python
from milvus_service import create_settings, with_settings, RetrieverService, SemanticSearchRequest

# 定义一套生产环境配置
prod_settings = create_settings(
    MILVUS_HOST="prod-milvus.example.com",
    MILVUS_PORT=19530,
    MILVUS_USER="prod_user",
    MILVUS_PASSWORD="prod_pwd",
    MILVUS_DB_NAME="prod_db",
    TOP_K=20,
    RERANK_ENABLED=True,
)

def semantic_search_on_prod(query: str, query_vector: list[float]):
    with with_settings(prod_settings):  # 在这个上下文中使用 prod_settings
        req = SemanticSearchRequest(
            query=query,
            query_vector=query_vector,
            # 不写 top_k 则使用 prod_settings.TOP_K（20）
        )
        return RetrieverService.semantic_search(req)
```

> 一般情况下，你只要用环境变量 / `.env` 即可。`create_settings` / `with_settings` 适合多环境、多租户场景。

### 3. 每个请求内显式覆盖配置（优先级最高）

所有 Service 的 Request 模型都支持在**单次调用内覆盖部分配置**，例如：

- `SemanticSearchRequest.top_k / rerank_enabled / similarity_threshold / milvus_search_params / milvus_connection`
- `ChunkRequest.strategy / chunk_size / chunk_overlap`
- `CreateCollectionRequest.dense_index_params / sparse_index_params / analyzer_params` 等

这些字段的值会优先于全局 Settings。

---

## 端到端使用示例（第三方调用方视角）

下面示例展示一个典型的 RAG 流程：

1. 一次性创建 Milvus 集合
2. 解析文档 → 切片 → 向量化 → 写入 Milvus
3. 使用查询向量进行语义检索

### 0. 准备：安装额外依赖

```bash
pip install sentence-transformers
```

### 1. 创建集合（只需要执行一次）

```python
from milvus_service import StorageService, CreateCollectionRequest
from pymilvus import FieldSchema, DataType


def ensure_collection():
    fields = [
        # 主键字段
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=False,
        ),
        # 向量字段
        FieldSchema(
            name="vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=768,  # 向量维度需与你的 embedding 模型一致
        ),
        # 文本字段
        FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=65535,
        ),
    ]

    req = CreateCollectionRequest(
        collection_name="docs_demo",
        dimension=768,
        description="demo collection",
        auto_id=False,
        primary_field="id",
        dense_vector_field="vector",
        metadata_fields=fields[1:],  # 除主键外的字段
    )

    StorageService.create_collection(req)


if __name__ == "__main__":
    ensure_collection()
```

### 2. 解析 + 切片 + 向量化 + 写入 Milvus

```python
from milvus_service import (
    ParserService,
    ParseRequest,
    ChunkerService,
    ChunkRequest,
    StorageService,
    InsertRequest,
)
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("BAAI/bge-base-zh-v1.5")


def index_file(file_path: str):
    # 1. 解析文档
    parse_req = ParseRequest(file_path=file_path)
    parse_res = ParserService.parse(parse_req)
    full_text = parse_res.content

    # 2. 切片（使用全局默认策略，也可显式指定）
    chunk_req = ChunkRequest(
        text=full_text,
        # strategy="fixed",
        # chunk_size=512,
        # chunk_overlap=50,
    )
    chunks = ChunkerService.chunk(chunk_req)  # List[ChunkDTO]

    # 3. 向量化
    texts = [c.content for c in chunks]
    vectors = model.encode(texts).tolist()

    # 4. 写入 Milvus
    records = []
    for i, (c, vec) in enumerate(zip(chunks, vectors), start=1):
        records.append(
            {
                "id": i,
                "vector": vec,
                "content": c.content,
                "document_id": 1,
                "chunk_index": c.chunk_index,
            }
        )

    insert_req = InsertRequest(
        collection_name="docs_demo",
        records=records,
    )
    ids = StorageService.insert(insert_req)
    print("insert ids:", ids)


if __name__ == "__main__":
    index_file("your_doc.md")  # 替换为你的文档路径
```

### 3. 语义检索

```python
from milvus_service import RetrieverService, SemanticSearchRequest
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("BAAI/bge-base-zh-v1.5")


def search(query: str):
    query_vector = model.encode([query])[0].tolist()

    req = SemanticSearchRequest(
        query=query,
        query_vector=query_vector,
        top_k=5,                # 不写则使用 Settings.TOP_K
        # 可选：覆盖重排序与阈值
        # rerank_enabled=True,
        # similarity_threshold=0.3,
        # milvus_search_params={"metric_type": "IP", "params": {"nprobe": 16}},
        # 可选：使用 Milvus 表达式做过滤（支持所有检索模式）
        # 例如只检索子块：
        # milvus_expr="chunk_type == 'child'",
    )

    results = RetrieverService.semantic_search(req)
    for r in results:
        print(f"[score={r.score:.3f}] {r.content}")


if __name__ == "__main__":
    search("什么是向量数据库？")
```

---

## 文档解析能力（简要说明）

- **对应 Service**：`ParserService`
- **推荐调用入口**：`ParserService.parse(ParseRequest)`

```python
from milvus_service import ParserService, ParseRequest

req = ParseRequest(
    file_path="document.pdf",
    parser_config={
        "extract_images": True,
        "extract_tables": True,
    },
)
res = ParserService.parse(req)
print(res.content)   # 解析后的纯文本
print(res.metadata)  # 元数据
```

更多字段说明与各格式特定参数详见 `API_DOCUMENTATION.md` 的“4.1 文档解析器 (Parsers)”章节。

---

## 文本切片能力（简要说明）

- **对应 Service**：`ChunkerService`
- **推荐调用入口**：`ChunkerService.chunk(ChunkRequest)`

```python
from milvus_service import ChunkerService, ChunkRequest

req = ChunkRequest(
    text="长文本内容...",
    strategy="fixed",      # 不填则使用 Settings.CHUNK_STRATEGY
    chunk_size=512,        # 不填则使用 Settings.CHUNK_SIZE
    chunk_overlap=50,      # 不填则使用 Settings.CHUNK_OVERLAP
)
chunks = ChunkerService.chunk(req)
for c in chunks:
    print(c.chunk_index, c.content[:50])
```

---

## 检索能力（简要说明）

- **检索 Service**：`RetrieverService`
- **常用请求模型**：
  - `SemanticSearchRequest`
  - `KeywordSearchRequest`
  - `HybridSearchRequest`

示例：关键词检索 & 混合检索：

```python
from milvus_service import (
    RetrieverService,
    KeywordSearchRequest,
    HybridSearchRequest,
)


def keyword_search(query: str):
    req = KeywordSearchRequest(
        query=query,
        top_k=5,
        # 使用 milvus_expr 进行 Milvus 侧过滤（推荐写法）
        milvus_expr="chunk_type == 'child'",
        # extra_params 可透传给具体检索器，老版本也支持在这里传 milvus_expr
        # extra_params={"min_match_count": 1, "milvus_expr": "chunk_type == 'child'"},
        extra_params={"min_match_count": 1},
    )
    return RetrieverService.keyword_search(req)


def hybrid_search(query: str, query_vector: list[float]):
    req = HybridSearchRequest(
        query=query,
        query_vector=query_vector,
        top_k=5,
        # 对语义 + 关键词两路统一做 Milvus 过滤
        milvus_expr="chunk_type == 'child'",
        extra_params={
            "semantic_weight": 0.7,
            "keyword_weight": 0.3,
        },
    )
    return RetrieverService.hybrid_search(req)
```

> 提示：  
> - `milvus_expr` 会在所有检索模式中统一生效（语义 / 关键词 / 混合 / 全文 / 文本匹配 / 短语匹配）。  
> - 表达式是在 Milvus 侧执行的结构化过滤，适合做「按字段过滤 + 文本检索」的组合场景。

---

## 自定义算子（插件）注册

第三方可以扩展自定义解析器 / 切片器 / 检索器 / 存储算子，并通过 `RegistryService` 注册。

```python
from milvus_service import RegistryService, RegisterParserRequest

# 显式注册一个自定义 Parser
RegistryService.register_parser(
    RegisterParserRequest(
        extension=".mydoc",
        module="my_project.my_parsers",
        class_name="MyDocParser",
    )
)

# 或从单个插件文件自动加载
from milvus_service import RegistryService, LoadPluginRequest

RegistryService.load_plugin(
    LoadPluginRequest(plugin_path="plugins/my_parsers.py")
)
```

插件实现需要满足一定约定（解析器 / 切片器 / 检索器 / 存储算子的输入输出接口），
具体规则和示例请参考 `API_DOCUMENTATION.md` 中的插件章节。

---

## 许可证

MIT License
