# Milvus RAG 系统 API 文档

## 目录

1. [项目文件结构](#1-项目文件结构)
2. [继承关系图](#2-继承关系图)
3. [功能模块清单](#3-功能模块清单)
4. [详细功能说明](#4-详细功能说明)
   - [4.1 文档解析器 (Parsers)](#41-文档解析器-parsers)
   - [4.2 文档切片器 (Chunkers)](#42-文档切片器-chunkers)
   - [4.3 检索器 (Retrievers)](#43-检索器-retrievers)
   - [4.4 存储 (Storage Operators)](#44-存储-storage-operators)
   - [4.5 存储客户端 (Storage Client)](#45-存储客户端-storage-client)
   - [4.6 工具函数 (Utils)](#46-工具函数-utils)

---

## 1. 项目文件结构

### 1.1 目录结构

> 说明：本项目已经精简为 **SDK 形态**，只保留文档解析 / 文档切片 / 文档检索 / 文档存储四大核心能力相关代码。

```
BaseVector-Core/
├── ability/                       # 核心算子与基础能力（原 app/）
│   ├── __init__.py
│   ├── config.py                 # 配置管理模块（Settings / get_settings 等）
│   │
│   ├── operators/                # 算子模块（核心功能）
│   │   ├── __init__.py
│   │   ├── base.py               # 算子基类 (BaseOperator)
│   │   ├── decorators.py         # 插件注册装饰器
│   │   ├── plugin_registry.py    # 插件注册机制
│   │   │
│   │   ├── parsers/              # 文档解析器模块
│   │   │   ├── __init__.py
│   │   │   ├── base_parser.py
│   │   │   ├── parser_factory.py
│   │   │   ├── pdf_parser.py
│   │   │   ├── docx_parser.py
│   │   │   ├── pptx_parser.py
│   │   │   ├── html_parser.py
│   │   │   ├── markdown_parser.py
│   │   │   └── txt_parser.py
│   │   │
│   │   ├── chunkers/             # 文档切片器模块
│   │   │   ├── __init__.py
│   │   │   ├── base_chunker.py
│   │   │   ├── chunker_factory.py
│   │   │   ├── fixed_chunker.py
│   │   │   ├── semantic_chunker.py
│   │   │   ├── title_chunker.py
│   │   │   └── parent_child_chunker.py
│   │   │
│   │   ├── retrievers/           # 检索器模块
│   │   │   ├── __init__.py
│   │   │   ├── base_retriever.py
│   │   │   ├── retriever_factory.py
│   │   │   ├── semantic_retriever.py
│   │   │   ├── keyword_retriever.py
│   │   │   ├── hybrid_retriever.py
│   │   │   ├── fulltext_retriever.py
│   │   │   ├── text_match_retriever.py
│   │   │   └── phrase_match_retriever.py
│   │   │
│   │   ├── storage/              # 存储算子模块
│   │   │   ├── __init__.py
│   │   │   ├── base_storage.py
│   │   │   ├── storage_factory.py
│   │   │   ├── collection_operator.py
│   │   │   ├── insert_operator.py
│   │   │   ├── update_operator.py
│   │   │   └── delete_operator.py
│   │
│   ├── storage/                  # 存储层封装
│   │   ├── __init__.py
│   │   └── milvus_client.py      # Milvus 客户端封装
│   │
│   └── utils/                    # 工具函数
│       ├── __init__.py
│       ├── logger.py             # 日志系统
│       ├── text_processing.py    # 文本处理工具
│       ├── hash.py               # 哈希工具函数
│       └── filter_validation.py  # 过滤表达式校验
│
├── milvus_service/               # 对外 SDK 入口（pip install git+ 之后导入此包）
│   ├── __init__.py               # 统一导出 Settings / 工厂 / Service 等
│   └── service/
│       ├── __init__.py
│       ├── parser_service.py     # 文档解析 Service 封装
│       ├── chunker_service.py    # 文本切片 Service 封装
│       ├── retriever_service.py  # 检索 Service 封装
│       ├── storage_service.py    # 存储 Service 封装
│       ├── registry_service.py   # 插件注册 Service 封装
│       └── ability_descriptor.py # 能力自描述
│
├── README.md                     # 项目说明（含使用示例）
├── pyproject.toml                # 打包配置（用于 pip install git+）
├── requirements.txt              # 核心依赖列表（本地开发/运行）
└── API_DOCUMENTATION.md          # API 文档（本文档）
```

### 1.2 核心文件功能说明

| 文件/模块 | 功能描述 |
|----------|---------|
| `ability/config.py` | 配置管理模块，支持环境变量、.env 文件、YAML 配置文件，提供统一的配置接口（Settings / get_settings / create_settings / with_settings 等） |
| `ability/operators/base.py` | 所有算子的抽象基类，提供统一的初始化、验证、错误处理机制 |
| `ability/operators/plugin_registry.py` | 插件注册机制，支持动态注册和加载自定义算子 |
| `ability/operators/decorators.py` | 提供装饰器简化插件注册（@register_parser、@register_chunker 等） |
| `ability/operators/storage/base_storage.py` | 存储算子基类，所有 Milvus 存储操作算子都继承此类 |
| `ability/operators/storage/storage_factory.py` | 存储算子工厂，根据操作类型创建对应的存储算子 |
| `ability/storage/milvus_client.py` | Milvus 向量数据库客户端封装，提供集合管理、数据插入、向量检索等功能 |
| `ability/utils/logger.py` | 基于 loguru 的日志系统，支持控制台和文件输出 |
| `ability/utils/text_processing.py` | 文本处理工具函数（清洗、句子分割、文本截断等） |
| `ability/utils/hash.py` | 文件/流/文本的 MD5 哈希计算工具 |
| `ability/utils/filter_validation.py` | Milvus 表达式和 SQL WHERE 子句的安全校验工具 |

---

## 2. 继承关系图

### 2.1 核心继承关系

```
BaseOperator (抽象基类)
│
├── BaseParser (文档解析器基类)
│   ├── PDFParser
│   ├── DocxParser
│   ├── PPTXParser
│   ├── HTMLParser
│   ├── MarkdownParser
│   └── TXTParser
│
├── BaseChunker (文档切片器基类)
│   ├── FixedChunker (固定窗口切片)
│   ├── SemanticChunker (语义切片)
│   ├── TitleChunker (标题切片)
│   └── ParentChildChunker (父子切片)
│
├── BaseRetriever (检索器基类)
│   ├── SemanticRetriever (语义检索)
│   ├── KeywordRetriever (关键词检索)
│   ├── HybridRetriever (混合检索)
│   ├── FullTextRetriever (全文检索)
│   ├── TextMatchRetriever (文本匹配)
│   └── PhraseMatchRetriever (短语匹配)
│
├── BaseStorageOperator (存储算子基类)
│   ├── CollectionOperator (集合操作)
│   ├── InsertOperator (数据插入)
│   ├── UpdateOperator (数据更新)
│   └── DeleteOperator (数据删除)
```

### 2.2 工厂类关系

所有工厂类都是独立的静态类，提供统一的创建接口：

```
ParserFactory          # 文档解析器工厂
ChunkerFactory         # 文档切片器工厂
RetrieverFactory       # 检索器工厂
StorageFactory         # 存储算子工厂
```

### 2.3 数据类

```
Chunk                  # 文档块数据类（chunkers使用）
RetrievalResult        # 检索结果数据类（retrievers使用）
```

---

## 3. 功能模块清单

### 3.1 文档解析器 (Parsers)

| 功能 | 支持格式 | 自定义支持 |
|------|---------|----------|
| PDF解析 | `.pdf` | ✅ 支持插件扩展 |
| Word解析 | `.docx`, `.doc` | ✅ 支持插件扩展 |
| PPT解析 | `.pptx`, `.ppt` | ✅ 支持插件扩展 |
| HTML解析 | `.html`, `.htm` | ✅ 支持插件扩展 |
| Markdown解析 | `.md`, `.markdown` | ✅ 支持插件扩展 |
| TXT解析 | `.txt` | ✅ 支持插件扩展 |

**自定义方式**: 继承 `BaseParser` 并实现 `_parse()` 方法，使用 `@register_parser` 装饰器注册

### 3.2 文档切片器 (Chunkers)

| 功能 | 策略 | 自定义支持 |
|------|------|----------|
| 固定窗口切片 | `fixed` | ✅ 支持插件扩展 |
| 语义切片 | `semantic` | ✅ 支持插件扩展 |
| 标题切片 | `title` | ✅ 支持插件扩展 |
| 父子切片 | `parent_child` | ✅ 支持插件扩展 |

**自定义方式**: 继承 `BaseChunker` 并实现 `_chunk()` 方法，使用 `@register_chunker` 装饰器注册

### 3.3 检索器 (Retrievers)

| 功能 | 模式 | 自定义支持 |
|------|------|----------|
| 语义检索 | `semantic` | ✅ 支持插件扩展 |
| 关键词检索 | `keyword` | ✅ 支持插件扩展 |
| 混合检索 | `hybrid` (RRF融合) | ✅ 支持插件扩展 |
| 全文检索 | `fulltext` | ✅ 支持插件扩展 |
| 文本匹配 | `text_match` | ✅ 支持插件扩展 |
| 短语匹配 | `phrase_match` | ✅ 支持插件扩展 |

**自定义方式**: 继承 `BaseRetriever` 并实现 `_retrieve()` 方法，使用 `@register_retriever` 装饰器注册

---

## 4. 详细功能说明

## 4.1 文档解析器 (Parsers)

### 4.1.1 ParserFactory - 解析器工厂

#### 创建解析器

**方法**: `create_parser(file_path: Path | str, config: Optional[Dict] = None) -> BaseParser`

**功能**: 根据文件扩展名自动创建对应的解析器

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `file_path` | `Path \| str` | ✅ | - | 待解析文件的路径 |
| `config` | `Optional[Dict]` | ❌ | `None` | 解析器配置字典，格式特定的配置参数 |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `BaseParser` | 解析器实例，根据文件扩展名自动选择对应的解析器类型 |

**示例**:

```python
from ability.operators.parsers.parser_factory import ParserFactory

# 创建PDF解析器
parser = ParserFactory.create_parser("document.pdf")

# 带配置创建
parser = ParserFactory.create_parser(
   "document.pdf",
   config={"extract_images": True, "extract_tables": True}
)
```

#### 获取支持的文件格式

**方法**: `get_supported_extensions() -> list[str]`

**功能**: 获取所有支持的文件扩展名（包括插件）

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `list[str]` | 扩展名列表，例如：`[".pdf", ".docx", ".txt", ".md", ...]`，包括内置解析器和已注册的插件解析器 |

#### 检查文件格式是否支持

**方法**: `is_supported(file_path: Path | str) -> bool`

**功能**: 检查文件格式是否支持

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `file_path` | `Path \| str` | ✅ | - | 待检查的文件路径 |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `bool` | `True` 表示文件格式支持，`False` 表示不支持 |

---

### 4.1.2 BaseParser - 解析器基类方法

#### 解析文档

**方法**: `process(input_data: Any, **kwargs) -> Dict[str, Any]`

**功能**: 解析文档，返回结构化的内容、元数据和文档结构

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `input_data` | `Path \| str` | ✅ | - | 待解析文件的路径 |
| `**kwargs` | `Any` | ❌ | - | 额外的处理参数，不同格式的解析器可能有不同的参数 |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `content` | `str` | 文档内容，以Markdown格式输出 |
| `metadata` | `Dict[str, Any]` | 文档元数据，包含标题、作者、创建时间、修改时间等信息（不同格式字段可能不同） |
| `structure` | `Dict[str, Any]` | 文档结构信息，包含页面、段落、表格、图片等结构化信息（不同格式字段可能不同） |

**示例**:
```python
result = parser.process("document.pdf")
print(result["content"])      # Markdown格式的文档内容
print(result["metadata"])     # 元数据字典
print(result["structure"])    # 结构信息
```

#### 提取纯文本

**方法**: `extract_text(file_path: Path) -> str`

**功能**: 快速提取文档的纯文本内容

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `file_path` | `Path` | ✅ | - | 待提取文本的文件路径 |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `str` | 文档的纯文本内容 |

#### 提取元数据

**方法**: `extract_metadata(file_path: Path) -> Dict[str, Any]`

**功能**: 快速提取文档元数据

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `file_path` | `Path` | ✅ | - | 待提取元数据的文件路径 |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `Dict[str, Any]` | 文档元数据字典，包含标题、作者、创建时间等字段（不同格式字段可能不同） |

---

### 4.1.3 各格式解析器特定参数

#### PDFParser

**配置参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `extract_images` | `bool` | ❌ | `False` | 是否提取图片信息（提取图片的元数据，不提取图片二进制数据） |
| `extract_tables` | `bool` | ❌ | `True` | 是否提取表格（表格会转换为Markdown格式） |

**返回的metadata字段**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `title` | `str` | PDF标题 |
| `author` | `str` | 作者 |
| `subject` | `str` | 主题 |
| `creator` | `str` | 创建者 |
| `producer` | `str` | 生成器 |
| `creation_date` | `str` | 创建日期 |
| `modification_date` | `str` | 修改日期 |
| `page_count` | `int` | 总页数 |

**返回的structure字段**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `pages` | `List[Dict]` | 页面列表，每个页面包含 `page_number` 和 `blocks`（文本块列表） |
| `tables` | `List[Dict]` | 表格列表，每个表格包含 `page`（页码）、`table_index`、`rows`、`cols` |
| `images` | `List[Dict]` | 图片列表（当 `extract_images=True` 时），每个图片包含 `page`、`image_index`、`xref` |

**示例**:
```python
parser = PDFParser(config={
    "extract_images": True,
    "extract_tables": True
})
```

#### DocxParser / PPTXParser

**配置参数**: 无特殊配置参数

**返回的metadata字段**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `title` | `str` | 文档标题 |
| `author` | `str` | 作者 |
| `subject` | `str` | 主题 |
| `keywords` | `str` | 关键词 |
| `comments` | `str` | 注释 |
| `created` | `str` | 创建时间 |
| `modified` | `str` | 修改时间 |

**DocxParser返回的structure字段**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `paragraphs` | `List[Dict]` | 段落列表，每个段落包含 `index`、`type`（"heading"或"paragraph"）、`text`，如果是标题还包含 `level` |
| `tables` | `List[Dict]` | 表格列表，每个表格包含 `table_index`、`rows`、`cols` |

**PPTXParser返回的structure字段**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `slides` | `List[Dict]` | 幻灯片列表，每个幻灯片包含 `slide_number`、`shapes`（形状列表，包含 `type` 和 `text`） |

#### HTMLParser

**配置参数**: 无特殊配置参数

**返回的metadata字段**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `title` | `str` | HTML标题（来自`<title>`标签） |
| `author` | `str` | 作者（来自`<meta name="author">`） |
| `description` | `str` | 描述（来自`<meta name="description">`） |

**返回的structure字段**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `headings` | `List[Dict]` | 标题列表，每个标题包含 `level`（1-6）和 `text` |
| `paragraphs` | `List[Dict]` | 段落列表，每个段落包含 `text` |
| `links` | `List[Dict]` | 链接列表，每个链接包含 `text` 和 `url` |

#### MarkdownParser

**配置参数**: 无特殊配置参数

**返回的metadata字段**: 如果文档包含YAML Front Matter，则解析其中的键值对

**返回的structure字段**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `headings` | `List[Dict]` | 标题列表，每个标题包含 `level` 和 `text` |
| `code_blocks` | `List[Dict]` | 代码块列表（当前版本未实现详细解析） |

#### TXTParser

**配置参数**: 无特殊配置参数

**返回的metadata字段**: 空字典

**返回的structure字段**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `lines` | `int` | 总行数 |
| `paragraphs` | `int` | 段落数（按空行分隔） |

---

## 4.2 文档切片器 (Chunkers)

### 4.2.1 ChunkerFactory - 切片器工厂

#### 创建切片器

**方法**: `create_chunker(strategy: Optional[str] = None, config: Optional[Dict] = None) -> BaseChunker`

**功能**: 根据策略创建对应的切片器

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `strategy` | `Optional[str]` | ❌ | 配置中的 `CHUNK_STRATEGY` | 切片策略，可选值：<br/>- `"fixed"`: 固定窗口切片<br/>- `"semantic"`: 语义切片<br/>- `"title"`: 标题切片<br/>- `"parent_child"`: 父子切片 |
| `config` | `Optional[Dict]` | ❌ | `None` | 切片器配置字典，不同策略的配置参数不同 |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `BaseChunker` | 切片器实例，根据strategy参数选择对应的切片器类型 |

**示例**:

```python
from ability.operators.chunkers.chunker_factory import ChunkerFactory

# 创建固定窗口切片器
chunker = ChunkerFactory.create_chunker(
   strategy="fixed",
   config={"chunk_size": 512, "chunk_overlap": 50}
)

# 创建语义切片器
chunker = ChunkerFactory.create_chunker(
   strategy="semantic",
   config={
      "similarity_threshold": 0.7,
      "min_chunk_size": 100,
      "max_chunk_size": 1000
   }
)
```

#### 获取支持的策略

**方法**: `get_supported_strategies() -> list[str]`

**功能**: 获取所有支持的切片策略（包括插件）

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `list[str]` | 支持的策略列表，例如：`["fixed", "semantic", "title", "parent_child"]`，包括内置切片器和已注册的插件切片器 |

---

### 4.2.2 BaseChunker - 切片器基类方法

#### 切片文档

**方法**: `process(input_data: str, **kwargs) -> List[Chunk]`

**功能**: 将文本切分成文档块列表

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `input_data` | `str` | ✅ | - | 待切片的文本内容 |
| `whitespace_pattern` | `Optional[str]` | ❌ | `None` | 空白字符正则表达式，用于文本清洗，默认使用内置模式 |
| `sentence_delimiters` | `Optional[str]` | ❌ | `None` | 句子分隔符正则表达式，用于句子分割，默认：`r"[。！？.!?]\s*"` |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `List[Chunk]` | 文档块列表，按顺序排列 |

**Chunk对象字段**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `content` | `str` | 块的内容文本 |
| `chunk_index` | `int` | 块的索引（从0开始） |
| `start_index` | `int` | 块在原文中的起始字符位置 |
| `end_index` | `int` | 块在原文中的结束字符位置 |
| `metadata` | `Dict[str, Any]` | 块的元数据，包含策略信息、块大小等 |
| `parent_chunk_id` | `Optional[int]` | 父块ID（用于父子切片策略，普通切片为`None`） |

**示例**:
```python
chunks = chunker.process("这是一段很长的文本...")
for chunk in chunks:
    print(f"块 {chunk.chunk_index}: {chunk.content[:50]}...")
    print(f"位置: {chunk.start_index}-{chunk.end_index}")
```

---

### 4.2.3 FixedChunker - 固定窗口切片器

**功能**: 按固定大小（字符数或Token数）切分文档

**配置参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `chunk_size` | `int` | ❌ | `512` | 块大小，单位：字符数（如果`use_tokens=False`）或Token数（如果`use_tokens=True`） |
| `chunk_overlap` | `int` | ❌ | `50` | 重叠大小，单位与`chunk_size`相同，相邻块之间的重叠字符/Token数 |
| `use_tokens` | `bool` | ❌ | `False` | 是否使用Token计数（`True`）而非字符数（`False`）。当前Token计数为简化实现，实际应用中建议使用tiktoken等库 |
| `min_content_ratio` | `float` | ❌ | `0.5` | 最小内容保留比例（0-1），在句子边界截断时，如果找到的边界位置小于此比例，则不截断 |

**示例**:
```python
chunker = FixedChunker(config={
    "chunk_size": 512,
    "chunk_overlap": 50,
    "use_tokens": False,
    "min_content_ratio": 0.5
})
```

---

### 4.2.4 SemanticChunker - 语义切片器

**功能**: 基于语义相似度在话题切换处自动断句

**配置参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `chunk_size` | `int` | ❌ | `512` | 基础块大小参考值（字符数） |
| `chunk_overlap` | `int` | ❌ | `50` | 重叠大小（字符数），实际会根据语义边界调整 |
| `similarity_threshold` | `float` | ❌ | `0.7` | 语义相似度阈值（0-1），相邻句子相似度低于此值会在该处分割 |
| `min_chunk_size` | `int` | ❌ | `100` | 最小块大小（字符数），即使相似度低也不会分割小于此大小的块 |
| `max_chunk_size` | `int` | ❌ | `1000` | 最大块大小（字符数），超过此大小会强制分割 |
| `model_name` | `str` | ❌ | `"BAAI/bge-small-zh-v1.5"` | 用于计算语义相似度的模型名称（需支持sentence-transformers） |

**kwargs参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `sentence_delimiters` | `Optional[str]` | ❌ | `None` | 句子分隔符正则表达式，默认：`r"[。！？.!?]\s*"` |

**示例**:
```python
chunker = SemanticChunker(config={
    "similarity_threshold": 0.7,
    "min_chunk_size": 100,
    "max_chunk_size": 1000,
    "model_name": "BAAI/bge-small-zh-v1.5"
})
```

---

### 4.2.5 TitleChunker - 标题切片器

**功能**: 按照Markdown标题层级进行切片

**配置参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `chunk_size` | `int` | ❌ | `512` | 基础块大小参考值（字符数），实际按标题边界分割 |
| `chunk_overlap` | `int` | ❌ | `50` | 重叠大小（字符数），此参数在当前实现中未使用 |
| `max_depth` | `int` | ❌ | `4` | 最大标题层级，对应Markdown的H1-H4（1-4级标题），超过此层级的标题不会被识别为分割点 |
| `include_headers` | `bool` | ❌ | `True` | 是否在块中包含标题行，`True`表示包含，`False`表示只包含标题下的内容 |

**返回说明**: 如果文本中没有标题，会将整个文本作为一个块返回。

**示例**:
```python
chunker = TitleChunker(config={
    "max_depth": 4,
    "include_headers": True
})
```

---

### 4.2.6 ParentChildChunker - 父子切片器

**功能**: 生成大粒度父块（含完整上下文）和小粒度子块（用于精准匹配）

**配置参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `parent_size` | `int` | ❌ | `2000` | 父块大小（字符数），用于保留更多上下文信息 |
| `child_size` | `int` | ❌ | `512` | 子块大小（字符数），用于精准匹配 |
| `child_overlap` | `int` | ❌ | `50` | 子块之间的重叠大小（字符数） |
| `min_content_ratio` | `float` | ❌ | `0.5` | 最小内容保留比例（0-1），在句子边界截断时使用 |

**返回说明**: 
- 返回的Chunk列表包含父块和子块，顺序为：父块1、父块1的所有子块、父块2、父块2的所有子块...
- 子块的 `parent_chunk_id` 指向父块的 `chunk_index`
- 父块的 `parent_chunk_id` 为 `None`

**示例**:
```python
chunker = ParentChildChunker(config={
    "parent_size": 2000,
    "child_size": 512,
    "child_overlap": 50
})
```

---

## 4.3 检索器 (Retrievers)

### 4.3.1 RetrieverFactory - 检索器工厂

#### 创建检索器

**方法**: `create_retriever(mode: Optional[str] = None, config: Optional[Dict] = None) -> BaseRetriever`

**功能**: 根据模式创建对应的检索器

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `mode` | `Optional[str]` | ❌ | 配置中的 `SEARCH_MODE` | 检索模式，可选值：<br/>- `"semantic"`: 语义检索（向量检索）<br/>- `"keyword"`: 关键词检索<br/>- `"hybrid"`: 混合检索（RRF融合）<br/>- `"fulltext"`: 全文检索（基于LIKE操作符）<br/>- `"text_match"`: 文本匹配（精确/模糊匹配）<br/>- `"phrase_match"`: 短语匹配（短语精确匹配） |
| `config` | `Optional[Dict]` | ❌ | `None` | 检索器配置字典，不同模式的配置参数不同 |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `BaseRetriever` | 检索器实例，根据mode参数选择对应的检索器类型 |

**示例**:

```python
from ability.operators.retrievers.retriever_factory import RetrieverFactory

# 创建语义检索器
retriever = RetrieverFactory.create_retriever(
   mode="semantic",
   config={"top_k": 10}
)

# 创建混合检索器
retriever = RetrieverFactory.create_retriever(
   mode="hybrid",
   config={
      "top_k": 10,
      "semantic_weight": 0.6,
      "keyword_weight": 0.4,
      "fusion_method": "rrf"
   }
)
```

#### 获取支持的模式

**方法**: `get_supported_modes() -> list[str]`

**功能**: 获取所有支持的检索模式（包括插件）

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `list[str]` | 支持的检索模式列表，例如：`["semantic", "keyword", "hybrid", "fulltext", "text_match", "phrase_match"]`，包括内置检索器和已注册的插件检索器 |

---

### 4.3.2 BaseRetriever - 检索器基类方法

#### 检索文档

**方法**: `process(query: str, top_k: Optional[int] = None, tenant_id: Optional[str] = None, **kwargs) -> List[RetrievalResult]`

**功能**: 检索相关文档块

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `query` | `str` | ✅ | - | 查询文本 |
| `top_k` | `Optional[int]` | ❌ | 配置中的 `TOP_K` 或实例的 `top_k` | 返回Top-K结果的数量 |
| `tenant_id` | `Optional[str]` | ❌ | `None` | 租户ID，用于多租户数据隔离，如果不指定则使用配置中的 `DEFAULT_TENANT_ID` |
| `milvus_expr` | `Optional[str]` | ❌ | `None` | Milvus过滤表达式（通过kwargs传入），需要先通过 `validate_milvus_expr()` 安全校验，例如：`'document_id > 100'` |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `List[RetrievalResult]` | 检索结果列表，按相似度分数降序排列 |

**RetrievalResult对象字段**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `chunk_id` | `int` | 文档块的ID（Milvus中的主键） |
| `document_id` | `int` | 文档ID（所属的原始文档ID） |
| `content` | `str` | 块的内容文本 |
| `score` | `float` | 相似度分数（0-1之间，分数越高越相似） |
| `metadata` | `Dict[str, Any]` | 元数据字典，包含：<br/>- `chunk_index`: 块在文档中的索引<br/>- `parent_chunk_id`: 父块ID（如果有）<br/>- `tenant_id`: 租户ID<br/>- 其他自定义元数据字段 |

**示例**:
```python
results = retriever.process(
    query="人工智能的应用",
    top_k=10,
    tenant_id="tenant_001"
)

for result in results:
    print(f"得分: {result.score:.4f}")
    print(f"内容: {result.content[:100]}...")
    print(f"文档ID: {result.document_id}")
```

---

### 4.3.3 SemanticRetriever - 语义检索器

**功能**: 基于Milvus向量检索的语义检索

**配置参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `top_k` | `int` | ❌ | `10` | 返回Top-K结果的数量 |

**kwargs参数**: 
- `query_vector`: 查询向量（List[float]），**必需参数**，用户需要自行使用嵌入模型生成向量

**示例**:
```python
from sentence_transformers import SentenceTransformer

# 用户自行选择和使用嵌入模型
model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
query_text = "查询文本"
query_vector = model.encode(query_text).tolist()

retriever = SemanticRetriever(config={"top_k": 10})

results = retriever.process(
    query=query_text,
    query_vector=query_vector,  # 必需：用户提供的查询向量
    top_k=10,
    tenant_id="tenant_001",
    milvus_expr='document_id > 100'  # 需要先校验
)
```

**注意**: 
- 用户需要自行选择和使用嵌入模型（如 sentence-transformers、OpenAI API 等）生成查询向量
- Milvus 本身支持稠密向量和稀疏向量（BM25），用户可以根据需求选择合适的向量类型

---

### 4.3.4 KeywordRetriever - 关键词检索器

**功能**: 基于关键词匹配和TF-IDF风格评分的检索

**配置参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `top_k` | `int` | ❌ | `10` | 返回Top-K结果的数量 |
| `min_match_count` | `int` | ❌ | `1` | 最少匹配关键词数量，如果匹配的关键词数量少于此值，则返回分数0 |
| `tf_normalization_factor` | `float` | ❌ | `100.0` | TF归一化因子，用于计算词频权重，影响分数计算 |
| `candidate_multiplier` | `int` | ❌ | `10` | 候选结果倍数，用于从Milvus获取更多候选结果（`top_k * candidate_multiplier`）以便进行评分排序 |

**kwargs参数**: 同 `BaseRetriever.process()` 的kwargs参数

**示例**:
```python
retriever = KeywordRetriever(config={
    "top_k": 10,
    "min_match_count": 1,
    "tf_normalization_factor": 100.0,
    "candidate_multiplier": 10
})
```

**工作原理**:
1. 对查询文本进行分词（支持中英文）
2. 使用Milvus的LIKE操作符进行关键词匹配
3. 对匹配结果进行TF-IDF风格评分
4. 按分数排序并返回Top-K结果

---

### 4.3.5 HybridRetriever - 混合检索器

**功能**: 结合语义检索和关键词检索，使用RRF（倒数排名融合）算法

**配置参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `top_k` | `int` | ❌ | `10` | 返回Top-K结果的数量 |
| `semantic_weight` | `float` | ❌ | `0.5` | 语义检索权重，用于融合分数计算，建议总和为1.0 |
| `keyword_weight` | `float` | ❌ | `0.5` | 关键词检索权重，用于融合分数计算，建议总和为1.0 |
| `fusion_method` | `str` | ❌ | `"rrf"` | 融合方法，可选值：<br/>- `"rrf"`: RRF（倒数排名融合）算法<br/>- `"weighted"`: 加权融合算法<br/>- `"custom"`: 自定义融合函数（需要提供 `fusion_function`）<br/>- 也可以直接传入callable对象作为自定义融合函数 |
| `fusion_function` | `Optional[Callable]` | ❌ | `None` | 自定义融合函数，当`fusion_method="custom"`时使用，函数签名：<br/>`(semantic_results: List[RetrievalResult], keyword_results: List[RetrievalResult], top_k: int) -> List[RetrievalResult]` |
| `rrf_k` | `int` | ❌ | `60` | RRF算法的k参数，用于计算RRF分数（公式：1 / (k + rank)） |
| `semantic_config` | `Dict` | ❌ | `{}` | 语义检索器配置字典，传递给内部的SemanticRetriever（注意：SemanticRetriever 需要用户提供 query_vector） |
| `keyword_config` | `Dict` | ❌ | `{}` | 关键词检索器配置字典，传递给内部的KeywordRetriever |

**kwargs参数**: 同 `BaseRetriever.process()` 的kwargs参数，会传递给内部的语义检索器和关键词检索器  
- `query_vector`: 查询向量（List[float]），**必需参数**（用于语义检索），用户需要自行使用嵌入模型生成向量

**示例**:
```python
retriever = HybridRetriever(config={
    "top_k": 10,
    "semantic_weight": 0.6,
    "keyword_weight": 0.4,
    "fusion_method": "rrf",
    "rrf_k": 60,
    "semantic_config": {
        "top_k": 20
    },
    "keyword_config": {
        "top_k": 20
    }
})

# 自定义融合函数
def custom_fusion(semantic_results, keyword_results, top_k):
    # 自定义融合逻辑
    return fused_results

retriever = HybridRetriever(config={
    "fusion_method": "custom",
    "fusion_function": custom_fusion
})
```

**RRF算法说明**:
- RRF分数 = 1 / (k + rank)，其中k是 `rrf_k` 参数
- 融合分数 = `semantic_weight * semantic_rrf_score + keyword_weight * keyword_rrf_score`

---

### 4.3.6 FullTextRetriever - 全文检索器

**功能**: 基于 Milvus LIKE 操作符实现全文检索，支持对查询文本进行分词，然后使用 LIKE 操作符进行模糊匹配

**配置参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `top_k` | `int` | ❌ | `10` | 返回Top-K结果的数量 |
| `min_match_count` | `int` | ❌ | `1` | 最少匹配的关键词数量，如果匹配的关键词数量少于此值，则返回分数0 |
| `candidate_multiplier` | `int` | ❌ | `10` | 候选结果倍数，用于从Milvus获取更多候选结果（`top_k * candidate_multiplier`）以便进行评分排序 |
| `match_mode` | `str` | ❌ | `"or"` | 匹配模式，可选值：<br/>- `"or"`: 任一关键词匹配即可<br/>- `"and"`: 所有关键词都必须匹配 |

**kwargs参数**: 同 `BaseRetriever.process()` 的kwargs参数

**示例**:
```python
retriever = FullTextRetriever(config={
    "top_k": 10,
    "min_match_count": 1,
    "candidate_multiplier": 10,
    "match_mode": "or"  # 或 "and"
})

results = retriever.process(
    query="人工智能 机器学习",
    top_k=10,
    tenant_id="tenant_001"
)
```

**工作原理**:
1. 对查询文本进行分词（支持中英文）
2. 使用Milvus的LIKE操作符构建过滤表达式
3. 根据匹配模式（OR/AND）组合关键词过滤条件
4. 对匹配结果进行评分（基于关键词匹配频率）
5. 按分数排序并返回Top-K结果

---

### 4.3.7 TextMatchRetriever - 文本匹配检索器

**功能**: 支持精确匹配和模糊匹配的文本检索

**配置参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `top_k` | `int` | ❌ | `10` | 返回Top-K结果的数量 |
| `match_type` | `str` | ❌ | `"fuzzy"` | 匹配类型，可选值：<br/>- `"exact"`: 精确匹配（使用 `==` 操作符）<br/>- `"fuzzy"`: 模糊匹配（使用 `LIKE` 操作符） |
| `case_sensitive` | `bool` | ❌ | `False` | 是否区分大小写，`False`表示不区分大小写 |

**kwargs参数**: 同 `BaseRetriever.process()` 的kwargs参数

**示例**:
```python
# 模糊匹配检索器
retriever = TextMatchRetriever(config={
    "top_k": 10,
    "match_type": "fuzzy",
    "case_sensitive": False
})

# 精确匹配检索器
retriever = TextMatchRetriever(config={
    "top_k": 10,
    "match_type": "exact",
    "case_sensitive": False
})

results = retriever.process(
    query="查询文本",
    top_k=10,
    tenant_id="tenant_001"
)
```

**工作原理**:
1. 根据配置的匹配类型构建Milvus查询表达式
   - 精确匹配：使用 `==` 操作符（大小写不敏感时使用 `LIKE` 模拟）
   - 模糊匹配：使用 `LIKE` 操作符进行模糊匹配
2. 对匹配结果进行评分
   - 精确匹配：完全匹配时返回1.0，否则返回0.0
   - 模糊匹配：根据匹配次数和位置计算分数
3. 按分数排序并返回Top-K结果

---

### 4.3.8 PhraseMatchRetriever - 短语匹配检索器

**功能**: 支持短语精确匹配，用于查找包含完整短语的文档，短语中的词序和位置都会被考虑

**配置参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `top_k` | `int` | ❌ | `10` | 返回Top-K结果的数量 |
| `case_sensitive` | `bool` | ❌ | `False` | 是否区分大小写，`False`表示不区分大小写 |
| `allow_partial` | `bool` | ❌ | `False` | 是否允许部分匹配，`False`表示必须完全匹配短语 |

**kwargs参数**: 同 `BaseRetriever.process()` 的kwargs参数

**示例**:
```python
retriever = PhraseMatchRetriever(config={
    "top_k": 10,
    "case_sensitive": False,
    "allow_partial": False
})

results = retriever.process(
    query="人工智能技术",
    top_k=10,
    tenant_id="tenant_001"
)
```

**工作原理**:
1. 构建短语匹配表达式（使用Milvus的LIKE操作符）
2. 查询包含完整短语的文档
3. 对匹配结果进行评分
   - 计算短语出现次数
   - 计算第一个匹配位置（越靠前分数越高）
   - 计算短语长度比例（短语越长，匹配越精确）
4. 综合以上因素计算最终分数
5. 按分数排序并返回Top-K结果

**适用场景**:
- 需要精确匹配特定短语的场景
- 对词序敏感的场景
- 需要查找包含完整短语的文档

---

## 4.4 存储 (Storage Operators)

### 4.4.1 StorageFactory - 存储工厂

#### 创建存储

**方法**: `create_operator(operation: Optional[str] = None, config: Optional[Dict] = None) -> BaseStorageOperator`

**功能**: 根据操作类型创建对应的存储

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `operation` | `Optional[str]` | ✅ | - | 操作类型，可选值：<br/>- `"collection"`: 集合操作（创建、删除、列出、获取）<br/>- `"insert"`: 数据插入<br/>- `"update"`: 数据更新<br/>- `"delete"`: 数据删除 |
| `config` | `Optional[Dict]` | ❌ | `None` | 算子配置字典，不同操作的配置参数不同 |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `BaseStorageOperator` | 存储算子实例，根据operation参数选择对应的算子类型 |

**示例**:

```python
from ability.operators.storage import StorageFactory

# 创建集合操作算子
collection_op = StorageFactory.create_operator("collection")

# 创建数据插入算子
insert_op = StorageFactory.create_operator("insert", config={"tenant_id": "tenant001"})
```

#### 获取支持的操作类型

**方法**: `get_supported_operations() -> list[str]`

**功能**: 获取所有支持的操作类型

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `list[str]` | 操作类型列表，例如：`["collection", "insert", "update", "delete"]` |

---

### 4.4.2 CollectionOperator - 集合操作

**功能**: 支持创建、删除、列出、获取Milvus集合

#### 处理集合操作

**方法**: `process(input_data: Any, **kwargs) -> Any`

**功能**: 处理集合操作

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `input_data` | `Any` | ✅ | - | 操作类型（"create", "delete", "list", "get", "exists"）或从kwargs获取 |
| `**kwargs` | `Any` | ❌ | - | 操作参数<br/>- `operation`: 操作类型（如果input_data不是字符串，则从kwargs获取）<br/>- `collection_name`: 集合名称（可选）<br/>- `tenant_id`: 租户ID（可选）<br/>- 其他参数根据操作类型而定 |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `Any` | 操作结果：<br/>- `create`: Collection对象<br/>- `delete`: None<br/>- `list`: 集合名称列表<br/>- `get`: Collection对象<br/>- `exists`: bool |

**示例**:

```python
from ability.operators.storage import StorageFactory

collection_op = StorageFactory.create_operator("collection")

# 创建集合
collection = collection_op.process("create",
                                   collection_name="my_collection",
                                   dimension=768,
                                   dense_vector_field="vector",
                                   metadata_fields=[...]
                                   )

# 列出所有集合
collections = collection_op.process("list")

# 检查集合是否存在
exists = collection_op.process("exists",
                               collection_name="my_collection"
                               )
```

---

### 4.4.3 InsertOperator - 数据插入

**功能**: 支持批量插入数据到Milvus集合

#### 插入数据

**方法**: `process(input_data: Any, **kwargs) -> List[int]`

**功能**: 插入数据到Milvus集合

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `input_data` | `List[Dict[str, Any]]` | ✅ | - | 要插入的数据列表，每个元素是一个字典，必须包含集合schema中定义的所有字段 |
| `**kwargs` | `Any` | ❌ | - | 额外参数<br/>- `collection_name`: 集合名称（可选，从config或tenant_id生成）<br/>- `tenant_id`: 租户ID（可选） |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `List[int]` | 插入的ID列表 |

**示例**:

```python
from ability.operators.storage import StorageFactory

insert_op = StorageFactory.create_operator("insert")

data = [
   {
      "id": 1,
      "vector": [0.1, 0.2, ...],
      "content": "文档内容",
      "document_id": 1,
      "tenant_id": "tenant001",
      "metadata": '{"key": "value"}'
   }
]

ids = insert_op.process(data, collection_name="my_collection")
```

---

### 4.4.4 UpdateOperator - 数据更新

**功能**: 支持通过表达式更新Milvus集合中的数据（通过delete + insert方式实现）

#### 更新数据

**方法**: `process(input_data: Any, **kwargs) -> Dict[str, Any]`

**功能**: 更新数据

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `input_data` | `List[Dict[str, Any]]` | ✅ | - | 更新的数据列表（每个元素是字典，包含要更新的字段） |
| `**kwargs` | `Any` | ❌ | - | 额外参数<br/>- `collection_name`: 集合名称（可选）<br/>- `tenant_id`: 租户ID（可选）<br/>- `expr`: 过滤表达式，用于确定要更新的数据（如 "id in [1, 2, 3]"）<br/>- `new_data`: 新数据列表（如果input_data不是列表，从kwargs获取） |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `Dict[str, Any]` | 更新结果字典：<br/>- `deleted_count`: 删除的记录数<br/>- `inserted_count`: 插入的记录数<br/>- `inserted_ids`: 新插入的ID列表 |

**注意**: Milvus的更新操作实际上是通过delete + insert实现的。如果需要更新部分字段，请先查询原数据，合并字段后重新插入。

---

### 4.4.5 DeleteOperator - 数据删除

**功能**: 支持通过表达式或ID列表删除Milvus集合中的数据

#### 删除数据

**方法**: `process(input_data: Any, **kwargs) -> Dict[str, Any]`

**功能**: 删除数据

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `input_data` | `str \| List[Union[int, str]] \| None` | ✅ | - | 过滤表达式字符串或ID列表，或从kwargs获取 |
| `**kwargs` | `Any` | ❌ | - | 额外参数<br/>- `collection_name`: 集合名称（可选）<br/>- `tenant_id`: 租户ID（可选）<br/>- `expr`: 过滤表达式（可选，优先级低于input_data）<br/>- `ids`: ID列表（可选，优先级低于input_data） |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `Dict[str, Any]` | 删除结果字典：<br/>- `deleted_count`: 删除的记录数（估算，Milvus不直接返回）<br/>- `expr`: 使用的表达式 |

**示例**:

```python
from ability.operators.storage import StorageFactory

delete_op = StorageFactory.create_operator("delete")

# 通过表达式删除
result = delete_op.process("id in [1, 2, 3]", collection_name="my_collection")

# 通过ID列表删除
result = delete_op.process([1, 2, 3], collection_name="my_collection")
```

---

## 4.5 存储客户端 (Storage Client)

### 4.5.1 MilvusClient - Milvus客户端

#### 连接Milvus

**方法**: `connect() -> None`

**功能**: 连接到Milvus服务器

**环境变量/配置**:
- `MILVUS_HOST` (默认: "localhost")
- `MILVUS_PORT` (默认: 19530)
- `MILVUS_USER` (可选)
- `MILVUS_PASSWORD` (可选)

#### 创建集合

**方法**: `create_collection(collection_name: str, dimension: Optional[int] = None, description: str = "", auto_id: bool = True, primary_field: str = "id", dense_vector_field: Optional[str] = None, sparse_vector_field: Optional[str] = None, primary_dtype: DataType = DataType.INT64, metadata_fields: Optional[List[FieldSchema]] = None, varchar_max_length: int = 255, dense_index_params: Optional[Dict[str, Any]] = None, sparse_index_params: Optional[Dict[str, Any]] = None) -> Collection`

**功能**: 创建Milvus集合（Collection）

**重要说明**:
- **支持动态选择向量字段类型**：可以选择创建稠密向量字段和/或稀疏向量字段（BM25）
- **稠密向量**：用于语义检索，需要用户使用嵌入模型（如 sentence-transformers、OpenAI API）生成
- **稀疏向量**：用于BM25检索，需要用户使用BM25算法生成（Milvus 2.4+ 支持 SPARSE_FLOAT_VECTOR）
- 可以同时创建两种向量字段，用于混合检索场景

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `collection_name` | `str` | ✅ | - | 集合名称 |
| `dimension` | `Optional[int]` | 条件必需 | `None` | 稠密向量维度（当 `dense_vector_field` 不为 None 时必需） |
| `description` | `str` | ❌ | `""` | 集合描述 |
| `auto_id` | `bool` | ❌ | `True` | 是否自动生成ID |
| `primary_field` | `str` | ❌ | `"id"` | 主键字段名 |
| `dense_vector_field` | `Optional[str]` | ❌ | `None` | 稠密向量字段名；**统一命名建议**：启用稠密向量时传 `"vector"`；为 `None` 则不创建稠密向量字段 |
| `sparse_vector_field` | `Optional[str]` | ❌ | `None` | 稀疏向量字段名（BM25）；**统一命名建议**：启用稀疏向量时传 `"sparse_vector"`；为 `None` 则不创建稀疏向量字段 |
| `primary_dtype` | `DataType` | ❌ | `DataType.INT64` | 主键字段类型 |
| `metadata_fields` | `Optional[List[FieldSchema]]` | ❌ | `None` | 额外的元数据字段 |
| `varchar_max_length` | `int` | ❌ | `255` | VARCHAR字段的最大长度 |
| `dense_index_params` | `Optional[Dict[str, Any]]` | ❌ | `None` | 稠密向量索引参数字典，支持自定义索引类型和参数。如果为None，则使用全局配置（`MILVUS_INDEX_TYPE`、`MILVUS_METRIC_TYPE`、`MILVUS_NLIST`） |
| `sparse_index_params` | `Optional[Dict[str, Any]]` | ❌ | `None` | 稀疏向量索引参数字典，如果为None，则使用默认配置（`SPARSE_INVERTED_INDEX`） |

**注意**: 
- 本项目 **默认不创建向量字段**：`dense_vector_field=None` 且 `sparse_vector_field=None`（仅存储文本与元数据）
- 若启用稠密向量（`dense_vector_field="vector"`），则 `dimension` 参数必需
- 若启用稀疏向量（`sparse_vector_field="sparse_vector"`），需要 Milvus 2.4+（`SPARSE_FLOAT_VECTOR`）支持

**dense_index_params参数说明**（稠密向量索引）:

支持以下索引类型：

1. **FLAT**（精确搜索，适合小规模数据）:
```python
index_params = {
    "metric_type": "L2",  # 或 "IP"（内积）
    "index_type": "FLAT"
}
```

2. **IVF_FLAT**（倒排索引，适合大规模数据）:
```python
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}  # nlist: 聚类中心数量，建议为数据量的1/16到1/64
}
```

3. **IVF_SQ8**（量化索引，节省存储空间）:
```python
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_SQ8",
    "params": {"nlist": 1024}
}
```

4. **IVF_PQ**（乘积量化，进一步压缩）:
```python
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_PQ",
    "params": {
        "nlist": 1024,
        "m": 8,      # 乘积量化的子空间数量，必须是向量维度的约数
        "nbits": 8   # 每个子空间的量化位数
    }
}
```

5. **HNSW**（分层导航小世界图，适合高召回率场景）:
```python
index_params = {
    "metric_type": "L2",
    "index_type": "HNSW",
    "params": {
        "M": 16,              # 每个节点的最大连接数，范围4-64
        "efConstruction": 200  # 构建时的搜索范围，越大构建越慢但质量越好
    }
}
```

**metric_type说明**（稠密向量）:
- `"L2"`: 欧氏距离（越小越相似）
- `"IP"`: 内积（越大越相似，使用前需要对向量做L2归一化）
- `"COSINE"`: 余弦相似度（越大越相似）

**sparse_index_params参数说明**（稀疏向量索引）:

稀疏向量通常使用 `SPARSE_INVERTED_INDEX` 索引类型：

```python
sparse_index_params = {
    "index_type": "SPARSE_INVERTED_INDEX"
}
```

**返回参数**:
- `Collection`: Milvus集合对象

**示例**:

```python
from ability.storage.milvus_client import milvus_client
from pymilvus import FieldSchema, DataType

metadata_fields = [
   FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
   FieldSchema(name="document_id", dtype=DataType.INT64),
   FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=255),
]

# 示例1: 只创建稠密向量字段（用于语义检索）
collection = milvus_client.create_collection(
   collection_name="documents_dense",
   dimension=1024,
   description="文档向量集合（仅稠密向量）",
   dense_vector_field="vector",  # 创建稠密向量字段
   sparse_vector_field=None,  # 不创建稀疏向量字段
   metadata_fields=metadata_fields
)

# 示例2: 只创建稀疏向量字段（用于BM25检索，需要Milvus 2.4+）
collection = milvus_client.create_collection(
   collection_name="documents_sparse",
   description="文档向量集合（仅稀疏向量）",
   dense_vector_field=None,  # 不创建稠密向量字段
   sparse_vector_field="sparse_vector",  # 创建稀疏向量字段
   metadata_fields=metadata_fields
)

# 示例3: 同时创建两种向量字段（用于混合检索）
collection = milvus_client.create_collection(
   collection_name="documents_hybrid",
   dimension=1024,
   description="文档向量集合（稠密+稀疏向量）",
   dense_vector_field="vector",  # 创建稠密向量字段
   sparse_vector_field="sparse_vector",  # 创建稀疏向量字段
   metadata_fields=metadata_fields,
   dense_index_params={
      "metric_type": "L2",
      "index_type": "HNSW",
      "params": {"M": 16, "efConstruction": 200}
   },
   sparse_index_params={
      "index_type": "SPARSE_INVERTED_INDEX"
   }
)

# 示例4: 使用默认字段名（dense_vector_field="vector", sparse_vector_field="sparse_vector"）
collection = milvus_client.create_collection(
   collection_name="documents_default",
   dimension=1024,
   description="文档向量集合（默认字段名）",
   metadata_fields=metadata_fields
)
```

#### 插入数据

**方法**: `insert(collection_name: str, data: List[Dict[str, Any]]) -> List[int]`

**功能**: 向集合中插入数据

**重要说明**:
- **用户需要自行决定使用什么嵌入模型生成向量**（如 sentence-transformers、OpenAI API、BM25 等）
- Milvus 本身支持稠密向量（dense vector）和稀疏向量（sparse vector/BM25）
- 向量化是存储到 Milvus 的步骤，由用户在插入前完成
- 用户可以根据需求选择合适的嵌入模型和向量类型

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `collection_name` | `str` | ✅ | - | 集合名称 |
| `data` | `List[Dict[str, Any]]` | ✅ | - | 数据列表，每个元素是一个字典，**必须包含集合schema中的所有字段**，字段顺序无关，但字段名必须匹配 |

**data字典字段示例**:

| 字段名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `vector` 或 `dense_vector_field` | `List[float]` | 根据schema | 稠密向量数据（如果集合包含稠密向量字段）。**用户需要自行使用嵌入模型生成** |
| `sparse_vector` 或 `sparse_vector_field` | `Dict[int, float]` | 根据schema | 稀疏向量数据（如果集合包含稀疏向量字段）。格式：`{index: value}`，例如 `{0: 0.5, 5: 0.3}`。**用户需要自行使用BM25算法生成** |
| `content` | `str` | 根据schema | 内容字段（如果schema中包含） |
| `document_id` | `int` | 根据schema | 文档ID（如果schema中包含） |
| `tenant_id` | `str` | 根据schema | 租户ID（如果schema中包含） |
| `metadata` | `str` | 根据schema | 元数据JSON字符串（如果schema中包含） |
| 其他字段 | 根据schema | 根据schema | schema中定义的其他字段 |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `List[int]` | 插入的ID列表，如果`auto_id=True`，返回Milvus自动生成的ID；如果`auto_id=False`，返回传入的主键值 |

**示例**:
```python
# 示例1: 插入稠密向量（集合包含 dense_vector_field）
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
texts = ["文档1的内容", "文档2的内容"]
vectors = model.encode(texts).tolist()

data = [
    {
        "vector": vectors[0],  # 稠密向量字段
        "content": texts[0],
        "document_id": 1,
        "tenant_id": "tenant001",
        "metadata": '{"key": "value"}'
    },
    {
        "vector": vectors[1],
        "content": texts[1],
        "document_id": 1,
        "tenant_id": "tenant001",
        "metadata": '{"key": "value"}'
    }
]

ids = milvus_client.insert("documents_dense", data)

# 示例2: 插入稀疏向量（集合包含 sparse_vector_field，需要用户自行实现BM25）
# 稀疏向量格式：Dict[int, float]，key是词汇索引，value是权重
# 例如：{0: 0.5, 5: 0.3, 10: 0.8} 表示索引0、5、10的词汇权重分别为0.5、0.3、0.8
data = [
    {
        "sparse_vector": {0: 0.5, 5: 0.3, 10: 0.8},  # BM25稀疏向量
        "content": "文档1的内容",
        "document_id": 1,
        "tenant_id": "tenant001",
        "metadata": '{"key": "value"}'
    }
]
ids = milvus_client.insert("documents_sparse", data)

# 示例3: 同时插入稠密向量和稀疏向量（集合包含两种向量字段）
data = [
    {
        "vector": vectors[0],  # 稠密向量
        "sparse_vector": {0: 0.5, 5: 0.3},  # 稀疏向量
        "content": texts[0],
        "document_id": 1,
        "tenant_id": "tenant001",
        "metadata": '{"key": "value"}'
    }
]
ids = milvus_client.insert("documents_hybrid", data)

# 示例4: 使用 OpenAI API 生成稠密向量
# import openai
# response = openai.embeddings.create(model="text-embedding-ada-002", input=texts)
# vectors = [item.embedding for item in response.data]
# 然后使用 vectors 插入数据
```

#### 向量检索

**方法**: `search(collection_name: str, vectors: List[List[float]], top_k: int = 10, expr: Optional[str] = None, output_fields: Optional[List[str]] = None, search_params: Optional[Dict[str, Any]] = None, anns_field: str = "vector") -> List[Dict[str, Any]]`

**功能**: 在集合中进行向量检索

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `collection_name` | `str` | ✅ | - | 集合名称 |
| `vectors` | `List[List[float]]` | ✅ | - | 查询向量列表，每个向量的维度必须等于集合的dimension |
| `top_k` | `int` | ❌ | `10` | 返回Top-K结果的数量 |
| `expr` | `Optional[str]` | ❌ | `None` | 过滤表达式（Milvus表达式语法），用于过滤结果，例如：`'tenant_id == "tenant001"'` |
| `output_fields` | `Optional[List[str]]` | ❌ | `None` | 返回的字段列表，如果不指定则只返回主键和向量距离，指定后可以返回内容、元数据等字段 |
| `search_params` | `Optional[Dict[str, Any]]` | ❌ | `None` | 搜索参数字典，支持自定义搜索参数。如果为None，则使用全局配置（`MILVUS_METRIC_TYPE`、`MILVUS_NPROBE`） |
| `anns_field` | `str` | ❌ | `"vector"` | 向量字段名（ANN 检索字段）。若你的集合使用统一命名：稠密向量用 `"vector"`；如使用其他字段名需显式传入 |

**search_params参数说明**:

根据集合使用的索引类型，需要设置不同的搜索参数：

1. **IVF系列索引**（IVF_FLAT、IVF_SQ8、IVF_PQ）:
```python
search_params = {
    "metric_type": "L2",  # 必须与索引的metric_type一致
    "params": {"nprobe": 10}  # nprobe: 搜索的聚类中心数量，范围1到nlist，越大召回率越高但速度越慢
}
```

2. **HNSW索引**:
```python
search_params = {
    "metric_type": "L2",
    "params": {"ef": 100}  # ef: 搜索时的候选集大小，范围top_k到数据总量，越大召回率越高但速度越慢
}
```

3. **FLAT索引**:
```python
search_params = {
    "metric_type": "L2"  # FLAT索引不需要params
}
```

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `List[Dict[str, Any]]` | 检索结果列表，按相似度降序排列（距离升序） |

**返回字典字段**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `id` | `int` | 文档ID（主键） |
| `distance` | `float` | 距离值（L2距离或内积，取决于索引的metric_type） |
| `score` | `float` | 相似度分数（转换为0-1区间，计算公式：`1 / (1 + distance)`） |
| 其他字段 | 根据output_fields | output_fields中指定的其他字段值（如content、document_id、metadata等） |

**示例**:
```python
# 使用默认搜索参数
results = milvus_client.search(
    collection_name="documents_tenant001",
    vectors=[[0.1, 0.2, ...]],  # 查询向量
    top_k=10,
    expr='tenant_id == "tenant001"',
    output_fields=["content", "document_id", "metadata"]
)

# 自定义IVF索引的搜索参数（提高召回率）
results = milvus_client.search(
    collection_name="documents_tenant001",
    vectors=[[0.1, 0.2, ...]],
    top_k=10,
    expr='tenant_id == "tenant001"',
    output_fields=["content", "document_id", "metadata"],
    search_params={
        "metric_type": "L2",
        "params": {"nprobe": 50}  # 增加nprobe提高召回率
    }
)

# 自定义HNSW索引的搜索参数
results = milvus_client.search(
    collection_name="documents_tenant002",  # 使用HNSW索引的集合
    vectors=[[0.1, 0.2, ...]],
    top_k=10,
    search_params={
        "metric_type": "L2",
        "params": {"ef": 200}  # 增加ef提高召回率
    }
)
```

#### 查询数据

**方法**: `get_collection(collection_name: str) -> Collection`

**功能**: 获取集合对象，然后可以使用Milvus的query方法进行精确查询

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `collection_name` | `str` | ✅ | - | 集合名称 |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `Collection` | Milvus集合对象，可以调用`query()`、`load()`等方法 |

**Collection.query()方法参数**（Milvus原生方法）:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `expr` | `str` | ✅ | - | 过滤表达式，例如：`'tenant_id == "tenant001"'` |
| `output_fields` | `List[str]` | ❌ | `["id"]` | 返回的字段列表 |
| `limit` | `int` | ❌ | `100` | 返回结果的最大数量 |

**示例**:
```python
collection = milvus_client.get_collection("documents_tenant001")
collection.load()

# 查询所有数据
results = collection.query(
    expr='tenant_id == "tenant001"',
    output_fields=["id", "content", "document_id"],
    limit=100
)
```

#### 断开连接

**方法**: `disconnect() -> None`

**功能**: 断开Milvus连接

**请求参数**: 无

**返回参数**: 无返回值（`None`）

---

#### 删除集合

**方法**: `delete_collection(collection_name: str) -> None`

**功能**: 删除Milvus集合

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `collection_name` | `str` | ✅ | - | 要删除的集合名称 |

**返回参数**: 无返回值（`None`）

---

#### 列出所有集合

**方法**: `list_collections() -> List[str]`

**功能**: 列出所有Milvus集合

**请求参数**: 无

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `List[str]` | 集合名称列表 |

---

#### 上下文管理器支持

**方法**: `__enter__()` / `__exit__()`

**功能**: 支持使用 `with` 语句自动管理连接

**使用示例**:
```python
with milvus_client:
    collection = milvus_client.create_collection(...)
    # 自动连接和断开
```

---

## 4.6 工具函数 (Utils)

### 4.6.1 文本处理工具 (text_processing.py)

#### clean_text

**方法**: `clean_text(text: str, whitespace_pattern: Optional[str] = None) -> str`

**功能**: 清洗文本，去除多余空白字符

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `text` | `str` | ✅ | - | 原始文本 |
| `whitespace_pattern` | `Optional[str]` | ❌ | `r"\s+"` | 空白字符正则表达式，用于匹配需要替换的空白字符 |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `str` | 清洗后的文本，首尾空白已去除，多个连续空白字符已替换为单个空格 |

**示例**:

```python
from ability.utils import clean_text

text = "这是   一段   有  多余空格的   文本"
cleaned = clean_text(text)
# 结果: "这是 一段 有 多余空格的 文本"
```

#### split_by_sentences

**方法**: `split_by_sentences(text: str, sentence_delimiters: Optional[str] = None) -> List[str]`

**功能**: 按句子分割文本

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `text` | `str` | ✅ | - | 原始文本 |
| `sentence_delimiters` | `Optional[str]` | ❌ | `r"[。！？.!?]\s*"` | 句子分隔符正则表达式，用于识别句子边界 |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `List[str]` | 句子列表，已去除空字符串和首尾空白 |

**示例**:

```python
from ability.utils import split_by_sentences

text = "这是第一句。这是第二句！这是第三句？"
sentences = split_by_sentences(text)
# 结果: ["这是第一句", "这是第二句", "这是第三句"]
```

#### remove_empty_lines

**方法**: `remove_empty_lines(text: str) -> str`

**功能**: 移除空行

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `text` | `str` | ✅ | - | 原始文本 |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `str` | 处理后的文本，所有空行已移除 |

#### truncate_text

**方法**: `truncate_text(text: str, max_length: int, suffix: str = "...") -> str`

**功能**: 截断文本到指定长度

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `text` | `str` | ✅ | - | 原始文本 |
| `max_length` | `int` | ✅ | - | 最大长度（字符数），如果文本长度小于等于此值，则返回原文本 |
| `suffix` | `str` | ❌ | `"..."` | 截断后缀，当文本被截断时会附加此后缀 |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `str` | 截断后的文本，如果原文本长度小于等于`max_length`，返回原文本；否则返回截断后的文本（长度 = `max_length - len(suffix)`）+ 后缀 |

---

### 4.6.2 哈希工具 (hash.py)

#### calculate_file_hash

**方法**: `calculate_file_hash(file_path: Path | str, chunk_size: int = 4096) -> str`

**功能**: 计算文件的MD5哈希值

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `file_path` | `Path \| str` | ✅ | - | 文件路径 |
| `chunk_size` | `int` | ❌ | `4096` | 每次读取的块大小（字节），用于分块读取大文件，避免内存溢出 |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `str` | MD5哈希值（32位十六进制字符串） |

#### calculate_stream_hash

**方法**: `calculate_stream_hash(stream: BinaryIO, chunk_size: int = 4096) -> str`

**功能**: 计算文件流的MD5哈希值

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `stream` | `BinaryIO` | ✅ | - | 文件流对象（已打开的文件对象） |
| `chunk_size` | `int` | ❌ | `4096` | 每次读取的块大小（字节），用于分块读取大文件流 |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `str` | MD5哈希值（32位十六进制字符串），计算完成后流位置会重置到开头 |

#### calculate_text_hash

**方法**: `calculate_text_hash(text: str) -> str`

**功能**: 计算文本的MD5哈希值

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `text` | `str` | ✅ | - | 文本内容（UTF-8编码） |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `str` | MD5哈希值（32位十六进制字符串） |

---

### 4.6.3 过滤表达式校验 (filter_validation.py)

#### validate_milvus_expr

**方法**: `validate_milvus_expr(expr: str, *, max_len: int = 500, forbidden_chars: Optional[List[str]] = None) -> str`

**功能**: 验证Milvus表达式，防止SQL注入等安全问题

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `expr` | `str` | ✅ | - | Milvus表达式字符串 |
| `max_len` | `int` | ❌ | `500` | 最大长度限制（字符数），超过此长度会抛出异常 |
| `forbidden_chars` | `Optional[List[str]]` | ❌ | `[";"]` | 禁止字符列表，表达式包含这些字符会抛出异常 |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `str` | 验证后的表达式（去除首尾空白），如果输入为`None`或空字符串，返回空字符串 |

**异常**:
- `ValueError`: 如果表达式不合法（长度超限、包含禁止字符等）

**示例**:

```python
from ability.utils import validate_milvus_expr

expr = 'tenant_id == "tenant001" && document_id > 100'
validated = validate_milvus_expr(expr)
```

#### validate_sql_where

**方法**: `validate_sql_where(sql_where: str, *, max_len: int = 500, forbidden_tokens: Optional[List[str]] = None, forbidden_keywords: Optional[List[str]] = None, char_whitelist: Optional[str] = None) -> str`

**功能**: 验证SQL WHERE表达式

**请求参数**:

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `sql_where` | `str` | ✅ | - | SQL WHERE表达式字符串 |
| `max_len` | `int` | ❌ | `500` | 最大长度限制（字符数），超过此长度会抛出异常 |
| `forbidden_tokens` | `Optional[List[str]]` | ❌ | `[";", "--", "/*", "*/"]` | 禁止的SQL令牌列表，表达式包含这些令牌会抛出异常 |
| `forbidden_keywords` | `Optional[List[str]]` | ❌ | `["drop", "delete", "update", "insert", "alter", "create", "truncate", "grant", "revoke"]` | 禁止的SQL关键字列表（不区分大小写），表达式包含这些关键字会抛出异常 |
| `char_whitelist` | `Optional[str]` | ❌ | `r"[a-zA-Z0-9_\s\(\)\=\!\<\>\.\,\:\'\"\%\&\|\+\-\*/]+"` | 字符白名单正则表达式，只允许匹配此模式的字符，其他字符会抛出异常 |

**返回参数**:

| 字段名 | 类型 | 说明 |
|--------|------|------|
| 返回值 | `str` | 验证后的表达式（去除首尾空白），如果输入为`None`或空字符串，返回空字符串 |

**异常**:
- `ValueError`: 如果表达式不合法（长度超限、包含禁止令牌/关键字、包含非法字符等）

---

### 4.6.4 日志系统 (logger.py)

**功能**: 基于loguru的日志系统

**配置**:
- `LOG_LEVEL` (环境变量): 日志级别（默认: "INFO"）

**输出**:
- 控制台输出（带颜色）
- 文件输出：`logs/app_{YYYY-MM-DD}.log`
- 错误日志：`logs/error_{YYYY-MM-DD}.log`

**使用**:

```python
from ability.utils import logger

logger.info("信息日志")
logger.warning("警告日志")
logger.error("错误日志")
```

---

## 5. 配置管理

### 5.1 配置来源优先级

1. **环境变量** (最高优先级)
2. **.env文件**
3. **YAML配置文件** (`config/config.yaml`)
4. **代码默认值** (最低优先级)

### 5.2 主要配置项

详见 `app/config.py` 中的 `Settings` 类，主要配置项包括：

- **Milvus配置**: `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_INDEX_TYPE` 等
- **向量模型配置**: `EMBEDDING_MODEL_NAME`, `EMBEDDING_DIM`, `EMBEDDING_DEVICE` 等
- **切片配置**: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHUNK_STRATEGY` 等
- **检索配置**: `TOP_K`, `SEARCH_MODE`, `RERANK_ENABLED` 等

---

## 6. 插件扩展

### 6.1 注册自定义解析器

```python
from ability.operators.decorators import register_parser
from ability.operators.parsers.base_parser import BaseParser


@register_parser(".custom")
class CustomParser(BaseParser):
   def _parse(self, file_path, **kwargs):
      # 实现解析逻辑
      return {
         "content": "...",
         "metadata": {},
         "structure": {}
      }
```

### 6.2 注册自定义切片器

```python
from ability.operators.decorators import register_chunker
from ability.operators.chunkers import BaseChunker, Chunk


@register_chunker("custom")
class CustomChunker(BaseChunker):
   def _chunk(self, text, **kwargs):
      # 实现切片逻辑
      return [Chunk(content="...", chunk_index=0, ...)]
```

### 6.3 注册自定义检索器

```python
from ability.operators.decorators import register_retriever
from ability.operators.retrievers.base_retriever import BaseRetriever, RetrievalResult


@register_retriever("custom")
class CustomRetriever(BaseRetriever):
   def _retrieve(self, query, top_k, tenant_id, **kwargs):
      # 实现检索逻辑
      return [RetrievalResult(...)]
```

### 6.4 从文件加载插件

```python
from ability.operators.plugin_registry import PluginRegistry
from pathlib import Path

PluginRegistry.load_plugin(Path("plugins/custom_operator.py"))
```

---

## 7. 完整使用示例

### 7.1 文档处理流水线

```python
from ability.operators.parsers.parser_factory import ParserFactory
from ability.operators.chunkers.chunker_factory import ChunkerFactory
from sentence_transformers import SentenceTransformer
from ability.storage.milvus_client import milvus_client
from pymilvus import FieldSchema, DataType

# 1. 解析文档
parser = ParserFactory.create_parser("document.pdf")
result = parser.process("document.pdf")
content = result["content"]

# 2. 切片文档
chunker = ChunkerFactory.create_chunker(
   strategy="semantic",
   config={"chunk_size": 512, "chunk_overlap": 50}
)
chunks = chunker.process(content)

# 3. 向量化（用户自行选择和使用嵌入模型）
model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
vectors = model.encode([chunk.content for chunk in chunks]).tolist()

# 4. 存储到Milvus
metadata_fields = [
   FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
   FieldSchema(name="document_id", dtype=DataType.INT64),
   FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=255),
]

collection = milvus_client.create_collection(
   collection_name="documents",
   dimension=1024,
   metadata_fields=metadata_fields
)

data = [
   {
      "vector": vector,  # 用户生成的向量
      "content": chunk.content,
      "document_id": 1,
      "tenant_id": "tenant001"
   }
   for vector, chunk in zip(vectors, chunks)
]

ids = milvus_client.insert("documents", data)
```

### 7.2 检索流水线

```python
from ability.operators.retrievers.retriever_factory import RetrieverFactory
from sentence_transformers import SentenceTransformer

# 创建混合检索器
retriever = RetrieverFactory.create_retriever(
   mode="hybrid",
   config={
      "top_k": 10,
      "semantic_weight": 0.6,
      "keyword_weight": 0.4,
      "fusion_method": "rrf"
   }
)

# 执行检索（混合检索中的语义检索需要提供查询向量）
model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
query_text = "人工智能的应用"
query_vector = model.encode(query_text).tolist()

results = retriever.process(
   query=query_text,
   query_vector=query_vector,  # 语义检索需要此参数
   top_k=10,
   tenant_id="tenant001"
)

# 处理结果
for result in results:
   print(f"得分: {result.score:.4f}")
   print(f"内容: {result.content}")
   print(f"元数据: {result.metadata}")
```

### 7.3 不同检索模式示例

```python
from ability.operators.retrievers.retriever_factory import RetrieverFactory

# 1. 全文检索（基于关键词匹配）
fulltext_retriever = RetrieverFactory.create_retriever(
   mode="fulltext",
   config={
      "top_k": 10,
      "match_mode": "or",  # 任一关键词匹配
      "min_match_count": 1
   }
)
results = fulltext_retriever.process("人工智能 机器学习", top_k=10)

# 2. 文本匹配（精确/模糊匹配）
text_match_retriever = RetrieverFactory.create_retriever(
   mode="text_match",
   config={
      "top_k": 10,
      "match_type": "fuzzy",  # 模糊匹配
      "case_sensitive": False
   }
)
results = text_match_retriever.process("查询文本", top_k=10)

# 3. 短语匹配（短语精确匹配）
phrase_match_retriever = RetrieverFactory.create_retriever(
   mode="phrase_match",
   config={
      "top_k": 10,
      "case_sensitive": False,
      "allow_partial": False
   }
)
results = phrase_match_retriever.process("人工智能技术", top_k=10)
```

---

## 8. 注意事项

1. **初始化**: 所有算子在使用前需要调用 `initialize()` 方法，或通过工厂类创建（会自动初始化）
2. **多租户**: 使用 `tenant_id` 参数实现多租户数据隔离
3. **安全性**: 在使用 `milvus_expr` 参数前，务必先通过 `validate_milvus_expr()` 校验
4. **配置优先级**: 环境变量 > .env文件 > YAML配置 > 代码默认值
5. **插件加载**: 自定义插件需要在应用启动时加载，或在导入时通过装饰器自动注册

---

## 9. 版本信息

- **项目版本**: 1.0.0
- **Python要求**: 3.9+
- **主要依赖**: pymilvus, sentence-transformers, PyMuPDF, python-docx 等

---

**文档最后更新**: 2026年

**维护者**: Milvus RAG Team
