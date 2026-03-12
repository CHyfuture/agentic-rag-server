# `md_to_json.py` 使用教程

`md_to_json.py` 用于**批量读取论文 Markdown（`.md`）文件**，从中提取结构化信息并写入为 **JSON 文件**（每篇论文一个 `.json`）。

# 路径配置

MD_DIR为需要翻译的MD文件存放目录，JSON_DIR为生成的JSON文件存放的目录！！
一定要根据实际情况修改路径，否则无法使用！！

- **输入**：MD_DIR = Path(r"E:\My_Project\Python\RAG\md")
- **输出**：JSON_DIR = Path(r"E:\My_Project\Python\RAG\json")
## 注意当前的deepseek api是硬编码，，可考虑移动到环境配置中

# 功能概览

## 注意当原论文无关键词时会自动从标题、摘要和结论中提取！！

- **提取字段**：
  - `title`：标题
  - `authors`：作者数组（每个作者 `{name, school}`）
  - `abstract`：摘要
  - `keywords`：关键词
  - `conclusion`：结论（直接从原文 Conclusion/结论段落提取，不自行生成）
  - `original_text`：原始 Markdown 全文
- **特点**：
  - 先用正则做快速抽取（标题/摘要/关键词/结论），再调用 LLM（DeepSeek）补全/纠正
  - 内置轻量语言检测（中/英）
  - 线程池并发处理多个文件（默认并发 `MAX_WORKERS=5`）

# 依赖与环境准备

  - `requests`：用于调用 DeepSeek 的 HTTP 接口（`requests.Session().post(...)`）

## 运行方式
在项目根目录直接运行：

运行日志会输出：

- 找到多少个 `.md`
- 每个文件的处理进度
- 最终成功/失败统计与耗时

## 输出 JSON 结构

每个输入 `paper.md` 会生成：
- 输出文件：`JSON_DIR\paper.json`

示例（字段示意，内容仅示例）：

```json
{
  "title": "Example Paper Title",
  "authors": [
    { "name": "Alice", "school": "Example University" }
  ],
  "abstract": "This paper investigates ...",
  "keywords": ["Machine Learning", "Object Tracking", "SAM2"],
  "conclusion": "In conclusion, ...",
  "original_text": "# Example Paper Title\n\n..."
}
```
## 常见问题与排查
### 1) 找不到文件 / 找到 0 个文件
- 确认 `MD_DIR` 指向的目录存在
- 确认目录下确实有 `.md` 文件（不是 `.markdown` / `.txt`）
- 脚本不递归子目录：如果你的 md 在子目录里，需要把 `MD_DIR` 改到对应目录
### 2) API 调用失败 / 频繁报错 / 成功率低
- 检查 `API_KEY` 是否有效
- 检查网络是否可访问 `BASE_URL`
- 降低 `MAX_WORKERS`（例如从 5 改到 1-2）以避免触发限流
