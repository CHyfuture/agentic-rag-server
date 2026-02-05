#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 JSON 文件读取论文结构化信息 + Markdown 原文，
通过父子切片 (parent_child) + 多向量编码，写入 Milvus 集合。

功能：
1. 如果集合不存在则按父子切片 + 多向量 schema 创建集合；
   已存在则直接复用（不会报错）。
2. 读取指定目录下的 JSON 文件：
   - title / authors / abstract / keywords / conclusion / original_text
3. 使用 ChunkerService 的 parent_child 策略，对 original_text 切片。
4. 对 content / abstract / keywords / summary 分别向量化，写入集合。

依赖：
    pip install milvus-service  (提供 milvus_service)
    pip install python-dotenv
"""

import argparse
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from app.service.index_service import index_json_dir, index_json_file


def load_env() -> None:
    """加载项目根目录下的 .env，确保 BaseVector-Core 能读取到配置。"""
    project_root = Path(__file__).resolve().parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 JSON (含 original_text) 进行父子切片 + 多向量入库脚本",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"D:\project\agentic-rag-server\前100修正\json_chinese",
        help="JSON 文件所在目录，默认为示例路径",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="仅处理某一个 JSON 文件（绝对路径或相对路径），优先于 --input-dir",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="Milvus 集合名称（默认读取环境变量 COLLECTION_NAME）",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="用于编码文本的句向量模型名称（默认读取环境变量 EMBEDDING_MODEL）",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="向量维度，默认读取环境变量 EMBEDDING_DIM 或使用 768",
    )

    args = parser.parse_args()

    load_env()
    total_records = 0

    if args.file:
        json_path = Path(args.file)
        if not json_path.exists():
            print(f"指定文件不存在: {json_path}")
            return
        total_records += index_json_file(
            file_path=json_path,
            collection_name=args.collection_name,
            model_name=args.model_name,
            dim=args.dim,
        )
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"输入目录不存在: {input_dir}")
            return

        total_records += index_json_dir(
            input_dir=input_dir,
            collection_name=args.collection_name,
            model_name=args.model_name,
            dim=args.dim,
        )

    print(f"所有处理完成，总插入记录数: {total_records}")


if __name__ == "__main__":
    main()

