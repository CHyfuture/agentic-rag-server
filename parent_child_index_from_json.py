#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模拟调用 /api/v1/index/build 接口，完成知识库构建。

从指定目录读取 JSON 文件，调用 build_index_from_json_contents（与 API 相同逻辑），
将论文结构化信息 + Markdown 原文通过父子切片 + 多向量编码写入 Milvus 与 BaseDB。

功能：
1. 读取指定目录下的 JSON 文件（title / authors / abstract / keywords / conclusion / original_text）
2. 使用 ChunkerService 的 parent_child 策略切片，多向量编码写入 Milvus
3. 文档与切片元数据写入 BaseDB（与 POST /index/build 行为一致）

依赖：
    pip install milvus-service base-db python-dotenv
"""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from app.service.index_service import build_index_from_json_contents


def load_env() -> None:
    """加载项目根目录下的 .env。"""
    project_root = Path(__file__).resolve().parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="模拟 /index/build 接口，从本地 JSON 目录构建知识库",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"D:\project\agentic-rag-server\前100修正\json_chinese",
        help="JSON 文件所在目录",
    )
    parser.add_argument(
        "--kb-id",
        type=int,
        default=1,
        help="知识库 ID（对应 API 的 kb_id 参数）",
    )
    parser.add_argument(
        "--skip-base-db",
        action="store_true",
        default=True,
        help="仅写入 Milvus，不调用 BaseDB（默认开启，适用于本地脚本）",
    )
    parser.add_argument(
        "--use-base-db",
        action="store_true",
        default=True,
        help="同时写入 BaseDB（与 API 行为一致，需 BaseDB 服务可用）",
    )

    args = parser.parse_args()
    skip_base_db = not args.use_base_db

    load_env()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"输入目录不存在: {input_dir}")
        return

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"目录下无 JSON 文件: {input_dir}")
        return

    items: list[tuple[str, str]] = []
    for p in json_files:
        try:
            text = p.read_text(encoding="utf-8")
            items.append((p.name, text))
        except Exception as e:
            print(f"读取失败 {p}: {e}")

    if not items:
        print("无有效 JSON 文件可处理")
        return

    mode = "Milvus+BaseDB" if not skip_base_db else "仅 Milvus"
    print(f"共 {len(items)} 个 JSON 文件，开始构建索引（{mode}）...")
    result = asyncio.run(
        build_index_from_json_contents(
            kb_id=args.kb_id,
            items=items,
            skip_base_db=skip_base_db,
        )
    )
    print(f"构建完成: {result}")


if __name__ == "__main__":
    main()

