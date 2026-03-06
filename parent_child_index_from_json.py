#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通过调用 POST /api/v1/index/build 接口，完成知识库数据初始化。

从指定目录读取 JSON 文件，以 multipart/form-data 上传至服务端，
由服务端完成父子切片、多向量编码并写入 Milvus 与 BaseDB。

功能：
1. 读取指定目录下的 JSON 文件（title / authors / abstract / keywords / conclusion / original_text）
2. 调用 /api/v1/index/build 接口上传
3. 服务端完成 ChunkerService parent_child 切片、多向量编码及存储

依赖：
    pip install requests
"""

import argparse
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="调用 /api/v1/index/build 接口，从本地 JSON 目录构建知识库",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"D:\project\agentic-rag-server\前100修正\json_chinese_a",
        help="JSON 文件所在目录",
    )
    parser.add_argument(
        "--kb-id",
        type=int,
        default=1,
        help="知识库 ID（对应 API 的 kb_id 参数）",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://192.168.1.5:5010/api/v1/index/build",
        help="索引构建接口地址",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"输入目录不存在: {input_dir}")
        return

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"目录下无 JSON 文件: {input_dir}")
        return

    # 构建 multipart/form-data：kb_id + files
    files = [
        ("files", (p.name, p.read_bytes(), "application/json"))
        for p in json_files
    ]
    data = {"kb_id": args.kb_id}

    print(f"共 {len(json_files)} 个 JSON 文件，正在调用 {args.api_url} ...")
    try:
        resp = requests.post(
            args.api_url,
            data=data,
            files=files,
            timeout=3600,
        )
        resp.raise_for_status()
        result = resp.json()
        print(f"构建完成: {result}")
        if result.get("skipped_files"):
            print(f"跳过文件: {result['skipped_files']}")
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        if hasattr(e, "response") and e.response is not None:
            try:
                print(f"响应内容: {e.response.text}")
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
