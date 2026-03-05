#!/bin/bash

# 版本号，我希望通过 -t 参数来指定

VERSION=""

# 解析命令行参数
while getopts "t:" opt; do
  case $opt in
    t)
      VERSION="$OPTARG"
      ;;
    \?)
      echo "用法: $0 -t <版本号>"
      echo "示例: $0 -t 1.0.0"
      exit 1
      ;;
  esac
done

# 检查版本号是否提供
if [ -z "$VERSION" ]; then
  echo "错误: 必须通过 -t 参数指定版本号"
  echo "用法: $0 -t <版本号>"
  echo "示例: $0 -t 1.0.0"
  exit 1
fi

# 停止并删除旧容器（如果存在）
echo "正在停止旧容器..."
docker stop agentic-rag-new-server 2>/dev/null || true

echo "正在删除旧容器..."
docker rm agentic-rag-new-server 2>/dev/null || true

# 启动新容器
echo "正在启动容器: agentic-rag-new-server:$VERSION"
docker run -d --name agentic-rag-new-server -p 5010:5010 -v /home/general/agentic-rag-new-server/workspace:/app/workspace  -e RERANK_MODEL_NAME=workspace/jina-reranker-v3  agentic-rag-new-server:$VERSION

if [ $? -eq 0 ]; then
  echo "容器启动成功！"
  echo "容器名称: agentic-rag-new-server"
  echo "镜像标签: agentic-rag-new-server:$VERSION"
  echo "端口映射: 5010:5010"
else
  echo "容器启动失败！"
  exit 1
fi

# 查看日志
docker logs -f agentic-rag-new-server
