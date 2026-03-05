#!/bin/bash

# 版本号，我希望通过参数 -t 来指定

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
git pull
# 构建 Docker 镜像
echo "正在构建 Docker 镜像: agentic-rag-new-server:$VERSION"
docker build --build-arg REFRESH_DATE=$(date +%s) -t agentic-rag-new-server:$VERSION .

if [ $? -eq 0 ]; then
  echo "构建成功！镜像标签: agentic-rag-new-server:$VERSION"
else
  echo "构建失败！"
  exit 1
fi
