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
# 若 SSH 克隆失败（如 Connection closed / 网络受限），改用 HTTPS：GITHUB_TOKEN=ghp_xxx ./build.sh -t 1.0.0
echo "正在构建 Docker 镜像: agentic-rag-new-server:$VERSION"
if [ -n "$GITHUB_TOKEN" ]; then
  echo "使用 HTTPS 方式安装私有依赖（GITHUB_TOKEN 已设置）"
  DOCKER_BUILDKIT=1 docker build -f Dockerfile.https --secret id=github_token,env=GITHUB_TOKEN --build-arg REFRESH_DATE=$(date +%s) -t agentic-rag-new-server:$VERSION .
else
  DOCKER_BUILDKIT=1 docker build --ssh default=$SSH_AUTH_SOCK --build-arg REFRESH_DATE=$(date +%s) -t agentic-rag-new-server:$VERSION .
fi

if [ $? -eq 0 ]; then
  echo "构建成功！镜像标签: agentic-rag-new-server:$VERSION"
else
  echo "构建失败！"
  exit 1
fi
