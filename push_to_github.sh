#!/bin/bash
# 推送Auto Paper Pipeline到GitHub
# 使用方法: bash push_to_github.sh

cd /root/git/mimo/paper-pipeline

echo "=== 创建GitHub仓库 ==="
# 方式1: 使用gh CLI（需先登录: gh auth login）
gh repo create infanwang/auto-paper-pipeline --public --source=. --push --description "📚 Auto Paper Pipeline - 自动化论文爬取、复现、验证系统" 2>/dev/null

if [ $? -ne 0 ]; then
    echo ""
    echo "gh CLI不可用或未登录，请按以下步骤操作："
    echo ""
    echo "1. 先登录GitHub CLI:"
    echo "   gh auth login"
    echo ""
    echo "2. 然后运行:"
    echo "   bash push_to_github.sh"
    echo ""
    echo "或者手动创建仓库："
    echo "  1. 访问 https://github.com/new"
    echo "  2. 仓库名: auto-paper-pipeline"
    echo "  3. 然后运行: git push -u origin main"
else
    echo "仓库创建并推送成功！"
    echo ""
    echo "仓库地址: https://github.com/infanwang/auto-paper-pipeline"
fi
