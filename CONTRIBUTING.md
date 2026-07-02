# 贡献指南

感谢您对本项目的关注！我们欢迎任何形式的贡献。

## 如何贡献

### 报告问题

1. 使用 GitHub Issues 报告 bug
2. 提供详细的问题描述和复现步骤
3. 包含相关的错误信息和环境信息

### 提交代码

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

### 开发环境设置

1. 克隆仓库
```bash
git clone https://github.com/your-username/qwen3-3b-nsfw.git
cd qwen3-3b-nsfw
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

4. 安装 pre-commit hooks
```bash
pre-commit install
```

### 代码规范

- 使用 Black 进行代码格式化
- 使用 isort 进行 import 排序
- 使用 flake8 进行代码检查
- 使用 mypy 进行类型检查
- 遵循 PEP 8 编码规范

### 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_preprocess.py

# 运行测试并生成覆盖率报告
pytest tests/ --cov=scripts --cov-report=html
```

### 提交规范

使用语义化提交信息：

- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `style`: 代码格式调整（不影响功能）
- `refactor`: 代码重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

示例：
```
feat: 添加 QLoRA 支持
fix: 修复数据预处理中的内存泄漏
docs: 更新 README 安装说明
```

## 行为准则

- 尊重所有参与者
- 接受建设性批评
- 专注于对社区最有利的事情
- 对他人表示同理心

## 许可证

通过贡献代码，您同意您的贡献将在 MIT 许可证下获得许可。