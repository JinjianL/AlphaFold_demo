# AlphaFold 简化演示

这是一个AlphaFold的简化演示项目，用于展示蛋白质结构预测的基本流程。

## 功能

- 输入蛋白质序列
- 下载预训练模型
- 预测蛋白质结构
- 可视化预测结果

## 安装

1. 克隆此仓库
2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

1. 启动演示应用：

```bash
python scripts/run_demo.py
```

2. 在Web界面中输入或上传蛋白质序列
3. 点击"预测"按钮
4. 查看和下载预测结果

## 文件结构

- `scripts/`: 包含主要的Python脚本
- `data/`: 用于存储下载的模型和序列数据
- `output/`: 存储预测结果 