# deepdoc_standalone
## 项目介绍
计划从RAGflow这个强大项目中提取出文档解析和文本分块功能，并封装成接口 
## 一些修改
- excel内容提取将合并的单元格拆散后填充为合并内容，不然提取的键值对可能为空
## 运行环境搭建
``` uv venv -p 3.10```

```uv pip install -r requirements.txt```