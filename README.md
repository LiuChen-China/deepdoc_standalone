# deepdoc_standalone
## 项目介绍
计划从RAGflow这个强大项目中提取出文档解析和文本分块功能，并封装成接口 
## 一些相对原deepdoc的修改
- excel内容提取将合并的单元格拆散后填充为合并内容，不然提取的键值对可能为空
- table分块，excel重复列删除 源代码为直接报错
- 增加了libreoffice的docker服务，用来支持旧office文档格式的转换
## 运行环境搭建
- 安装uv
```
# macOS / Linux
curl -LsSf https://gitee.com/wangnov/uv-custom/releases/download/0.9.15/uv-installer-custom.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -c "irm https://gitee.com/wangnov/uv-custom/releases/download/0.9.15/uv-installer-custom.ps1 | iex"
```
- 安装python环境
```
uv venv -p 3.10
uv pip install -r requirements.txt
```
- 运行libreoffice的docker服务，如果不需要旧文档支持或格式转换就不需要运行
```
cd docker_libreoffice
docker compose up -d
```
- ubuntu可能需要安装openssl，win10好像不需要
```
sudo apt install -y libssl-dev openssl
```
## 接口
- 解析文档接口，查看脚本deepdoc_api.py
- 文档格式转换接口，查看脚本deepdoc/parser/libreoffice_convert.py