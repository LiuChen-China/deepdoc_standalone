import warnings
from sklearn.exceptions import ConvergenceWarning
# 忽略 ConvergenceWarning 警告
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import sys, os; __name__ == "__main__" and sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepdoc_standalone.deepdoc.parser.libreoffice_convert import OfficeConverter
from deepdoc_standalone.rag.app.naive import chunk as naive_chunk
from deepdoc_standalone.rag.app.presentation import chunk as presentation_chunk
from deepdoc_standalone.rag.app.table import chunk as table_chunk
import time

#对libreoffice的容器服务进行连接
try:
    converter = OfficeConverter(host="localhost",port=18976,timeout=30,retry_times=2)
except Exception as e:
    converter = None
    print(f"libreoffice服务连接失败,文件转换功能将不可用,错误信息:{e}")

def param_preprocess(file_path: str="",binary_content: bytes=None,ext: str = "auto"):
    '''预处理文档解析函数的参数'''
    if (ext == "auto") and (not file_path):
        raise Exception("file_path 或 ext 参数请至少提供一个用来判断文件类型")
    #获取扩展名
    ext = os.path.splitext(file_path)[1].lower() if ext == "auto" else ext.lower()
    #扩充小数点开头
    ext = "." + ext if not ext.startswith(".") else ext
    #读取文档
    if binary_content is None:
        with open(file_path,"rb") as f:
            binary_content = f.read()
    #如果是旧office文档，需要转换为新格式文档
    if ext in [".doc",".xls",".ppt"]:
        if converter is None:
            raise Exception(f"libreoffice服务未连接,无法转换旧office格式文件:{file_path}")
        binary_content =  converter.convert_old_to_new(input_data=binary_content,default_ext=ext)
        ext = ext + "x"
        file_path += "x"
    #支持的文档
    supported_ext = [".docx",".xlsx",".pptx",".pdf",".txt",".md",".html",".json"]
    if ext not in supported_ext:
        raise Exception(f"不支持的文件扩展名:{ext},当前支持的扩展名有:{supported_ext}")
    #提取文件名（包含扩展名）
    if file_path:
        file_name = os.path.basename(file_path)
    else:
        file_name = f"file_{int(time.time())}{ext}"
    return file_name, binary_content, ext


def printProgress(prog="",msg=""):
    if prog:
        #print(f"{str(int(prog*100))}% {msg}")
        pass
    else:
        #print(msg)
        pass

def parse_file(file_path: str="",binary_content=None,ext: str = "auto",layout_recognize="deepdoc"):
    '''
    解析提取文件内容，同时对文档内容进行切分
    :param file_path: 文件路径
    :param binary_content: 文件二进制内容
    :param ext: 文件扩展名,默认自动判断 小数点开头
    :return: 解析后的文本内容
    '''
    start_time = time.time()
    #预处理参数 读取文件 文件旧格式转换 格式检查 如果没输入文件路径 返回的文件名则是随机名
    file_name, binary_content, ext = param_preprocess(file_path, binary_content, ext)
    fileSizeM = len(binary_content)/1024/1024
    chunks = []
    #按格式选择切分策略
    if ext in ['.json','.docx','.txt','.md','.html','.pdf']:
        #先尝试纯文本提取 若提取到的知识块太少 再尝试deepdoc提取
        if ext == '.pdf':
            layout_recognize = "plaintext"
            chunks = naive_chunk(binary=binary_content, filename=file_name,callback=printProgress,layout_recognize="plaintext")
        if len(chunks) < 5:
            layout_recognize = "deepdoc"
            chunks = naive_chunk(binary=binary_content, filename=file_name,callback=printProgress,layout_recognize=layout_recognize)
    elif ext in ['.pptx']:
        chunks = presentation_chunk(binary=binary_content, filename=file_name,callback=printProgress)
    elif ext in ['.xlsx']:
        chunks = table_chunk(binary=binary_content, filename=file_name,callback=printProgress)
    content = "\n".join([chunk["content_with_weight"] for chunk in chunks if chunk.get("content_with_weight","").strip()])
    print(f"解析文件 {file_name} 大小:{fileSizeM:.2f}M 文本块:{len(chunks)}个 耗时: {(time.time() - start_time):.2f} 秒")
    return {'chunks':chunks,'content':content,'file_name':file_name}


if __name__ == "__main__":
    if 1:
        result = parse_file(file_path="static/测试文件/test.docx")
    if 0:
        result = parse_file(file_path="static/测试文件/test.ppt")
    if 0:
        result = parse_file(file_path="static/测试文件/test.xls")
    if 0:
        result = parse_file(file_path="static/测试文件/test.txt")
    if 0:
        result = parse_file(file_path="static/测试文件/test.html")
    if 0:
        result = parse_file(file_path="static/测试文件/test.pdf")


     



