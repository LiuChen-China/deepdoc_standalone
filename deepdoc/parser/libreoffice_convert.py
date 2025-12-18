import requests
import os
import time
import logging
import socket
from typing import Optional, List, Dict, Union, BinaryIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import io

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class OfficeConverter:
    """Office文件转换器（支持二进制输入输出）：
    1. 旧格式转新格式（仅doc/xls/ppt → docx/xlsx/pptx）
    2. 全格式转PDF（doc/docx、xls/xlsx、ppt/pptx 都支持）
    输入：文件路径 或 二进制字节流
    输出：转换后的二进制字节流（不再自动保存文件）
    """
    # 旧→新格式映射（仅针对旧格式）
    OLD_TO_NEW_MAP = {
        ".doc": "docx",
        ".xls": "xlsx",
        ".ppt": "pptx"
    }
    # PDF转换支持的所有格式（新旧都包含）
    PDF_SUPPORT_EXTS = [
        ".doc", ".docx",
        ".xls", ".xlsx",
        ".ppt", ".pptx"
    ]
    # 所有支持的Office格式汇总
    ALL_SUPPORT_EXTS = list(set(OLD_TO_NEW_MAP.keys()) | set(PDF_SUPPORT_EXTS))

    def __init__(self, host: str = "localhost", port: int = 2004, timeout: int = 60, retry_times: int = 2):
        """
        初始化转换器（适配官方REST API：/request）
        """
        self.api_url = f"http://{host}:{port}/request"
        self.host = host
        self.port = port
        self.timeout = timeout
        self.retry_times = retry_times
        self._check_port_connectivity()

    def _check_port_connectivity(self):
        """检查端口可达性"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        try:
            result = sock.connect_ex((self.host, self.port))
            if result != 0:
                raise ConnectionError(f"❌ {self.host}:{self.port} 端口不通（TCP连接失败）")
            logger.info(f"✅ {self.host}:{self.port} 端口连通性正常")
        finally:
            sock.close()

        try:
            resp = requests.post(self.api_url, timeout=5)
            logger.info(f"✅ REST API接口（/request）可达，响应码：{resp.status_code}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"❌ 无法连接到REST API：{self.api_url}\n"
                "请检查：\n1. Docker容器是否启动\n2. 端口是否映射\n3. 防火墙是否放行端口"
            )

    def _get_file_info(self, input_data: Union[str, bytes, BinaryIO], default_ext: str = "") -> tuple:
        """
        解析输入数据，返回（文件对象、文件名、文件扩展名）
        :param input_data: 文件路径 / 二进制字节 / 文件流
        :param default_ext: 当输入为二进制且无文件名时的默认扩展名
        :return: (file_obj, filename, file_ext)
        """
        file_obj = None
        filename = "temp_file"
        file_ext = default_ext.lower()

        try:
            # 情况1：输入是文件路径
            if isinstance(input_data, str):
                if not os.path.exists(input_data):
                    raise FileNotFoundError(f"文件不存在：{input_data}")
                filename = os.path.basename(input_data)
                file_ext = os.path.splitext(filename)[1].lower()
                file_obj = open(input_data, "rb")
            
            # 情况2：输入是二进制字节
            elif isinstance(input_data, bytes):
                filename = f"temp_{int(time.time())}{default_ext}"
                file_obj = io.BytesIO(input_data)
            
            # 情况3：输入是文件流（BytesIO/StringIO）
            elif hasattr(input_data, 'read'):
                filename = getattr(input_data, 'name', f"temp_{int(time.time())}{default_ext}")
                file_ext = os.path.splitext(filename)[1].lower()
                # 重置文件指针
                input_data.seek(0)
                file_obj = input_data
            
            else:
                raise TypeError(f"不支持的输入类型：{type(input_data)}，仅支持文件路径、二进制字节、文件流")

            return file_obj, filename, file_ext

        except Exception as e:
            if file_obj and isinstance(file_obj, (io.BytesIO, io.StringIO)):
                file_obj.close()
            raise e

    def _send_convert_request(self, input_data: Union[str, bytes, BinaryIO], convert_to: str, 
                             default_ext: str = "", opts: List[str] = None) -> bytes:
        """
        通用转换请求（内部方法）
        :param input_data: 文件路径 / 二进制字节 / 文件流
        :param convert_to: 转换目标格式（docx/xlsx/pptx/pdf）
        :param default_ext: 二进制输入时的默认扩展名
        :param opts: 转换参数
        :return: 转换后的二进制字节流
        """
        opts = opts or []
        file_obj = None

        for retry in range(self.retry_times + 1):
            try:
                # 解析输入数据
                file_obj, filename, file_ext = self._get_file_info(input_data, default_ext)
                
                #logger.info(f"📤 开始转换（第{retry+1}次）：{filename} → {convert_to}")
                start_time = time.time()

                # 构建请求表单
                form_data = {
                    "file": (filename, file_obj),
                    "convert-to": (None, convert_to),
                }
                for idx, opt in enumerate(opts):
                    form_data[f"opts[{idx}]"] = (None, opt)

                # 发送转换请求
                resp = requests.post(
                    self.api_url,
                    files=form_data,
                    timeout=self.timeout,
                    stream=True
                )

                if resp.status_code == 200:
                    # 读取二进制响应
                    output_bytes = b""
                    for chunk in resp.iter_content(chunk_size=8192):
                        output_bytes += chunk
                    
                    if len(output_bytes) == 0:
                        raise RuntimeError("转换后的文件为空（可能源文件损坏）")
                    
                    #logger.info(f"✅ 转换成功（耗时{round(time.time()-start_time,2)}秒）：{filename} → {convert_to}")
                    return output_bytes
                else:
                    raise RuntimeError(f"API返回错误：{resp.status_code} - {resp.text[:200]}")

            except Exception as e:
                # 清理资源
                if file_obj and isinstance(file_obj, io.BytesIO):
                    file_obj.close()
                elif file_obj and isinstance(input_data, str):
                    file_obj.close()

                if retry < self.retry_times:
                    logger.warning(f"⚠️ 转换失败（将重试）：{str(e)[:100]}")
                    time.sleep(1)
                else:
                    logger.error(f"❌ 最终转换失败（重试{self.retry_times}次）：{str(e)[:100]}")
                    raise e

            finally:
                # 确保文件对象关闭（除了传入的文件流）
                if file_obj and not isinstance(input_data, (bytes, io.BytesIO, io.StringIO)):
                    try:
                        file_obj.close()
                    except:
                        pass

    # ==================== 功能1：旧格式转新格式（仅doc/xls/ppt） ====================
    def convert_old_to_new(self, input_data: Union[str, bytes, BinaryIO], 
                          default_ext: str = "") -> bytes:
        """
        仅将旧Office文件转为新版（doc→docx、xls→xlsx、ppt→pptx）
        :param input_data: 文件路径 / 二进制字节 / 文件流
        :param default_ext: 二进制输入时指定源文件扩展名（如".doc"）
        :return: 转换后的二进制字节流
        """
        # 解析输入数据
        _, filename, file_ext = self._get_file_info(input_data, default_ext)
        
        # 验证格式
        if file_ext not in self.OLD_TO_NEW_MAP:
            raise ValueError(
                f"仅支持旧格式转换：{list(self.OLD_TO_NEW_MAP.keys())}，"
                f"当前文件：{filename}（{file_ext}）"
            )
        
        # 执行转换
        convert_to = self.OLD_TO_NEW_MAP[file_ext]
        return self._send_convert_request(
            input_data=input_data,
            convert_to=convert_to,
            default_ext=default_ext,
            opts=["--overwrite"]
        )

    # ==================== 功能2：全格式转PDF（新旧Office文件都支持） ====================
    def convert_to_pdf(self, input_data: Union[str, bytes, BinaryIO], 
                      default_ext: str = "", opts: List[str] = None) -> bytes:
        """
        全格式转PDF（支持doc/docx、xls/xlsx、ppt/pptx）
        :param input_data: 文件路径 / 二进制字节 / 文件流
        :param default_ext: 二进制输入时指定源文件扩展名（如".docx"）
        :param opts: PDF专属参数（如["--quality=90", "--landscape"]）
        :return: 转换后的二进制字节流
        """
        # 解析输入数据
        _, filename, file_ext = self._get_file_info(input_data, default_ext)
        
        # 验证格式
        if file_ext not in self.PDF_SUPPORT_EXTS:
            raise ValueError(
                f"不支持的PDF转换格式：{filename}（{file_ext}），"
                f"仅支持：{self.PDF_SUPPORT_EXTS}"
            )
        
        # PDF默认参数
        pdf_opts = ["--quality=95", "--overwrite"]
        if opts:
            pdf_opts.extend(opts)
        
        # 执行转换
        return self._send_convert_request(
            input_data=input_data,
            convert_to="pdf",
            default_ext=default_ext,
            opts=pdf_opts
        )

    # ==================== 批量转换（文件路径输入，返回二进制字典） ====================
    def batch_convert(self, input_dir: str, convert_type: str = "pdf", 
                     max_workers: int = 3, opts: List[str] = None) -> Dict[str, Dict[str, Union[bytes, Exception]]]:
        """
        批量转换（仅支持文件路径输入）
        :param input_dir: 源文件目录
        :param convert_type: 转换类型（new/pdf/both）
        :param max_workers: 最大线程数
        :param opts: 转换参数
        :return: 结果字典 {
            "new": {"文件路径": 二进制数据 | 异常对象},
            "pdf": {"文件路径": 二进制数据 | 异常对象}
        }
        """
        if convert_type not in ["new", "pdf", "both"]:
            raise ValueError(f"无效的转换类型：{convert_type}（仅支持new/pdf/both）")

        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"源目录不存在：{input_dir}")

        # 收集任务
        new_tasks = []  # 旧→新任务
        pdf_tasks = []   # 转PDF任务

        for root, _, files in os.walk(input_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                file_ext = os.path.splitext(filename)[1].lower()

                # 旧→新任务（仅处理旧格式）
                if convert_type in ["new", "both"] and file_ext in self.OLD_TO_NEW_MAP:
                    new_tasks.append(file_path)

                # 转PDF任务（处理所有支持的格式）
                if convert_type in ["pdf", "both"] and file_ext in self.PDF_SUPPORT_EXTS:
                    pdf_tasks.append(file_path)

        # 定义转换执行函数
        def _convert_new(file_path):
            try:
                return file_path, self.convert_old_to_new(file_path), None
            except Exception as e:
                logger.error(f"❌ 旧转新失败：{file_path} → {str(e)[:100]}")
                return file_path, None, e

        def _convert_pdf(file_path):
            try:
                return file_path, self.convert_to_pdf(file_path, opts=opts), None
            except Exception as e:
                logger.error(f"❌ 转PDF失败：{file_path} → {str(e)[:100]}")
                return file_path, None, e

        # 执行转换
        results = {"new": {}, "pdf": {}}
        
        # 旧→新转换
        if new_tasks:
            logger.info(f"\n📦 开始批量旧格式转新格式（共{len(new_tasks)}个文件）")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_convert_new, fp): fp for fp in new_tasks}
                for future in as_completed(futures):
                    file_path, data, error = future.result()
                    if error:
                        results["new"][file_path] = error
                    else:
                        results["new"][file_path] = data

        # 转PDF转换
        if pdf_tasks:
            logger.info(f"\n📦 开始批量转PDF（共{len(pdf_tasks)}个文件）")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_convert_pdf, fp): fp for fp in pdf_tasks}
                for future in as_completed(futures):
                    file_path, data, error = future.result()
                    if error:
                        results["pdf"][file_path] = error
                    else:
                        results["pdf"][file_path] = data

        # 汇总结果
        success_new = sum(1 for v in results["new"].values() if isinstance(v, bytes))
        success_pdf = sum(1 for v in results["pdf"].values() if isinstance(v, bytes))
        
        logger.info(f"\n📊 批量转换汇总：")
        logger.info(f"  - 旧格式转新格式：总数={len(new_tasks)} | 成功={success_new} | 失败={len(new_tasks)-success_new}")
        logger.info(f"  - 全格式转PDF：总数={len(pdf_tasks)} | 成功={success_pdf} | 失败={len(pdf_tasks)-success_pdf}")

        return results



# ==================== 测试调用（覆盖所有场景）====================
if __name__ == "__main__":

    # 初始化转换器
    converter = OfficeConverter(
        host="localhost",
        port=18996,
        timeout=120,
        retry_times=2
    )
    # -------------------- 场景1：文件路径输入 → 二进制输出 --------------------
    # 旧格式转新格式（doc→docx）
    try:
        docx_bytes = converter.convert_old_to_new(input_data="./test.doc")
        # 可选：将二进制保存为文件（如需）
        with open("./test_new.docx", "wb") as f:
            f.write(docx_bytes)
        logger.info(f"✅ 旧格式转新格式成功，二进制数据大小：{len(docx_bytes)} 字节")
    except Exception as e:
        logger.error(f"❌ 转换失败：{e}")

    # 新格式转PDF（docx→pdf）
    try:
        pdf_bytes = converter.convert_to_pdf(input_data="./test.docx")
        with open("./test_new.pdf", "wb") as f:
            f.write(pdf_bytes)
        logger.info(f"✅ 转PDF成功，二进制数据大小：{len(pdf_bytes)} 字节")
    except Exception as e:
        logger.error(f"❌ 转换失败：{e}")

    # -------------------- 场景2：二进制输入 → 二进制输出 --------------------
    # 读取文件为二进制，再转换
    # try:
    #     with open("./test.ppt", "rb") as f:
    #         ppt_bytes = f.read()
        
    #     # 二进制转PDF（需指定默认扩展名）
    #     pdf_bytes = converter.convert_to_pdf(
    #         input_data=ppt_bytes,
    #         default_ext=".ppt",
    #         opts=["--landscape"]
    #     )
    #     with open("./test_from_binary.pdf", "wb") as f:
    #         f.write(pdf_bytes)
    #     logger.info(f"✅ 二进制转PDF成功，数据大小：{len(pdf_bytes)} 字节")
    # except Exception as e:
    #     logger.error(f"❌ 二进制转换失败：{e}")

    # -------------------- 场景3：文件流输入 → 二进制输出 --------------------
    try:
        with open("./data.xlsx", "rb") as f:
            # 文件流直接传入
            pdf_bytes = converter.convert_to_pdf(input_data=f)
        with open("./data_from_stream.pdf", "wb") as f:
            f.write(pdf_bytes)
        logger.info(f"✅ 文件流转PDF成功，数据大小：{len(pdf_bytes)} 字节")
    except Exception as e:
        logger.error(f"❌ 文件流转换失败：{e}")
