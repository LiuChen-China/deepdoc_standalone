#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import io
import hashlib
import zipfile
import requests
from requests.exceptions import Timeout, RequestException
from io import BytesIO
from typing import List, Union, Tuple, Optional, Dict
import PyPDF2
from docx import Document
import olefile

def _is_zip(h: bytes) -> bool:
    return h.startswith(b"PK\x03\x04") or h.startswith(b"PK\x05\x06") or h.startswith(b"PK\x07\x08")

def _is_pdf(h: bytes) -> bool:
    return h.startswith(b"%PDF-")

def _is_ole(h: bytes) -> bool:
    return h.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1")

def _sha10(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:10]

def _guess_ext(b: bytes) -> str:
    h = b[:8]
    if _is_zip(h):
        try:
            with zipfile.ZipFile(io.BytesIO(b), "r") as z:
                names = [n.lower() for n in z.namelist()]
                if any(n.startswith("word/") for n in names):
                    return ".docx"
                if any(n.startswith("ppt/") for n in names):
                    return ".pptx"
                if any(n.startswith("xl/") for n in names):
                    return ".xlsx"
        except Exception:
            pass
        return ".zip"
    if _is_pdf(h):
        return ".pdf"
    if _is_ole(h):
        return ".doc"
    return ".bin"

# Try to extract the real embedded payload from OLE's Ole10Native
def _extract_ole10native_payload(data: bytes) -> bytes:
    try:
        pos = 0
        if len(data) < 4:
            return data
        _ = int.from_bytes(data[pos:pos+4], "little")
        pos += 4
        # filename/src/tmp (NUL-terminated ANSI)
        for _ in range(3):
            z = data.index(b"\x00", pos)
            pos = z + 1
        # skip unknown 4 bytes
        pos += 4
        if pos + 4 > len(data):
            return data
        size = int.from_bytes(data[pos:pos+4], "little")
        pos += 4
        if pos + size <= len(data):
            return data[pos:pos+size]
    except Exception:
        pass
    return data

def extract_embed_file(target: Union[bytes, bytearray]) -> List[Tuple[str, bytes]]:
    """
    从文档容器中提取嵌入的文件，支持Office文档中的"第一层"嵌入内容
    
    参数:
        target (Union[bytes, bytearray]): 目标文档的二进制数据，可以是bytes或bytearray类型
        
    返回:
        List[Tuple[str, bytes]]: 提取出的嵌入文件列表，每个元素为(文件名, 文件二进制数据)的元组
        
    支持的容器格式:
        - OOXML格式 (docx, xlsx, pptx): 基于ZIP的容器格式
        - OLE格式 (doc, xls, ppt): 旧版Office二进制容器格式
        
    工作原理:
        1. 通过文件头部特征识别容器类型
        2. 根据不同容器类型使用相应的方法提取嵌入文件
        3. 对提取的文件进行去重处理，避免重复添加
        4. 为每个文件生成合适的文件名
    """
    # 将目标数据转换为bytes类型以便统一处理
    top = bytes(target)  # 确保输入数据被标准化为bytes类型，无论原始类型是bytes还是bytearray
    head = top[:8]  # 提取文件头部前8个字节，用于文件类型识别
    out: List[Tuple[str, bytes]] = []  # 初始化输出列表，存储(文件名, 文件数据)的元组
    seen = set()  # 使用集合存储已处理文件的哈希值，用于去重

    def push(b: bytes, name_hint: str = ""):
        """
        内部辅助函数：处理并添加提取出的嵌入文件
        
        参数:
            b (bytes): 嵌入文件的二进制数据
            name_hint (str): 文件路径提示，用于生成文件名
        """
        h10 = _sha10(b)  # 计算文件数据的哈希值前10个字符作为唯一标识
        if h10 in seen:  # 检查文件是否已处理过（去重）
            return
        seen.add(h10)  # 记录已处理的文件哈希值
        ext = _guess_ext(b)  # 根据文件内容猜测文件扩展名
        # 根据name_hint是否有扩展名决定文件名生成策略
        if "." in name_hint:
            fname = name_hint.split("/")[-1]  # 从路径提示中提取文件名部分
        else:
            fname = f"{h10}{ext}"  # 使用哈希值和猜测的扩展名生成文件名
        out.append((fname, b))  # 将(文件名, 文件数据)添加到输出列表

    # 处理OOXML/ZIP容器格式 (docx/xlsx/pptx)
    if _is_zip(head):
        try:
            # 将二进制数据包装成文件对象并打开ZIP容器
            with zipfile.ZipFile(io.BytesIO(top), "r") as z:
                # 定义可能包含嵌入文件的目录路径
                embed_dirs = (
                    "word/embeddings/", "word/objects/", "word/activex/",
                    "xl/embeddings/", "ppt/embeddings/"
                )
                # 遍历ZIP中的所有文件
                for name in z.namelist():
                    low = name.lower()
                    # 检查文件是否位于嵌入文件目录下
                    if any(low.startswith(d) for d in embed_dirs):
                        try:
                            b = z.read(name)  # 读取嵌入文件的二进制数据
                            push(b, name)  # 处理并添加到输出列表
                        except Exception:
                            pass  # 忽略读取错误
        except Exception:
            pass  # 忽略ZIP处理错误
        return out

    # 处理OLE容器格式 (doc/ppt/xls)
    if _is_ole(head):
        try:
            # 将二进制数据包装成文件对象并打开OLE容器
            with olefile.OleFileIO(io.BytesIO(top)) as ole:
                # 遍历OLE中的所有条目
                for entry in ole.listdir():
                    p = "/".join(entry)  # 将条目路径列表转换为字符串路径
                    try:
                        data = ole.openstream(entry).read()  # 读取条目数据流
                    except Exception:
                        continue  # 忽略读取错误
                    if not data:  # 跳过空数据
                        continue
                    # 特殊处理Ole10Native格式，提取其实际内容
                    if "Ole10Native" in p or "ole10native" in p.lower():
                        data = _extract_ole10native_payload(data)
                    push(data, p)  # 处理并添加到输出列表
        except Exception:
            pass  # 忽略OLE处理错误
        return out

    return out  # 返回提取结果，若无嵌入文件则返回空列表


def extract_links_from_docx(docx_bytes: bytes):
    """
    Extract all hyperlinks from a Word (.docx) document binary stream.

    Args:
        docx_bytes (bytes): Raw bytes of a .docx file.

    Returns:
        set[str]: A set of unique hyperlink URLs.
    """
    links = set()
    with BytesIO(docx_bytes) as bio:
        document = Document(bio)

        # Each relationship may represent a hyperlink, image, footer, etc.
        for rel in document.part.rels.values():
            if rel.reltype == (
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"
            ):
                links.add(rel.target_ref)

    return links


def extract_links_from_pdf(pdf_bytes: bytes):
    """
    Extract all clickable hyperlinks from a PDF binary stream.

    Args:
        pdf_bytes (bytes): Raw bytes of a PDF file.

    Returns:
        set[str]: A set of unique hyperlink URLs (unordered).
    """
    links = set()
    with BytesIO(pdf_bytes) as bio:
        pdf = PyPDF2.PdfReader(bio)

        for page in pdf.pages:
            annots = page.get("/Annots")
            if not annots or isinstance(annots, PyPDF2.generic.IndirectObject):
                continue
            for annot in annots:
                obj = annot.get_object()
                a = obj.get("/A")
                if a and a.get("/URI"):
                    links.add(a["/URI"])

    return links


_GLOBAL_SESSION: Optional[requests.Session] = None
def _get_session(headers: Optional[Dict[str, str]] = None) -> requests.Session:
    """Get or create a global reusable session."""
    global _GLOBAL_SESSION
    if _GLOBAL_SESSION is None:
        _GLOBAL_SESSION = requests.Session()
        _GLOBAL_SESSION.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0 Safari/537.36"
            )
        })
    if headers:
        _GLOBAL_SESSION.headers.update(headers)
    return _GLOBAL_SESSION


def extract_html(
    url: str,
    timeout: float = 60.0,
    headers: Optional[Dict[str, str]] = None,
    max_retries: int = 2,
) -> Tuple[Optional[bytes], Dict[str, str]]:
    """
    Extract the full HTML page as raw bytes from a given URL.
    Automatically reuses a persistent HTTP session and applies robust timeout & retry logic.

    Args:
        url (str): Target webpage URL.
        timeout (float): Request timeout in seconds (applies to connect + read).
        headers (dict, optional): Extra HTTP headers.
        max_retries (int): Number of retries on timeout or transient errors.

    Returns:
        tuple(bytes|None, dict):
            - html_bytes: Raw HTML content (or None if failed)
            - metadata: HTTP info (status_code, content_type, final_url, error if any)
    """
    session = _get_session(headers=headers)
    metadata = {"final_url": url, "status_code": "", "content_type": "", "error": ""}

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()

            html_bytes = resp.content
            metadata.update({
                "final_url": resp.url,
                "status_code": str(resp.status_code),
                "content_type": resp.headers.get("Content-Type", ""),
            })
            return html_bytes, metadata

        except Timeout:
            metadata["error"] = f"Timeout after {timeout}s (attempt {attempt}/{max_retries})"
            if attempt >= max_retries:
                continue
        except RequestException as e:
            metadata["error"] = f"Request failed: {e}"
            continue

    return None, metadata