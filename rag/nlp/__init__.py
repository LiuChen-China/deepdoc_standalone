#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
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

import logging
import random
from collections import Counter
try:
    from deepdoc.parser.pdf_parser import RAGFlowPdfParser
    from common.token_utils import num_tokens_from_string
    from . import rag_tokenizer
except:
    from deepdoc_standalone.deepdoc.parser.pdf_parser import RAGFlowPdfParser
    from deepdoc_standalone.common.token_utils import num_tokens_from_string
    from deepdoc_standalone.rag.nlp import rag_tokenizer
import re
import copy
import roman_numbers as r
from word2number import w2n
from cn2an import cn2an
from PIL import Image

import chardet

all_codecs = [
    'utf-8', 'gb2312', 'gbk', 'utf_16', 'ascii', 'big5', 'big5hkscs',
    'cp037', 'cp273', 'cp424', 'cp437',
    'cp500', 'cp720', 'cp737', 'cp775', 'cp850', 'cp852', 'cp855', 'cp856', 'cp857',
    'cp858', 'cp860', 'cp861', 'cp862', 'cp863', 'cp864', 'cp865', 'cp866', 'cp869',
    'cp874', 'cp875', 'cp932', 'cp949', 'cp950', 'cp1006', 'cp1026', 'cp1125',
    'cp1140', 'cp1250', 'cp1251', 'cp1252', 'cp1253', 'cp1254', 'cp1255', 'cp1256',
    'cp1257', 'cp1258', 'euc_jp', 'euc_jis_2004', 'euc_jisx0213', 'euc_kr',
    'gb18030', 'hz', 'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2',
    'iso2022_jp_2004', 'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_kr', 'latin_1',
    'iso8859_2', 'iso8859_3', 'iso8859_4', 'iso8859_5', 'iso8859_6', 'iso8859_7',
    'iso8859_8', 'iso8859_9', 'iso8859_10', 'iso8859_11', 'iso8859_13',
    'iso8859_14', 'iso8859_15', 'iso8859_16', 'johab', 'koi8_r', 'koi8_t', 'koi8_u',
    'kz1048', 'mac_cyrillic', 'mac_greek', 'mac_iceland', 'mac_latin2', 'mac_roman',
    'mac_turkish', 'ptcp154', 'shift_jis', 'shift_jis_2004', 'shift_jisx0213',
    'utf_32', 'utf_32_be', 'utf_32_le', 'utf_16_be', 'utf_16_le', 'utf_7', 'windows-1250', 'windows-1251',
    'windows-1252', 'windows-1253', 'windows-1254', 'windows-1255', 'windows-1256',
    'windows-1257', 'windows-1258', 'latin-2'
]


def find_codec(blob):
    detected = chardet.detect(blob[:1024])
    if detected['confidence'] > 0.5:
        if detected['encoding'] == "ascii":
            return "utf-8"

    for c in all_codecs:
        try:
            blob[:1024].decode(c)
            return c
        except Exception:
            pass
        try:
            blob.decode(c)
            return c
        except Exception:
            pass

    return "utf-8"


QUESTION_PATTERN = [
    r"第([零一二三四五六七八九十百0-9]+)问",
    r"第([零一二三四五六七八九十百0-9]+)条",
    r"[\(（]([零一二三四五六七八九十百]+)[\)）]",
    r"第([0-9]+)问",
    r"第([0-9]+)条",
    r"([0-9]{1,2})[\. 、]",
    r"([零一二三四五六七八九十百]+)[ 、]",
    r"[\(（]([0-9]{1,2})[\)）]",
    r"QUESTION (ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)",
    r"QUESTION (I+V?|VI*|XI|IX|X)",
    r"QUESTION ([0-9]+)",
]


def has_qbullet(reg, box, last_box, last_index, last_bull, bull_x0_list):
    section, last_section = box['text'], last_box['text']
    q_reg = r'(\w|\W)*?(?:？|\?|\n|$)+'
    full_reg = reg + q_reg
    has_bull = re.match(full_reg, section)
    index_str = None
    if has_bull:
        if 'x0' not in last_box:
            last_box['x0'] = box['x0']
        if 'top' not in last_box:
            last_box['top'] = box['top']
        if last_bull and box['x0'] - last_box['x0'] > 10:
            return None, last_index
        if not last_bull and box['x0'] >= last_box['x0'] and box['top'] - last_box['top'] < 20:
            return None, last_index
        avg_bull_x0 = 0
        if bull_x0_list:
            avg_bull_x0 = sum(bull_x0_list) / len(bull_x0_list)
        else:
            avg_bull_x0 = box['x0']
        if box['x0'] - avg_bull_x0 > 10:
            return None, last_index
        index_str = has_bull.group(1)
        index = index_int(index_str)
        if last_section[-1] == ':' or last_section[-1] == '：':
            return None, last_index
        if not last_index or index >= last_index:
            bull_x0_list.append(box['x0'])
            return has_bull, index
        if section[-1] == '?' or section[-1] == '？':
            bull_x0_list.append(box['x0'])
            return has_bull, index
        if box['layout_type'] == 'title':
            bull_x0_list.append(box['x0'])
            return has_bull, index
        pure_section = section.lstrip(re.match(reg, section).group()).lower()
        ask_reg = r'(what|when|where|how|why|which|who|whose|为什么|为啥|哪)'
        if re.match(ask_reg, pure_section):
            bull_x0_list.append(box['x0'])
            return has_bull, index
    return None, last_index


def index_int(index_str):
    res = -1
    try:
        res = int(index_str)
    except ValueError:
        try:
            res = w2n.word_to_num(index_str)
        except ValueError:
            try:
                res = cn2an(index_str)
            except ValueError:
                try:
                    res = r.number(index_str)
                except ValueError:
                    return -1
    return res


def qbullets_category(sections):
    global QUESTION_PATTERN
    hits = [0] * len(QUESTION_PATTERN)
    for i, pro in enumerate(QUESTION_PATTERN):
        for sec in sections:
            if re.match(pro, sec) and not not_bullet(sec):
                hits[i] += 1
                break
    maximum = 0
    res = -1
    for i, h in enumerate(hits):
        if h <= maximum:
            continue
        res = i
        maximum = h
    return res, QUESTION_PATTERN[res]


BULLET_PATTERN = [[
    r"第[零一二三四五六七八九十百0-9]+(分?编|部分)",
    r"第[零一二三四五六七八九十百0-9]+章",
    r"第[零一二三四五六七八九十百0-9]+节",
    r"第[零一二三四五六七八九十百0-9]+条",
    r"[\(（][零一二三四五六七八九十百]+[\)）]",
], [
    r"第[0-9]+章",
    r"第[0-9]+节",
    r"[0-9]{,2}[\. 、]",
    r"[0-9]{,2}\.[0-9]{,2}[^a-zA-Z/%~-]",
    r"[0-9]{,2}\.[0-9]{,2}\.[0-9]{,2}",
    r"[0-9]{,2}\.[0-9]{,2}\.[0-9]{,2}\.[0-9]{,2}",
], [
    r"第[零一二三四五六七八九十百0-9]+章",
    r"第[零一二三四五六七八九十百0-9]+节",
    r"[零一二三四五六七八九十百]+[ 、]",
    r"[\(（][零一二三四五六七八九十百]+[\)）]",
    r"[\(（][0-9]{,2}[\)）]",
], [
    r"PART (ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)",
    r"Chapter (I+V?|VI*|XI|IX|X)",
    r"Section [0-9]+",
    r"Article [0-9]+"
], [
    r"^#[^#]",
    r"^##[^#]",
    r"^###.*",
    r"^####.*",
    r"^#####.*",
    r"^######.*",
]
]


def random_choices(arr, k):
    k = min(len(arr), k)
    return random.choices(arr, k=k)


def not_bullet(line):
    patt = [
        r"0", r"[0-9]+ +[0-9~个只-]", r"[0-9]+\.{2,}"
    ]
    return any([re.match(r, line) for r in patt])


def bullets_category(sections):
    global BULLET_PATTERN
    hits = [0] * len(BULLET_PATTERN)
    for i, pro in enumerate(BULLET_PATTERN):
        for sec in sections:
            sec = sec.strip()
            for p in pro:
                if re.match(p, sec) and not not_bullet(sec):
                    hits[i] += 1
                    break
    maximum = 0
    res = -1
    for i, h in enumerate(hits):
        if h <= maximum:
            continue
        res = i
        maximum = h
    return res


def is_english(texts):
    if not texts:
        return False

    pattern = re.compile(r"[`a-zA-Z0-9\s.,':;/\"?<>!\(\)\-]")

    if isinstance(texts, str):
        texts = list(texts)
    elif isinstance(texts, list):
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
    else:
        return False

    if not texts:
        return False

    eng = sum(1 for t in texts if pattern.fullmatch(t.strip()))
    return (eng / len(texts)) > 0.8


def is_chinese(text):
    if not text:
        return False
    chinese = 0
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            chinese += 1
    if chinese / len(text) > 0.2:
        return True
    return False


def tokenize(d, t, eng):
    d["content_with_weight"] = t
    t = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", t)
    d["content_ltks"] = rag_tokenizer.tokenize(t)
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])


def tokenize_chunks(chunks, doc, eng, pdf_parser=None):
    """
    对文档块进行标记化处理并转换为ES文档格式
    
    参数:
        chunks (list): 文档文本块列表，每个元素为需要处理的文本片段
        doc (dict): 文档模板字典，作为基础结构被深拷贝用于每个块
        eng: 引擎实例，可能包含NLP处理相关的配置和功能
        pdf_parser: PDF解析器实例（可选），用于从PDF中提取图像和位置信息
    
    返回:
        list: 处理后的文档列表，每个文档包含标记化内容和位置信息
    """
    res = []  # 存储处理后文档的结果列表
    # wrap up as es documents
    for ii, ck in enumerate(chunks):
        # 跳过空文本块
        if len(ck.strip()) == 0:
            continue
        logging.debug("-- {}".format(ck))  # 记录处理的文本块内容
        
        # 深拷贝文档模板，避免修改原始文档
        d = copy.deepcopy(doc)
        
        # 如果提供了PDF解析器，尝试提取图像和位置信息
        if pdf_parser:
            try:
                # 从文本块裁剪图像并获取位置信息
                d["image"], poss = pdf_parser.crop(ck, need_position=True)
                # 将位置信息添加到文档中
                add_positions(d, poss)
                # 移除文本块中的标签
                ck = pdf_parser.remove_tag(ck)
            except NotImplementedError:
                # 如果PDF解析器不支持某些功能，则跳过
                pass
        else:
            # 如果没有PDF解析器，使用默认位置信息
            add_positions(d, [[ii]*5])
        
        # 对文本块进行标记化处理
        tokenize(d, ck, eng)
        # 将处理后的文档添加到结果列表
        res.append(d)
    
    return res  # 返回处理后的文档列表

def tokenize_chunks_with_images(chunks, doc, eng, images):
    res = []
    # wrap up as es documents
    for ii, (ck, image) in enumerate(zip(chunks, images)):
        if len(ck.strip()) == 0:
            continue
        logging.debug("-- {}".format(ck))
        d = copy.deepcopy(doc)
        d["image"] = image
        add_positions(d, [[ii]*5])
        tokenize(d, ck, eng)
        res.append(d)
    return res


def tokenize_table(tbls, doc, eng, batch_size=10):
    res = []
    # add tables
    for (img, rows), poss in tbls:
        if not rows:
            continue
        if isinstance(rows, str):
            d = copy.deepcopy(doc)
            tokenize(d, rows, eng)
            d["content_with_weight"] = rows
            if img:
                d["image"] = img
                d["doc_type_kwd"] = "image"
            if poss:
                add_positions(d, poss)
            res.append(d)
            continue
        de = "; " if eng else "； "
        for i in range(0, len(rows), batch_size):
            d = copy.deepcopy(doc)
            r = de.join(rows[i:i + batch_size])
            tokenize(d, r, eng)
            if img:
                d["image"] = img
                d["doc_type_kwd"] = "image"
            add_positions(d, poss)
            res.append(d)
    return res


def add_positions(d, poss):
    """
    向字典中添加位置信息的函数
    
    该函数用于将文档元素（如文本、图像、表格等）的位置信息处理并添加到指定字典中。
    位置信息包括页码、左右边界坐标、上下边界坐标等，这些信息对于文档解析、元素定位和
    可视化非常重要。
    
    参数:
        d (dict): 要添加位置信息的目标字典
        poss (list): 位置信息列表，每个元素是一个元组，包含(pn, left, right, top, bottom)
                    其中pn为页码（从0开始），其他为坐标信息
    
    返回值:
        None: 函数直接修改传入的字典，没有返回值
    """
    # 如果位置信息列表为空，则直接返回
    if not poss:
        return
    
    # 初始化存储转换后位置信息的列表
    page_num_int = []  # 存储页码（整数）的列表
    position_int = []  # 存储完整位置信息（整数）的列表
    top_int = []       # 存储顶部坐标（整数）的列表
    
    # 遍历每个位置信息，进行类型转换并存储
    for pn, left, right, top, bottom in poss:
        # 页码从1开始（原始页码从0开始，所以+1）
        page_num_int.append(int(pn + 1))
        # 添加顶部坐标
        top_int.append(int(top))
        # 添加完整的位置信息元组
        position_int.append((int(pn + 1), int(left), int(right), int(top), int(bottom)))
    
    # 将处理后的位置信息添加到字典中
    d["page_num_int"] = page_num_int  # 页码信息
    d["position_int"] = position_int  # 完整位置信息
    d["top_int"] = top_int           # 顶部坐标信息


def remove_contents_table(sections, eng=False):
    i = 0
    while i < len(sections):
        def get(i):
            nonlocal sections
            return (sections[i] if isinstance(sections[i],
                                              type("")) else sections[i][0]).strip()

        if not re.match(r"(contents|目录|目次|table of contents|致谢|acknowledge)$",
                        re.sub(r"( | |\u3000)+", "", get(i).split("@@")[0], flags=re.IGNORECASE)):
            i += 1
            continue
        sections.pop(i)
        if i >= len(sections):
            break
        prefix = get(i)[:3] if not eng else " ".join(get(i).split()[:2])
        while not prefix:
            sections.pop(i)
            if i >= len(sections):
                break
            prefix = get(i)[:3] if not eng else " ".join(get(i).split()[:2])
        sections.pop(i)
        if i >= len(sections) or not prefix:
            break
        for j in range(i, min(i + 128, len(sections))):
            if not re.match(prefix, get(j)):
                continue
            for _ in range(i, j):
                sections.pop(i)
            break


def make_colon_as_title(sections):
    if not sections:
        return []
    if isinstance(sections[0], type("")):
        return sections
    i = 0
    while i < len(sections):
        txt, layout = sections[i]
        i += 1
        txt = txt.split("@")[0].strip()
        if not txt:
            continue
        if txt[-1] not in ":：":
            continue
        txt = txt[::-1]
        arr = re.split(r"([。？！!?;；]| \.)", txt)
        if len(arr) < 2 or len(arr[1]) < 32:
            continue
        sections.insert(i - 1, (arr[0][::-1], "title"))
        i += 1


def title_frequency(bull, sections):
    bullets_size = len(BULLET_PATTERN[bull])
    levels = [bullets_size + 1 for _ in range(len(sections))]
    if not sections or bull < 0:
        return bullets_size + 1, levels

    for i, (txt, layout) in enumerate(sections):
        for j, p in enumerate(BULLET_PATTERN[bull]):
            if re.match(p, txt.strip()) and not not_bullet(txt):
                levels[i] = j
                break
        else:
            if re.search(r"(title|head)", layout) and not not_title(txt.split("@")[0]):
                levels[i] = bullets_size
    most_level = bullets_size + 1
    for level, c in sorted(Counter(levels).items(), key=lambda x: x[1] * -1):
        if level <= bullets_size:
            most_level = level
            break
    return most_level, levels


def not_title(txt):
    if re.match(r"第[零一二三四五六七八九十百0-9]+条", txt):
        return False
    if len(txt.split()) > 12 or (txt.find(" ") < 0 and len(txt) >= 32):
        return True
    return re.search(r"[,;，。；！!]", txt)

def tree_merge(bull, sections, depth):

    if not sections or bull < 0:
        return sections
    if isinstance(sections[0], type("")):
        sections = [(s, "") for s in sections]

    # filter out position information in pdf sections
    sections = [(t, o) for t, o in sections if
                t and len(t.split("@")[0].strip()) > 1 and not re.match(r"[0-9]+$", t.split("@")[0].strip())]

    def get_level(bull, section):
        text, layout = section
        text = re.sub(r"\u3000", " ",   text).strip()

        for i, title in enumerate(BULLET_PATTERN[bull]):
            if re.match(title, text.strip()):
                return i+1, text
        else:
            if re.search(r"(title|head)", layout) and not not_title(text):
                return len(BULLET_PATTERN[bull])+1, text
            else:
                return len(BULLET_PATTERN[bull])+2, text
    level_set = set()
    lines = []
    for section in sections:
        level, text = get_level(bull, section)
        if not text.strip("\n"):
            continue

        lines.append((level, text))
        level_set.add(level)

    sorted_levels = sorted(list(level_set))

    if depth <= len(sorted_levels):
        target_level = sorted_levels[depth - 1]
    else:
        target_level = sorted_levels[-1]

    if target_level == len(BULLET_PATTERN[bull]) + 2:
        target_level = sorted_levels[-2] if len(sorted_levels) > 1 else sorted_levels[0]

    root = Node(level=0, depth=target_level, texts=[])
    root.build_tree(lines)

    return [element for element in root.get_tree() if element]

def hierarchical_merge(bull, sections, depth):

    if not sections or bull < 0:
        return []
    if isinstance(sections[0], type("")):
        sections = [(s, "") for s in sections]
    sections = [(t, o) for t, o in sections if
                t and len(t.split("@")[0].strip()) > 1 and not re.match(r"[0-9]+$", t.split("@")[0].strip())]
    bullets_size = len(BULLET_PATTERN[bull])
    levels = [[] for _ in range(bullets_size + 2)]

    for i, (txt, layout) in enumerate(sections):
        for j, p in enumerate(BULLET_PATTERN[bull]):
            if re.match(p, txt.strip()):
                levels[j].append(i)
                break
        else:
            if re.search(r"(title|head)", layout) and not not_title(txt):
                levels[bullets_size].append(i)
            else:
                levels[bullets_size + 1].append(i)
    sections = [t for t, _ in sections]

    # for s in sections: print("--", s)

    def binary_search(arr, target):
        if not arr:
            return -1
        if target > arr[-1]:
            return len(arr) - 1
        if target < arr[0]:
            return -1
        s, e = 0, len(arr)
        while e - s > 1:
            i = (e + s) // 2
            if target > arr[i]:
                s = i
                continue
            elif target < arr[i]:
                e = i
                continue
            else:
                assert False
        return s

    cks = []
    readed = [False] * len(sections)
    levels = levels[::-1]
    for i, arr in enumerate(levels[:depth]):
        for j in arr:
            if readed[j]:
                continue
            readed[j] = True
            cks.append([j])
            if i + 1 == len(levels) - 1:
                continue
            for ii in range(i + 1, len(levels)):
                jj = binary_search(levels[ii], j)
                if jj < 0:
                    continue
                if levels[ii][jj] > cks[-1][-1]:
                    cks[-1].pop(-1)
                cks[-1].append(levels[ii][jj])
            for ii in cks[-1]:
                readed[ii] = True

    if not cks:
        return cks

    for i in range(len(cks)):
        cks[i] = [sections[j] for j in cks[i][::-1]]
        logging.debug("\n* ".join(cks[i]))

    res = [[]]
    num = [0]
    for ck in cks:
        if len(ck) == 1:
            n = num_tokens_from_string(re.sub(r"@@[0-9]+.*", "", ck[0]))
            if n + num[-1] < 218:
                res[-1].append(ck[0])
                num[-1] += n
                continue
            res.append(ck)
            num.append(n)
            continue
        res.append(ck)
        num.append(218)

    return res


"""
将文本段落合并为大小适中的文本块，支持自定义分隔符和重叠处理

参数:
    sections: str | list - 要合并的文本段落，可以是单个字符串或字符串列表，
                         也可以是包含(文本,位置信息)元组的列表
    chunk_token_num: int - 每个合并后文本块的最大token数量，默认128
    delimiter: str - 用于分割文本的分隔符，默认为"\n。；！？"，
                    支持使用反引号定义自定义分隔符，如`\n\n`
    overlapped_percent: int - 相邻文本块之间的重叠百分比，默认0%

返回:
    list - 合并后的文本块列表
"""
def naive_merge(sections: str | list, chunk_token_num=128, delimiter="\n。；！？", overlapped_percent=0):
    # 导入用于处理标签的工具类
    
    
    # 处理空输入情况
    if not sections:
        return []
    
    # 统一输入格式：如果输入是单个字符串，转换为列表
    if isinstance(sections, str):
        sections = [sections]
    
    # 统一输入格式：如果列表元素是字符串，转换为(文本,位置信息)元组形式
    if isinstance(sections[0], str):
        sections = [(s, "") for s in sections]
    
    # 初始化结果列表和token计数列表
    cks = [""]
    tk_nums = [0]
    
    """
    内部函数：将文本添加到合适的文本块中
    
    参数:
        t: str - 要添加的文本内容
        pos: str - 文本的位置信息
    """
    def add_chunk(t, pos):
        # 使用nonlocal引用外部函数的变量
        nonlocal cks, tk_nums, delimiter
        # 计算当前文本的token数量
        tnum = num_tokens_from_string(t)
        
        # 处理空位置信息
        if not pos:
            pos = ""
        # 对于过短的文本，不保留位置信息
        if tnum < 8:
            pos = ""
        
        # 检查是否需要创建新的文本块
        # 条件：当前块为空 或 当前块token数超过阈值(考虑重叠百分比)
        if cks[-1] == "" or tk_nums[-1] > chunk_token_num * (100 - overlapped_percent)/100.:
            # 处理重叠情况：如果不是第一个块，提取前一个块的尾部作为重叠部分
            if cks:
                overlapped = RAGFlowPdfParser.remove_tag(cks[-1])
                # 截取前一个块的尾部(根据重叠百分比)并与当前文本拼接
                t = overlapped[int(len(overlapped)*(100-overlapped_percent)/100.):] + t
            
            # 确保位置信息被添加到文本中
            if t.find(pos) < 0:
                t += pos
            
            # 创建新块并更新token计数
            cks.append(t)
            tk_nums.append(tnum)
        else:
            # 将文本添加到现有块
            if cks[-1].find(pos) < 0:
                t += pos
            cks[-1] += t
            tk_nums[-1] += tnum
    
    # 处理自定义分隔符情况 (格式为`分隔符`)
    custom_delimiters = [m.group(1) for m in re.finditer(r"`([^`]+)`", delimiter)]
    has_custom = bool(custom_delimiters)
    
    # 如果存在自定义分隔符，使用特殊逻辑处理
    if has_custom:
        # 构建自定义分隔符的正则表达式模式
        # 按长度降序排列，确保较长的分隔符先匹配
        custom_pattern = "|".join(re.escape(t) for t in sorted(set(custom_delimiters), key=len, reverse=True))
        # 重新初始化结果列表
        cks, tk_nums = [], []
        
        # 对每个段落进行处理
        for sec, pos in sections:
            # 使用自定义分隔符分割文本
            split_sec = re.split(r"(%s)" % custom_pattern, sec, flags=re.DOTALL)
            
            # 处理分割后的每个子段落
            for sub_sec in split_sec:
                # 跳过纯分隔符部分
                if re.fullmatch(custom_pattern, sub_sec or ""):
                    continue
                
                # 添加换行符并处理位置信息
                text = "\n" + sub_sec
                local_pos = pos
                # 对于过短的文本，不保留位置信息
                if num_tokens_from_string(text) < 8:
                    local_pos = ""
                # 确保位置信息被添加到文本中
                if local_pos and text.find(local_pos) < 0:
                    text += local_pos
                
                # 添加到结果列表
                cks.append(text)
                tk_nums.append(num_tokens_from_string(text))
        
        # 返回使用自定义分隔符处理的结果
        return cks
    
    # 使用默认合并逻辑处理所有段落
    for sec, pos in sections:
        add_chunk("\n"+sec, pos)
    
    # 返回合并后的文本块列表
    return cks


def naive_merge_with_images(texts, images, chunk_token_num=128, delimiter="\n。；！？", overlapped_percent=0):
    from deepdoc.parser.pdf_parser import RAGFlowPdfParser
    if not texts or len(texts) != len(images):
        return [], []
    cks = [""]
    result_images = [None]
    tk_nums = [0]

    def add_chunk(t, image, pos=""):
        nonlocal cks, result_images, tk_nums, delimiter
        tnum = num_tokens_from_string(t)
        if not pos:
            pos = ""
        if tnum < 8:
            pos = ""
        # Ensure that the length of the merged chunk does not exceed chunk_token_num
        if cks[-1] == "" or tk_nums[-1] > chunk_token_num * (100 - overlapped_percent)/100.:
            if cks:
                overlapped = RAGFlowPdfParser.remove_tag(cks[-1])
                t = overlapped[int(len(overlapped)*(100-overlapped_percent)/100.):] + t
            if t.find(pos) < 0:
                t += pos
            cks.append(t)
            result_images.append(image)
            tk_nums.append(tnum)
        else:
            if cks[-1].find(pos) < 0:
                t += pos
            cks[-1] += t
            if result_images[-1] is None:
                result_images[-1] = image
            else:
                result_images[-1] = concat_img(result_images[-1], image)
            tk_nums[-1] += tnum

    custom_delimiters = [m.group(1) for m in re.finditer(r"`([^`]+)`", delimiter)]
    has_custom = bool(custom_delimiters)
    if has_custom:
        custom_pattern = "|".join(re.escape(t) for t in sorted(set(custom_delimiters), key=len, reverse=True))
        cks, result_images, tk_nums = [], [], []
        for text, image in zip(texts, images):
            text_str = text[0] if isinstance(text, tuple) else text
            text_pos = text[1] if isinstance(text, tuple) and len(text) > 1 else ""
            split_sec = re.split(r"(%s)" % custom_pattern, text_str)
            for sub_sec in split_sec:
                if re.fullmatch(custom_pattern, sub_sec or ""):
                    continue
                text_seg = "\n" + sub_sec
                local_pos = text_pos
                if num_tokens_from_string(text_seg) < 8:
                    local_pos = ""
                if local_pos and text_seg.find(local_pos) < 0:
                    text_seg += local_pos
                cks.append(text_seg)
                result_images.append(image)
                tk_nums.append(num_tokens_from_string(text_seg))
        return cks, result_images

    for text, image in zip(texts, images):
        # if text is tuple, unpack it
        if isinstance(text, tuple):
            text_str = text[0]
            text_pos = text[1] if len(text) > 1 else ""
            add_chunk("\n"+text_str, image, text_pos)
        else:
            add_chunk("\n"+text, image)

    return cks, result_images


def docx_question_level(p, bull=-1):
    txt = re.sub(r"\u3000", " ", p.text).strip()
    if p.style.name.startswith('Heading'):
        return int(p.style.name.split(' ')[-1]), txt
    else:
        if bull < 0:
            return 0, txt
        for j, title in enumerate(BULLET_PATTERN[bull]):
            if re.match(title, txt):
                return j + 1, txt
    return len(BULLET_PATTERN[bull])+1, txt


def concat_img(img1, img2):
    if img1 and not img2:
        return img1
    if not img1 and img2:
        return img2
    if not img1 and not img2:
        return None

    if img1 is img2:
        return img1

    if isinstance(img1, Image.Image) and isinstance(img2, Image.Image):
        pixel_data1 = img1.tobytes()
        pixel_data2 = img2.tobytes()
        if pixel_data1 == pixel_data2:
            return img1

    width1, height1 = img1.size
    width2, height2 = img2.size

    new_width = max(width1, width2)
    new_height = height1 + height2
    new_image = Image.new('RGB', (new_width, new_height))

    new_image.paste(img1, (0, 0))
    new_image.paste(img2, (0, height1))
    return new_image



def naive_merge_docx(sections, chunk_token_num=128, delimiter="\n。；！？"):
    """
    简单合并docx文档的文本段落和对应图像

    该函数将docx文档的文本段落和对应图像进行合并处理，根据指定的令牌数量限制进行分块，
    支持自定义分隔符进行文本分割，同时保持文本和图像的对应关系。

    参数:
        sections: 包含文本段落和对应图像的列表，每个元素为(段落文本, 图像)元组
        chunk_token_num: 每个文本块的最大令牌数量，默认128
        delimiter: 用于分割文本的分隔符字符串，默认包含换行符和中文标点符号

    返回:
        tuple: (分块后的文本列表, 对应图像列表)
    """
    # 如果没有段落，直接返回空列表
    if not sections:
        return [], []

    # 初始化结果列表
    cks = []  # 存储文本块
    images = []  # 存储对应图像
    tk_nums = []  # 存储每个文本块的令牌数量

    """
    将文本和图像添加到适当的块中
    
    参数:
        t: 要添加的文本
        image: 要添加的图像
        pos: 可选的位置标记文本
    """
    def add_chunk(t, image, pos=""):
        nonlocal cks, images, tk_nums
        # 计算文本的令牌数量
        tnum = num_tokens_from_string(t)
        # 如果令牌数量太少，忽略位置标记
        if tnum < 8:
            pos = ""

        # 如果这是第一个块或者上一个块的令牌数量已超过限制，则创建新块
        if not cks or tk_nums[-1] > chunk_token_num:
            # new chunk
            # 如果有位置标记且文本中不包含，则添加位置标记
            if pos and t.find(pos) < 0:
                t += pos
            cks.append(t)
            images.append(image)
            tk_nums.append(tnum)
        else:
            # add to last chunk
            # 添加到最后一个块
            # 如果有位置标记且最后一个块中不包含，则添加位置标记
            if pos and cks[-1].find(pos) < 0:
                t += pos
            cks[-1] += t
            # 合并图像
            images[-1] = concat_img(images[-1], image)
            # 更新令牌数量
            tk_nums[-1] += tnum

    # 提取自定义分隔符（用反引号包裹的部分）
    custom_delimiters = [m.group(1) for m in re.finditer(r"`([^`]+)`", delimiter)]
    has_custom = bool(custom_delimiters)
    # 如果有自定义分隔符，使用它们进行文本分割
    if has_custom:
        # 构建自定义分隔符的正则表达式模式，按长度降序排序避免子模式匹配问题
        custom_pattern = "|".join(re.escape(t) for t in sorted(set(custom_delimiters), key=len, reverse=True))
        # 重置结果列表
        cks, images, tk_nums = [], [], []
        pattern = r"(%s)" % custom_pattern
        # 遍历每个段落和图像
        for sec, image in sections:
            # 使用自定义分隔符分割文本
            split_sec = re.split(pattern, sec)
            for sub_sec in split_sec:
                # 跳过空文本或仅包含分隔符的文本
                if not sub_sec or re.fullmatch(custom_pattern, sub_sec):
                    continue
                text_seg = "\n" + sub_sec
                cks.append(text_seg)
                images.append(image)
                tk_nums.append(num_tokens_from_string(text_seg))
        return cks, images

    # 使用默认方式处理：将每个段落作为整体添加
    for sec, image in sections:
        add_chunk("\n" + sec, image, "")

    return cks, images

def extract_between(text: str, start_tag: str, end_tag: str) -> list[str]:
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    return re.findall(pattern, text, flags=re.DOTALL)


def get_delimiters(delimiters: str):
    dels = []
    s = 0
    for m in re.finditer(r"`([^`]+)`", delimiters, re.I):
        f, t = m.span()
        dels.append(m.group(1))
        dels.extend(list(delimiters[s: f]))
        s = t
    if s < len(delimiters):
        dels.extend(list(delimiters[s:]))

    dels.sort(key=lambda x: -len(x))
    dels = [re.escape(d) for d in dels if d]
    dels = [d for d in dels if d]
    dels_pattern = "|".join(dels)

    return dels_pattern


class Node:
    def __init__(self, level, depth=-1, texts=None):
        self.level = level
        self.depth = depth
        self.texts = texts or []
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def get_children(self):
        return self.children

    def get_level(self):
        return self.level

    def get_texts(self):
        return self.texts

    def set_texts(self, texts):
        self.texts = texts

    def add_text(self, text):
        self.texts.append(text)

    def clear_text(self):
        self.texts = []

    def __repr__(self):
        return f"Node(level={self.level}, texts={self.texts}, children={len(self.children)})"

    def build_tree(self, lines):
        stack = [self]
        for level, text in lines:
            if self.depth != -1 and level > self.depth:
                # Beyond target depth: merge content into the current leaf instead of creating deeper nodes
                stack[-1].add_text(text)
                continue

            # Move up until we find the proper parent whose level is strictly smaller than current
            while len(stack) > 1 and level <= stack[-1].get_level():
                stack.pop()

            node = Node(level=level, texts=[text])
            # Attach as child of current parent and descend
            stack[-1].add_child(node)
            stack.append(node)

        return self

    def get_tree(self):
        tree_list = []
        self._dfs(self, tree_list, [])
        return tree_list

    def _dfs(self, node, tree_list, titles):
        level = node.get_level()
        texts = node.get_texts()
        child = node.get_children()

        if level == 0 and texts:
            tree_list.append("\n".join(titles+texts))

        # Titles within configured depth are accumulated into the current path
        if 1 <= level <= self.depth:
            path_titles = titles + texts
        else:
            path_titles = titles

        # Body outside the depth limit becomes its own chunk under the current title path
        if level > self.depth and texts:
            tree_list.append("\n".join(path_titles + texts))

        # A leaf title within depth emits its title path as a chunk (header-only section)
        elif not child and (1 <= level <= self.depth):
            tree_list.append("\n".join(path_titles))

        # Recurse into children with the updated title path
        for c in child:
            self._dfs(c, tree_list, path_titles)