import sys, os; __name__ == "__main__" and sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.file_utils import get_project_base_directory
from common import settings
import nltk
nltk.data.path.insert(0, os.path.join(get_project_base_directory(), "static/nltk_data"))
import logging
import copy
import datrie
import math

import re
import string

from hanziconv import HanziConv

from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer



class RagTokenizer:
    def key_(self, line):
        # 将输入字符串转为小写，使用UTF-8编码为字节序列，
        # 然后将字节序列转为字符串表示并去掉开头的"b'"和结尾的"'"
        # 用于在字典树(Trie)中作为统一的键值格式存储和查找词汇
        return str(line.lower().encode("utf-8"))[2:-1]

    def rkey_(self, line):
        return str(("DD" + (line[::-1].lower())).encode("utf-8"))[2:-1]

    def _load_dict(self, fnm):
        """从词典文件加载词汇数据并构建字典树
        
        参数:
            fnm: 词典文件路径
        
        功能:
            1. 读取词典文件中的词汇数据
            2. 计算词汇权重并构建字典树
            3. 为反向最大匹配准备反向词汇索引
            4. 将构建好的字典树保存为缓存文件
        """
        # 记录日志：开始从指定文件构建字典树
        logging.info(f"[HUQIE]:Build trie from {fnm}")
        
        try:
            # 打开词典文件，使用UTF-8编码
            of = open(fnm, "r", encoding="utf-8")
            
            # 逐行读取词典文件
            while True:
                line = of.readline()
                # 文件读取完毕，退出循环
                if not line:
                    break
                    
                # 数据预处理：
                # 1. 移除换行符
                line = re.sub(r"[\r\n]+", "", line)
                # 2. 按空格或制表符分割行数据
                #    每行格式应为：词汇 频率 词性/标签
                line = re.split(r"[ \t]", line)
                
                # 获取词汇的键值（经过key_方法处理，转为小写并编码）
                k = self.key_(line[0])
                
                # 计算词汇权重：
                # 使用对数转换处理频率值，使权重分布更加合理
                # 加0.5是为了四舍五入取整
                F = int(math.log(float(line[1]) / self.DENOMINATOR) + 0.5)
                # 只有当词汇不存在于字典树中，或者当前词汇权重更高时，才更新字典树
                if k not in self.trie_ or self.trie_[k][0] < F:
                    # 在字典树中存储词汇的权重和标签信息
                    self.trie_[self.key_(line[0])] = (F, line[2])
                
                # 为反向最大匹配算法准备反向词汇索引
                # 这里只存储标记值1，表示该反向词汇存在
                self.trie_[self.rkey_(line[0])] = 1
        
            # 构建字典树缓存文件名
            dict_file_cache = fnm + ".trie"
            # 记录日志：保存字典树缓存
            logging.info(f"[HUQIE]:Build trie cache to {dict_file_cache}")
            # 保存字典树到缓存文件，以便下次快速加载
            self.trie_.save(dict_file_cache)
            # 关闭文件
            of.close()
            
        except Exception:
            # 捕获所有异常并记录日志
            logging.exception(f"[HUQIE]:Build trie {fnm} failed")

    def __init__(self, debug=False):
        """初始化分词器实例
        
        参数:
            debug: 布尔值，是否启用调试模式，默认为False
        
        功能:
            1. 设置基本参数和调试标志
            2. 初始化词干提取器和词形还原器
            3. 定义文本分割正则表达式
            4. 加载或构建字典树(Trie)结构
        """
        # 设置调试标志
        self.DEBUG = debug
        # 频率计算的分母值，用于词汇频率的对数转换
        self.DENOMINATOR = 1000000
        # 字典文件所在目录路径
        self.DIR_ = os.path.join(get_project_base_directory(), "rag/res", "huqie")
        
        # 初始化英文词干提取器（Porter词干算法）比如happiness → happi workers → worker 更考虑语义
        self.stemmer = PorterStemmer()
        # 初始化英文词形还原器（WordNet词形还原） 比如running → run 确保结果是语言中实际存在的合法单词，而非机械切割后的片段。
        self.lemmatizer = WordNetLemmatizer()
        
        # 文本分割正则表达式，用于：
        # 1. 匹配各种标点符号、空白字符（中文和英文）
        # 2. 匹配英文单词、数字和相关标点组合
        self.SPLIT_CHAR = r"([ ,\.<>/?;:'\[\]\\`!@#$%^&*\(\)\{\}\|_+=《》，。？、；‘’：“”【】~！￥%……（）——-]+|[a-zA-Z0-9,\.-]+)"
        # 构建字典树文件名路径
        trie_file_name = self.DIR_ + ".txt.trie"
        # 检查字典树缓存文件是否存在
        if os.path.exists(trie_file_name):
            try:
                # 尝试从缓存文件加载字典树
                self.trie_ = datrie.Trie.load(trie_file_name)
                return
            except Exception:
                # 加载失败时记录异常并创建新的字典树
                logging.exception(f"[HUQIE]:Fail to load trie file {trie_file_name}, build the default trie file")
                self.trie_ = datrie.Trie(string.printable)
        else:
            # 缓存文件不存在时，记录信息并创建新的字典树
            logging.info(f"[HUQIE]:Trie file {trie_file_name} not found, build the default trie file")
            self.trie_ = datrie.Trie(string.printable)
        # 从字典文本文件加载数据并构建字典树
        self._load_dict(self.DIR_ + ".txt")

    def load_user_dict(self, fnm):
        try:
            self.trie_ = datrie.Trie.load(fnm + ".trie")
            return
        except Exception:
            self.trie_ = datrie.Trie(string.printable)
        self._load_dict(fnm)

    def add_user_dict(self, fnm):
        self._load_dict(fnm)

    def _strQ2B(self, ustring):
        """Convert full-width characters to half-width characters"""
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xFEE0
            if inside_code < 0x0020 or inside_code > 0x7E:  # After the conversion, if it's not a half-width character, return the original character.
                rstring += uchar
            else:
                rstring += chr(inside_code)
        return rstring

    def _tradi2simp(self, line):
        return HanziConv.toSimplified(line)

    def dfs_(self, chars, s, preTks, tkslist, _depth=0, _memo=None):
        if _memo is None:
            _memo = {}
        MAX_DEPTH = 10
        if _depth > MAX_DEPTH:
            if s < len(chars):
                copy_pretks = copy.deepcopy(preTks)
                remaining = "".join(chars[s:])
                copy_pretks.append((remaining, (-12, "")))
                tkslist.append(copy_pretks)
            return s

        state_key = (s, tuple(tk[0] for tk in preTks)) if preTks else (s, None)
        if state_key in _memo:
            return _memo[state_key]

        res = s
        if s >= len(chars):
            tkslist.append(preTks)
            _memo[state_key] = s
            return s
        if s < len(chars) - 4:
            is_repetitive = True
            char_to_check = chars[s]
            for i in range(1, 5):
                if s + i >= len(chars) or chars[s + i] != char_to_check:
                    is_repetitive = False
                    break
            if is_repetitive:
                end = s
                while end < len(chars) and chars[end] == char_to_check:
                    end += 1
                mid = s + min(10, end - s)
                t = "".join(chars[s:mid])
                k = self.key_(t)
                copy_pretks = copy.deepcopy(preTks)
                if k in self.trie_:
                    copy_pretks.append((t, self.trie_[k]))
                else:
                    copy_pretks.append((t, (-12, "")))
                next_res = self.dfs_(chars, mid, copy_pretks, tkslist, _depth + 1, _memo)
                res = max(res, next_res)
                _memo[state_key] = res
                return res

        S = s + 1
        if s + 2 <= len(chars):
            t1 = "".join(chars[s : s + 1])
            t2 = "".join(chars[s : s + 2])
            if self.trie_.has_keys_with_prefix(self.key_(t1)) and not self.trie_.has_keys_with_prefix(self.key_(t2)):
                S = s + 2
        if len(preTks) > 2 and len(preTks[-1][0]) == 1 and len(preTks[-2][0]) == 1 and len(preTks[-3][0]) == 1:
            t1 = preTks[-1][0] + "".join(chars[s : s + 1])
            if self.trie_.has_keys_with_prefix(self.key_(t1)):
                S = s + 2

        for e in range(S, len(chars) + 1):
            t = "".join(chars[s:e])
            k = self.key_(t)
            if e > s + 1 and not self.trie_.has_keys_with_prefix(k):
                break
            if k in self.trie_:
                pretks = copy.deepcopy(preTks)
                pretks.append((t, self.trie_[k]))
                res = max(res, self.dfs_(chars, e, pretks, tkslist, _depth + 1, _memo))

        if res > s:
            _memo[state_key] = res
            return res

        t = "".join(chars[s : s + 1])
        k = self.key_(t)
        copy_pretks = copy.deepcopy(preTks)
        if k in self.trie_:
            copy_pretks.append((t, self.trie_[k]))
        else:
            copy_pretks.append((t, (-12, "")))
        result = self.dfs_(chars, s + 1, copy_pretks, tkslist, _depth + 1, _memo)
        _memo[state_key] = result
        return result

    def freq(self, tk):
        k = self.key_(tk)
        if k not in self.trie_:
            return 0
        return int(math.exp(self.trie_[k][0]) * self.DENOMINATOR + 0.5)

    def tag(self, tk):
        k = self.key_(tk)
        if k not in self.trie_:
            return ""
        return self.trie_[k][1]

    def score_(self, tfts):
        """计算分词结果的评分
        参数:
            tfts: list - 分词结果列表，每个元素是(词汇, (频率, 标签))的元组
        返回:
            tuple - (分词结果列表, 评分值)
        评分算法说明:
            综合考虑了以下三个因素来评估分词质量：
            1. 词汇频率权重(F) - 词典中词汇的频率值之和
            2. 长词汇比例(L) - 长度≥2的词汇占总词汇数的比例
            3. 词汇数量影响(B/len(tks)) - 词汇数量越少得分越高，系数B为常数30
            最终得分计算公式: B/len(tks) + L + F
        """
        # 常数B，用于计算词汇数量对得分的影响权重
        B = 30
        # 初始化变量：
        # F: 词汇频率总和
        # L: 长词汇(长度≥2)数量
        # tks: 分词结果列表
        F, L, tks = 0, 0, []
        
        # 遍历所有分词结果
        for tk, (freq, tag) in tfts:
            # 累加词汇频率权重
            F += freq
            # 计算长词汇数量（长度≥2的词汇）
            L += 0 if len(tk) < 2 else 1
            # 收集分词结果
            tks.append(tk)
        
        # 注释掉的代码：计算平均频率（可能在算法优化过程中被弃用）
        # F /= len(tks)
        
        # 计算长词汇比例：长词汇数量 / 总词汇数
        L /= len(tks)
        
        # 调试日志：记录分词结果、词汇数量、长词汇比例、频率总和和最终得分
        logging.debug("[SC] {} {} {} {} {}".format(tks, len(tks), L, F, B / len(tks) + L + F))
        
        # 返回分词结果列表和计算得到的评分
        return tks, B / len(tks) + L + F

    def _sort_tokens(self, tkslist):
        res = []
        for tfts in tkslist:
            tks, s = self.score_(tfts)
            res.append((tks, s))
        return sorted(res, key=lambda x: x[1], reverse=True)

    def merge_(self, tks):
        # if split chars is part of token
        res = []
        tks = re.sub(r"[ ]+", " ", tks).split()
        s = 0
        while True:
            if s >= len(tks):
                break
            E = s + 1
            for e in range(s + 2, min(len(tks) + 2, s + 6)):
                tk = "".join(tks[s:e])
                if re.search(self.SPLIT_CHAR, tk) and self.freq(tk):
                    E = e
            res.append("".join(tks[s:E]))
            s = E

        return " ".join(res)

    def _max_forward(self, line):
        """正向最大匹配分词算法实现
        
        参数:
            line: str - 待分词的文本行
        
        返回:
            tuple - 包含分词结果和得分的元组，具体格式取决于score_方法的实现
        
        算法流程:
            1. 从文本开头开始，尝试匹配最长的词汇
            2. 利用Trie树(trie_)进行前缀匹配和完整词汇验证
            3. 记录每个匹配到的词汇及其在字典中的权重信息
            4. 最后通过score_方法对分词结果进行评分并返回
        """
        # 存储分词结果的列表
        res = []
        # 文本处理起始位置指针
        s = 0
        
        # 遍历整个文本
        while s < len(line):
            # 初始化为当前字符
            e = s + 1
            t = line[s:e]
            
            # 第一阶段：最大正向匹配 - 尽可能匹配最长的前缀
            while e < len(line) and self.trie_.has_keys_with_prefix(self.key_(t)):
                e += 1
                t = line[s:e]
            
            # 第二阶段：回溯验证 - 确保匹配到的是完整词汇而非仅前缀
            while e - 1 > s and self.key_(t) not in self.trie_:
                e -= 1
                t = line[s:e]
            
            # 将匹配到的词汇及其权重信息添加到结果列表
            # 如果词汇存在于词典中，记录其权重和标签
            # 否则，权重设为0，标签为空字符串
            if self.key_(t) in self.trie_:
                res.append((t, self.trie_[self.key_(t)]))
            else:
                res.append((t, (0, "")))
            
            # 更新起始位置，继续处理剩余文本
            s = e
        
        # 将分词结果传入评分函数进行评分并返回
        return self.score_(res)

    def _max_backward(self, line):
        res = []
        s = len(line) - 1
        while s >= 0:
            e = s + 1
            t = line[s:e]
            while s > 0 and self.trie_.has_keys_with_prefix(self.rkey_(t)):
                s -= 1
                t = line[s:e]

            while s + 1 < e and self.key_(t) not in self.trie_:
                s += 1
                t = line[s:e]

            if self.key_(t) in self.trie_:
                res.append((t, self.trie_[self.key_(t)]))
            else:
                res.append((t, (0, "")))

            s -= 1

        return self.score_(res[::-1])

    def english_normalize_(self, tks):
        return [self.stemmer.stem(self.lemmatizer.lemmatize(t)) if re.match(r"[a-zA-Z_-]+$", t) else t for t in tks]

    def _split_by_lang(self, line):
        """
        根据语言类型分割文本，将文本分成连续的中文片段和非中文片段
        
        参数:
            line: str - 要分割的原始文本
            
        返回:
            list - 包含(文本片段, 语言标识)元组的列表，其中语言标识为True表示中文，False表示非中文
        """
        # 初始化文本-语言配对列表
        txt_lang_pairs = []
        
        # 使用分割字符将文本分成多个部分
        arr = re.split(self.SPLIT_CHAR, line)
        
        # 遍历每个分割后的部分
        for a in arr:
            # 跳过空字符串
            if not a:
                continue
                
            # 初始化开始索引和结束索引
            s = 0
            e = s + 1
            
            # 确定起始字符的语言类型（是否为中文）
            zh = is_chinese(a[s])
            
            # 遍历当前文本段，寻找语言变化点
            while e < len(a):
                # 检查当前结束位置字符的语言类型
                _zh = is_chinese(a[e])
                
                # 如果语言类型相同，继续向后检查
                if _zh == zh:
                    e += 1
                    continue
                    
                # 当检测到语言变化时，记录之前的连续同语言片段
                txt_lang_pairs.append((a[s:e], zh))
                
                # 更新起始位置到语言变化点，重置结束位置
                s = e
                e = s + 1
                # 更新当前处理的语言类型
                zh = _zh
            
            # 确保处理完最后一个片段
            if s >= len(a):
                continue
                
            # 添加最后一个语言片段
            txt_lang_pairs.append((a[s:e], zh))
            
        # 返回所有文本-语言配对
        return txt_lang_pairs

    def tokenize(self, line: str) -> str:
        """文本分词处理函数
        
        参数:
            line: 输入文本字符串
        返回:
            分词后的文本字符串，词之间用空格分隔
        """
        # 如果使用INFINITY文档引擎，则直接返回原始文本
        if settings.DOC_ENGINE_INFINITY:
            return line
        
        # 文本预处理：
        # 1. 将所有非单词字符替换为空格
        line = re.sub(r"\W+", " ", line)
        # 2. 全角字符转半角字符，并转为小写
        line = self._strQ2B(line).lower()
        # 3. 繁体中文转为简体中文
        line = self._tradi2simp(line)
    
        # 根据语言类型（中文/非中文）分割文本 连续的中文会被分成一段
        arr = self._split_by_lang(line)
        res = []
        
        # 对每个语言片段进行处理 L是英文单词或一段中文
        for L, lang in arr:
            # 如果是非中文文本（lang=False）
            if not lang:
                # 使用NLTK的word_tokenize进行分词，然后进行词干提取和词形还原
                res.extend([self.stemmer.stem(self.lemmatizer.lemmatize(t)) for t in word_tokenize(L)])
                continue
                
            # 对于中文文本，如果长度小于2或者是纯字母/数字，则直接添加到结果中
            if len(L) < 2 or re.match(r"[a-z\.-]+\$", L) or re.match(r"[0-9\.-]+\$", L):
                res.append(L)
                continue
    
            # 双向最大匹配算法：
            # 1. 首先使用正向最大匹配
            tks, s = self._max_forward(L)
            # 2. 然后使用反向最大匹配
            tks1, s1 = self._max_backward(L)
            
            # 调试模式下记录两种匹配的结果
            if self.DEBUG:
                logging.debug("[FW] {} {}".format(tks, s))
                logging.debug("[BW] {} {}".format(tks1, s1))
    
            # 初始化指针，用于合并两种匹配结果
            i, j, _i, _j = 0, 0, 0, 0  # i,j指向当前处理位置，_i,_j指向上一个相同片段的结束位置
            same = 0  # 记录连续相同的词数
            
            # 找出从当前位置开始的连续相同词
            while i + same < len(tks1) and j + same < len(tks) and tks1[i + same] == tks[j + same]:
                same += 1
            
            # 如果找到相同词，添加到结果中
            if same > 0:
                res.append(" ".join(tks[j : j + same]))
            
            # 更新指针位置
            _i = i + same
            _j = j + same
            j = _j + 1
            i = _i + 1
    
            # 处理两种分词结果中不同的部分
            while i < len(tks1) and j < len(tks):
                # 获取当前比较的文本片段
                tk1, tk = "".join(tks1[_i:i]), "".join(tks[_j:j])
                
                # 如果片段不匹配，移动指针继续比较
                if tk1 != tk:
                    if len(tk1) > len(tk):
                        j += 1
                    else:
                        i += 1
                    continue
    
                # 如果当前词不匹配，移动指针继续比较
                if tks1[i] != tks[j]:
                    i += 1
                    j += 1
                    continue
                    
                # 当发现两种分词结果在某一段落结束点一致但分词方式不同时
                # 使用深度优先搜索(DFS)找出最优分词方案
                tkslist = []
                self.dfs_("".join(tks[_j:j]), 0, [], tkslist)
                # 选择得分最高的分词方案并添加到结果中
                res.append(" ".join(self._sort_tokens(tkslist)[0][0]))
    
                # 继续寻找下一段连续相同的词
                same = 1
                while i + same < len(tks1) and j + same < len(tks) and tks1[i + same] == tks[j + same]:
                    same += 1
                    
                # 添加连续相同的词到结果中
                res.append(" ".join(tks[j : j + same]))
                # 更新指针位置
                _i = i + same
                _j = j + same
                j = _j + 1
                i = _i + 1
    
            # 处理剩余的文本部分
            if _i < len(tks1):
                # 确保两种分词结果的剩余部分文本相同
                assert _j < len(tks)
                assert "".join(tks1[_i:]) == "".join(tks[_j:])
                
                # 对剩余部分使用DFS找出最优分词方案
                tkslist = []
                self.dfs_("".join(tks[_j:]), 0, [], tkslist)
                # 选择得分最高的分词方案并添加到结果中
                res.append(" ".join(self._sort_tokens(tkslist)[0][0]))
    
        # 合并所有分词结果
        res = " ".join(res)
        # 调试模式下记录最终合并后的结果
        logging.debug("[TKS] {}".format(self.merge_(res)))
        # 使用merge_方法进行最终处理并返回
        return self.merge_(res)

    def fine_grained_tokenize(self, tks: str) -> str:
        """
        对输入文本进行细粒度分词处理
        该方法根据文本的语言特性（中文/英文比例）选择不同的分词策略，
        对于含有较多中文的文本，使用深度优先搜索进行更细粒度的分词处理。
        参数:
            tks: str - 待分词的文本字符串
        返回:
            str - 分词后的文本字符串，单词之间用空格分隔
        """
        # 特殊情况处理：如果使用Infinity文档引擎，则直接返回原始文本
        if settings.DOC_ENGINE_INFINITY:
            return tks
        
        # 将输入文本按空格分割成单词列表
        tks = tks.split()
        
        # 统计以中文字符开头的单词数量
        zh_num = len([1 for c in tks if c and is_chinese(c[0])])
        
        # 如果中文单词比例小于20%，认为是英文文本，只进行简单的斜杠分割
        if zh_num < len(tks) * 0.2:
            res = []
            for tk in tks:
                res.extend(tk.split("/"))
            return " ".join(res)

        # 中文比例较高的文本处理
        res = []
        for tk in tks:
            # 跳过短词（长度小于3）或纯数字词（包含数字、逗号、小数点、连字符）
            if len(tk) < 3 or re.match(r"[0-9,\.-]+$", tk):
                res.append(tk)
                continue
            
            # 存储分词候选结果
            tkslist = []
            # 对于过长的词（长度超过10），不再进行分词，直接保留
            if len(tk) > 10:
                tkslist.append(tk)
            else:
                # 使用深度优先搜索进行分词，获取所有可能的分词结果
                self.dfs_(tk, 0, [], tkslist)
            
            # 如果只有一种分词结果，直接保留原词
            if len(tkslist) < 2:
                res.append(tk)
                continue
            
            # 对分词结果按评分排序，取第二好的结果（索引为1）
            # 这里不取最高分结果可能是为了避免过度分词
            stk = self._sort_tokens(tkslist)[1][0]
            
            # 如果分词结果长度与原词相同，说明没有有效分词，保留原词
            if len(stk) == len(tk):
                stk = tk
            else:
                # 对于纯英文词（包含小写字母、小数点、连字符）
                if re.match(r"[a-z\.-]+$", tk):
                    # 检查分词后的每个子词长度，如果有小于3的，则保留原词
                    for t in stk:
                        if len(t) < 3:
                            stk = tk
                            break
                    else:
                        # 所有子词长度都不小于3，将子词用空格连接
                        stk = " ".join(stk)
                else:
                    # 非纯英文词，直接将子词用空格连接
                    stk = " ".join(stk)

            res.append(stk)

        # 对结果列表中的英文词进行词形规范化（词干提取和词性还原）
        return " ".join(self.english_normalize_(res))

def is_chinese(s):
    if s >= "\u4e00" and s <= "\u9fa5":
        return True
    else:
        return False


def is_number(s):
    if s >= "\u0030" and s <= "\u0039":
        return True
    else:
        return False


def is_alphabet(s):
    if ("\u0041" <= s <= "\u005a") or ("\u0061" <= s <= "\u007a"):
        return True
    else:
        return False


def naive_qie(txt):
    tks = []
    for t in txt.split():
        if tks and re.match(r".*[a-zA-Z]$", tks[-1]) and re.match(r".*[a-zA-Z]$", t):
            tks.append(" ")
        tks.append(t)
    return tks


tokenizer = RagTokenizer()
tokenize = tokenizer.tokenize
fine_grained_tokenize = tokenizer.fine_grained_tokenize
tag = tokenizer.tag
freq = tokenizer.freq
load_user_dict = tokenizer.load_user_dict
add_user_dict = tokenizer.add_user_dict
tradi2simp = tokenizer._tradi2simp
strQ2B = tokenizer._strQ2B

if __name__ == "__main__":
    tknzr = RagTokenizer(debug=True)
    # huqie.add_user_dict("/tmp/tmp.new.tks.dict")
    texts = [
        "ISIM-chat这个系统如何",
        "哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈",
        "公开征求意见稿提出，境外投资者可使用自有人民币或外汇投资。使用外汇投资的，可通过债券持有人在香港人民币业务清算行及香港地区经批准可进入境内银行间外汇市场进行交易的境外人民币业务参加行（以下统称香港结算行）办理外汇资金兑换。香港结算行由此所产生的头寸可到境内银行间外汇市场平盘。使用外汇投资的，在其投资的债券到期或卖出后，原则上应兑换回外汇。",
        "多校划片就是一个小区对应多个小学初中，让买了学区房的家庭也不确定到底能上哪个学校。目的是通过这种方式为学区房降温，把就近入学落到实处。南京市长江大桥",
        "实际上当时他们已经将业务中心偏移到安全部门和针对政府企业的部门 Scripts are compiled and cached aaaaaaaaa",
        "虽然我不怎么玩",
        "蓝月亮如何在外资夹击中生存,那是全宇宙最有意思的",
        "涡轮增压发动机num最大功率,不像别的共享买车锁电子化的手段,我们接过来是否有意义,黄黄爱美食,不过，今天阿奇要讲到的这家农贸市场，说实话，还真蛮有特色的！不仅环境好，还打出了",
        "这周日你去吗？这周日你有空吗？",
        "Unity3D开发经验 测试开发工程师 c++双11双11 985 211 ",
        "数据分析项目经理|数据分析挖掘|数据分析方向|商品数据分析|搜索数据分析 sql python hive tableau Cocos2d-",
    ]
    for text in texts:
        print('='*50)
        print('原文:',text)
        tks1 = tknzr.tokenize(text)
        tks2 = tknzr.fine_grained_tokenize(tks1)
        print('初步分词结果:',tks1)
        print('细粒度分词结果:',tks2)
    if len(sys.argv) < 2:
        sys.exit()
    tknzr.load_user_dict(sys.argv[1])
    of = open(sys.argv[2], "r")
    while True:
        line = of.readline()
        if not line:
            break
        print(tknzr.tokenize(line))
    of.close()