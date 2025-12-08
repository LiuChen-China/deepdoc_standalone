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
import math
import json
import re
import os
import numpy as np
from rag.nlp import rag_tokenizer
from common.file_utils import get_project_base_directory


class Dealer:
    def __init__(self):
        self.stop_words = set(["请问",
                               "您",
                               "你",
                               "我",
                               "他",
                               "是",
                               "的",
                               "就",
                               "有",
                               "于",
                               "及",
                               "即",
                               "在",
                               "为",
                               "最",
                               "有",
                               "从",
                               "以",
                               "了",
                               "将",
                               "与",
                               "吗",
                               "吧",
                               "中",
                               "#",
                               "什么",
                               "怎么",
                               "哪个",
                               "哪些",
                               "啥",
                               "相关"])

        def load_dict(fnm):
            res = {}
            f = open(fnm, "r")
            while True:
                line = f.readline()
                if not line:
                    break
                arr = line.replace("\n", "").split("\t")
                if len(arr) < 2:
                    res[arr[0]] = 0
                else:
                    res[arr[0]] = int(arr[1])

            c = 0
            for _, v in res.items():
                c += v
            if c == 0:
                return set(res.keys())
            return res

        fnm = os.path.join(get_project_base_directory(), "rag/res")
        self.ne, self.df = {}, {}
        try:
            self.ne = json.load(open(os.path.join(fnm, "ner.json"), "r",encoding="utf-8"))
        except Exception as e:
            logging.warning("Load ner.json FAIL!")
        try:
            self.df = load_dict(os.path.join(fnm, "term.freq"))
        except Exception:
            logging.warning("Load term.freq FAIL!")

    def pretoken(self, txt, num=False, stpwd=True):
        patt = [
            r"[~—\t @#%!<>,\.\?\":;'\{\}\[\]_=\(\)\|，。？》•●○↓《；‘’：“”【¥ 】…￥！、·（）×`&\\/「」\\]"
        ]
        rewt = [
        ]
        for p, r in rewt:
            txt = re.sub(p, r, txt)

        res = []
        for t in rag_tokenizer.tokenize(txt).split():
            tk = t
            if (stpwd and tk in self.stop_words) or (
                    re.match(r"[0-9]$", tk) and not num):
                continue
            for p in patt:
                if re.match(p, t):
                    tk = "#"
                    break
            #tk = re.sub(r"([\+\\-])", r"\\\1", tk)
            if tk != "#" and tk:
                res.append(tk)
        return res

    def token_merge(self, tks):
        def one_term(t): return len(t) == 1 or re.match(r"[0-9a-z]{1,2}$", t)

        res, i = [], 0
        while i < len(tks):
            j = i
            if i == 0 and one_term(tks[i]) and len(
                    tks) > 1 and (len(tks[i + 1]) > 1 and not re.match(r"[0-9a-zA-Z]", tks[i + 1])):  # 多 工位
                res.append(" ".join(tks[0:2]))
                i = 2
                continue

            while j < len(
                    tks) and tks[j] and tks[j] not in self.stop_words and one_term(tks[j]):
                j += 1
            if j - i > 1:
                if j - i < 5:
                    res.append(" ".join(tks[i:j]))
                    i = j
                else:
                    res.append(" ".join(tks[i:i + 2]))
                    i = i + 2
            else:
                if len(tks[i]) > 0:
                    res.append(tks[i])
                i += 1
        return [t for t in res if t]

    def ner(self, t):
        if not self.ne:
            return ""
        res = self.ne.get(t, "")
        if res:
            return res

    def split(self, txt):
        tks = []
        for t in re.sub(r"[ \t]+", " ", txt).split():
            if tks and re.match(r".*[a-zA-Z]$", tks[-1]) and \
               re.match(r".*[a-zA-Z]$", t) and tks and \
               self.ne.get(t, "") != "func" and self.ne.get(tks[-1], "") != "func":
                tks[-1] = tks[-1] + " " + t
            else:
                tks.append(t)
        return tks

    def weights(self, tks, preprocess=True):
        """
        计算输入tokens的权重，结合多种特征（命名实体、词性、词频、文档频率等）
        
        参数:
            tks: 输入的token列表
            preprocess: 是否需要对输入tokens进行预处理，默认为True
                       - True: 对每个token进行预处理(pretoken)和合并(token_merge)
                       - False: 直接使用原始tokens计算权重
        
        返回:
            列表，每个元素为(token, weight)元组，权重已归一化
        """
        # 定义各种正则表达式模式用于识别不同类型的文本
        num_pattern = re.compile(r"[0-9,.]{2,}$")  # 匹配数字序列
        short_letter_pattern = re.compile(r"[a-z]{1,2}$")  # 匹配短字母序列(1-2个字符)
        num_space_pattern = re.compile(r"[0-9. -]{2,}$")  # 匹配数字和空格混合序列
        letter_pattern = re.compile(r"[a-z. -]+$")  # 匹配字母和空格混合序列

        def ner(t):
            """
            根据命名实体(Named Entity)类型计算权重
            
            参数:
                t: 待评估的token
            
            返回:
                权重值，基于token的实体类型
            """
            # 数字类型权重设为2
            if num_pattern.match(t):
                return 2
            # 短字母序列权重设为0.01(降低重要性)
            if short_letter_pattern.match(t):
                return 0.01
            # 如果没有命名实体信息或token不在实体词典中，返回默认权重1
            if not self.ne or t not in self.ne:
                return 1
            # 不同实体类型的权重映射
            m = {"toxic": 2, "func": 1, "corp": 3, "loca": 3, "sch": 3, "stock": 3,
                 "firstnm": 1}
            return m[self.ne[t]]

        def postag(t):
            """
            根据词性标签(Part-of-speech)计算权重
            
            参数:
                t: 待评估的token
            
            返回:
                权重值，基于token的词性
            """
            t = rag_tokenizer.tag(t)  # 获取token的词性标签
            # 代词、连词、副词权重较低
            if t in set(["r", "c", "d"]):
                return 0.3
            # 地名、机构名权重较高
            if t in set(["ns", "nt"]):
                return 3
            # 名词权重中等
            if t in set(["n"]):
                return 2
            # 数字序列权重较高
            if re.match(r"[0-9-]+", t):
                return 2
            # 默认权重
            return 1

        def freq(t):
            """
            计算词频(term frequency)
            
            参数:
                t: 待评估的token
            
            返回:
                词频值，确保最小为10
            """
            # 数字和空格混合序列返回3
            if num_space_pattern.match(t):
                return 3
            # 获取token的词频
            s = rag_tokenizer.freq(t)
            # 没有词频信息且是字母序列，返回300
            if not s and letter_pattern.match(t):
                return 300
            # 没有词频信息，设为0
            if not s:
                s = 0

            # 对于长token(>=4个字符)且没有词频信息，尝试细粒度分词
            if not s and len(t) >= 4:
                # 细粒度分词并过滤掉短token(<=1个字符)
                s = [tt for tt in rag_tokenizer.fine_grained_tokenize(t).split() if len(tt) > 1]
                if len(s) > 1:
                    # 取子token频率的最小值并除以6
                    s = np.min([freq(tt) for tt in s]) / 6.
                else:
                    s = 0

            # 确保词频最小为10
            return max(s, 10)

        def df(t):
            """
            计算文档频率(document frequency)
            
            参数:
                t: 待评估的token
            
            返回:
                文档频率值
            """
            # 数字和空格混合序列返回5
            if num_space_pattern.match(t):
                return 5
            # 如果token在文档频率词典中，返回文档频率加3
            if t in self.df:
                return self.df[t] + 3
            # 字母序列返回300
            elif letter_pattern.match(t):
                return 300
            # 对于长token(>=4个字符)，尝试细粒度分词
            elif len(t) >= 4:
                # 细粒度分词并过滤掉短token(<=1个字符)
                s = [tt for tt in rag_tokenizer.fine_grained_tokenize(t).split() if len(tt) > 1]
                if len(s) > 1:
                    # 取子token文档频率的最小值除以6，且不小于3
                    return max(3, np.min([df(tt) for tt in s]) / 6.)

            # 默认返回3
            return 3

        # 计算逆文档频率(Inverse Document Frequency)
        def idf(s, N): return math.log10(10 + ((N - s + 0.5) / (s + 0.5)))

        tw = []  # 存储(token, weight)对的列表
        if not preprocess:
            # 不进行预处理，直接计算权重
            idf1 = np.array([idf(freq(t), 10000000) for t in tks])  # 使用词频计算IDF
            idf2 = np.array([idf(df(t), 1000000000) for t in tks])  # 使用文档频率计算IDF
            # 加权组合：30%词频IDF + 70%文档频率IDF，再乘以命名实体和词性权重
            wts = (0.3 * idf1 + 0.7 * idf2) * \
                np.array([ner(t) * postag(t) for t in tks])
            wts = [s for s in wts]
            tw = list(zip(tks, wts))
        else:
            # 进行预处理，对每个token单独处理
            for tk in tks:
                # 预处理并合并token
                tt = self.token_merge(self.pretoken(tk, True))
                # 计算预处理后每个token的权重
                idf1 = np.array([idf(freq(t), 10000000) for t in tt])
                idf2 = np.array([idf(df(t), 1000000000) for t in tt])
                wts = (0.3 * idf1 + 0.7 * idf2) * \
                    np.array([ner(t) * postag(t) for t in tt])
                wts = [s for s in wts]
                # 将结果添加到列表中
                tw.extend(zip(tt, wts))

        # 计算所有权重的总和
        S = np.sum([s for _, s in tw])
        # 返回归一化后的(token, weight)列表
        return [(t, s / S) for t, s in tw]