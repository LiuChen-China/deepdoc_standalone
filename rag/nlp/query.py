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
import sys, os; __name__ == "__main__" and sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import logging
import json
import re
from collections import defaultdict
from abc import ABC
try:
    from rag.nlp import rag_tokenizer, term_weight, synonym
except:
    from deepdoc_standalone.rag.nlp import rag_tokenizer, term_weight, synonym

class MatchTextExpr(ABC):
    def __init__(
        self,
        fields: list[str],
        matching_text: str,
        topn: int,
        extra_options: dict = dict(),
    ):
        self.fields = fields
        self.matching_text = matching_text
        self.topn = topn
        self.extra_options = extra_options



class FulltextQueryer:
    def __init__(self):
        self.tw = term_weight.Dealer()
        self.syn = synonym.Dealer()
        self.query_fields = [
            "title_tks^10",
            "title_sm_tks^5",
            "important_kwd^30",
            "important_tks^20",
            "question_tks^20",
            "content_ltks^2",
            "content_sm_ltks",
        ]

    @staticmethod
    def sub_special_char(line):
        return re.sub(r"([:\{\}/\[\]\-\*\"\(\)\|\+~\^])", r"\\\1", line).strip()

    @staticmethod
    def is_chinese(line):
        arr = re.split(r"[ \t]+", line)
        if len(arr) <= 3:
            return True
        e = 0
        for t in arr:
            if not re.match(r"[a-zA-Z]+$", t):
                e += 1
        return e * 1.0 / len(arr) >= 0.7

    @staticmethod
    def rmWWW(txt):
        patts = [
            (
                r"是*(怎么办|什么样的|哪家|一下|那家|请问|啥样|咋样了|什么时候|何时|何地|何人|是否|是不是|多少|哪里|怎么|哪儿|怎么样|如何|哪些|是啥|啥是|啊|吗|呢|吧|咋|什么|有没有|呀|谁|哪位|哪个)是*",
                "",
            ),
            (r"(^| )(what|who|how|which|where|why)('re|'s)? ", " "),
            (
                r"(^| )('s|'re|is|are|were|was|do|does|did|don't|doesn't|didn't|has|have|be|there|you|me|your|my|mine|just|please|may|i|should|would|wouldn't|will|won't|done|go|for|with|so|the|a|an|by|i'm|it's|he's|she's|they|they're|you're|as|by|on|in|at|up|out|down|of|to|or|and|if) ",
                " ")
        ]
        otxt = txt
        for r, p in patts:
            txt = re.sub(r, p, txt, flags=re.IGNORECASE)
        if not txt:
            txt = otxt
        return txt

    @staticmethod
    def add_space_between_eng_zh(txt):
        # (ENG/ENG+NUM) + ZH
        txt = re.sub(r'([A-Za-z]+[0-9]+)([\u4e00-\u9fa5]+)', r'\1 \2', txt)
        # ENG + ZH
        txt = re.sub(r'([A-Za-z])([\u4e00-\u9fa5]+)', r'\1 \2', txt)
        # ZH + (ENG/ENG+NUM)
        txt = re.sub(r'([\u4e00-\u9fa5]+)([A-Za-z]+[0-9]+)', r'\1 \2', txt)
        txt = re.sub(r'([\u4e00-\u9fa5]+)([A-Za-z])', r'\1 \2', txt)
        return txt

    def question(self, txt, tbl="qa", min_match: float = 0.6):
        """
        构建全文搜索查询表达式
        
        Args:
            txt (str): 原始查询文本
            tbl (str): 表名，默认为"qa"
            min_match (float): 最小匹配分数，默认为0.6
            
        Returns:
            tuple: (MatchTextExpr对象, 关键词列表)，失败时返回(None, 关键词列表)
        """
        # 保存原始查询文本
        original_query = txt
        
        # 在中英文之间添加空格
        txt = FulltextQueryer.add_space_between_eng_zh(txt)
        
        # 文本预处理：转换为小写 -> 全角转半角 -> 繁体转简体 -> 替换特殊字符为空格 -> 去除首尾空格
        txt = re.sub(
            r"[ :|\r\n\t,，。？?/`!！&^%%()\[\]{}<>]+",
            " ",
            rag_tokenizer.tradi2simp(rag_tokenizer.strQ2B(txt.lower())),
        ).strip()
        
        # 保存处理后的文本副本
        otxt = txt
        
        # 移除常见的疑问词和停用词
        txt = FulltextQueryer.rmWWW(txt)

        # 处理非中文查询
        if not self.is_chinese(txt):
            # 再次移除疑问词和停用词（可能是冗余操作）
            txt = FulltextQueryer.rmWWW(txt)
            
            # 分词并过滤空值
            tks = rag_tokenizer.tokenize(txt).split()
            keywords = [t for t in tks if t]
            
            # 计算词权重
            tks_w = self.tw.weights(tks, preprocess=False)
            
            # 清理词项：移除特殊字符、单字符数字/字母、符号前缀等
            tks_w = [(re.sub(r"[ \\\"'^]", "", tk), w) for tk, w in tks_w]
            tks_w = [(re.sub(r"^[a-z0-9]$", "", tk), w) for tk, w in tks_w if tk]
            tks_w = [(re.sub(r"^[\+-]", "", tk), w) for tk, w in tks_w if tk]
            tks_w = [(tk.strip(), w) for tk, w in tks_w if tk.strip()]
            
            syns = []
            # 处理前256个词项
            for tk, w in tks_w[:256]:
                # 查找同义词
                syn = self.syn.lookup(tk)
                syn = rag_tokenizer.tokenize(" ".join(syn)).split()
                syn = [word for word in syn if word not in keywords]
                #keywords.extend(syn)
                # 格式化同义词并设置权重（原词权重的1/4）
                syn = ["\"{}\"^{:.4f}".format(s, w / 4.) for s in syn if s.strip()]
                syns.append(" ".join(syn))

            # 构建查询表达式：关键词+同义词组合
            q = ["({}^{:.4f}".format(tk, w) + " {})".format(syn) for (tk, w), syn in zip(tks_w, syns) if
                 tk and not re.match(r"[.^+\(\)-]", tk)]
            
            # 添加相邻词组合，权重设为两者权重最大值的2倍
            for i in range(1, len(tks_w)):
                left, right = tks_w[i - 1][0].strip(), tks_w[i][0].strip()
                if not left or not right:
                    continue
                q.append(
                    '"%s %s"^%.4f'
                    % (
                        tks_w[i - 1][0],
                        tks_w[i][0],
                        max(tks_w[i - 1][1], tks_w[i][1]) * 2,
                    )
                )
            
            # 确保查询不为空
            if not q:
                q.append(txt)
            
            query = " ".join(q)
            return MatchTextExpr(
                self.query_fields, query, 100, {"original_query": original_query}
            ), keywords

        # 判断是否需要细粒度分词的内部函数
        def need_fine_grained_tokenize(tk):
            # 小于3个字符的词不分词
            if len(tk) < 3:
                return False
            # 纯数字字母或特定符号组合的词不分词
            if re.match(r"[0-9a-z\.\+#_\*-]+", tk):
                return False
            return True

        # 再次移除疑问词和停用词
        txt = FulltextQueryer.rmWWW(txt)
        qs, keywords = [], []
        
        # 处理中文查询，最多处理256个词项
        for tt in self.tw.split(txt)[:256]:
            if not tt:
                continue
            
            keywords.append(tt)
            # 计算词权重
            twts = self.tw.weights([tt])
            # 查找同义词
            syns = self.syn.lookup(tt)
            if syns and len(keywords) < 32:
                syns = [word for word in syns if word not in keywords]
                #keywords.extend(syns)
            
            logging.debug(json.dumps(twts, ensure_ascii=False))
            tms = []
            
            # 按权重降序处理每个词项
            for tk, w in sorted(twts, key=lambda x: x[1] * -1):
                # 根据需要进行细粒度分词
                sm = (
                    rag_tokenizer.fine_grained_tokenize(tk).split()
                    if need_fine_grained_tokenize(tk)
                    else []
                )
                
                # 清理分词结果：移除特殊字符、过滤长度<2的词
                sm = [
                    re.sub(
                        r"[ ,\./;'\[\]\\`~!@#$%\^&\*\(\)=\+_<>\?:\"\{\}\|，。；‘’【】、！￥……（）——《》？：“”-]+",
                        "",
                        m,
                    )
                    for m in sm
                ]
                sm = [FulltextQueryer.sub_special_char(m) for m in sm if len(m) > 1]
                sm = [m for m in sm if len(m) > 1]

                # 收集关键词（最多32个）
                if len(keywords) < 32 and (re.sub(r"[ \\\"']+", "", tk) not in keywords):
                    keywords.append(re.sub(r"[ \\\"']+", "", tk))
                    keywords.extend(sm)

                # 处理词项的同义词
                tk_syns = self.syn.lookup(tk)
                tk_syns = [FulltextQueryer.sub_special_char(s) for s in tk_syns]
                if len(keywords) < 32:
                    keywords.extend([s for s in tk_syns if s and s not in keywords])
                tk_syns = [rag_tokenizer.fine_grained_tokenize(s) for s in tk_syns if s]
                tk_syns = [f"\"{s}\"" if s.find(" ") > 0 else s for s in tk_syns]

                # 限制关键词数量
                if len(keywords) >= 32:
                    break

                # 构建查询表达式：处理特殊字符、空格词项加引号、添加同义词和分词结果
                tk = FulltextQueryer.sub_special_char(tk)
                if tk.find(" ") > 0:
                    tk = '"%s"' % tk
                if tk_syns:
                    tk = f"({tk} OR (%s)^0.2)" % " ".join(tk_syns)
                if sm:
                    tk = f'{tk} OR "%s" OR ("%s"~2)^0.5' % (" ".join(sm), " ".join(sm))
                if tk.strip():
                    tms.append((tk, w))

            # 合并带权重的词项查询
            tms = " ".join([f"({t})^{w}" for t, w in tms])

            # 添加相邻词项组合，设置较高权重
            if len(twts) > 1:
                tms += ' ("%s"~2)^1.5' % rag_tokenizer.tokenize(tt)

            # 处理同义词
            syns = " OR ".join(
                [
                    '"%s"'
                    % rag_tokenizer.tokenize(FulltextQueryer.sub_special_char(s))
                    for s in syns
                ]
            )
            if syns and tms:
                # 主查询权重为5，同义词权重为0.7
                tms = f"({tms})^5 OR ({syns})^0.7"

            qs.append(tms)

        # 构建最终查询表达式
        if qs:
            query = " OR ".join([f"({t})" for t in qs if t])
            if not query:
                query = otxt
            return MatchTextExpr(
                self.query_fields, query, 100, {"minimum_should_match": min_match, "original_query": original_query}
            ), keywords
        keywords = list(set(keywords))
        return None, keywords

    def hybrid_similarity(self, avec, bvecs, atks, btkss, tkweight=0.3, vtweight=0.7):
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        sims = cosine_similarity([avec], bvecs)
        tksim = self.token_similarity(atks, btkss)
        if np.sum(sims[0]) == 0:
            return np.array(tksim), tksim, sims[0]
        return np.array(sims[0]) * vtweight + np.array(tksim) * tkweight, tksim, sims[0]

    def token_similarity(self, atks, btkss):
        def to_dict(tks):
            if isinstance(tks, str):
                tks = tks.split()
            d = defaultdict(int)
            wts = self.tw.weights(tks, preprocess=False)
            for i, (t, c) in enumerate(wts):
                d[t] += c
            return d

        atks = to_dict(atks)
        btkss = [to_dict(tks) for tks in btkss]
        return [self.similarity(atks, btks) for btks in btkss]

    def similarity(self, qtwt, dtwt):
        if isinstance(dtwt, type("")):
            dtwt = {t: w for t, w in self.tw.weights(self.tw.split(dtwt), preprocess=False)}
        if isinstance(qtwt, type("")):
            qtwt = {t: w for t, w in self.tw.weights(self.tw.split(qtwt), preprocess=False)}
        s = 1e-9
        for k, v in qtwt.items():
            if k in dtwt:
                s += v #* dtwt[k]
        q = 1e-9
        for k, v in qtwt.items():
            q += v #* v
        return s/q #math.sqrt(3. * (s / q / math.log10( len(dtwt.keys()) + 512 )))

    def paragraph(self, content_tks: str, keywords: list = [], keywords_topn=30):
        if isinstance(content_tks, str):
            content_tks = [c.strip() for c in content_tks.strip() if c.strip()]
        tks_w = self.tw.weights(content_tks, preprocess=False)

        origin_keywords = keywords.copy()
        keywords = [f'"{k.strip()}"' for k in keywords]
        for tk, w in sorted(tks_w, key=lambda x: x[1] * -1)[:keywords_topn]:
            tk_syns = self.syn.lookup(tk)
            tk_syns = [FulltextQueryer.sub_special_char(s) for s in tk_syns]
            tk_syns = [rag_tokenizer.fine_grained_tokenize(s) for s in tk_syns if s]
            tk_syns = [f"\"{s}\"" if s.find(" ") > 0 else s for s in tk_syns]
            tk = FulltextQueryer.sub_special_char(tk)
            if tk.find(" ") > 0:
                tk = '"%s"' % tk
            if tk_syns:
                tk = f"({tk} OR (%s)^0.2)" % " ".join(tk_syns)
            if tk:
                keywords.append(f"{tk}^{w}")

        return MatchTextExpr(self.query_fields, " ".join(keywords), 100,
                             {"minimum_should_match": min(3, len(keywords) / 10), "original_query": " ".join(origin_keywords)})

if __name__ == "__main__":
    f = FulltextQueryer()
    '''
    1. 对查询文本进行预处理（添加英中间的空格、繁体转简体、全角转半角等）
    2. 移除无关词（如疑问词）
    3. 分词并提取关键词
    4. 计算词权重
    5. 添加同义词扩展
    6. 构建匹配表达式
    '''
    matchExpr,keywords = f.question("请问这个系统是干嘛的")
    pass