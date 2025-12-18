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
import sys, os; __name__ == "__main__" and sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import re
try:
    from common.token_utils import num_tokens_from_string

    from rag.nlp import find_codec
except:
    from deepdoc_standalone.common.token_utils import num_tokens_from_string

    from deepdoc_standalone.rag.nlp import find_codec


def get_text(fnm: str, binary=None) -> str:
    txt = ""
    if binary:
        encoding = find_codec(binary)
        txt = binary.decode(encoding, errors="ignore")
    else:
        try:
            with open(fnm, "r",encoding="utf-8") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    txt += line
        except:
            with open(fnm, "r",encoding="gbk") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    txt += line
    return txt


class RAGFlowTxtParser:
    def __call__(self, fnm, binary=None, chunk_token_num=128, delimiter="\n!?;。；！？"):
        txt = get_text(fnm, binary)
        return self.parser_txt(txt, chunk_token_num, delimiter)

    @classmethod
    def parser_txt(cls, txt, chunk_token_num=128, delimiter="\n!?;。；！？"):
        if not isinstance(txt, str):
            raise TypeError("txt type should be str!")
        cks = [""]
        tk_nums = [0]
        delimiter = delimiter.encode('utf-8').decode('unicode_escape').encode('latin1').decode('utf-8')

        def add_chunk(t):
            nonlocal cks, tk_nums, delimiter
            tnum = num_tokens_from_string(t)
            if tk_nums[-1] > chunk_token_num:
                cks.append(t)
                tk_nums.append(tnum)
            else:
                cks[-1] += t
                tk_nums[-1] += tnum

        dels = []
        s = 0
        for m in re.finditer(r"`([^`]+)`", delimiter, re.I):
            f, t = m.span()
            dels.append(m.group(1))
            dels.extend(list(delimiter[s: f]))
            s = t
        if s < len(delimiter):
            dels.extend(list(delimiter[s:]))
        dels = [re.escape(d) for d in dels if d]
        dels = [d for d in dels if d]
        dels = "|".join(dels)
        secs = re.split(r"(%s)" % dels, txt)
        for sec in secs:
            if re.match(f"^{dels}$", sec):
                continue
            add_chunk(sec)

        return [[c, ""] for c in cks]

if __name__ == "__main__":
    from common.file_utils import get_project_base_directory
    path = os.path.join(get_project_base_directory(), "static/测试文件/test.txt")
    psr = RAGFlowTxtParser()
    result = psr(path)
    pass