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
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from common.constants import LLMType
from api.db.services.llm_service import LLMBundle
from common.connection_utils import timeout
from rag.app.picture import vision_llm_chunk as picture_vision_llm_chunk
from rag.prompts.generator import vision_llm_figure_describe_prompt


def vision_figure_parser_figure_data_wrapper(figures_data_without_positions):
    return [
        (
            (figure_data[1], [figure_data[0]]),
            [(0, 0, 0, 0, 0)],
        )
        for figure_data in figures_data_without_positions
        if isinstance(figure_data[1], Image.Image)
    ]


def vision_figure_parser_docx_wrapper(sections,tbls,callback=None,**kwargs):
    try:
        vision_model = LLMBundle(kwargs["tenant_id"], LLMType.IMAGE2TEXT)
        callback(0.7, "Visual model detected. Attempting to enhance figure extraction...")
    except Exception:
        vision_model = None
    if vision_model:
        figures_data = vision_figure_parser_figure_data_wrapper(sections)
        try:
            docx_vision_parser = VisionFigureParser(vision_model=vision_model, figures_data=figures_data, **kwargs)
            boosted_figures = docx_vision_parser(callback=callback)
            tbls.extend(boosted_figures)
        except Exception as e:
            callback(0.8, f"Visual model error: {e}. Skipping figure parsing enhancement.")
    return tbls


def vision_figure_parser_pdf_wrapper(tbls, callback=None, **kwargs):
    """使用视觉模型增强PDF文档中的图表解析
    
    该函数是一个包装器，尝试使用AI视觉模型(IMAGE2TEXT)对PDF中的图表进行增强解析，
    提取更丰富的图表信息。如果视觉模型可用，则处理表格数据中的图片项，否则返回原始数据。
    
    参数:
        tbls (list): 包含表格和图表数据的列表，每项可能是文本表格或图片数据
        callback (callable, optional): 回调函数，用于报告处理进度和状态消息
        **kwargs: 额外参数，必须包含以下关键字:
            - tenant_id: 租户ID，用于初始化视觉模型
            - 其他可能被VisionFigureParser使用的参数
    
    返回:
        list: 处理后的表格和图表数据列表，如果视觉模型可用，则包含增强后的图表信息
    """
    try:
        # 尝试初始化视觉模型(IMAGE2TEXT类型)
        vision_model = LLMBundle(kwargs["tenant_id"], LLMType.IMAGE2TEXT)
        # 通知回调函数视觉模型已检测到，准备增强图表提取
        callback(0.7, "Visual model detected. Attempting to enhance figure extraction...")
    except Exception:
        # 如果初始化失败，设置视觉模型为None
        vision_model = None
    
    if vision_model:
        # 定义一个内部函数，用于判断表格项是否为图片类型
        def is_figure_item(item):
            """判断一个表格项是否为图片数据
            
            图片项的特征是: 第一项是一个包含PIL.Image对象和坐标列表的元组
            
            参数:
                item: 表格中的一项数据
            
            返回:
                bool: 如果是图片项返回True，否则返回False
            """
            return (
                isinstance(item[0][0], Image.Image) and
                isinstance(item[0][1], list)
            )
        
        # 过滤出所有图片类型的表格项
        figures_data = [item for item in tbls if is_figure_item(item)]
        
        try:
            # 使用VisionFigureParser处理图片数据，增强图表解析
            docx_vision_parser = VisionFigureParser(vision_model=vision_model, figures_data=figures_data, **kwargs)
            # 调用视觉解析器处理图片，并传入回调函数
            boosted_figures = docx_vision_parser(callback=callback)
            # 移除原始图片数据，保留其他类型的表格数据
            tbls = [item for item in tbls if not is_figure_item(item)]
            # 将增强后的图片数据添加到结果中
            tbls.extend(boosted_figures)
        except Exception as e:
            # 如果处理过程中发生错误，通知回调函数并继续使用原始数据
            callback(0.8, f"Visual model error: {e}. Skipping figure parsing enhancement.")
    
    # 返回处理后的表格和图表数据列表
    return tbls


shared_executor = ThreadPoolExecutor(max_workers=10)


class VisionFigureParser:
    def __init__(self, vision_model, figures_data, *args, **kwargs):
        self.vision_model = vision_model
        self._extract_figures_info(figures_data)
        assert len(self.figures) == len(self.descriptions)
        assert not self.positions or (len(self.figures) == len(self.positions))

    def _extract_figures_info(self, figures_data):
        self.figures = []
        self.descriptions = []
        self.positions = []

        for item in figures_data:
            # position
            if len(item) == 2 and isinstance(item[0], tuple) and len(item[0]) == 2 and isinstance(item[1], list) and isinstance(item[1][0], tuple) and len(item[1][0]) == 5:
                img_desc = item[0]
                assert len(img_desc) == 2 and isinstance(img_desc[0], Image.Image) and isinstance(img_desc[1], list), "Should be (figure, [description])"
                self.figures.append(img_desc[0])
                self.descriptions.append(img_desc[1])
                self.positions.append(item[1])
            else:
                assert len(item) == 2 and isinstance(item[0], Image.Image) and isinstance(item[1], list), f"Unexpected form of figure data: get {len(item)=}, {item=}"
                self.figures.append(item[0])
                self.descriptions.append(item[1])

    def _assemble(self):
        self.assembled = []
        self.has_positions = len(self.positions) != 0
        for i in range(len(self.figures)):
            figure = self.figures[i]
            desc = self.descriptions[i]
            pos = self.positions[i] if self.has_positions else None

            figure_desc = (figure, desc)

            if pos is not None:
                self.assembled.append((figure_desc, pos))
            else:
                self.assembled.append((figure_desc,))

        return self.assembled

    def __call__(self, **kwargs):
        callback = kwargs.get("callback", lambda prog, msg: None)

        @timeout(30, 3)
        def process(figure_idx, figure_binary):
            description_text = picture_vision_llm_chunk(
                binary=figure_binary,
                vision_model=self.vision_model,
                prompt=vision_llm_figure_describe_prompt(),
                callback=callback,
            )
            return figure_idx, description_text

        futures = []
        for idx, img_binary in enumerate(self.figures or []):
            futures.append(shared_executor.submit(process, idx, img_binary))

        for future in as_completed(futures):
            figure_num, txt = future.result()
            if txt:
                self.descriptions[figure_num] = txt + "\n".join(self.descriptions[figure_num])

        self._assemble()

        return self.assembled
