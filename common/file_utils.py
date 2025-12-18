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

import os,sys
def get_project_base():
    """
    计算项目根目录（即main.py/EXE所在的目录），兼容源码和打包场景
    匹配目录结构：main.py（根） → module/common.py（子目录）
    """
    # 场景1：PyInstaller打包后的环境（单文件/文件夹EXE）
    if hasattr(sys, '_MEIPASS'):
        # sys.executable 指向EXE文件的绝对路径（如 D:/dist/main.exe）
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        # EXE所在目录就是项目根目录（对应源码中main.py的目录）
        project_base = exe_dir
    # 场景2：源码运行环境
    else:
        project_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.exists(os.path.join(project_base, "static")):
        return project_base
    project_base = os.path.join(project_base, "deepdoc_standalone")
    if os.path.exists(os.path.join(project_base, "static")):
        return project_base    
    raise Exception("deepdoc_standalone工作目录未找到static目录")



# 初始化PROJECT_BASE，供main.py导入使用
PROJECT_BASE = get_project_base()

print(f'deepdoc_standalone工作目录:{PROJECT_BASE}')
def get_project_base_directory(*args):
    global PROJECT_BASE
    if args:
        return os.path.join(PROJECT_BASE, *args)
    return PROJECT_BASE

def traversal_files(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname
