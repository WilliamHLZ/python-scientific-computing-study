# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 09:51:47 2018

@author: William
"""

import os
import zipfile
from pathlib import Path

source_path = 'F:\study_video\python'
target_path = 'F:\study_video\python\pyvideo'

filenames = os.listdir(source_path)

for seq1 in range(len(filenames)-1,-1,-1):  # 用倒叙的方式来删除元素，避免了元素删除后自动补位的问题
    temp1 = str.split(filenames[seq1], '.')
    if (temp1[len(temp1)-1] == 'zip'): 
        ziptemp = zipfile.ZipFile(source_path+'\\'+filenames[seq1])
        for fn in ziptemp.namelist():
            print(fn)
            extracted_path = Path(ziptemp.extract(fn, path=target_path))
            extracted_path.rename(target_path+'\\'+fn.encode('cp437').decode('gbk'))

