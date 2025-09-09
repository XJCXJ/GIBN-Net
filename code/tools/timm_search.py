# -*- coding: utf-8 -*-
"""
@Time    : 2023/3/14/014 12:33
@Author  : NDWX
@File    : timm_search.py
@Software: PyCharm
"""
import timm
avail_pretrained_models = timm.list_models(pretrained=True)
all_models = timm.list_models("*swin*")
print(all_models)