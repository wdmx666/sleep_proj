"""
@Name:        test4
@Description: ''
@Author:      Lucas Yu
@Created:     2018/11/6
@Copyright:   (c) GYENNO Science,Shenzhen,Guangdong 2018
@Licence:
"""
from typing import Callable, Iterable, Union, Optional, List
from typing import List, Set, Dict, Tuple, Optional

# def greeting(name):
#     return 'Hello, {}'.format(name)

# def greeting(name: str) -> str:
#     return 'Hello, {}'.format(name)

import pandas as pd


def greeting(name: str) -> str:
    return 'Hello, {}'.format(name)


def show(df: pd.DataFrame) ->str:
    print(df.head())
    return "Good"


#print(greeting("World!"))
x: str = "12"
print(greeting(x))

from sklearn.preprocessing import MaxAbsScaler
import numpy

scaler =StandardScaler()
scaler.fit(numpy.arange(10))
scaler.transform(numpy.arange(10))