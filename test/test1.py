import traceback
import cgitb

cgitb.enable(display=True, logdir=r'E:\my_proj\ms10\logs',format='text')

def add2(a, b):
    if not isinstance(a, float):
        raise TypeError('类型错误')
    return a + b


def f(a, b):
    try:
       return add2(a,b)

    except Exception as e:
        traceback.print_exc()

    print("tst")
    return -1

def f2(a, b):
    add2(a,b)

if __name__=="__main__":
    print(f("A",3.4))
    print(f2("A",3.4))
    print(sum(12,'a'))



import pandas
import numpy as np
df=pandas.DataFrame(data=np.arange(12).reshape(-1,3))
df.to_parquet()
from pandas import io as pio

