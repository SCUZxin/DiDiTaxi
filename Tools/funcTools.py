import os

# 读取非隐藏文件的函数
def listdir_nohidden(path):
    list = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            list.append(f)
    return list

