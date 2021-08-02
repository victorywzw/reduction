import numpy as np 
import cantera as ct
import os, sys
import time
import shutil
import random

def ReLU(x):
    return x * (x > 0)



# 用于创建目录的函数
def mkdir(fn):
    if not os.path.isdir(fn):                   # os.path.isdir()用于判断对象是否为一个目录
        os.mkdir(fn) 


'''
    输出格式：20210327_102133
'''
def get_time_str():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

'''
    在当前目录下生成 result 和 result_test 两个文件夹
    每个文件夹下生成实验文件
    实验文件下生成 pic, parameter, result, code四个文件夹
'''
def mkdir_x(path = '/home/wangzhiwei/combustion', is_serious = 'False'):
    # 创建目录
    test_index = get_time_str()
    print('folder index:', test_index)
    if is_serious:
        mkdir(path + '/result')
        FolderName = path + '/result/%s' % (test_index)
        mkdir(FolderName)
    else:
        mkdir(path + '/result_test')
        FolderName = path + '/result_test/%s' % (test_index)
        mkdir(FolderName)

    pic_fold = '%s/pic' % (FolderName)
    mkdir(pic_fold)

    parameter_fold = '%s/model' % (FolderName)
    mkdir(parameter_fold)

    result_fold = '%s/result' % (FolderName)
    mkdir(result_fold)

    code_fold = '%s/code' % (FolderName)
    mkdir(code_fold)

    return FolderName

'''
    将文件夹 file_path 中的 py 文件复制到 target_file_path 中
'''
def savecode(file_path, target_file_path):
    for files in os.listdir(file_path):
        sourceFile = os.path.join(file_path, files)
        targetFile = os.path.join(target_file_path, files)
        if os.path.isfile(sourceFile) and sourceFile.find('.py')>0: # 要求是文件且后缀是py
            shutil.copy(sourceFile,targetFile)
    print('copy all the code......\n source File: %s\n target File: %s\n Done!'% (file_path, target_file_path))

'''
    程序完成后将文件名加一个后缀 _done
'''
def rename(FolderName):
    os.rename('%s' % (FolderName), '%s_done' % (FolderName))
    print('-'*50)
    print('program is done, and the data will be saved in %s_done' % (FolderName))
    print('-'*50)




















