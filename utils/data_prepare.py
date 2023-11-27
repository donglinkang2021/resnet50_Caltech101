import os
import random

def get_data_list(
        train_data_path="./data/dataset/train.txt",
        train_list_path="./data/process/train_list.txt",
        val_list_path="./data/process/val_list.txt"
    ):
    """
    生成训练集和验证集的数据列表
    
    Parameters:
    @param train_data_path: str, 训练集数据路径
    @param train_list_path: str, 划分训练集列表路径
    @param val_list_path: str, 划分验证集列表路径
    """

    with open(train_list_path, 'w', encoding='utf-8') as f_train:
        f_train.seek(0)
        f_train.truncate() 

    with open(val_list_path, 'w', encoding='utf-8') as f_val:
        f_val.seek(0)
        f_val.truncate()
        

    train_list = []
    val_list = []

    with open(train_data_path, "r") as f:
        train_data = f.read().strip("\n")

    for i, line in enumerate(train_data.split("\n")):
        img_path = line.split("\t")[0]
        class_index = line.split("\t")[1].strip()
        if i % 10 == 0:
            val_list.append(img_path + "\t" + class_index + "\n")
        else:
            train_list.append(img_path + "\t" + class_index + "\n")

    with open(train_list_path, 'a+', encoding='utf-8') as f_train:
        f_train.writelines(train_list)

    with open(val_list_path, 'a+', encoding='utf-8') as f_val:
        f_val.writelines(val_list)


def load_class_list(text_path="./data/dataset/class.txt"):
    """
    加载类别列表

    Parameters:
    @param text_path: str, 类别列表文件路径
    @return class_list: list, 类别列表
    """
    with open(text_path, "r") as f:
        class_text = f.read().strip("\n")
    class_list = []
    for line in class_text.split("\n"):
        class_name = line.split("\t")[0]
        class_list.append(class_name)
    return class_list