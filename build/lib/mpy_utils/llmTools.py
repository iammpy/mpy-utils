# llmTools.py


def build_prompt(*args)->list[dict]:
    """_summary_
    将输入的一系列字符串转化为一个字典列表
    Returns:
        list[dict]: _description_
    """
    
    # 如果没有导入json5模块，就导入json5
    # import json5
    list_dict = []
    for i,arg in enumerate(args):
        if i%2==0:
            # 偶数索引为用户输入
            temp={
                "role": "user",
                "content": arg
            }
        else:
            temp={
                "role": "assistant",
                "content": arg
            }
        list_dict.append(temp)
    
    return list_dict
        