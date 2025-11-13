"""
答案提取和比较工具模块

用于统一处理 GSM8K 数据集的答案提取、标准化和比较逻辑。
确保训练和推理阶段使用一致的评估标准。
"""

import re
import ast
import json


def parse_answer_field(answer_field):
    """
    解析 answer 字段，处理多种格式：
    - 普通字符串: "128"
    - 列表: [128] 或 ["128"]
    - 字符串化的列表: "['Final Answer: 128']"
    - JSON字符串: '["128"]'
    
    返回: 清理后的字符串
    """
    # 1. 如果已经是列表，直接取第一个元素
    if isinstance(answer_field, list):
        if len(answer_field) > 0:
            answer_field = answer_field[0]
        else:
            return ""
    
    # 2. 如果是字符串，尝试解析为列表
    if isinstance(answer_field, str):
        # 尝试 JSON 解析
        try:
            parsed = json.loads(answer_field)
            if isinstance(parsed, list) and len(parsed) > 0:
                answer_field = parsed[0]
        except (json.JSONDecodeError, ValueError):
            # 尝试 ast.literal_eval
            try:
                parsed = ast.literal_eval(answer_field)
                if isinstance(parsed, list) and len(parsed) > 0:
                    answer_field = parsed[0]
            except (ValueError, SyntaxError):
                pass  # 保持原样
    
    # 3. 转为字符串并清理
    answer_str = str(answer_field).strip()
    
    # 4. 去除常见前缀
    prefixes = [
        r'^\s*Final Answer\s*:\s*',
        r'^\s*The answer is\s*:\s*',
        r'^\s*Answer\s*:\s*',
        r'^\s*####\s*',  # GSM8K 有时用 #### 作为答案标记
    ]
    for prefix in prefixes:
        answer_str = re.sub(prefix, '', answer_str, flags=re.IGNORECASE)
    
    return answer_str.strip()


def extract_number_from_text(text):
    """
    从文本中提取最终数值答案（支持多种格式，优先级递减）
    
    支持的格式（按优先级）：
    1. <ans>...</ans> 标签（方案 3）
    2. 标准格式: <think>...</think>\n42
    3. Final Answer 格式: <think>...Final Answer: 42</think>
    4. 无格式: 直接提取最后一个数字
    
    返回: (number_str, number_value)
        - number_str: 标准化的数字字符串（如 "128"）
        - number_value: 数值（float），如果无法提取则为 None
    """
    # 1. 优先提取 <ans>...</ans> 标签（方案 3 的标准格式）
    ans_pattern = r'<ans>(.+?)</ans>'
    ans_matches = re.findall(ans_pattern, text, re.IGNORECASE | re.DOTALL)
    if ans_matches:
        text = ans_matches[-1].strip()  # 取最后一个 <ans> 标签
    
    # 2. 回退：尝试提取 </think> 之后的内容（旧格式兼容）
    elif "</think>" in text:
        after_think = text.split("</think>")[-1].strip()
        
        # 如果 </think> 后面有实质内容，优先使用
        if after_think:
            text = after_think
        else:
            # 如果 </think> 后面是空的，尝试从内部提取 "Final Answer: xxx"
            final_answer_pattern = r'Final Answer[:\s]+(.+?)(?:\n|</think>|$)'
            matches = re.findall(final_answer_pattern, text, re.IGNORECASE)
            if matches:
                text = matches[-1].strip()
    
    # 3. 用正则提取数字（支持负数、小数、千位分隔符）
    # 匹配模式：-?[\d,]+\.?\d*
    # 例如：-1,234.56、1234、-5.5、1,000,000
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    
    if not numbers:
        return text.strip(), None
    
    # 4. 取最后一个数字（通常是最终答案）
    last_number = numbers[-1]
    
    # 5. 清理并标准化
    # 移除千位分隔符
    cleaned = last_number.replace(',', '')
    
    # 6. 转换为数值
    try:
        num_value = float(cleaned)
        
        # 7. 标准化格式：整数不带小数点，小数保留原样
        if num_value.is_integer():
            return str(int(num_value)), num_value
        else:
            return str(num_value), num_value
    except ValueError:
        return cleaned, None


def extract_answer(text_or_field):
    """
    统一的答案提取函数
    
    参数:
        text_or_field: 可能是生成的文本、final_answer字段等
    
    返回:
        标准化的答案字符串（数字格式统一）
    """
    # 1. 先解析字段（处理列表、JSON等格式）
    parsed_text = parse_answer_field(text_or_field)
    
    # 2. 提取数字
    answer_str, answer_num = extract_number_from_text(parsed_text)
    
    # 3. 优先返回标准化的数字字符串
    return answer_str


def compare_answers(predicted, ground_truth, tolerance=1e-6):
    """
    比较两个答案是否相等（优先数值比较）
    
    参数:
        predicted: 预测答案（可能是文本或数字）
        ground_truth: 标准答案（可能是文本或数字）
        tolerance: 浮点数比较容差
    
    返回:
        bool: 是否相等
    """
    # 1. 提取标准化答案
    pred_str = parse_answer_field(predicted)
    gt_str = parse_answer_field(ground_truth)
    
    # 2. 提取数值
    _, pred_num = extract_number_from_text(pred_str)
    _, gt_num = extract_number_from_text(gt_str)
    
    # 3. 优先数值比较
    if pred_num is not None and gt_num is not None:
        return abs(pred_num - gt_num) < tolerance
    
    # 4. 回退到字符串比较（忽略大小写和空白）
    return pred_str.lower().strip() == gt_str.lower().strip()
