# -*- coding: utf-8 -*-
"""工具函数 - 完整版"""

import re
import json
import time
import logging
from typing import Dict, List, Any, Optional, Set
from functools import wraps

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# 全局tokenizer
_tokenizer = None

def get_tokenizer():
    """获取tokenizer（单例）"""
    global _tokenizer
    if _tokenizer is None and TIKTOKEN_AVAILABLE:
        try:
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            _tokenizer = None
    return _tokenizer

def calculate_tokens(text: str) -> int:
    """计算文本token数量"""
    if not text:
        return 0
    
    tokenizer = get_tokenizer()
    if tokenizer:
        return len(tokenizer.encode(text))
    
    # 回退方案：估算
    chinese = len(re.findall(r'[\u4e00-\u9fff]', text))
    other = len(text) - chinese
    return int(chinese / 1.5 + other / 4)

def clean_text(text: str) -> str:
    """清洗文本"""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_json_from_response(text: str) -> Optional[Dict]:
    """从LLM响应中提取JSON"""
    if not text:
        return None
    
    # 直接解析
    try:
        return json.loads(text.strip())
    except:
        pass
    
    # 从代码块提取
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except:
            pass
    
    # 提取第一个完整JSON对象
    brace_start = text.find('{')
    if brace_start != -1:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start:i+1])
                    except:
                        break
    return None

def text_similarity(text1: str, text2: str) -> float:
    """计算两个文本的Jaccard相似度"""
    words1 = set(text1.split())
    words2 = set(text2.split())
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)

def setup_logger(name: str, level: str = "INFO", log_file: str = "dkdf.log") -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if logger.handlers:
        return logger
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """失败重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
            raise last_exception
        return wrapper
    return decorator
