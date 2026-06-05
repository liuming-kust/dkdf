# -*- coding: utf-8 -*-
"""DKDF 配置文件 - 完整版"""

from typing import Dict, Any

DEFAULT_CONFIG: Dict[str, Any] = {
    # HKG配置
    "hkg": {
        "entity_extraction": {
            "rule_based": True,
            "llm_based": True,
            "min_confidence": 0.6,
        },
        "relation_extraction": {
            "co_occurrence_window": 200,
            "co_occurrence_confidence": 0.5,
            "rule_based_confidence": 0.8,
            "llm_based_confidence": 0.7,
        },
        "rule_extraction": {
            "min_support": 0.3,
            "min_confidence": 0.6,
            "max_path_length": 3,
        },
    },
    
    # DRD配置
    "drd": {
        "inner_loop": {
            "min_content_length": 20,
            "conflict_threshold": 0.3,
            "similarity_threshold": 0.85,
            "max_llm_calls": 100,
        },
        "outer_loop": {
            "update_interval": 3600,      # 1小时
            "retain_threshold": 0.6,      # 保留阈值
            "expire_days": 30,             # 过期天数
            "low_value_threshold": 0.4,    # 低价值淘汰阈值
            "value_weights": {             # V(k)权重
                "novelty": 0.4,
                "coverage": 0.3,
                "conflict": 0.3,
            },
        },
    },
    
    # CCA配置
    "cca": {
        "thresholds": {
            "ocr": 0.05,      # 本体冲突率阈值
            "pcr": 0.70,      # 路径连贯性阈值
            "pmc": 0.80,      # 原型匹配度阈值
            "cec_min": 0.40,  # 认知经济度最低值
        },
        "overall_weights": {
            "ocr": 0.20,
            "pcr": 0.20,
            "pmc": 0.20,
            "cec": 0.15,
            "ker": 0.15,
            "okcr": 0.10,
        },
    },
    
    # 处理配置
    "processing": {
        "max_tokens": 1024,
        "chunk_size": 2000,
        "chunk_overlap": 400,
        "target_compression": 0.3,
    },
    
    # LLM配置
    "llm": {
        "model": "qwen2.5:14b",
        "temperature": 0.2,
        "base_url": "http://localhost:11434",
        "timeout": 120,
    },
    
    # 向量存储配置
    "vector_store": {
        "embedding_model": "nomic-embed-text",
        "index_type": "faiss",
    },
    
    # 日志配置
    "logging": {
        "level": "INFO",
        "file": "dkdf.log",
    },
}

def load_config(config_path: str = None) -> Dict[str, Any]:
    """加载配置文件"""
    import copy
    import json
    
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
        _deep_update(config, user_config)
    return config

def _deep_update(base: Dict, updates: Dict) -> None:
    """深度更新字典"""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
