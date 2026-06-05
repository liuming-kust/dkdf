# -*- coding: utf-8 -*-
"""
DKDF: Dynamic Knowledge Distillation Framework
动态知识精炼框架 - 构建-蒸馏-评估三位一体

论文贡献全覆盖：
1. HKG：层次化知识图谱构建（实体+关系+规则）
2. DRD：双循环蒸馏机制（内环提纯+外环进化+价值评估V(k)）
3. CCA：认知一致性评估（OCR/PCR/PMC/CEC/KER/OKCR）
4. 全链路制导：分块→过滤→蒸馏→重组
"""

from .version import __version__
from .config import DEFAULT_CONFIG, load_config
from .utils import setup_logger, calculate_tokens, clean_text
from .kg_loader import KnowledgeGraphLoader
from .hkg_builder import HKGProducer
from .drd_distiller import DRDDistiller
from .cca_evaluator import CCAEvaluator
from .main_distiller import DKDFDistiller

__all__ = [
    "__version__",
    "DEFAULT_CONFIG",
    "load_config",
    "setup_logger",
    "calculate_tokens",
    "clean_text",
    "KnowledgeGraphLoader",
    "HKGProducer",
    "DRDDistiller",
    "CCAEvaluator",
    "DKDFDistiller",
]
