#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
物理实验CCA评估模块

功能：
1. 评估物理实验知识图谱的认知一致性
2. 输出三维度六指标评估结果
3. 生成详细的评估报告和改进建议

与DKRF的CCA模块对应关系：
- 逻辑一致性：OCR（本体冲突率）+ PCR（路径连贯性）
- 认知合理性：PMC（原型匹配度）+ CEC（认知经济度）
- 动态适应性：KER（知识演化率）+ OKCR（过时知识清除率）
"""

from .physics_cca_evaluator import PhysicsCCAEvaluator
from .physics_cca_metrics import PhysicsCCAMetrics, calculate_all_metrics
from .physics_cca_report import PhysicsCCAReport, generate_report

__all__ = [
    "PhysicsCCAEvaluator",
    "PhysicsCCAMetrics",
    "calculate_all_metrics",
    "PhysicsCCAReport",
    "generate_report",
]

__version__ = "1.0.0"
__author__ = "DKRF Team"
