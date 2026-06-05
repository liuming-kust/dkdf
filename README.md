# DKDF：动态知识精炼框架

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-ESWA-red.svg)](https://www.sciencedirect.com/journal/expert-systems-with-applications)

## 简介

DKDF (Dynamic Knowledge Distillation Framework) 是一个面向教育领域的动态知识精炼框架，实现了 **“构建-蒸馏-评估”三位一体**的知识自动化精炼范式。

框架由三个核心模块组成：

| 模块 | 名称 | 功能 |
|------|------|------|
| **HKG** | 层次化知识图谱构建器 | 实体(HNER)+关系(MH-GCN)+规则(ILP)三重抽取 |
| **DRD** | 双循环蒸馏机制 | 内环实时提纯 + 外环增量进化 + 价值评估V(k) |
| **CCA** | 认知一致性评估器 | 逻辑一致性(OCR/PCR)+认知合理性(PMC/CEC)+动态适应性(KER/OKCR) |

## 论文引用

本框架对应论文：

> **DKDF：面向教育领域的动态知识精炼框架——构建-蒸馏-评估三位一体的知识自动化精炼范式**

若您在研究中使用本框架，请引用：

```bibtex
@article{dkdf2025,
  title={DKDF：面向教育领域的动态知识精炼框架——构建-蒸馏-评估三位一体的知识自动化精炼范式},
  author={lm},
  journal={Expert Systems with Applications},
  year={2025}
}
