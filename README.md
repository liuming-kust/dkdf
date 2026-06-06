# DKDF：动态知识精炼框架

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 简介

DKDF (Dynamic Knowledge Distillation Framework) 是一个面向教育领域的动态知识精炼框架，实现了 **“构建-蒸馏-评估”三位一体**的知识自动化精炼范式。

框架由三个核心模块组成：

| 模块 | 名称 | 功能 |
|------|------|------|
| **HKG** | 层次化知识图谱构建器 | 实体(HNER)+关系(MH-GCN)+规则(ILP)三重抽取 |
| **DRD** | 双循环蒸馏机制 | 内环实时提纯 + 外环增量进化 + 价值评估V(k) |
| **CCA** | 认知一致性评估器 | 逻辑一致性(OCR/PCR)+认知合理性(PMC/CEC)+动态适应性(KER/OKCR) |

本框架对应论文：

> **DKDF：面向教育领域的动态知识精炼框架——构建-蒸馏-评估三位一体的知识自动化精炼范式**

DKDF（A Tripartite Paradigm of Construction-Distillation-Evaluation for Knowledge Refinement in Educational Documents，动态知识精炼框架），将知识处理解构为三个协同模块：层次化知识图谱构建器（HKG）实现实体、关系、规则的三重抽取；双循环蒸馏机制（DRD）通过内环实时提纯与外环增量进化实现知识的动态演化；认知一致性评估器（CCA）从逻辑一致性、认知合理性、动态适应性三个维度系统性保障知识质量。

## 论文引用
## ⚠️ 重要声明：数据版权与使用限制

**本文涉及的教务文件和物理实验讲义均为本校内部资料，受版权保护和保密协议约束，无法公开。**

- ❌ 仓库中**不包含**任何原始文档（PDF、DOCX等）
- ✅ 仓库提供**数据格式模板**（JSON格式）和**模拟数据**（内容完全虚构）
- ✅ 用户可使用**自己的文档**按照模板格式进行测试

如您需要验证框架效果，请：
1. 参考 `examples/sample_data/` 中的格式模板
2. 准备您自己的文档（各高校均有类似材料）
3. 按照示例代码运行测试


若您在研究中使用本框架，请引用：

```bibtex
@article{dkdf2025,
  title={DKDF：面向教育领域的动态知识精炼框架——构建-蒸馏-评估三位一体的知识自动化精炼范式},
  author={lm},
  year={2025}
}
