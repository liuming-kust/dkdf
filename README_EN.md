# DKDF: Dynamic Knowledge Distillation Framework

## Introduction

DKDF (Dynamic Knowledge Distillation Framework) is a dynamic knowledge distillation framework for the educational domain, implementing a "Construction-Distillation-Evaluation" tripartite closed-loop paradigm for automated knowledge refinement.

The framework consists of three core modules:

| Module | Name | Function |
|--------|------|----------|
| HKG | Hierarchical Knowledge Graph Builder | Triple extraction: entities (HNER) + relations (MH-GCN) + rules (ILP) |
| DRD | Dual-loop Rectification Distillation | Inner-loop real-time purification + outer-loop incremental evolution + value function V(k) |
| CCA | Cognitive Consistency Assessment | Logical consistency (OCR/PCR) + cognitive rationality (PMC/CEC) + dynamic adaptability (KER/OKCR) |

This framework accompanies the paper:

**DKDF: A Tripartite Closed-Loop Framework for Dynamic Knowledge Distillation from Educational Documents**  
(Applied Intelligence, APIN-D-26-07834)

*Terminological note*: In this context, "knowledge distillation" refers to extracting, purifying, and organizing domain knowledge from unstructured documents into a structured knowledge base. This is distinct from model distillation, although both share the idea of "extracting core information" but differ fundamentally in their objects and application scenarios.

## Reproducibility & Result Notes

- The numerical results reported in the paper (e.g., Precision 4.2/5.0, Coverage 91.5%, CCA Overall 0.862) are based on **full internal datasets** that cannot be released due to institutional confidentiality.
- Our provided **synthetic data** and **code** reproduce the **complete pipeline**; exact numbers will differ when applied to other document sets.
- We observe **stable trends** across 5‑fold cross‑validation and two distinct datasets, confirming the framework is not over‑fitted to a single collection.

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{dkdf2025,
  title={DKDF：面向教育领域的动态知识精炼框架——构建-蒸馏-评估三位一体的知识自动化精炼范式},
  author={Liu, Ming and Liu, ChunYin and Guo, Ping},
  year={2025}
}
