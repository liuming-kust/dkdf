#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
物理实验CCA评估报告生成模块

生成详细的评估报告，包括：
1. 综合评分和评级
2. 各指标详细得分
3. 可视化图表数据
4. 改进建议
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from .physics_cca_metrics import PhysicsCCAMetrics


class PhysicsCCAReport:
    """
    物理实验CCA评估报告生成器
    """
    
    def __init__(self, metrics: PhysicsCCAMetrics, 
                 kg_metadata: Optional[Dict] = None,
                 recommendations: Optional[List[str]] = None):
        """
        初始化报告生成器
        
        Args:
            metrics: 评估指标
            kg_metadata: 知识图谱元数据
            recommendations: 改进建议列表
        """
        self.metrics = metrics
        self.kg_metadata = kg_metadata or {}
        self.recommendations = recommendations or []
    
    def generate(self, format: str = "json") -> Dict[str, Any]:
        """
        生成评估报告
        
        Args:
            format: 输出格式 ('json' 或 'dict')
        
        Returns:
            报告字典
        """
        grade_code, grade_name = self.metrics.get_grade()
        
        report = {
            "report_metadata": {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "report_version": "1.0",
                "evaluator": "PhysicsCCA"
            },
            "kg_metadata": self.kg_metadata,
            "summary": {
                "overall_score": round(self.metrics.overall, 4),
                "grade_code": grade_code,
                "grade_name": grade_name,
                "status": "合格" if self.metrics.overall >= 0.6 else "不合格"
            },
            "dimension_scores": {
                "logical_consistency": {
                    "score": (1 - self.metrics.ocr) * 0.5 + self.metrics.pcr * 0.5,
                    "weight": 0.40,
                    "metrics": {
                        "ocr": {
                            "value": round(self.metrics.ocr, 4),
                            "description": "本体冲突率",
                            "ideal_range": "[0, 0.05]",
                            "status": self.metrics.ocr_details.get("status", "unknown")
                        },
                        "pcr": {
                            "value": round(self.metrics.pcr, 4),
                            "description": "路径连贯性",
                            "ideal_range": "[0.7, 1.0]",
                            "status": self.metrics.pcr_details.get("status", "unknown")
                        }
                    }
                },
                "cognitive_rationality": {
                    "score": (self.metrics.pmc + self.metrics.cec) / 2,
                    "weight": 0.35,
                    "metrics": {
                        "pmc": {
                            "value": round(self.metrics.pmc, 4),
                            "description": "原型匹配度",
                            "ideal_range": "[0.8, 1.0]",
                            "status": self.metrics.pmc_details.get("status", "unknown")
                        },
                        "cec": {
                            "value": round(self.metrics.cec, 4),
                            "description": "认知经济度",
                            "ideal_range": "[0.4, 1.0]",
                            "status": self.metrics.cec_details.get("status", "unknown")
                        }
                    }
                },
                "dynamic_adaptability": {
                    "score": (self.metrics.ker + self.metrics.okcr) / 2,
                    "weight": 0.25,
                    "metrics": {
                        "ker": {
                            "value": round(self.metrics.ker, 4),
                            "description": "知识演化率",
                            "ideal_range": "[0.1, 1.0]",
                            "status": self.metrics.ker_details.get("status", "unknown")
                        },
                        "okcr": {
                            "value": round(self.metrics.okcr, 4),
                            "description": "过时知识清除率",
                            "ideal_range": "[0.8, 1.0]",
                            "status": self.metrics.okcr_details.get("status", "unknown")
                        }
                    }
                }
            },
            "details": {
                "ocr": self.metrics.ocr_details,
                "pcr": self.metrics.pcr_details,
                "pmc": self.metrics.pmc_details,
                "cec": self.metrics.cec_details,
                "ker": self.metrics.ker_details,
                "okcr": self.metrics.okcr_details
            },
            "recommendations": self.recommendations,
            "raw_metrics": self.metrics.to_dict()
        }
        
        return report
    
    def save(self, output_path: str) -> None:
        """保存报告到JSON文件"""
        report = self.generate(format="json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    
    def print_summary(self) -> None:
        """打印报告摘要"""
        report = self.generate()
        
        print("\n" + "=" * 60)
        print("物理实验知识图谱CCA评估报告")
        print("=" * 60)
        
        print(f"\n生成时间: {report['report_metadata']['generated_at']}")
        
        summary = report['summary']
        print(f"\n【综合评分】: {summary['overall_score']} ({summary['grade_name']})")
        print(f"【评估状态】: {summary['status']}")
        
        print("\n【维度得分】")
        for dim_name, dim_data in report['dimension_scores'].items():
            dim_name_cn = {
                "logical_consistency": "逻辑一致性",
                "cognitive_rationality": "认知合理性",
                "dynamic_adaptability": "动态适应性"
            }.get(dim_name, dim_name)
            print(f"  - {dim_name_cn}: {dim_data['score']:.4f} (权重: {dim_data['weight']})")
        
        print("\n【改进建议】")
        for rec in self.recommendations[:5]:
            print(f"  • {rec}")
        
        print("\n" + "=" * 60)


def generate_report(metrics: PhysicsCCAMetrics,
                    kg_metadata: Optional[Dict] = None,
                    recommendations: Optional[List[str]] = None,
                    output_path: Optional[str] = None) -> PhysicsCCAReport:
    """
    生成评估报告的便捷函数
    
    Args:
        metrics: 评估指标
        kg_metadata: 知识图谱元数据
        recommendations: 改进建议
        output_path: 输出文件路径
    
    Returns:
        PhysicsCCAReport对象
    """
    report = PhysicsCCAReport(metrics, kg_metadata, recommendations)
    
    if output_path:
        report.save(output_path)
    
    return report
