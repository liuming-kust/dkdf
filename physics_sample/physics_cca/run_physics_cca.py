#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
物理实验CCA评估 - 运行示例

演示如何评估物理实验知识图谱的认知一致性
"""

import sys
import os
import json
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.physics_cca.physics_cca_evaluator import PhysicsCCAEvaluator
from examples.physics_cca.physics_cca_report import generate_report


def load_sample_kg() -> dict:
    """加载示例知识图谱数据"""
    # 创建示例物理实验知识图谱
    return {
        "metadata": {
            "source": "大学物理实验教程",
            "experiment_count": 36,
            "version": "1.0"
        },
        "entities": {
            "exp_0001": {
                "id": "exp_0001",
                "name": "3.3 刚体转动惯量的测定",
                "type": "concept",
                "attributes": {
                    "purpose": "通过测量刚体的转动惯量，验证平行轴定理",
                    "theory": "转动惯量是刚体转动惯性的量度，计算公式为 J = ∫r²dm",
                    "page": 45
                }
            },
            "inst_0001": {
                "id": "inst_0001",
                "name": "转动惯量测试仪",
                "type": "instrument",
                "attributes": {"quantity": 1}
            },
            "inst_0002": {
                "id": "inst_0002",
                "name": "电子天平",
                "type": "instrument",
                "attributes": {"quantity": 1}
            },
            "step_0001": {
                "id": "step_0001",
                "name": "调整实验装置水平",
                "type": "procedure",
                "attributes": {"order": 0}
            },
            "step_0002": {
                "id": "step_0002",
                "name": "测量刚体质量",
                "type": "procedure",
                "attributes": {"order": 1}
            },
            "step_0003": {
                "id": "step_0003",
                "name": "记录不同位置的转动周期",
                "type": "procedure",
                "attributes": {"order": 2}
            }
        },
        "relations": [
            {
                "id": "rel_0001",
                "source_id": "exp_0001",
                "source_name": "3.3 刚体转动惯量的测定",
                "target_id": "inst_0001",
                "target_name": "转动惯量测试仪",
                "type": "contains",
                "confidence": 0.9
            },
            {
                "id": "rel_0002",
                "source_id": "exp_0001",
                "source_name": "3.3 刚体转动惯量的测定",
                "target_id": "inst_0002",
                "target_name": "电子天平",
                "type": "contains",
                "confidence": 0.9
            },
            {
                "id": "rel_0003",
                "source_id": "exp_0001",
                "source_name": "3.3 刚体转动惯量的测定",
                "target_id": "step_0001",
                "target_name": "调整实验装置水平",
                "type": "contains",
                "confidence": 0.85
            }
        ],
        "rules": []
    }


def simulate_updates(evaluator: PhysicsCCAEvaluator):
    """模拟知识更新（用于KER计算）"""
    evaluator.record_update(added=5, updated=2)
    evaluator.record_update(added=3, updated=1)
    evaluator.record_update(added=2, updated=3)
    
    evaluator.record_clearance(expired_count=2, cleared_count=2)
    evaluator.record_clearance(expired_count=3, cleared_count=2)
    evaluator.record_clearance(expired_count=1, cleared_count=1)


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("物理实验CCA评估 - 运行示例")
    print("=" * 60)
    
    # 创建评估器
    evaluator = PhysicsCCAEvaluator()
    
    # 加载示例知识图谱
    kg = load_sample_kg()
    print(f"\n加载知识图谱: {kg['metadata']['source']}")
    print(f"  - 实体数: {len(kg['entities'])}")
    print(f"  - 关系数: {len(kg['relations'])}")
    
    # 模拟知识更新记录
    simulate_updates(evaluator)
    
    # 执行评估
    print("\n开始CCA评估...")
    metrics = evaluator.evaluate(kg)
    
    # 生成改进建议
    recommendations = evaluator.generate_recommendations(metrics)
    
    # 生成报告
    report = generate_report(
        metrics=metrics,
        kg_metadata=kg.get("metadata", {}),
        recommendations=recommendations,
        output_path="physics_cca_report.json"
    )
    
    # 打印摘要
    report.print_summary()
    
    # 获取统计信息
    stats = evaluator.get_statistics()
    print(f"\n【评估统计】")
    print(f"  - 总评估次数: {stats['total_evaluations']}")
    print(f"  - 知识更新记录: {stats['update_history']['total_updates']}")
    print(f"  - 过期清除记录: {stats['update_history']['total_clearances']}")
    
    print(f"\n完整报告已保存至: physics_cca_report.json")
    print("\n" + "=" * 60)


def evaluate_real_kg(kg_path: str):
    """评估真实知识图谱"""
    print(f"\n评估真实知识图谱: {kg_path}")
    
    evaluator = PhysicsCCAEvaluator()
    
    try:
        metrics = evaluator.evaluate_from_file(kg_path)
        recommendations = evaluator.generate_recommendations(metrics)
        
        report = generate_report(
            metrics=metrics,
            kg_metadata={"source": kg_path},
            recommendations=recommendations,
            output_path=f"{Path(kg_path).stem}_report.json"
        )
        
        report.print_summary()
        
    except FileNotFoundError:
        print(f"文件不存在: {kg_path}")
    except json.JSONDecodeError:
        print(f"JSON解析错误: {kg_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="物理实验CCA评估")
    parser.add_argument("--kg", type=str, help="知识图谱JSON文件路径")
    args = parser.parse_args()
    
    if args.kg:
        evaluate_real_kg(args.kg)
    else:
        main()
