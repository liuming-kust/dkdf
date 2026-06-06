#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
物理实验知识图谱完整处理流程

流程：
1. 从PDF提取实验信息（PhysicsKGExtractor）
2. 转换为DKRF标准格式（PhysicsToDKRFConverter）
3. 使用DKRF框架进行评估（可选）
"""

import sys
import os
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.physics_kg_extractor import PhysicsKGExtractor
from examples.convert_physics_to_dkrf import PhysicsToDKRFConverter


def run_extraction(pdf_path: str, output_json: str = "experiments.json"):
    """步骤1：从PDF提取实验信息"""
    print("\n" + "=" * 60)
    print("步骤1：从PDF提取实验信息")
    print("=" * 60)
    
    extractor = PhysicsKGExtractor(output_path=output_json)
    experiments = extractor.process_pdf(pdf_path)
    
    print(f"\n提取完成，共 {len(experiments)} 个实验")
    print(f"结果保存至: {output_json}")
    
    return experiments


def run_conversion(input_json: str, output_json: str = "dkrf_physics_kg.json"):
    """步骤2：转换为DKRF格式"""
    print("\n" + "=" * 60)
    print("步骤2：转换为DKRF标准格式")
    print("=" * 60)
    
    converter = PhysicsToDKRFConverter(input_json)
    converter.save_dkrf_format(output_json)
    
    # 打印统计信息
    with open(output_json, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    stats = kg_data.get("statistics", {})
    print(f"\n转换统计:")
    print(f"  - 实验数: {stats.get('experiments', 0)}")
    print(f"  - 仪器实体: {stats.get('instruments', 0)}")
    print(f"  - 步骤实体: {stats.get('steps', 0)}")
    print(f"  - 关系数: {stats.get('relations', 0)}")
    print(f"  - 规则数: {stats.get('rules', 0)}")
    
    return kg_data


def run_dkrf_evaluation(kg_path: str):
    """步骤3：使用DKRF框架评估（可选）"""
    print("\n" + "=" * 60)
    print("步骤3：DKRF认知一致性评估")
    print("=" * 60)
    
    try:
        from dkrf.cca_evaluator import CCAEvaluator
        
        # 加载知识图谱
        with open(kg_path, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)
        
        # 创建CCA评估器
        evaluator = CCAEvaluator()
        
        # 执行评估
        metrics = evaluator.evaluate(kg_data)
        
        print(f"\nCCA评估结果:")
        print(f"  - 本体冲突率 (OCR): {metrics.ocr:.4f}")
        print(f"  - 路径连贯性 (PCR): {metrics.pcr:.4f}")
        print(f"  - 原型匹配度 (PMC): {metrics.pmc:.4f}")
        print(f"  - 认知经济度 (CEC): {metrics.cec:.4f}")
        print(f"  - 知识演化率 (KER): {metrics.ker:.4f}")
        print(f"  - 过时清除率 (OKCR): {metrics.okcr:.4f}")
        print(f"  - 综合评分: {metrics.overall:.4f}")
        
        # 生成报告
        report = evaluator.get_evaluation_report(metrics)
        print(f"\n评级: {report['summary']['grade']}")
        
        if report['recommendations']:
            print(f"\n改进建议:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        return metrics
        
    except ImportError as e:
        print(f"\n无法导入DKRF模块: {e}")
        print("请确保已安装dkrf包: pip install -e .")
        return None
    except Exception as e:
        print(f"\n评估失败: {e}")
        return None


def generate_summary(extraction_json: str, dkrf_json: str):
    """生成处理摘要"""
    print("\n" + "=" * 60)
    print("处理摘要")
    print("=" * 60)
    
    # 读取提取结果
    with open(extraction_json, 'r', encoding='utf-8') as f:
        extraction = json.load(f)
    
    # 读取DKRF格式结果
    with open(dkrf_json, 'r', encoding='utf-8') as f:
        dkrf = json.load(f)
    
    metadata = extraction.get("metadata", {})
    stats = dkrf.get("statistics", {})
    
    print(f"\n源文件: {metadata.get('source', '未知')}")
    print(f"生成时间: {metadata.get('generated_at', '未知')}")
    print(f"处理耗时: {metadata.get('processing_time', '未知')}")
    print(f"\n提取统计:")
    print(f"  - 原始实验数: {metadata.get('experiment_count', 0)}")
    print(f"\n转换统计:")
    print(f"  - 实体总数: {stats.get('experiments', 0) + stats.get('instruments', 0) + stats.get('steps', 0)}")
    print(f"    - 概念实体: {stats.get('experiments', 0)}")
    print(f"    - 仪器实体: {stats.get('instruments', 0)}")
    print(f"    - 步骤实体: {stats.get('steps', 0)}")
    print(f"  - 关系总数: {stats.get('relations', 0)}")
    print(f"  - 规则总数: {stats.get('rules', 0)}")


def main():
    """主流程"""
    print("\n" + "=" * 60)
    print("物理实验知识图谱 → DKRF框架 完整处理流程")
    print("=" * 60)
    
    # 配置路径
    pdf_path = "基础文件/大学物理实验教程（无目录）.pdf"
    extraction_json = "experiments.json"
    dkrf_json = "dkrf_physics_kg.json"
    
    # 检查PDF是否存在
    if not os.path.exists(pdf_path):
        print(f"\n错误: PDF文件不存在 - {pdf_path}")
        print("请修改pdf_path变量为实际路径")
        return
    
    # 步骤1：提取
    run_extraction(pdf_path, extraction_json)
    
    # 步骤2：转换
    run_conversion(extraction_json, dkrf_json)
    
    # 步骤3：DKRF评估（可选）
    run_dkrf_evaluation(dkrf_json)
    
    # 生成摘要
    generate_summary(extraction_json, dkrf_json)
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
