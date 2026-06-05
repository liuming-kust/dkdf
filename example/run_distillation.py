#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DKDF 知识蒸馏运行脚本

论文贡献全覆盖运行演示：
1. HKG：层次化知识图谱构建（实体+关系+规则抽取）
2. DRD：双循环蒸馏机制（内环提纯+外环进化+价值评估V(k)）
3. CCA：认知一致性评估（OCR/PCR/PMC/CEC/KER/OKCR）
4. 全链路制导：分块→过滤→蒸馏→重组

使用前准备：
1. 启动 Ollama: ollama serve
2. 拉取模型: ollama pull qwen2.5:14b
3. 拉取嵌入模型: ollama pull nomic-embed-text

运行命令：
    python examples/run_distillation.py
    python examples/run_distillation.py --quick
    python examples/run_distillation.py --config custom_config.json
"""

import sys
import os
import json
import argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dkdf import (
    __version__,
    DEFAULT_CONFIG,
    load_config,
    setup_logger,
    calculate_tokens,
    KnowledgeGraphLoader,
    HKGProducer,
    DRDDistiller,
    CCAEvaluator,
)


class OllamaClient:
    """Ollama LLM 客户端封装"""
    
    def __init__(self, config: dict):
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("model", "qwen2.5:14b")
        self.temperature = config.get("temperature", 0.2)
        self.timeout = config.get("timeout", 120)
        self.logger = setup_logger("OllamaClient")
    
    def generate(self, prompt: str, system: str = None) -> str:
        """生成文本"""
        import requests
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            return response.json().get("message", {}).get("content", "")
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            return ""
    
    def extract_json(self, prompt: str, system: str = None) -> dict:
        """提取JSON格式响应"""
        from dkdf.utils import extract_json_from_response
        
        response = self.generate(prompt, system)
        return extract_json_from_response(response)


def check_environment():
    """检查运行环境"""
    logger = setup_logger("EnvCheck")
    
    logger.info("检查运行环境...")
    
    # 检查 Ollama
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            logger.info(f"✓ Ollama 服务运行中，可用模型: {model_names}")
            
            # 检查必要模型
            if not any("qwen2.5:14b" in m for m in model_names):
                logger.warning("⚠ qwen2.5:14b 模型未安装，请运行: ollama pull qwen2.5:14b")
            if not any("nomic-embed-text" in m for m in model_names):
                logger.warning("⚠ nomic-embed-text 模型未安装，请运行: ollama pull nomic-embed-text")
        else:
            logger.error("✗ Ollama 服务异常")
            return False
    except Exception as e:
        logger.error(f"✗ Ollama 服务未启动: {e}")
        logger.info("  请执行: ollama serve")
        return False
    
    # 检查必要的Python包
    required_packages = ["langchain_community", "langchain_text_splitters", "faiss"]
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    
    if missing:
        logger.warning(f"缺少Python包: {missing}")
        logger.info("  请执行: pip install -r requirements.txt")
    
    logger.info("环境检查完成")
    return True


def create_sample_kg(kg_path: str = "experiments.json"):
    """创建示例知识图谱文件"""
    sample_kg = {
        "experiments": [
            {
                "data": {
                    "name": "刚体转动惯量测量",
                    "purpose": "掌握用转动惯量仪测量刚体转动惯量的方法",
                    "theory": "根据刚体转动定律，转动惯量I = M/α，其中M为力矩，α为角加速度",
                    "equipment": [
                        {"name": "刚体转动仪"},
                        {"name": "数字毫秒计"},
                        {"name": "砝码"}
                    ],
                    "steps": [
                        "调节仪器水平",
                        "测量不同质量下的角加速度",
                        "计算转动惯量"
                    ]
                },
                "metadata": {
                    "pages": [23, 28]
                }
            },
            {
                "data": {
                    "name": "霍尔效应实验",
                    "purpose": "测量霍尔电压与磁场的关系",
                    "theory": "霍尔电压V_H = (R_H * I * B) / d",
                    "equipment": [
                        {"name": "霍尔元件"},
                        {"name": "电磁铁"},
                        {"name": "直流电源"},
                        {"name": "数字电压表"}
                    ],
                    "steps": [
                        "连接实验电路",
                        "调节电流大小",
                        "测量霍尔电压",
                        "绘制V_H-B曲线"
                    ]
                },
                "metadata": {
                    "pages": [45, 52]
                }
            },
            {
                "data": {
                    "name": "光的干涉实验",
                    "purpose": "观察光的干涉现象，测量波长",
                    "theory": "干涉条纹间距Δx = λL/d",
                    "equipment": [
                        {"name": "双缝装置"},
                        {"name": "激光器"},
                        {"name": "光屏"},
                        {"name": "测微目镜"}
                    ],
                    "steps": [
                        "调节光路",
                        "观察干涉条纹",
                        "测量条纹间距",
                        "计算波长"
                    ]
                },
                "metadata": {
                    "pages": [67, 75]
                }
            }
        ]
    }
    
    with open(kg_path, 'w', encoding='utf-8') as f:
        json.dump(sample_kg, f, ensure_ascii=False, indent=2)
    
    return kg_path


def create_sample_doc(doc_path: str = "sample_document.txt"):
    """创建示例文档（用于快速测试）"""
    sample_doc = """
    ========================================
    实验一：刚体转动惯量测量
    ========================================
    
    【实验目的】
    掌握用转动惯量仪测量刚体转动惯量的方法，验证转动定律。
    
    【实验原理】
    根据刚体转动定律，刚体绕固定轴转动的角加速度α与所受合外力矩M成正比，
    与转动惯量I成反比，即 M = I·α。
    
    本实验中，砝码通过细绳绕在塔轮上，产生重力矩 M = mgr。
    测得角加速度α后，可计算转动惯量 I = mgr/α。
    
    【实验仪器】
    刚体转动仪、数字毫秒计、砝码、游标卡尺、秒表。
    
    【实验步骤】
    1. 将转动仪调节至水平状态。
    2. 选择一定质量的砝码，记录其质量m。
    3. 测量塔轮半径r。
    4. 释放砝码，用数字毫秒计测量角加速度α。
    5. 改变砝码质量，重复测量3-5次。
    6. 计算转动惯量I，并与理论值比较。
    
    【数据处理】
    转动惯量计算公式：I = mgr/α
    相对误差 = |I_测量 - I_理论| / I_理论 × 100%
    
    ========================================
    实验二：霍尔效应实验
    ========================================
    
    【实验目的】
    测量霍尔电压与磁场的关系，计算霍尔系数。
    
    【实验原理】
    霍尔效应：当电流通过置于磁场中的导体时，在垂直于电流和磁场的方向上
    会产生横向电势差，称为霍尔电压：V_H = (R_H · I · B) / d
    
    【实验仪器】
    霍尔元件、电磁铁、直流电源、数字电压表、高斯计。
    
    【实验步骤】
    1. 按电路图连接实验线路。
    2. 调节励磁电流至设定值，产生磁场。
    3. 调节工作电流，测量霍尔电压。
    4. 改变磁场方向，测量反向霍尔电压。
    5. 计算霍尔系数和载流子浓度。
    """
    
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(sample_doc)
    
    return doc_path


def run_distillation(args):
    """执行完整的知识蒸馏流程"""
    logger = setup_logger("DKDF-Main", "INFO")
    
    logger.info("=" * 80)
    logger.info(f"DKDF 动态知识精炼框架 v{__version__}")
    logger.info("知识图谱全链路制导的大模型文档智能处理方案")
    logger.info("构建-蒸馏-评估 三位一体")
    logger.info("=" * 80)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    # 加载配置
    config = load_config(args.config) if args.config else DEFAULT_CONFIG
    
    # 准备文件路径
    kg_path = args.kg_path or "experiments.json"
    doc_path = args.doc_path or "sample_document.txt"
    output_path = args.output or "distilled_knowledge.json"
    
    # 检查文件是否存在，不存在则创建示例
    if not os.path.exists(kg_path):
        logger.info(f"知识图谱文件不存在，创建示例: {kg_path}")
        create_sample_kg(kg_path)
    
    if not os.path.exists(doc_path):
        logger.info(f"文档文件不存在，创建示例: {doc_path}")
        create_sample_doc(doc_path)
    
    # ==================== 步骤0：环境检查 ====================
    logger.info("[步骤0] 环境检查")
    if not check_environment():
        logger.error("环境检查失败，退出")
        return
    
    # ==================== 步骤1：加载知识图谱 ====================
    logger.info("\n[步骤1] 加载知识图谱 (KG Loader)")
    kg_loader = KnowledgeGraphLoader(kg_path)
    core_concepts = kg_loader.get_core_concepts()
    entities = kg_loader.get_entities()
    relations = kg_loader.get_relations()
    rules = kg_loader.get_rules()
    
    logger.info(f"  核心概念: {len(core_concepts)} 个")
    logger.info(f"  实体: {len(entities)} 个")
    logger.info(f"  关系: {len(relations)} 条")
    logger.info(f"  规则: {len(rules)} 条")
    
    for name, data in list(core_concepts.items())[:3]:
        logger.info(f"    - {name}: {data.get('purpose', '')[:50]}...")
    
    # ==================== 步骤2：初始化组件 ====================
    logger.info("\n[步骤2] 初始化组件")
    
    # 初始化LLM客户端
    llm_client = OllamaClient(config.get("llm", {}))
    logger.info(f"  LLM: {config.get('llm', {}).get('model', 'qwen2.5:14b')}")
    
    # 初始化HKG构建器
    hkg = HKGProducer(llm_client=llm_client, config=config)
    logger.info("  HKG: 层次化知识图谱构建器")
    
    # 初始化DRD蒸馏器
    drd = DRDDistiller(llm_client=llm_client, config=config)
    logger.info("  DRD: 双循环蒸馏器")
    
    # 初始化CCA评估器
    cca = CCAEvaluator(config=config)
    logger.info("  CCA: 认知一致性评估器")
    
    # ==================== 步骤3：读取文档 ====================
    logger.info("\n[步骤3] 读取文档")
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        doc_content = f.read()
    
    logger.info(f"  文档: {doc_path}")
    logger.info(f"  大小: {len(doc_content)} 字符, {calculate_tokens(doc_content)} tokens")
    
    # ==================== 步骤4：HKG知识图谱构建 ====================
    logger.info("\n[步骤4] HKG - 知识图谱构建")
    
    # 从文档中提取实体
    extracted_entities = hkg.extract_entities(doc_content)
    logger.info(f"  实体抽取: {len(extracted_entities)} 个")
    for e in extracted_entities[:5]:
        logger.info(f"    - {e['name']} ({e['type']})")
    
    # 从文档中提取关系
    extracted_relations = hkg.extract_relations(extracted_entities, doc_content)
    logger.info(f"  关系抽取: {len(extracted_relations)} 条")
    for r in extracted_relations[:5]:
        logger.info(f"    - {r['source_name']} → {r['target_name']} ({r['type']})")
    
    # 从关系中提取规则
    extracted_rules = hkg.extract_rules(extracted_relations)
    logger.info(f"  规则抽取: {len(extracted_rules)} 条")
    for r in extracted_rules[:3]:
        logger.info(f"    - {r['horn_clause'][:60]}...")
    
    # 获取HKG统计
    hkg_stats = hkg.get_statistics()
    logger.info(f"  HKG统计: {hkg_stats}")
    
    # ==================== 步骤5：DRD知识蒸馏 ====================
    logger.info("\n[步骤5] DRD - 知识蒸馏")
    
    # 准备原始知识（按段落分割）
    paragraphs = [p.strip() for p in doc_content.split('\n\n') if len(p.strip()) > 50]
    logger.info(f"  原始知识片段: {len(paragraphs)} 个")
    
    # 执行完整蒸馏流程
    distilled_knowledge = drd.distill(paragraphs)
    logger.info(f"  蒸馏后知识单元: {len(distilled_knowledge)} 个")
    
    for unit in distilled_knowledge[:3]:
        logger.info(f"    - [{unit.id}] V(k)={unit.value_score:.3f}: {unit.content[:80]}...")
    
    # 获取DRD统计
    drd_stats = drd.get_statistics()
    logger.info(f"  DRD统计: {drd_stats}")
    
    # ==================== 步骤6：CCA认知一致性评估 ====================
    logger.info("\n[步骤6] CCA - 认知一致性评估")
    
    # 构建评估用的知识图谱
    eval_kg = {
        "entities": hkg.entities,
        "relations": hkg.relations,
        "rules": hkg.rules,
    }
    
    # 执行评估
    metrics = cca.evaluate(eval_kg)
    
    logger.info("  评估结果:")
    logger.info(f"    【逻辑一致性】")
    logger.info(f"      - OCR (本体冲突率): {metrics.ocr:.4f} (阈值≤{cca.ocr_threshold})")
    logger.info(f"      - PCR (路径连贯性): {metrics.pcr:.4f} (阈值≥{cca.pcr_threshold})")
    logger.info(f"    【认知合理性】")
    logger.info(f"      - PMC (原型匹配度): {metrics.pmc:.4f} (阈值≥{cca.pmc_threshold})")
    logger.info(f"      - CEC (认知经济度): {metrics.cec:.4f}")
    logger.info(f"    【动态适应性】")
    logger.info(f"      - KER (知识进化率): {metrics.ker:.4f}")
    logger.info(f"      - OKCR (过时清除率): {metrics.okcr:.4f}")
    logger.info(f"    【综合评分】")
    logger.info(f"      - Overall: {metrics.overall:.4f}")
    
    # 评级
    if metrics.overall >= 0.9:
        grade = "A (优秀)"
    elif metrics.overall >= 0.8:
        grade = "B (良好)"
    elif metrics.overall >= 0.7:
        grade = "C (中等)"
    elif metrics.overall >= 0.6:
        grade = "D (及格)"
    else:
        grade = "F (不及格)"
    
    logger.info(f"      评级: {grade}")
    
    # ==================== 步骤7：保存结果 ====================
    logger.info("\n[步骤7] 保存结果")
    
    # 构建输出结果
    result = {
        "version": __version__,
        "distilled_at": datetime.now().isoformat(),
        "source_document": doc_path,
        "knowledge_graph_source": kg_path,
        "statistics": {
            "hkg": hkg_stats,
            "drd": drd_stats,
            "cca": metrics.to_dict(),
        },
        "hkg_output": {
            "entities": {k: v for k, v in list(hkg.entities.items())[:20]},
            "relations": hkg.relations[:20],
            "rules": hkg.rules[:10],
        },
        "drd_output": [
            {
                "id": u.id,
                "content": u.content,
                "value_score": u.value_score,
                "novelty": u.novelty,
                "coverage": u.coverage,
                "conflict": u.conflict,
                "entities": u.entities,
                "relations": u.relations,
            }
            for u in distilled_knowledge[:20]
        ],
        "cca_metrics": metrics.to_dict(),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"  结果保存至: {output_path}")
    
    # ==================== 结果摘要 ====================
    logger.info("\n" + "=" * 80)
    logger.info("知识蒸馏完成!")
    logger.info("=" * 80)
    logger.info(f"\n【结果摘要】")
    logger.info(f"  输入: {len(paragraphs)} 个段落")
    logger.info(f"  HKG: {hkg_stats['entity_count']}实体, {hkg_stats['relation_count']}关系, {hkg_stats['rule_count']}规则")
    logger.info(f"  DRD: {len(distilled_knowledge)} 个精炼知识单元")
    logger.info(f"  CCA综合评分: {metrics.overall:.4f} ({grade})")
    logger.info(f"  总耗时: {drd_stats.get('inner_loops', 0)} 内环, {drd_stats.get('outer_loops', 0)} 外环")
    logger.info(f"\n输出文件: {output_path}")
    logger.info("=" * 80)
    
    return result


def run_quick_test(args):
    """快速测试模式（不依赖PDF和Ollama）"""
    logger = setup_logger("QuickTest", "INFO")
    
    logger.info("=" * 60)
    logger.info("DKDF 快速测试模式")
    logger.info("=" * 60)
    
    # 创建测试文件
    kg_path = "test_kg.json"
    doc_path = "test_doc.txt"
    output_path = args.output or "test_output.json"
    
    create_sample_kg(kg_path)
    create_sample_doc(doc_path)
    
    # 加载知识图谱
    logger.info("\n[测试1] 知识图谱加载")
    kg_loader = KnowledgeGraphLoader(kg_path)
    logger.info(f"  ✓ 加载 {len(kg_loader.get_core_concepts())} 个核心概念")
    logger.info(f"  ✓ 加载 {len(kg_loader.get_entities())} 个实体")
    
    # 测试HKG（不使用LLM）
    logger.info("\n[测试2] HKG实体抽取（规则模式）")
    hkg = HKGProducer(llm_client=None)
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    entities = hkg.extract_entities(content)
    logger.info(f"  ✓ 抽取 {len(entities)} 个实体")
    for e in entities[:5]:
        logger.info(f"    - {e['name']} ({e['type']})")
    
    # 测试DRD
    logger.info("\n[测试3] DRD蒸馏（简化版）")
    drd = DRDDistiller(llm_client=None)
    
    paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 30]
    logger.info(f"  ✓ 输入 {len(paragraphs)} 个段落")
    
    # 创建知识单元
    from dkdf.drd_distiller import KnowledgeUnit
    units = []
    for i, p in enumerate(paragraphs[:5]):
        unit = KnowledgeUnit(id=f"test_{i}", content=p)
        unit.value_score = 0.7
        units.append(unit)
    
    logger.info(f"  ✓ 创建 {len(units)} 个知识单元")
    
    # 测试CCA
    logger.info("\n[测试4] CCA评估")
    cca = CCAEvaluator()
    
    eval_kg = {
        "entities": hkg.entities,
        "relations": hkg.relations,
        "rules": hkg.rules,
    }
    metrics = cca.evaluate(eval_kg)
    logger.info(f"  ✓ CCA综合评分: {metrics.overall:.4f}")
    
    # 保存结果
    result = {
        "mode": "quick_test",
        "hkg_entity_count": len(entities),
        "cca_overall": metrics.overall,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n测试结果保存至: {output_path}")
    logger.info("\n快速测试完成！")
    
    # 清理测试文件（可选）
    if not args.keep_files:
        os.remove(kg_path)
        os.remove(doc_path)
        logger.info("已清理测试文件")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="DKDF 知识蒸馏工具 - 构建-蒸馏-评估三位一体",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 完整运行（需要Ollama）
    python run_distillation.py
    
    # 快速测试（不依赖Ollama）
    python run_distillation.py --quick
    
    # 指定文件
    python run_distillation.py --kg my_kg.json --doc my_doc.pdf --output result.json
    
    # 使用自定义配置
    python run_distillation.py --config my_config.json
        """
    )
    
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="快速测试模式（不依赖Ollama）"
    )
    parser.add_argument(
        "--kg", "-k",
        dest="kg_path",
        help="知识图谱JSON文件路径"
    )
    parser.add_argument(
        "--doc", "-d",
        dest="doc_path",
        help="待处理文档路径（支持txt/pdf）"
    )
    parser.add_argument(
        "--output", "-o",
        help="输出结果JSON路径"
    )
    parser.add_argument(
        "--config", "-c",
        help="配置文件路径"
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="快速测试模式下保留临时文件"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test(args)
    else:
        run_distillation(args)


if __name__ == "__main__":
    main()
