# -*- coding: utf-8 -*-
"""
DKDF 主蒸馏器 - 整合全链路

论文贡献全覆盖：
1. HKG：层次化知识图谱构建（实体+关系+规则抽取）
2. DRD：双循环蒸馏机制（内环提纯+外环进化+价值评估V(k)）
3. CCA：认知一致性评估（OCR/PCR/PMC/CEC/KER/OKCR）
4. 全链路制导：分块→过滤→蒸馏→重组（图谱贯穿全流程）
"""

import os
import time
import json
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

from .utils import (
    setup_logger, calculate_tokens, clean_text,
    text_similarity, extract_json_from_response
)
from .kg_loader import KnowledgeGraphLoader
from .hkg_builder import HKGProducer
from .drd_distiller import DRDDistiller, KnowledgeUnit
from .cca_evaluator import CCAEvaluator, CCAMetrics


@dataclass
class DistillationResult:
    """蒸馏结果"""
    # 输入统计
    source_file: str = ""
    raw_chunks: int = 0
    raw_tokens: int = 0
    
    # HKG输出
    hkg_entities: int = 0
    hkg_relations: int = 0
    hkg_rules: int = 0
    
    # DRD输出
    drd_units: int = 0
    drd_kept: int = 0
    drd_value_scores: List[float] = field(default_factory=list)
    
    # CCA输出
    cca_metrics: Optional[CCAMetrics] = None
    
    # 时间统计
    total_time: float = 0.0
    hkg_time: float = 0.0
    drd_time: float = 0.0
    cca_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "source_file": self.source_file,
            "raw_chunks": self.raw_chunks,
            "raw_tokens": self.raw_tokens,
            "hkg": {
                "entities": self.hkg_entities,
                "relations": self.hkg_relations,
                "rules": self.hkg_rules,
                "time": round(self.hkg_time, 2),
            },
            "drd": {
                "units": self.drd_units,
                "kept": self.drd_kept,
                "avg_value_score": round(sum(self.drd_value_scores) / max(len(self.drd_value_scores), 1), 4),
                "time": round(self.drd_time, 2),
            },
            "cca": self.cca_metrics.to_dict() if self.cca_metrics else {},
            "total_time": round(self.total_time, 2),
        }


class DKDFDistiller:
    """
    DKDF 主蒸馏器
    
    整合 HKG + DRD + CCA 全链路流程
    实现知识图谱全链路制导的文档精炼
    """
    
    def __init__(
        self,
        kg_path: str = "experiments.json",
        config: Dict = None,
        llm_client = None
    ):
        """
        初始化蒸馏器
        
        Args:
            kg_path: 知识图谱文件路径
            config: 配置字典
            llm_client: LLM客户端（Ollama等）
        """
        self.config = config or {}
        self.logger = setup_logger("DKDFDistiller")
        
        # 加载知识图谱
        self.logger.info(f"加载知识图谱: {kg_path}")
        self.kg_loader = KnowledgeGraphLoader(kg_path)
        
        # 初始化三个核心模块
        self.hkg = HKGProducer(llm_client=llm_client, config=config)
        self.drd = DRDDistiller(llm_client=llm_client, config=config)
        self.cca = CCAEvaluator(config=config)
        
        # LLM客户端（供外部使用）
        self.llm_client = llm_client
        
        # 处理配置
        proc_config = config.get("processing", {}) if config else {}
        self.max_tokens = proc_config.get("max_tokens", 1024)
        self.chunk_size = proc_config.get("chunk_size", 2000)
        self.chunk_overlap = proc_config.get("chunk_overlap", 400)
        self.target_compression = proc_config.get("target_compression", 0.3)
        
        self.logger.info("DKDF蒸馏器初始化完成")
    
    # ==================== 1. 文档分块（图谱制导） ====================
    
    def _kg_guided_chunking(self, text: str) -> List[str]:
        """
        知识图谱制导的文档分块
        
        使用知识图谱中的实体和概念作为动态分隔符
        """
        # 获取核心概念和实体
        core_concepts = self.kg_loader.get_core_concepts()
        all_entities = self.kg_loader.get_all_entity_names()
        
        # 构建动态分隔符
        separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        
        # 添加概念名称作为分隔标记
        for concept in core_concepts.keys():
            if concept and len(concept) > 2:
                separators.insert(0, concept)
        
        # 添加实体名称作为分隔标记
        for entity in all_entities[:20]:  # 限制数量
            if entity and len(entity) > 2:
                separators.insert(0, entity)
        
        # 去重
        separators = list(dict.fromkeys(separators))
        
        # 分块
        chunks = []
        current_chunk = ""
        
        for line in text.split('\n'):
            # 检查是否遇到分隔符
            is_separator = False
            for sep in separators[:10]:  # 只检查前10个
                if sep and sep in line and len(sep) > 2:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    is_separator = True
                    break
            
            # 添加到当前块
            if len(current_chunk) + len(line) < self.chunk_size:
                current_chunk += line + "\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # 过滤过短的块
        chunks = [c for c in chunks if len(c) > 50]
        
        self.logger.info(f"图谱制导分块: {len(chunks)} 个文本块")
        return chunks
    
    # ==================== 2. 概念相关性过滤 ====================
    
    def _concept_filtering(self, chunks: List[str]) -> List[tuple]:
        """
        概念相关性过滤
        
        只保留与知识图谱中核心概念相关的文本块
        返回 (chunk, related_concepts) 列表
        """
        filtered = []
        
        for chunk in chunks:
            # 查找相关概念
            related = self.kg_loader.find_related_concepts(chunk)
            
            if related:
                filtered.append((chunk, related))
                self.logger.debug(f"保留块: 相关概念 {related}")
            else:
                self.logger.debug(f"过滤块: 无相关概念")
        
        self.logger.info(f"概念过滤: {len(filtered)}/{len(chunks)} 保留")
        return filtered
    
    # ==================== 3. 知识图谱增强蒸馏 ====================
    
    def _kg_enhanced_distill(self, chunk: str, related_concepts: Set[str]) -> str:
        """
        知识图谱增强蒸馏
        
        使用知识图谱上下文增强蒸馏提示词
        """
        # 获取概念上下文
        context = self.kg_loader.get_context_for_concepts(related_concepts)
        
        # 构建蒸馏提示
        prompt = f"""你是一个教育领域知识精炼专家。请基于知识图谱上下文，从原始文本中提取核心知识。

## 知识图谱上下文（相关概念）
{context}

## 蒸馏要求
1. 只保留与上述概念直接相关的内容
2. 去除冗余描述、重复说明
3. 按「核心原理→关键步骤→重要参数」的逻辑组织
4. 输出长度控制在原文的30%以内
5. 使用中文输出

## 原始文本
{chunk}

## 蒸馏输出
"""
        
        if self.llm_client:
            try:
                distilled = self.llm_client.generate(prompt)
                return distilled.strip()
            except Exception as e:
                self.logger.warning(f"LLM蒸馏失败: {e}")
        
        # 回退：简单提取
        return self._simple_extract(chunk, related_concepts)
    
    def _simple_extract(self, chunk: str, related_concepts: Set[str]) -> str:
        """简单提取（无LLM时的回退方案）"""
        lines = chunk.split('\n')
        extracted = []
        
        for line in lines:
            # 保留包含相关概念的句子
            for concept in related_concepts:
                if concept in line:
                    extracted.append(line)
                    break
            
            # 保留包含关键词的句子
            keywords = ["原理", "步骤", "目的", "方法", "数据", "结果"]
            for kw in keywords:
                if kw in line:
                    extracted.append(line)
                    break
        
        result = "\n".join(extracted)
        
        # 控制长度
        if len(result) > len(chunk) * self.target_compression:
            result = result[:int(len(chunk) * self.target_compression)]
        
        return result if result else chunk[:500]
    
    # ==================== 4. 蒸馏主流程 ====================
    
    def distill_document(self, text: str) -> DistillationResult:
        """
        对单个文档执行完整蒸馏流程
        
        Args:
            text: 文档文本内容
        
        Returns:
            蒸馏结果
        """
        start_time = time.time()
        result = DistillationResult()
        
        # 统计原始输入
        result.raw_tokens = calculate_tokens(text)
        self.logger.info(f"开始蒸馏，文本长度: {len(text)}字符, {result.raw_tokens}tokens")
        
        # ===== 步骤1：HKG 知识图谱构建 =====
        hkg_start = time.time()
        self.logger.info("=" * 50)
        self.logger.info("[HKG] 层次化知识图谱构建")
        
        # 实体抽取
        entities = self.hkg.extract_entities(text)
        result.hkg_entities = len(entities)
        self.logger.info(f"  实体抽取: {result.hkg_entities} 个")
        
        # 关系抽取
        relations = self.hkg.extract_relations(entities, text)
        result.hkg_relations = len(relations)
        self.logger.info(f"  关系抽取: {result.hkg_relations} 条")
        
        # 规则抽取
        rules = self.hkg.extract_rules(relations)
        result.hkg_rules = len(rules)
        self.logger.info(f"  规则抽取: {result.hkg_rules} 条")
        
        result.hkg_time = time.time() - hkg_start
        
        # ===== 步骤2：图谱制导分块 =====
        chunks = self._kg_guided_chunking(text)
        result.raw_chunks = len(chunks)
        self.logger.info(f"  文档分块: {result.raw_chunks} 个")
        
        # ===== 步骤3：概念相关性过滤 =====
        filtered_chunks = self._concept_filtering(chunks)
        
        # ===== 步骤4：知识图谱增强蒸馏 =====
        drd_start = time.time()
        self.logger.info("=" * 50)
        self.logger.info("[DRD] 双循环蒸馏")
        
        # 对每个相关块进行蒸馏
        distilled_units = []
        for chunk, concepts in filtered_chunks:
            distilled_content = self._kg_enhanced_distill(chunk, concepts)
            
            if distilled_content and len(distilled_content) > 20:
                unit = KnowledgeUnit(
                    id=f"distilled_{len(distilled_units)}",
                    content=distilled_content,
                    entities=list(concepts),
                )
                distilled_units.append(unit)
        
        self.logger.info(f"  蒸馏片段: {len(distilled_units)} 个")
        
        # ===== 步骤5：DRD 内环蒸馏 =====
        inner_result = self.drd.inner_loop_distill(distilled_units)
        self.logger.info(f"  内环蒸馏后: {len(inner_result)} 个")
        
        # ===== 步骤6：DRD 外环进化（价值评估） =====
        final_units = self.drd.outer_loop_evolve(inner_result)
        result.drd_units = len(final_units)
        result.drd_kept = len([u for u in final_units if u.value_score >= 0.6])
        result.drd_value_scores = [u.value_score for u in final_units]
        
        avg_score = sum(result.drd_value_scores) / max(len(result.drd_value_scores), 1)
        self.logger.info(f"  外环进化后: {result.drd_units} 个")
        self.logger.info(f"  平均价值评分: {avg_score:.4f}")
        
        result.drd_time = time.time() - drd_start
        
        # ===== 步骤7：CCA 认知一致性评估 =====
        cca_start = time.time()
        self.logger.info("=" * 50)
        self.logger.info("[CCA] 认知一致性评估")
        
        # 构建评估用知识图谱
        eval_kg = {
            "entities": self.hkg.entities,
            "relations": self.hkg.relations,
            "rules": self.hkg.rules,
        }
        
        result.cca_metrics = self.cca.evaluate(eval_kg)
        self.logger.info(f"  综合评分: {result.cca_metrics.overall:.4f}")
        
        result.cca_time = time.time() - cca_start
        result.total_time = time.time() - start_time
        
        # 输出摘要
        self.logger.info("=" * 50)
        self.logger.info("蒸馏完成!")
        self.logger.info(f"  总耗时: {result.total_time:.2f}秒")
        self.logger.info(f"  HKG: {result.hkg_entities}实体, {result.hkg_relations}关系, {result.hkg_rules}规则")
        self.logger.info(f"  DRD: {result.drd_units}知识单元, 平均V(k)={avg_score:.4f}")
        self.logger.info(f"  CCA: {result.cca_metrics.overall:.4f}")
        self.logger.info("=" * 50)
        
        return result
    
    # ==================== 5. 批量处理 ====================
    
    def distill_batch(
        self,
        texts: List[str],
        source_names: List[str] = None
    ) -> List[DistillationResult]:
        """
        批量蒸馏多个文档
        
        Args:
            texts: 文档文本列表
            source_names: 文档名称列表（可选）
        
        Returns:
            蒸馏结果列表
        """
        results = []
        
        for i, text in enumerate(texts):
            name = source_names[i] if source_names else f"doc_{i}"
            self.logger.info(f"\n处理文档: {name}")
            
            result = self.distill_document(text)
            result.source_file = name
            results.append(result)
        
        # 汇总统计
        if len(results) > 1:
            self._print_batch_summary(results)
        
        return results
    
    def _print_batch_summary(self, results: List[DistillationResult]):
        """打印批量处理摘要"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("批量处理摘要")
        self.logger.info("=" * 60)
        
        avg_entities = sum(r.hkg_entities for r in results) / len(results)
        avg_relations = sum(r.hkg_relations for r in results) / len(results)
        avg_rules = sum(r.hkg_rules for r in results) / len(results)
        avg_units = sum(r.drd_units for r in results) / len(results)
        avg_cca = sum(r.cca_metrics.overall for r in results) / len(results)
        total_time = sum(r.total_time for r in results)
        
        self.logger.info(f"  文档数: {len(results)}")
        self.logger.info(f"  平均HKG: {avg_entities:.1f}实体, {avg_relations:.1f}关系, {avg_rules:.1f}规则")
        self.logger.info(f"  平均DRD: {avg_units:.1f}知识单元")
        self.logger.info(f"  平均CCA: {avg_cca:.4f}")
        self.logger.info(f"  总耗时: {total_time:.2f}秒")
        self.logger.info("=" * 60)
    
    # ==================== 6. 结果保存 ====================
    
    def save_result(self, result: DistillationResult, output_path: str):
        """保存蒸馏结果到JSON文件"""
        output = {
            "result": result.to_dict(),
            "hkg_entities": {k: v for k, v in list(self.hkg.entities.items())[:50]},
            "hkg_relations": self.hkg.relations[:50],
            "hkg_rules": self.hkg.rules[:20],
            "drd_knowledge": [
                {
                    "id": u.id,
                    "content": u.content,
                    "value_score": u.value_score,
                    "entities": u.entities,
                }
                for u in list(self.drd.knowledge_base.values())[:50]
            ],
            "cca_metrics": result.cca_metrics.to_dict() if result.cca_metrics else {},
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"结果保存至: {output_path}")
    
    # ==================== 7. 获取统计信息 ====================
    
    def get_statistics(self) -> Dict:
        """获取所有模块的统计信息"""
        return {
            "hkg": self.hkg.get_statistics(),
            "drd": self.drd.get_statistics(),
            "kg_loader": {
                "core_concepts": len(self.kg_loader.get_core_concepts()),
                "entities": len(self.kg_loader.get_entities()),
                "relations": len(self.kg_loader.get_relations()),
                "rules": len(self.kg_loader.get_rules()),
            }
        }
    
    # ==================== 8. 便捷方法 ====================
    
    def distill_file(
        self,
        file_path: str,
        output_path: str = None
    ) -> DistillationResult:
        """
        从文件读取并蒸馏
        
        Args:
            file_path: 文件路径（支持txt）
            output_path: 输出路径（可选）
        
        Returns:
            蒸馏结果
        """
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 执行蒸馏
        result = self.distill_document(text)
        result.source_file = file_path
        
        # 保存结果
        if output_path:
            self.save_result(result, output_path)
        
        return result
    
    def query_knowledge(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        查询蒸馏后的知识
        
        Args:
            query: 查询文本
            top_k: 返回数量
        
        Returns:
            相关知识列表
        """
        # 基于价值评分排序
        sorted_units = sorted(
            self.drd.knowledge_base.values(),
            key=lambda u: u.value_score,
            reverse=True
        )
        
        # 简单的关键词匹配
        results = []
        query_words = set(query.split())
        
        for unit in sorted_units[:top_k * 3]:
            # 计算匹配度
            unit_words = set(unit.content.split())
            if query_words and unit_words:
                similarity = len(query_words & unit_words) / len(query_words | unit_words)
            else:
                similarity = 0
            
            if similarity > 0.1 or len(results) < top_k:
                results.append({
                    "content": unit.content,
                    "value_score": unit.value_score,
                    "entities": unit.entities,
                    "similarity": similarity,
                })
        
        return results[:top_k]


# ==================== 便捷函数 ====================

def create_distiller(
    kg_path: str = "experiments.json",
    config_path: str = None,
    llm_client = None
) -> DKDFDistiller:
    """
    创建蒸馏器的便捷函数
    
    Args:
        kg_path: 知识图谱路径
        config_path: 配置文件路径
        llm_client: LLM客户端
    
    Returns:
        DKDFDistiller实例
    """
    config = None
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    return DKDFDistiller(
        kg_path=kg_path,
        config=config,
        llm_client=llm_client
    )


def quick_distill(text: str, kg_path: str = "experiments.json") -> DistillationResult:
    """
    快速蒸馏一段文本
    
    Args:
        text: 文本内容
        kg_path: 知识图谱路径
    
    Returns:
        蒸馏结果
    """
    distiller = DKDFDistiller(kg_path=kg_path)
    return distiller.distill_document(text)
