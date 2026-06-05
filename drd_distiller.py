# -*- coding: utf-8 -*-
"""
DRD: 双循环蒸馏机制
论文贡献：
- 内环：噪声过滤 + 语义增强 + 分块重组（实时提纯）
- 外环：价值评估V(k) + 高价值保留 + 过时淘汰（增量进化）
"""

import time
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field

from .utils import (
    setup_logger, calculate_tokens, clean_text,
    text_similarity, extract_json_from_response
)


@dataclass
class KnowledgeUnit:
    """知识单元"""
    id: str
    content: str
    entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    novelty: float = 0.0          # 新颖性
    coverage: float = 0.0         # 覆盖率
    conflict: float = 0.0         # 冲突度
    value_score: float = 0.0      # 综合价值 V(k)
    timestamp: float = field(default_factory=time.time)
    source: str = "extracted"


class DRDDistiller:
    """
    双循环蒸馏器
    
    内环（Inner Loop）：实时提纯与重组
    外环（Outer Loop）：增量进化与反馈
    """
    
    # 蒸馏系统提示词
    DISTILL_SYSTEM = """你是教育领域知识精炼专家。提取核心知识，去除冗余。"""

    def __init__(self, llm_client=None, config: Dict = None):
        self.llm = llm_client
        self.config = config or {}
        self.logger = setup_logger("DRD")
        
        # 知识库
        self.knowledge_base: Dict[str, KnowledgeUnit] = {}
        
        # DRD配置
        drd_config = self.config.get("drd", {})
        
        # 内环配置
        inner = drd_config.get("inner_loop", {})
        self.min_content_length = inner.get("min_content_length", 20)
        self.conflict_threshold = inner.get("conflict_threshold", 0.3)
        self.similarity_threshold = inner.get("similarity_threshold", 0.85)
        
        # 外环配置
        outer = drd_config.get("outer_loop", {})
        self.retain_threshold = outer.get("retain_threshold", 0.6)
        self.expire_days = outer.get("expire_days", 30)
        self.low_value_threshold = outer.get("low_value_threshold", 0.4)
        self.update_interval = outer.get("update_interval", 3600)
        self.value_weights = outer.get("value_weights", {
            "novelty": 0.4,
            "coverage": 0.3,
            "conflict": 0.3,
        })
        
        # 统计
        self.stats = {
            "inner_loops": 0,
            "outer_loops": 0,
            "noise_removed": 0,
            "units_added": 0,
            "units_removed": 0,
            "llm_calls": 0,
        }
        
        self.last_update_time = time.time()
    
    # ==================== 内环蒸馏 ====================
    
    def inner_loop_distill(self, knowledge_units: List[KnowledgeUnit]) -> List[KnowledgeUnit]:
        """
        内环蒸馏：实时提纯与重组
        
        步骤：
        1. 噪声过滤
        2. 语义增强
        3. 分块重组
        """
        self.logger.info(f"内环蒸馏开始，输入: {len(knowledge_units)}")
        start_time = time.time()
        
        # 步骤1：噪声过滤
        filtered = self._noise_filtering(knowledge_units)
        self.stats["noise_removed"] = len(knowledge_units) - len(filtered)
        self.logger.info(f"噪声过滤: 移除 {self.stats['noise_removed']} 个")
        
        # 步骤2：语义增强
        enhanced = self._semantic_enhancement(filtered)
        
        # 步骤3：分块重组
        reorganized = self._chunk_reorganization(enhanced)
        
        self.stats["inner_loops"] += 1
        elapsed = time.time() - start_time
        self.logger.info(f"内环蒸馏完成，输出: {len(reorganized)}，耗时: {elapsed:.2f}s")
        
        return reorganized
    
    def _noise_filtering(self, units: List[KnowledgeUnit]) -> List[KnowledgeUnit]:
        """噪声过滤：剔除低质量和冲突知识"""
        filtered = []
        
        for unit in units:
            # 长度过滤
            if len(unit.content) < self.min_content_length:
                self.logger.debug(f"过滤过短单元: {unit.id}")
                continue
            
            # 冲突检测
            if self._has_conflict(unit):
                self.logger.debug(f"过滤冲突单元: {unit.id}")
                continue
            
            filtered.append(unit)
        
        return filtered
    
    def _has_conflict(self, unit: KnowledgeUnit) -> bool:
        """检查与已有知识的冲突"""
        for existing in self.knowledge_base.values():
            similarity = text_similarity(unit.content, existing.content)
            
            # 高相似度但内容不同 -> 冲突
            if similarity > 0.6 and unit.content != existing.content:
                common_entities = set(unit.entities) & set(existing.entities)
                if len(common_entities) > 0:
                    return True
        return False
    
    def _semantic_enhancement(self, units: List[KnowledgeUnit]) -> List[KnowledgeUnit]:
        """语义增强（可选LLM精炼）"""
        if not self.llm:
            return units
        
        enhanced = []
        for unit in units[:20]:  # 限制LLM调用
            try:
                refined = self._refine_with_llm(unit.content)
                if refined and len(refined) > 10:
                    unit.content = refined
                    self.stats["llm_calls"] += 1
            except Exception as e:
                self.logger.warning(f"LLM精炼失败: {e}")
            enhanced.append(unit)
        
        return enhanced
    
    def _refine_with_llm(self, content: str) -> Optional[str]:
        """使用LLM精炼知识"""
        if not self.llm or len(content) > 2000:
            return None
        
        prompt = f"请精炼以下知识，保留核心：\n\n{content}"
        
        try:
            return self.llm.generate(prompt, system=self.DISTILL_SYSTEM)
        except Exception as e:
            self.logger.warning(f"精炼失败: {e}")
            return None
    
    def _chunk_reorganization(self, units: List[KnowledgeUnit]) -> List[KnowledgeUnit]:
        """分块重组：基于相似度合并"""
        if not units:
            return units
        
        merged = []
        used = set()
        
        for i, unit in enumerate(units):
            if i in used:
                continue
            
            cluster = [unit]
            for j, other in enumerate(units[i+1:], i+1):
                if j in used:
                    continue
                
                # 实体重叠合并
                common = set(unit.entities) & set(other.entities)
                # 内容相似度合并
                sim = text_similarity(unit.content, other.content)
                
                if len(common) > 0 or sim > self.similarity_threshold:
                    cluster.append(other)
                    used.add(j)
            
            if len(cluster) > 1:
                merged.append(self._merge_units(cluster))
            else:
                merged.append(unit)
        
        return merged
    
    def _merge_units(self, units: List[KnowledgeUnit]) -> KnowledgeUnit:
        """合并多个知识单元"""
        combined_content = "\n\n".join([u.content for u in units])
        
        all_entities = []
        for u in units:
            all_entities.extend(u.entities)
        all_entities = list(set(all_entities))
        
        all_relations = []
        for u in units:
            all_relations.extend(u.relations)
        all_relations = list(set(all_relations))
        
        avg_value = sum(u.value_score for u in units) / len(units)
        
        return KnowledgeUnit(
            id=f"merged_{units[0].id}",
            content=combined_content,
            entities=all_entities,
            relations=all_relations,
            value_score=avg_value,
            timestamp=max(u.timestamp for u in units)
        )
    
    # ==================== 外环进化 ====================
    
    def outer_loop_evolve(self, new_knowledge: List[KnowledgeUnit]) -> List[KnowledgeUnit]:
        """
        外环进化：增量更新与反馈
        
        核心公式：V(k) = ω1·新颖性 + ω2·覆盖率 - ω3·冲突度
        """
        self.logger.info(f"外环进化开始，新知识: {len(new_knowledge)}")
        start_time = time.time()
        
        # 检查更新间隔
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval and self.knowledge_base:
            self.logger.debug("未到更新间隔，跳过")
            return list(self.knowledge_base.values())
        
        # 步骤1：价值评估 V(k)
        for unit in new_knowledge:
            unit.novelty = self._calculate_novelty(unit)
            unit.coverage = self._calculate_coverage(unit)
            unit.conflict = self._calculate_conflict(unit)
            unit.value_score = self._evaluate_knowledge_value(unit)
            
            self.logger.debug(
                f"价值评估 {unit.id}: "
                f"新颖={unit.novelty:.3f}, "
                f"覆盖={unit.coverage:.3f}, "
                f"冲突={unit.conflict:.3f}, "
                f"V(k)={unit.value_score:.3f}"
            )
        
        # 步骤2：筛选高价值知识（V(k) >= 阈值）
        valuable = [u for u in new_knowledge if u.value_score >= self.retain_threshold]
        
        # 步骤3：更新知识库
        for unit in valuable:
            self.knowledge_base[unit.id] = unit
            self.stats["units_added"] += 1
        
        # 步骤4：淘汰过时低价值知识
        self._remove_outdated()
        
        self.last_update_time = current_time
        self.stats["outer_loops"] += 1
        
        elapsed = time.time() - start_time
        self.logger.info(
            f"外环进化完成: 新增 {len(valuable)}/{len(new_knowledge)}, "
            f"移除 {self.stats['units_removed']}, 耗时 {elapsed:.2f}s"
        )
        
        return list(self.knowledge_base.values())
    
    def _calculate_novelty(self, unit: KnowledgeUnit) -> float:
        """
        新颖性计算：与已有知识的最大差异度
        novelty = 1 - max_similarity
        """
        if not self.knowledge_base:
            return 1.0
        
        max_sim = 0.0
        for existing in self.knowledge_base.values():
            sim = text_similarity(unit.content, existing.content)
            max_sim = max(max_sim, sim)
        
        return 1.0 - max_sim
    
    def _calculate_coverage(self, unit: KnowledgeUnit) -> float:
        """
        覆盖率计算：新实体占已有实体的比例
        coverage = |新实体 ∩ 已有实体| / |已有实体|
        """
        if not self.knowledge_base:
            return 0.5
        
        all_entities = set()
        for existing in self.knowledge_base.values():
            all_entities.update(existing.entities)
        
        if not all_entities:
            return 0.5
        
        new_entities = set(unit.entities)
        overlap = len(new_entities & all_entities)
        
        return min(1.0, overlap / len(all_entities))
    
    def _calculate_conflict(self, unit: KnowledgeUnit) -> float:
        """
        冲突度计算：与已有知识的矛盾程度
        """
        conflict_score = 0.0
        count = 0
        
        for existing in self.knowledge_base.values():
            # 相反结论检测
            if self._has_opposite(unit.content, existing.content):
                conflict_score += 0.8
                count += 1
            
            # 实体冲突
            common = set(unit.entities) & set(existing.entities)
            if common:
                conflict_score += 0.5
                count += 1
        
        return conflict_score / max(count, 1)
    
    def _has_opposite(self, content1: str, content2: str) -> bool:
        """检测是否有相反结论"""
        positive = ["有效", "正确", "可行", "通过", "成正比"]
        negative = ["无效", "错误", "不可行", "不通过", "成反比"]
        
        pos1 = any(w in content1 for w in positive)
        neg1 = any(w in content1 for w in negative)
        pos2 = any(w in content2 for w in positive)
        neg2 = any(w in content2 for w in negative)
        
        return (pos1 and neg2) or (neg1 and pos2)
    
    def _evaluate_knowledge_value(self, unit: KnowledgeUnit) -> float:
        """
        知识价值评估函数 V(k)
        V(k) = ω1·新颖性 + ω2·覆盖率 - ω3·冲突度
        """
        value = (
            self.value_weights["novelty"] * unit.novelty +
            self.value_weights["coverage"] * unit.coverage -
            self.value_weights["conflict"] * unit.conflict
        )
        return max(0.0, min(1.0, value))
    
    def _remove_outdated(self) -> None:
        """淘汰过时低价值知识"""
        current_time = time.time()
        max_age = self.expire_days * 24 * 3600
        
        to_remove = []
        for kid, unit in self.knowledge_base.items():
            if current_time - unit.timestamp > max_age:
                if unit.value_score < self.low_value_threshold:
                    to_remove.append(kid)
        
        for kid in to_remove:
            del self.knowledge_base[kid]
            self.stats["units_removed"] += 1
    
    # ==================== 完整蒸馏流程 ====================
    
    def distill(self, raw_texts: List[str]) -> List[KnowledgeUnit]:
        """执行完整蒸馏流程"""
        # 转换为知识单元
        units = []
        for i, text in enumerate(raw_texts):
            unit = KnowledgeUnit(
                id=f"raw_{i}",
                content=clean_text(text),
            )
            units.append(unit)
        
        # 内环蒸馏
        distilled = self.inner_loop_distill(units)
        
        # 外环进化
        final = self.outer_loop_evolve(distilled)
        
        return final
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            "knowledge_base_size": len(self.knowledge_base),
        }
