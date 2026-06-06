#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
物理实验CCA评估指标计算模块

实现三维度六指标：
1. 逻辑一致性
   - OCR (Ontology Conflict Rate): 本体冲突率
   - PCR (Path Coherence Rate): 路径连贯性
2. 认知合理性
   - PMC (Prototype Matching Consistency): 原型匹配度
   - CEC (Cognitive Economy Coefficient): 认知经济度
3. 动态适应性
   - KER (Knowledge Evolution Rate): 知识演化率
   - OKCR (Obsolete Knowledge Clearance Rate): 过时知识清除率
"""

import math
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class PhysicsCCAMetrics:
    """物理实验CCA评估指标"""
    
    # 逻辑一致性
    ocr: float = 0.0          # 本体冲突率 (Ontology Conflict Rate)
    pcr: float = 0.0          # 路径连贯性 (Path Coherence Rate)
    
    # 认知合理性
    pmc: float = 0.0          # 原型匹配度 (Prototype Matching Consistency)
    cec: float = 0.0          # 认知经济度 (Cognitive Economy Coefficient)
    
    # 动态适应性
    ker: float = 0.0          # 知识演化率 (Knowledge Evolution Rate)
    okcr: float = 0.0         # 过时知识清除率 (Obsolete Knowledge Clearance Rate)
    
    # 综合评分
    overall: float = 0.0
    
    # 详细分解
    ocr_details: Dict[str, Any] = field(default_factory=dict)
    pcr_details: Dict[str, Any] = field(default_factory=dict)
    pmc_details: Dict[str, Any] = field(default_factory=dict)
    cec_details: Dict[str, Any] = field(default_factory=dict)
    ker_details: Dict[str, Any] = field(default_factory=dict)
    okcr_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "logical_consistency": {
                "ocr": round(self.ocr, 4),
                "pcr": round(self.pcr, 4),
                "ocr_details": self.ocr_details,
                "pcr_details": self.pcr_details
            },
            "cognitive_rationality": {
                "pmc": round(self.pmc, 4),
                "cec": round(self.cec, 4),
                "pmc_details": self.pmc_details,
                "cec_details": self.cec_details
            },
            "dynamic_adaptability": {
                "ker": round(self.ker, 4),
                "okcr": round(self.okcr, 4),
                "ker_details": self.ker_details,
                "okcr_details": self.okcr_details
            },
            "overall": round(self.overall, 4)
        }
    
    def get_grade(self) -> Tuple[str, str]:
        """获取评级"""
        if self.overall >= 0.9:
            return ("A", "优秀")
        elif self.overall >= 0.8:
            return ("B", "良好")
        elif self.overall >= 0.7:
            return ("C", "中等")
        elif self.overall >= 0.6:
            return ("D", "及格")
        else:
            return ("F", "不及格")


class PhysicsCCAMetricsCalculator:
    """
    物理实验CCA指标计算器
    
    专门针对物理实验知识图谱的评估指标计算
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化计算器
        
        Args:
            config: 配置参数，包含阈值、权重等
        """
        self.config = config or {}
        
        # 评分权重（与论文一致）
        self.weights = self.config.get("weights", {
            "ocr": 0.20,
            "pcr": 0.20,
            "pmc": 0.20,
            "cec": 0.15,
            "ker": 0.15,
            "okcr": 0.10
        })
        
        # 阈值设置
        self.thresholds = self.config.get("thresholds", {
            "ocr_max": 0.05,      # OCR不超过5%
            "pcr_min": 0.70,      # PCR不低于70%
            "pmc_min": 0.80,      # PMC不低于80%
            "cec_min": 0.40,      # CEC不低于40%
            "ker_min": 0.10,      # KER不低于10%
            "okcr_min": 0.80      # OKCR不低于80%
        })
        
        # 物理实验领域原型特征
        self._init_prototypes()
    
    def _init_prototypes(self):
        """初始化物理实验领域原型特征"""
        self.prototypes = {
            "experiment": {
                "key_features": ["实验目的", "实验原理", "实验步骤", "数据处理", "实验结果"],
                "required_fields": ["name", "purpose", "theory", "steps"],
                "keywords": ["测量", "验证", "探究", "分析", "计算", "记录"]
            },
            "instrument": {
                "key_features": ["名称", "型号", "量程", "精度", "数量"],
                "required_fields": ["name", "quantity"],
                "keywords": ["仪", "计", "器", "表", "电源", "传感器"]
            },
            "procedure": {
                "key_features": ["步骤序号", "操作描述", "注意事项"],
                "required_fields": ["order", "description"],
                "keywords": ["连接", "调节", "测量", "记录", "计算", "重复"]
            },
            "formula": {
                "key_features": ["公式", "符号说明", "单位"],
                "required_fields": ["expression", "variables"],
                "keywords": ["=", "+", "-", "*", "/", "∫", "∑"]
            }
        }
    
    def calculate_ocr(self, kg: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        计算本体冲突率
        
        OCR = 违反约束的三元组数 / 总三元组数
        
        检查的约束类型：
        1. 实体类型符合性
        2. 关系参数类型匹配
        3. 属性值范围
        """
        entities = kg.get("entities", {})
        relations = kg.get("relations", [])
        
        if not entities:
            return 0.0, {"error": "无实体数据"}
        
        violations = []
        total_checks = 0
        
        # 1. 检查实体类型符合性
        valid_types = {"concept", "instrument", "procedure", "parameter", "formula", "organization"}
        for e_id, entity in entities.items():
            total_checks += 1
            e_type = entity.get("type", "unknown")
            if e_type not in valid_types and e_type != "unknown":
                violations.append({
                    "type": "invalid_entity_type",
                    "entity_id": e_id,
                    "message": f"无效实体类型: {e_type}"
                })
        
        # 2. 检查关系参数类型匹配
        for rel in relations:
            total_checks += 1
            rel_type = rel.get("type", "unknown")
            
            # 检查关系类型是否在预定义集合中
            valid_relation_types = {"contains", "requires", "determines", "equals", 
                                    "based_on", "co_occurrence", "related"}
            if rel_type not in valid_relation_types:
                violations.append({
                    "type": "invalid_relation_type",
                    "relation": rel,
                    "message": f"无效关系类型: {rel_type}"
                })
            
            # 检查源和目标实体是否存在
            source_id = rel.get("source_id")
            target_id = rel.get("target_id")
            if source_id and source_id not in entities:
                violations.append({
                    "type": "missing_source_entity",
                    "relation": rel,
                    "message": f"源实体不存在: {source_id}"
                })
            if target_id and target_id not in entities:
                violations.append({
                    "type": "missing_target_entity",
                    "relation": rel,
                    "message": f"目标实体不存在: {target_id}"
                })
        
        # 3. 检查属性值范围
        for e_id, entity in entities.items():
            attrs = entity.get("attributes", {})
            
            # 检查仪器数量
            if entity.get("type") == "instrument":
                quantity = attrs.get("quantity")
                if quantity is not None and (not isinstance(quantity, (int, float)) or quantity <= 0):
                    violations.append({
                        "type": "invalid_quantity",
                        "entity_id": e_id,
                        "message": f"无效数量: {quantity}"
                    })
            
            # 检查步骤顺序
            if entity.get("type") == "procedure":
                order = attrs.get("order")
                if order is not None and (not isinstance(order, int) or order < 0):
                    violations.append({
                        "type": "invalid_step_order",
                        "entity_id": e_id,
                        "message": f"无效步骤顺序: {order}"
                    })
        
        ocr = len(violations) / max(total_checks, 1)
        
        details = {
            "total_checks": total_checks,
            "violations_count": len(violations),
            "violations": violations[:20],  # 只保留前20条
            "status": "pass" if ocr <= self.thresholds["ocr_max"] else "fail"
        }
        
        return ocr, details
    
    def calculate_pcr(self, kg: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        计算路径连贯性
        
        PCR = 平均语义连贯性（基于路径长度和关系合理性）
        """
        entities = kg.get("entities", {})
        relations = kg.get("relations", [])
        
        if len(entities) < 2:
            return 1.0, {"message": "实体数量不足，无法计算路径连贯性"}
        
        # 构建关系图
        graph = defaultdict(list)
        relation_types = {}
        
        for rel in relations:
            source = rel.get("source_id")
            target = rel.get("target_id")
            rel_type = rel.get("type", "unknown")
            if source and target:
                graph[source].append((target, rel_type))
                relation_types[(source, target)] = rel_type
        
        # 计算所有实体对之间的最短路径
        entity_ids = list(entities.keys())
        total_coherence = 0.0
        valid_pairs = 0
        path_lengths = []
        
        # 路径合理性权重
        type_weights = {
            "contains": 1.0,
            "requires": 0.9,
            "determines": 0.85,
            "equals": 1.0,
            "based_on": 0.8,
            "co_occurrence": 0.5,
            "related": 0.6
        }
        
        for i, start in enumerate(entity_ids[:50]):  # 限制计算量
            for end in entity_ids[i+1:50]:
                path = self._find_path(graph, start, end)
                if path:
                    path_length = len(path) - 1
                    path_lengths.append(path_length)
                    
                    # 计算路径合理性
                    path_reasonability = 1.0
                    for j in range(len(path) - 1):
                        rel_type = relation_types.get((path[j], path[j+1]), "unknown")
                        path_reasonability *= type_weights.get(rel_type, 0.5)
                    
                    # 连贯性 = (1/(1+log(length))) * 合理性
                    coherence = (1.0 / (1.0 + math.log(path_length + 1))) * path_reasonability
                    total_coherence += coherence
                    valid_pairs += 1
        
        pcr = total_coherence / max(valid_pairs, 1)
        
        avg_path_length = sum(path_lengths) / max(len(path_lengths), 1)
        
        details = {
            "valid_pairs": valid_pairs,
            "average_path_length": round(avg_path_length, 2),
            "max_path_length": max(path_lengths) if path_lengths else 0,
            "status": "pass" if pcr >= self.thresholds["pcr_min"] else "fail"
        }
        
        return pcr, details
    
    def _find_path(self, graph: Dict, start: str, end: str) -> Optional[List[str]]:
        """BFS查找最短路径"""
        if start not in graph:
            return None
        
        visited = {start}
        queue = [(start, [start])]
        
        while queue:
            node, path = queue.pop(0)
            if node == end:
                return path
            for neighbor, _ in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def calculate_pmc(self, kg: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        计算原型匹配度
        
        PMC = 实体特征与领域原型的平均匹配率
        """
        entities = kg.get("entities", {})
        
        if not entities:
            return 0.0, {"error": "无实体数据"}
        
        total_match = 0.0
        entity_scores = []
        
        for e_id, entity in entities.items():
            e_type = entity.get("type", "unknown")
            if e_type not in self.prototypes:
                continue
            
            prototype = self.prototypes[e_type]
            required_fields = prototype.get("required_fields", [])
            key_features = prototype.get("key_features", [])
            keywords = prototype.get("keywords", [])
            
            # 检查必填字段
            attrs = entity.get("attributes", {})
            name = entity.get("name", "")
            description = entity.get("description", "")
            
            # 计算字段匹配度
            field_score = 0.0
            for field in required_fields:
                if field in attrs or field == "name" and name:
                    field_score += 1.0
                elif field == "description" and description:
                    field_score += 1.0
            field_score = field_score / max(len(required_fields), 1)
            
            # 计算特征匹配度（基于关键词）
            all_text = name + " " + description + " " + str(attrs)
            keyword_score = 0.0
            for kw in keywords:
                if kw in all_text:
                    keyword_score += 1.0
            keyword_score = keyword_score / max(len(keywords), 1)
            
            # 综合得分
            entity_score = field_score * 0.6 + keyword_score * 0.4
            total_match += entity_score
            entity_scores.append({
                "entity_id": e_id,
                "name": name,
                "type": e_type,
                "score": round(entity_score, 4),
                "field_score": round(field_score, 4),
                "keyword_score": round(keyword_score, 4)
            })
        
        pmc = total_match / max(len(entity_scores), 1)
        
        # 找出低分实体
        low_score_entities = [e for e in entity_scores if e["score"] < 0.5]
        
        details = {
            "total_entities_evaluated": len(entity_scores),
            "average_entity_score": round(pmc, 4),
            "low_score_entities": low_score_entities[:10],
            "status": "pass" if pmc >= self.thresholds["pmc_min"] else "fail"
        }
        
        return pmc, details
    
    def calculate_cec(self, kg: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        计算认知经济度
        
        CEC = 1 - (结构化表示大小 / 原始文本估算大小)
        值越高表示知识表示越简洁高效
        """
        entities = kg.get("entities", {})
        relations = kg.get("relations", [])
        
        # 计算结构化表示大小
        structured_size = 0
        for e_id, entity in entities.items():
            # 实体名称长度
            structured_size += len(entity.get("name", ""))
            # 实体类型
            structured_size += len(entity.get("type", ""))
            # 属性
            for k, v in entity.get("attributes", {}).items():
                structured_size += len(k) + len(str(v))
        
        for rel in relations:
            structured_size += len(rel.get("source_name", ""))
            structured_size += len(rel.get("target_name", ""))
            structured_size += len(rel.get("type", ""))
        
        # 估算原始文本大小（假设每个实体对应约200字符的描述）
        raw_size = len(entities) * 200 + len(relations) * 50
        
        if raw_size == 0:
            cec = 1.0
        else:
            compression_ratio = structured_size / raw_size
            cec = max(0.0, 1.0 - min(1.0, compression_ratio))
        
        details = {
            "structured_size": structured_size,
            "estimated_raw_size": raw_size,
            "compression_ratio": round(structured_size / max(raw_size, 1), 4),
            "status": "pass" if cec >= self.thresholds["cec_min"] else "fail"
        }
        
        return cec, details
    
    def calculate_ker(self, kg: Dict[str, Any], 
                      history: Optional[List[Dict]] = None) -> Tuple[float, Dict[str, Any]]:
        """
        计算知识演化率
        
        KER = (新增知识 + 更新知识) / (总知识 × 时间窗口)
        """
        # 如果没有历史数据，返回默认值
        if not history or len(history) < 2:
            return 0.15, {"message": "历史数据不足，使用默认值0.15"}
        
        # 计算时间窗口内的更新率
        current_time = time.time()
        window = 3600 * 24 * 7  # 7天窗口
        
        recent_updates = [
            u for u in history
            if current_time - u.get("timestamp", 0) < window
        ]
        
        if not recent_updates:
            return 0.0, {"message": "最近无更新"}
        
        # 计算当前知识库大小
        total_knowledge = len(kg.get("entities", {})) + len(kg.get("relations", []))
        
        # 计算更新量
        added = sum(u.get("added", 0) for u in recent_updates)
        updated = sum(u.get("updated", 0) for u in recent_updates)
        
        ker = (added + updated) / max(total_knowledge * (window / (24*3600)), 1)
        ker = min(1.0, ker)
        
        details = {
            "time_window_days": window / (24*3600),
            "updates_in_window": len(recent_updates),
            "total_knowledge": total_knowledge,
            "added": added,
            "updated": updated,
            "status": "pass" if ker >= self.thresholds["ker_min"] else "fail"
        }
        
        return ker, details
    
    def calculate_okcr(self, kg: Dict[str, Any], 
                       expired_history: Optional[List[Dict]] = None) -> Tuple[float, Dict[str, Any]]:
        """
        计算过时知识清除率
        
        OKCR = 已清除的过时知识 / 应清除的过时知识
        """
        # 如果没有历史数据，返回默认值
        if not expired_history:
            return 0.12, {"message": "历史数据不足，使用默认值0.12"}
        
        total_expired = sum(e.get("expired_count", 0) for e in expired_history)
        total_cleared = sum(e.get("cleared_count", 0) for e in expired_history)
        
        if total_expired == 0:
            okcr = 1.0
        else:
            okcr = total_cleared / total_expired
            okcr = min(1.0, okcr)
        
        details = {
            "total_expired": total_expired,
            "total_cleared": total_cleared,
            "clearance_rate": round(okcr, 4),
            "status": "pass" if okcr >= self.thresholds["okcr_min"] else "fail"
        }
        
        return okcr, details
    
    def calculate_overall(self, metrics: PhysicsCCAMetrics) -> float:
        """
        计算综合评分
        
        Overall = w1·(1-OCR) + w2·PCR + w3·PMC + w4·CEC + w5·KER + w6·OKCR
        """
        score = (
            self.weights["ocr"] * (1 - min(1.0, metrics.ocr)) +
            self.weights["pcr"] * metrics.pcr +
            self.weights["pmc"] * metrics.pmc +
            self.weights["cec"] * metrics.cec +
            self.weights["ker"] * metrics.ker +
            self.weights["okcr"] * metrics.okcr
        )
        return min(1.0, max(0.0, score))


def calculate_all_metrics(kg: Dict[str, Any],
                          history: Optional[List[Dict]] = None,
                          expired_history: Optional[List[Dict]] = None,
                          config: Optional[Dict] = None) -> PhysicsCCAMetrics:
    """
    计算所有CCA指标的便捷函数
    
    Args:
        kg: 知识图谱数据
        history: 知识更新历史
        expired_history: 过期知识清除历史
        config: 配置参数
    
    Returns:
        PhysicsCCAMetrics对象
    """
    calculator = PhysicsCCAMetricsCalculator(config)
    
    metrics = PhysicsCCAMetrics()
    
    # 逻辑一致性
    metrics.ocr, metrics.ocr_details = calculator.calculate_ocr(kg)
    metrics.pcr, metrics.pcr_details = calculator.calculate_pcr(kg)
    
    # 认知合理性
    metrics.pmc, metrics.pmc_details = calculator.calculate_pmc(kg)
    metrics.cec, metrics.cec_details = calculator.calculate_cec(kg)
    
    # 动态适应性
    metrics.ker, metrics.ker_details = calculator.calculate_ker(kg, history)
    metrics.okcr, metrics.okcr_details = calculator.calculate_okcr(kg, expired_history)
    
    # 综合评分
    metrics.overall = calculator.calculate_overall(metrics)
    
    return metrics
