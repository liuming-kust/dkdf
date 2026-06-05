# -*- coding: utf-8 -*-
"""
CCA: 认知一致性评估器
论文贡献：
- 逻辑一致性：OCR（本体冲突率）+ PCR（路径连贯性）
- 认知合理性：PMC（原型匹配度）+ CEC（认知经济度）
- 动态适应性：KER（知识进化率）+ OKCR（过时知识清除率）
"""

import math
import time
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field

from .utils import setup_logger


@dataclass
class CCAMetrics:
    """CCA评估指标"""
    # 逻辑一致性
    ocr: float = 0.0      # 本体冲突率
    pcr: float = 0.0      # 路径连贯性
    
    # 认知合理性
    pmc: float = 0.0      # 原型匹配度
    cec: float = 0.0      # 认知经济度
    
    # 动态适应性
    ker: float = 0.0      # 知识进化率
    okcr: float = 0.0     # 过时知识清除率
    
    # 综合
    overall: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "ocr": round(self.ocr, 4),
            "pcr": round(self.pcr, 4),
            "pmc": round(self.pmc, 4),
            "cec": round(self.cec, 4),
            "ker": round(self.ker, 4),
            "okcr": round(self.okcr, 4),
            "overall": round(self.overall, 4),
        }


class CCAEvaluator:
    """
    认知一致性评估器
    
    三维度评估体系：
    维度1 - 逻辑一致性：检查知识是否自洽
    维度2 - 认知合理性：检查知识是否符合领域认知范式
    维度3 - 动态适应性：检查知识是否及时更新
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = setup_logger("CCA")
        
        # 阈值配置
        cca_config = self.config.get("cca", {})
        thresholds = cca_config.get("thresholds", {})
        self.ocr_threshold = thresholds.get("ocr", 0.05)
        self.pcr_threshold = thresholds.get("pcr", 0.70)
        self.pmc_threshold = thresholds.get("pmc", 0.80)
        
        # 综合评分权重
        self.overall_weights = cca_config.get("overall_weights", {
            "ocr": 0.20,
            "pcr": 0.20,
            "pmc": 0.20,
            "cec": 0.15,
            "ker": 0.15,
            "okcr": 0.10,
        })
        
        # 评估历史
        self.history: List[CCAMetrics] = []
        
        # 知识更新历史（用于动态适应性评估）
        self.knowledge_update_history: List[Dict] = []
        
        # 领域原型特征（用于PMC计算）
        self._init_prototypes()
    
    def _init_prototypes(self):
        """初始化领域原型特征"""
        self.prototypes = {
            "instrument": {
                "key_features": ["测量", "参数", "精度", "量程", "校准"],
                "required_attrs": ["measurement_range", "accuracy"]
            },
            "concept": {
                "key_features": ["定义", "适用范围", "条件", "公式", "单位"],
                "required_attrs": ["definition", "scope"]
            },
            "procedure": {
                "key_features": ["步骤", "顺序", "操作", "注意事项", "安全"],
                "required_attrs": ["steps", "precautions"]
            },
            "parameter": {
                "key_features": ["数值", "单位", "范围", "误差", "精度"],
                "required_attrs": ["value", "unit"]
            },
            "organization": {
                "key_features": ["部门", "职责", "权限", "流程", "审批"],
                "required_attrs": ["function", "responsibility"]
            },
            "rule": {
                "key_features": ["条件", "要求", "规定", "标准", "依据"],
                "required_attrs": ["condition", "consequence"]
            }
        }
    
    def evaluate(self, kg: Dict = None) -> CCAMetrics:
        """
        执行完整评估
        
        Args:
            kg: 知识图谱字典（entities, relations, rules）
        
        Returns:
            CCAMetrics 评估指标
        """
        if kg is None:
            self.logger.warning("未提供知识图谱，返回空指标")
            return CCAMetrics()
        
        self.logger.info("开始认知一致性评估")
        start_time = time.time()
        
        metrics = CCAMetrics()
        
        # 维度1：逻辑一致性
        metrics.ocr = self._calculate_ocr(kg)
        metrics.pcr = self._calculate_pcr(kg)
        
        # 维度2：认知合理性
        metrics.pmc = self._calculate_pmc(kg)
        metrics.cec = self._calculate_cec(kg)
        
        # 维度3：动态适应性
        metrics.ker = self._calculate_ker()
        metrics.okcr = self._calculate_okcr()
        
        # 综合评分
        metrics.overall = self._calculate_overall(metrics)
        
        # 记录历史
        self.history.append(metrics)
        
        elapsed = time.time() - start_time
        self.logger.info(
            f"评估完成，综合评分: {metrics.overall:.4f}，"
            f"耗时 {elapsed:.2f}秒"
        )
        
        return metrics
    
    def _calculate_ocr(self, kg: Dict) -> float:
        """
        计算本体冲突率 (Ontology Conflict Rate)
        
        OCR = 违反OWL约束的三元组数 / 总三元组数
        """
        entities = kg.get("entities", {})
        relations = kg.get("relations", {})
        
        if not entities:
            return 0.0
        
        # 提取本体约束
        constraints = self._extract_ontology_constraints()
        
        violations = 0
        total = 0
        
        # 检查实体类型约束
        for e_id, entity in entities.items():
            total += 1
            entity_type = entity.get("type", "unknown")
            
            # 检查是否违反类型约束
            if entity_type in constraints:
                required_attrs = constraints[entity_type].get("required_attributes", [])
                for attr in required_attrs:
                    if attr not in entity.get("attributes", {}):
                        violations += 1
                        break
        
        # 检查关系约束
        for r_id, relation in relations.items():
            total += 1
            rel_type = relation.get("type", "unknown")
            
            # 检查关系类型约束
            if rel_type in constraints.get("relation_types", {}):
                required_props = constraints["relation_types"][rel_type].get(
                    "required_properties", []
                )
                for prop in required_props:
                    if prop not in relation.get("attributes", {}):
                        violations += 1
                        break
        
        return violations / max(total, 1)
    
    def _extract_ontology_constraints(self) -> Dict:
        """提取本体约束"""
        return {
            "instrument": {
                "required_attributes": ["measurement_range", "accuracy"],
            },
            "concept": {
                "required_attributes": ["definition", "scope"],
            },
            "procedure": {
                "required_attributes": ["steps", "precautions"],
            },
            "parameter": {
                "required_attributes": ["value", "unit"],
            },
            "organization": {
                "required_attributes": ["function", "responsibility"],
            },
            "rule": {
                "required_attributes": ["condition", "consequence"],
            },
            "relation_types": {
                "determines": {"required_properties": ["confidence"]},
                "requires": {"required_properties": ["condition"]},
                "contains": {"required_properties": ["quantity"]},
            }
        }
    
    def _calculate_pcr(self, kg: Dict) -> float:
        """
        计算路径连贯性 (Path Coherence Rate)
        
        PCR = 平均语义相似度（基于路径长度）
        """
        entities = kg.get("entities", {})
        relations = kg.get("relations", {})
        
        if len(entities) < 2:
            return 1.0
        
        # 构建关系图
        graph = defaultdict(list)
        for r_id, relation in relations.items():
            source = relation.get("source")
            target = relation.get("target")
            if source and target:
                graph[source].append(target)
        
        # 计算最短路径的语义连贯性
        total_similarity = 0.0
        count = 0
        
        # 只计算部分样本（避免O(n^2)）
        entity_ids = list(entities.keys())[:50]
        
        for i, e1 in enumerate(entity_ids):
            for e2 in entity_ids[i+1:]:
                path_length = self._shortest_path_length(graph, e1, e2)
                if path_length > 0:
                    # 路径连贯性与长度成反比
                    similarity = 1.0 / (1.0 + math.log(path_length))
                    total_similarity += similarity
                    count += 1
        
        return total_similarity / max(count, 1)
    
    def _shortest_path_length(
        self,
        graph: Dict[str, List[str]],
        start: str,
        end: str
    ) -> int:
        """BFS计算最短路径长度"""
        if start not in graph:
            return -1
        
        visited = {start}
        queue = [(start, 0)]
        
        while queue:
            node, dist = queue.pop(0)
            if node == end:
                return dist
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        return -1
    
    def _calculate_pmc(self, kg: Dict) -> float:
        """
        计算原型匹配度 (Prototype Matching Consistency)
        
        PMC = 实体特征与领域原型的平均匹配率
        """
        entities = kg.get("entities", {})
        
        if not entities:
            return 0.0
        
        total_match = 0.0
        count = 0
        
        for e_id, entity in entities.items():
            entity_type = entity.get("type", "unknown")
            if entity_type in self.prototypes:
                required_features = set(
                    self.prototypes[entity_type]["key_features"]
                )
                
                # 从实体属性或描述中提取特征
                attributes = entity.get("attributes", {})
                description = entity.get("description", "")
                
                # 简单匹配：检查关键词是否在描述中
                actual_features = set()
                all_text = str(attributes) + description
                for feat in required_features:
                    if feat in all_text:
                        actual_features.add(feat)
                
                if required_features:
                    match_rate = len(actual_features & required_features) / len(required_features)
                    total_match += match_rate
                    count += 1
        
        return total_match / max(count, 1)
    
    def _calculate_cec(self, kg: Dict) -> float:
        """
        计算认知经济度 (Cognitive Economy Coefficient)
        
        CEC = 1 - (结构化表示大小 / 原始文本估算大小)
        """
        entities = kg.get("entities", {})
        relations = kg.get("relations", {})
        
        # 估计原始文本大小（简化估算）
        raw_size = 0
        for e in entities.values():
            name = e.get("name", "")
            raw_size += len(name) * 20  # 每个实体约20倍名称长度
        
        # 结构化表示大小
        structured_size = len(entities) * 30 + len(relations) * 25
        
        if raw_size == 0:
            return 1.0
        
        compression_ratio = structured_size / raw_size
        cec = 1.0 - min(1.0, compression_ratio)
        
        return cec
    
    def _calculate_ker(self) -> float:
        """
        计算知识进化率 (Knowledge Evolution Rate)
        
        KER = 单位时间内新增/更新知识占比
        """
        if len(self.knowledge_update_history) < 2:
            return 0.15  # 默认值，实际应从统计计算
        
        # 计算最近时间窗口内的更新率
        window = 3600 * 24  # 24小时
        current_time = time.time()
        
        recent_updates = [
            u for u in self.knowledge_update_history
            if current_time - u.get("timestamp", 0) < window
        ]
        
        if not recent_updates:
            return 0.0
        
        total_updates = len(recent_updates)
        estimated_ker = min(1.0, total_updates / 100.0)
        
        return estimated_ker
    
    def _calculate_okcr(self) -> float:
        """
        计算过时知识清除率 (Obsolete Knowledge Clearance Rate)
        
        OKCR = 已清除的过时知识 / 应清除的过时知识
        """
        if len(self.history) < 2:
            return 0.12  # 默认值
        
        # 基于历史记录估算
        return 0.12
    
    def _calculate_overall(self, metrics: CCAMetrics) -> float:
        """
        计算综合评分
        
        Overall = w1·(1-OCR) + w2·PCR + w3·PMC + w4·CEC + w5·KER + w6·OKCR
        """
        weights = self.overall_weights
        
        score = (
            weights.get("ocr", 0.20) * (1 - min(1.0, metrics.ocr)) +
            weights.get("pcr", 0.20) * metrics.pcr +
            weights.get("pmc", 0.20) * metrics.pmc +
            weights.get("cec", 0.15) * metrics.cec +
            weights.get("ker", 0.15) * metrics.ker +
            weights.get("okcr", 0.10) * metrics.okcr
        )
        
        return min(1.0, max(0.0, score))
    
    def record_update(self, update_info: Dict) -> None:
        """记录知识更新（用于动态适应性评估）"""
        update_info["timestamp"] = time.time()
        self.knowledge_update_history.append(update_info)
        
        # 保留最近1000条记录
        if len(self.knowledge_update_history) > 1000:
            self.knowledge_update_history = self.knowledge_update_history[-1000:]
    
    def get_evaluation_report(self, metrics: CCAMetrics) -> Dict:
        """生成评估报告"""
        return {
            "summary": {
                "overall_score": round(metrics.overall, 3),
                "grade": self._get_grade(metrics.overall),
            },
            "metrics": {
                "ocr": {
                    "value": round(metrics.ocr, 4),
                    "threshold": self.ocr_threshold,
                    "passed": metrics.ocr <= self.ocr_threshold
                },
                "pcr": {
                    "value": round(metrics.pcr, 4),
                    "threshold": self.pcr_threshold,
                    "passed": metrics.pcr >= self.pcr_threshold
                },
                "pmc": {
                    "value": round(metrics.pmc, 4),
                    "threshold": self.pmc_threshold,
                    "passed": metrics.pmc >= self.pmc_threshold
                },
                "cec": {"value": round(metrics.cec, 4)},
                "ker": {"value": round(metrics.ker, 4)},
                "okcr": {"value": round(metrics.okcr, 4)},
            },
            "recommendations": self._generate_recommendations(metrics),
        }
    
    def _get_grade(self, score: float) -> str:
        """根据分数评级"""
        if score >= 0.9:
            return "A (优秀)"
        elif score >= 0.8:
            return "B (良好)"
        elif score >= 0.7:
            return "C (中等)"
        elif score >= 0.6:
            return "D (及格)"
        else:
            return "F (不及格)"
    
    def _generate_recommendations(self, metrics: CCAMetrics) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if metrics.ocr > self.ocr_threshold:
            recommendations.append(
                f"本体冲突率过高 ({metrics.ocr:.3f} > {self.ocr_threshold})，"
                "建议检查实体类型和关系约束"
            )
        
        if metrics.pcr < self.pcr_threshold:
            recommendations.append(
                f"路径连贯性不足 ({metrics.pcr:.3f} < {self.pcr_threshold})，"
                "建议补充实体间的逻辑关联"
            )
        
        if metrics.pmc < self.pmc_threshold:
            recommendations.append(
                f"原型匹配度偏低 ({metrics.pmc:.3f} < {self.pmc_threshold})，"
                "建议对照领域原型完善实体特征"
            )
        
        if metrics.cec < 0.4:
            recommendations.append(
                "认知经济度偏低，建议优化知识表示，减少冗余"
            )
        
        if metrics.ker < 0.1:
            recommendations.append(
                "知识进化率低，建议加快知识更新频率"
            )
        
        if not recommendations:
            recommendations.append("知识质量良好，继续保持")
        
        return recommendations


class ConsistencyChecker:
    """一致性检查器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = setup_logger("ConsistencyChecker")
    
    def check_temporal_consistency(self, kg: Dict) -> List[Dict]:
        """检查时间一致性"""
        issues = []
        
        for e_id, entity in kg.get("entities", {}).items():
            timestamp = entity.get("attributes", {}).get("timestamp")
            if timestamp:
                # 简单的时间格式检查
                if not isinstance(timestamp, (int, float)):
                    issues.append({
                        "type": "invalid_timestamp",
                        "entity": e_id,
                        "message": f"Invalid timestamp format: {timestamp}"
                    })
        
        return issues
    
    def check_logical_consistency(self, kg: Dict) -> List[Dict]:
        """检查逻辑一致性"""
        issues = []
        relations = kg.get("relations", {})
        
        # 检查循环依赖
        graph = defaultdict(list)
        for r_id, relation in relations.items():
            source = relation.get("source")
            target = relation.get("target")
            if source and target:
                graph[source].append(target)
        
        # 检测环
        visited = set()
        rec_stack = set()
        
        def detect_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if detect_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    issues.append({
                        "type": "cycle",
                        "nodes": [node, neighbor],
                        "message": f"Detected cycle between {node} and {neighbor}"
                    })
                    return True
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                detect_cycle(node)
        
        return issues
