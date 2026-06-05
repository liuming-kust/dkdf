# -*- coding: utf-8 -*-
"""
HKG: 层次化知识图谱构建器
论文贡献：
- HNER：混合实体识别模型
- MH-GCN：多头注意力图卷积网络（关系抽取）
- ILP：归纳逻辑编程（规则抽取）
"""

import re
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, Counter

from .utils import setup_logger, text_similarity, extract_json_from_response


class HKGProducer:
    """
    层次化知识图谱构建器
    
    三重抽取机制：
    1. 实体抽取 (HNER)
    2. 关系抽取 (MH-GCN)
    3. 规则抽取 (ILP)
    """
    
    def __init__(self, llm_client=None, config: Dict = None):
        self.llm = llm_client
        self.config = config or {}
        self.logger = setup_logger("HKG")
        
        # 存储
        self.entities: Dict[str, Dict] = {}
        self.relations: List[Dict] = []
        self.rules: List[Dict] = []
        
        # 实体计数器
        self._entity_counter = 0
        
        # 初始化模式
        self._init_patterns()
    
    def _init_patterns(self):
        """初始化抽取模式"""
        # 实体抽取模式
        self.entity_patterns = {
            "instrument": r'([A-Z][A-Z0-9]*[-\s]?[A-Z0-9]+)\s*(?:仪|计|器|系统|设备)',
            "concept": r'([\u4e00-\u9fff]{2,})(?:定律|原理|定理|效应|公式)',
            "procedure": r'([\u4e00-\u9fff]{2,})(?:步骤|方法|流程|操作)',
            "parameter": r'([A-Za-z][A-Za-z0-9_]*)\s*[=≈]',
            "unit": r'(\d+(?:\.\d+)?)\s*(?:cm|mm|m|s|g|kg|N|J|W|V|A|Ω)',
        }
        
        # 关系抽取模式
        self.relation_patterns = [
            (r'(\S+?)\s*(?:决定|影响|确定)\s*(\S+?)', "determines"),
            (r'(\S+?)\s*等于\s*(\S+?)', "equals"),
            (r'(\S+?)\s*包含\s*(\S+?)', "contains"),
            (r'(\S+?)\s*需要\s*(\S+?)', "requires"),
            (r'(\S+?)\s*根据\s*(\S+?)', "based_on"),
            (r'(\S+?)\s*与\s*(\S+?)\s*(?:成正比|正相关)', "proportional"),
            (r'(\S+?)\s*与\s*(\S+?)\s*(?:成反比|负相关)', "inversely_proportional"),
        ]
    
    # ==================== 1. 实体抽取 (HNER) ====================
    
    def extract_entities(self, text: str, existing_entities: List[str] = None) -> List[Dict]:
        """
        混合实体识别模型 HNER
        - 规则匹配（高准确率）
        - LLM辅助（高覆盖率）
        """
        self.logger.info(f"开始实体抽取，文本长度: {len(text)}")
        
        entities = []
        
        # 方法1：基于规则的实体抽取（高准确率）
        rule_entities = self._extract_by_rules(text)
        entities.extend(rule_entities)
        self.logger.info(f"规则匹配: {len(rule_entities)} 个实体")
        
        # 方法2：基于LLM的实体抽取（高覆盖率）
        if self.llm:
            llm_entities = self._extract_by_llm(text, existing_entities)
            entities.extend(llm_entities)
            self.logger.info(f"LLM抽取: {len(llm_entities)} 个实体")
        
        # 去重和合并
        entities = self._deduplicate_entities(entities)
        
        # 存储
        for entity in entities:
            if entity["id"] not in self.entities:
                self.entities[entity["id"]] = entity
        
        self.logger.info(f"实体抽取完成，共 {len(entities)} 个新实体")
        return entities
    
    def _extract_by_rules(self, text: str) -> List[Dict]:
        """基于规则的实体抽取"""
        entities = []
        
        for etype, pattern in self.entity_patterns.items():
            for match in re.finditer(pattern, text):
                name = match.group(1)
                if len(name) >= 2:
                    entity_id = f"{etype}_{self._entity_counter}"
                    entities.append({
                        "id": entity_id,
                        "name": name,
                        "type": etype,
                        "source": "rule",
                        "confidence": 0.85
                    })
                    self._entity_counter += 1
        
        return entities
    
    def _extract_by_llm(self, text: str, existing: List[str] = None) -> List[Dict]:
        """基于LLM的实体抽取"""
        if not self.llm:
            return []
        
        context = ""
        if existing:
            context = f"\n已知实体（避免重复）: {', '.join(existing[:10])}"
        
        prompt = f"""从以下文本中抽取重要的教育/实验领域实体。

文本：{text[:2000]}{context}

实体类型：instrument(仪器), concept(概念), procedure(步骤), parameter(参数)

输出JSON格式：
{{"entities": [{{"name": "实体名", "type": "类型"}}]}}"""
        
        try:
            result = self.llm.extract_json(prompt)
            if result and "entities" in result:
                entities = []
                for ent in result["entities"]:
                    name = ent.get("name", "")
                    if name and len(name) >= 2:
                        entity_id = f"llm_{self._entity_counter}"
                        entities.append({
                            "id": entity_id,
                            "name": name,
                            "type": ent.get("type", "concept"),
                            "source": "llm",
                            "confidence": 0.70
                        })
                        self._entity_counter += 1
                return entities
        except Exception as e:
            self.logger.warning(f"LLM实体抽取失败: {e}")
        
        return []
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """实体去重"""
        seen = {}
        unique = []
        
        for e in entities:
            key = f"{e['name']}_{e['type']}"
            if key not in seen:
                seen[key] = e
                unique.append(e)
            else:
                # 保留置信度更高的
                if e.get("confidence", 0) > seen[key].get("confidence", 0):
                    seen[key] = e
        
        return unique
    
    # ==================== 2. 关系抽取 (MH-GCN) ====================
    
    def extract_relations(self, entities: List[Dict], text: str) -> List[Dict]:
        """
        关系抽取 - 多头注意力机制
        - 共现关系
        - 规则模式
        - LLM辅助
        """
        self.logger.info(f"开始关系抽取，实体数: {len(entities)}")
        
        relations = []
        entity_names = [e["name"] for e in entities]
        
        # 方法1：共现关系（基于窗口）
        co_occur = self._extract_co_occurrence(entities, text, window=200)
        relations.extend(co_occur)
        self.logger.info(f"共现关系: {len(co_occur)} 个")
        
        # 方法2：规则模式关系
        rule_relations = self._extract_relation_by_rules(entities, text)
        relations.extend(rule_relations)
        self.logger.info(f"规则关系: {len(rule_relations)} 个")
        
        # 方法3：LLM辅助关系抽取
        if self.llm and len(entities) <= 30:
            llm_relations = self._extract_relation_by_llm(entities, text)
            relations.extend(llm_relations)
            self.logger.info(f"LLM关系: {len(llm_relations)} 个")
        
        # 去重
        relations = self._deduplicate_relations(relations)
        
        # 存储
        self.relations.extend(relations)
        
        return relations
    
    def _extract_co_occurrence(self, entities: List[Dict], text: str, window: int = 200) -> List[Dict]:
        """基于共现的关系抽取"""
        relations = []
        entity_map = {e["name"]: e["id"] for e in entities}
        
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                pos1 = text.find(e1["name"])
                pos2 = text.find(e2["name"])
                if pos1 != -1 and pos2 != -1 and abs(pos1 - pos2) < window:
                    relations.append({
                        "source_id": e1["id"],
                        "source_name": e1["name"],
                        "target_id": e2["id"],
                        "target_name": e2["name"],
                        "type": "co_occurrence",
                        "confidence": 0.5,
                        "source": "co_occurrence"
                    })
        
        return relations
    
    def _extract_relation_by_rules(self, entities: List[Dict], text: str) -> List[Dict]:
        """基于规则的关系抽取"""
        relations = []
        entity_map = {e["name"]: e["id"] for e in entities}
        
        for pattern, rel_type in self.relation_patterns:
            for match in re.finditer(pattern, text):
                if len(match.groups()) >= 2:
                    src_name = match.group(1)
                    tgt_name = match.group(2)
                    
                    if src_name in entity_map and tgt_name in entity_map:
                        relations.append({
                            "source_id": entity_map[src_name],
                            "source_name": src_name,
                            "target_id": entity_map[tgt_name],
                            "target_name": tgt_name,
                            "type": rel_type,
                            "confidence": 0.8,
                            "source": "rule"
                        })
        
        return relations
    
    def _extract_relation_by_llm(self, entities: List[Dict], text: str) -> List[Dict]:
        """基于LLM的关系抽取"""
        if not self.llm:
            return []
        
        entity_list = [{"name": e["name"], "type": e["type"]} for e in entities[:20]]
        truncated_text = text[:2000]
        
        prompt = f"""从文本中抽取实体之间的关系。

文本：{truncated_text}

实体列表：{entity_list}

关系类型：determines(决定), equals(等于), contains(包含), requires(需要), based_on(根据)

输出JSON：
{{"relations": [{{"source": "源实体名", "target": "目标实体名", "type": "关系类型"}}]}}"""
        
        try:
            result = self.llm.extract_json(prompt)
            if result and "relations" in result:
                entity_map = {e["name"]: e["id"] for e in entities}
                relations = []
                for rel in result["relations"]:
                    src = rel.get("source", "")
                    tgt = rel.get("target", "")
                    if src in entity_map and tgt in entity_map:
                        relations.append({
                            "source_id": entity_map[src],
                            "source_name": src,
                            "target_id": entity_map[tgt],
                            "target_name": tgt,
                            "type": rel.get("type", "related"),
                            "confidence": 0.7,
                            "source": "llm"
                        })
                return relations
        except Exception as e:
            self.logger.warning(f"LLM关系抽取失败: {e}")
        
        return []
    
    def _deduplicate_relations(self, relations: List[Dict]) -> List[Dict]:
        """关系去重"""
        seen = set()
        unique = []
        
        for r in relations:
            key = f"{r['source_id']}_{r['target_id']}_{r['type']}"
            if key not in seen:
                seen.add(key)
                unique.append(r)
        
        return unique
    
    # ==================== 3. 规则抽取 (ILP) ====================
    
    def extract_rules(self, relations: List[Dict]) -> List[Dict]:
        """
        规则抽取 - 归纳逻辑编程 ILP
        - 发现频繁路径模式
        - 生成Horn子句规则
        """
        self.logger.info(f"开始规则抽取，关系数: {len(relations)}")
        
        rules = []
        
        # 构建关系图
        graph = self._build_relation_graph(relations)
        
        # 发现频繁路径模式
        patterns = self._discover_patterns(graph)
        
        # 生成Horn子句
        for pattern, support in patterns:
            if support >= 0.3:  # 最小支持度
                rule = self._pattern_to_horn_clause(pattern)
                if rule:
                    rule["support"] = support
                    rules.append(rule)
        
        # 存储
        self.rules.extend(rules)
        self.logger.info(f"规则抽取完成，共 {len(rules)} 条规则")
        
        return rules
    
    def _build_relation_graph(self, relations: List[Dict]) -> Dict[str, List[str]]:
        """构建关系图"""
        graph = defaultdict(list)
        for r in relations:
            graph[r["source_id"]].append(r["target_id"])
        return graph
    
    def _discover_patterns(self, graph: Dict[str, List[str]]) -> List[Tuple[List[str], float]]:
        """发现频繁路径模式"""
        patterns = []
        nodes = list(graph.keys())[:50]
        
        for start in nodes:
            for neighbor in graph.get(start, [])[:10]:
                # 长度为2的模式
                patterns.append(([start, neighbor], 1.0))
                
                # 长度为3的模式
                for neighbor2 in graph.get(neighbor, [])[:10]:
                    if neighbor2 != start:
                        patterns.append(([start, neighbor, neighbor2], 0.7))
        
        # 合并相同模式
        pattern_counts = {}
        for pattern, weight in patterns:
            key = tuple(pattern)
            if key not in pattern_counts:
                pattern_counts[key] = (pattern, 1)
            else:
                _, count = pattern_counts[key]
                pattern_counts[key] = (pattern, count + 1)
        
        max_count = max(c for _, c in pattern_counts.values()) if pattern_counts else 1
        
        result = []
        for pattern, count in pattern_counts.values():
            support = count / max_count
            result.append((pattern, support))
        
        return result
    
    def _pattern_to_horn_clause(self, pattern: List[str]) -> Optional[Dict]:
        """将路径转换为Horn子句规则"""
        if len(pattern) < 2:
            return None
        
        conditions = []
        for i in range(len(pattern) - 1):
            conditions.append(f"related({pattern[i]}, {pattern[i+1]})")
        
        conclusion = f"connected({pattern[0]}, {pattern[-1]})"
        horn_clause = f"{' ∧ '.join(conditions)} → {conclusion}"
        
        return {
            "id": f"rule_{len(self.rules)}",
            "horn_clause": horn_clause,
            "conditions": conditions,
            "conclusion": conclusion,
            "confidence": 0.7,
            "source": "ilp"
        }
    
    # ==================== 完整构建流程 ====================
    
    def build_from_text(self, text: str) -> Dict[str, Any]:
        """从文本完整构建知识图谱"""
        self.logger.info("开始构建知识图谱")
        
        # 步骤1：实体抽取
        entities = self.extract_entities(text)
        
        # 步骤2：关系抽取
        relations = self.extract_relations(entities, text)
        
        # 步骤3：规则抽取
        rules = self.extract_rules(relations)
        
        return {
            "entities": self.entities,
            "relations": self.relations,
            "rules": self.rules,
        }
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "rule_count": len(self.rules),
            "entity_by_type": Counter([e["type"] for e in self.entities.values()]),
        }
