# -*- coding: utf-8 -*-
"""知识图谱加载器 - 从 experiments.json 加载图谱数据"""

import json
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

from .utils import setup_logger


class KnowledgeGraphLoader:
    """
    知识图谱加载器
    加载 experiments.json 并提取实体、关系、规则
    """
    
    def __init__(self, kg_path: str):
        self.kg_path = kg_path
        self.logger = setup_logger("KGLoader")
        self.raw_data = self._load()
        self.entities = {}      # 实体字典
        self.relations = []     # 关系列表
        self.rules = []         # 规则列表
        self.core_concepts = {} # 核心概念
        
        # 构建图谱
        self._build_graph()
    
    def _load(self) -> Dict:
        """加载JSON文件"""
        with open(self.kg_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_graph(self):
        """构建知识图谱"""
        entity_counter = 0
        
        for exp in self.raw_data.get("experiments", []):
            exp_data = exp.get("data", {})
            name = exp_data.get("name", "")
            if not name:
                continue
            
            # 核心概念（实验名称）
            self.core_concepts[name] = {
                "purpose": exp_data.get("purpose", ""),
                "theory": exp_data.get("theory", ""),
                "key_entities": [],
                "key_relations": [],
                "source_pages": exp.get("metadata", {}).get("pages", []),
            }
            
            # 实体抽取：设备
            for equipment in exp_data.get("equipment", []):
                eq_name = equipment.get("name", "")
                if eq_name:
                    entity_id = f"entity_{entity_counter}"
                    self.entities[entity_id] = {
                        "id": entity_id,
                        "name": eq_name,
                        "type": "instrument",
                        "belongs_to": name
                    }
                    self.core_concepts[name]["key_entities"].append(eq_name)
                    entity_counter += 1
            
            # 实体抽取：步骤中的关键操作
            for step in exp_data.get("steps", []):
                step_name = step if isinstance(step, str) else step.get("name", "")
                if step_name and len(step_name) > 2:
                    entity_id = f"entity_{entity_counter}"
                    self.entities[entity_id] = {
                        "id": entity_id,
                        "name": step_name[:30],
                        "type": "procedure",
                        "belongs_to": name
                    }
                    entity_counter += 1
            
            # 关系抽取：基于理论文本
            theory = exp_data.get("theory", "")
            if theory:
                # 从理论中提取隐含关系
                relations_found = self._extract_relations_from_text(theory, name)
                self.relations.extend(relations_found)
                self.core_concepts[name]["key_relations"] = relations_found
            
            # 规则抽取：从步骤中提取逻辑规则
            steps = exp_data.get("steps", [])
            if len(steps) >= 2:
                rule = self._extract_rule_from_steps(steps, name)
                if rule:
                    self.rules.append(rule)
        
        self.logger.info(f"知识图谱构建完成: {len(self.entities)}实体, {len(self.relations)}关系, {len(self.rules)}规则")
    
    def _extract_relations_from_text(self, text: str, concept_name: str) -> List[Dict]:
        """从文本中抽取关系"""
        relations = []
        
        # 定义关系模式
        patterns = [
            (r'(\S+?)\s*决定\s*(\S+?)', "determines"),
            (r'(\S+?)\s*影响\s*(\S+?)', "affects"),
            (r'(\S+?)\s*等于\s*(\S+?)', "equals"),
            (r'(\S+?)\s*与\s*(\S+?)\s*成正比', "proportional_to"),
            (r'(\S+?)\s*与\s*(\S+?)\s*成反比', "inversely_proportional_to"),
        ]
        
        for pattern, rel_type in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) >= 2:
                    relations.append({
                        "source": match[0],
                        "target": match[1],
                        "type": rel_type,
                        "concept": concept_name
                    })
        
        return relations
    
    def _extract_rule_from_steps(self, steps: List, concept_name: str) -> Optional[Dict]:
        """从步骤中提取规则"""
        if len(steps) < 2:
            return None
        
        # 提取步骤名称
        step_names = []
        for step in steps:
            if isinstance(step, dict):
                step_names.append(step.get("name", ""))
            else:
                step_names.append(step)
        
        step_names = [s for s in step_names if s]
        
        if len(step_names) >= 2:
            return {
                "id": f"rule_{concept_name}",
                "horn_clause": f"完成({step_names[0]}) → 进行({step_names[1]})",
                "conditions": [f"完成({step_names[0]})"],
                "conclusion": f"进行({step_names[1]})",
                "confidence": 0.8,
                "concept": concept_name
            }
        
        return None
    
    def get_core_concepts(self) -> Dict:
        """获取核心概念"""
        return self.core_concepts
    
    def get_entities(self) -> Dict:
        """获取实体"""
        return self.entities
    
    def get_relations(self) -> List:
        """获取关系"""
        return self.relations
    
    def get_rules(self) -> List:
        """获取规则"""
        return self.rules
    
    def find_related_concepts(self, text: str) -> Set[str]:
        """根据文本内容找到相关概念"""
        related = set()
        
        for concept_name, concept_data in self.core_concepts.items():
            # 检查实验名称
            if concept_name in text:
                related.add(concept_name)
                continue
            
            # 检查关键实体
            for entity in concept_data.get("key_entities", []):
                if entity and entity in text:
                    related.add(concept_name)
                    break
            
            # 检查关键关系
            for relation in concept_data.get("key_relations", []):
                source = relation.get("source", "")
                target = relation.get("target", "")
                if source and source in text:
                    related.add(concept_name)
                    break
                if target and target in text:
                    related.add(concept_name)
                    break
        
        return related
    
    def get_context_for_concepts(self, concepts: Set[str]) -> str:
        """获取概念上下文"""
        if not concepts:
            return "无相关概念"
        
        lines = []
        for name in concepts:
            data = self.core_concepts.get(name, {})
            lines.append(
                f"- {name}: 目的={data.get('purpose', '未知')[:80]}, "
                f"关键实体={', '.join(data.get('key_entities', [])[:3])}"
            )
        return "\n".join(lines)
    
    def get_all_entity_names(self) -> List[str]:
        """获取所有实体名称"""
        return [e["name"] for e in self.entities.values()]
    
    def get_concept_by_entity(self, entity_name: str) -> Optional[str]:
        """根据实体名称查找所属概念"""
        for concept_name, concept_data in self.core_concepts.items():
            if entity_name in concept_data.get("key_entities", []):
                return concept_name
            if concept_name in entity_name:
                return concept_name
        return None
    
    def export_to_dict(self) -> Dict:
        """导出为字典格式"""
        return {
            "entities": self.entities,
            "relations": self.relations,
            "rules": self.rules,
            "core_concepts": self.core_concepts,
        }
