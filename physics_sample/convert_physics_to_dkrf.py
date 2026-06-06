#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
物理实验知识图谱 → DKRF标准格式转换器

将PhysicsKGExtractor提取的experiments.json转换为DKRF框架的KnowledgeGraph格式
"""

import json
import hashlib
from typing import Dict, List, Any
from pathlib import Path


class PhysicsToDKRFConverter:
    """物理实验KG到DKRF格式转换器"""
    
    def __init__(self, physics_json_path: str):
        """
        初始化转换器
        
        Args:
            physics_json_path: 物理实验JSON文件路径
        """
        self.physics_json_path = physics_json_path
        self.physics_data = self._load_physics_data()
    
    def _load_physics_data(self) -> Dict[str, Any]:
        """加载物理实验数据"""
        with open(self.physics_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _generate_id(self, prefix: str, counter: int) -> str:
        """生成实体ID"""
        return f"{prefix}_{counter:04d}"
    
    def convert(self) -> Dict[str, Any]:
        """
        转换为DKRF标准格式
        
        Returns:
            DKRF格式的知识图谱字典，包含：
            - metadata: 元数据
            - entities: 实体字典 {id: {name, type, attributes}}
            - relations: 关系列表 [{source, target, type, confidence}]
            - rules: 规则列表 [{horn_clause, conditions, conclusion, confidence}]
        """
        entities = {}
        relations = []
        rules = []
        
        # 计数器
        exp_counter = 0
        inst_counter = 0
        step_counter = 0
        rel_counter = 0
        rule_counter = 0
        
        # 统计
        stats = {
            "experiments": 0,
            "instruments": 0,
            "steps": 0,
            "relations": 0,
            "rules": 0
        }
        
        for exp in self.physics_data.get("experiments", []):
            exp_data = exp["data"]
            exp_name = exp_data["name"]
            page = exp["metadata"].get("page", 0)
            
            # ========== 实体：实验（concept类型） ==========
            exp_id = self._generate_id("exp", exp_counter)
            entities[exp_id] = {
                "id": exp_id,
                "name": exp_name,
                "type": "concept",
                "attributes": {
                    "purpose": exp_data.get("purpose", ""),
                    "theory": exp_data.get("theory", "")[:500],  # 限制长度
                    "page": page,
                    "source": exp["metadata"].get("source", "大学物理实验教程")
                }
            }
            stats["experiments"] += 1
            exp_counter += 1
            
            # ========== 实体：仪器设备（instrument类型） ==========
            for eq in exp_data.get("equipment", []):
                eq_name = eq.get("name", "")
                if not eq_name:
                    continue
                
                inst_id = self._generate_id("inst", inst_counter)
                entities[inst_id] = {
                    "id": inst_id,
                    "name": eq_name,
                    "type": "instrument",
                    "attributes": {
                        "quantity": eq.get("quantity", 1),
                        "belongs_to": exp_name
                    }
                }
                stats["instruments"] += 1
                inst_counter += 1
                
                # 关系：实验包含仪器 (contains)
                rel_id = self._generate_id("rel", rel_counter)
                relations.append({
                    "id": rel_id,
                    "source_id": exp_id,
                    "source_name": exp_name,
                    "target_id": inst_id,
                    "target_name": eq_name,
                    "type": "contains",
                    "confidence": 0.9,
                    "source": "extraction"
                })
                stats["relations"] += 1
                rel_counter += 1
            
            # ========== 实体：实验步骤（procedure类型） ==========
            steps = exp_data.get("steps", [])
            for step_idx, step in enumerate(steps):
                if not step or len(step) < 5:
                    continue
                
                step_name = step[:50] if len(step) > 50 else step
                step_id = self._generate_id("step", step_counter)
                entities[step_id] = {
                    "id": step_id,
                    "name": step_name,
                    "type": "procedure",
                    "attributes": {
                        "order": step_idx,
                        "full_text": step,
                        "belongs_to": exp_name
                    }
                }
                stats["steps"] += 1
                step_counter += 1
                
                # 关系：实验包含步骤 (contains)
                rel_id = self._generate_id("rel", rel_counter)
                relations.append({
                    "id": rel_id,
                    "source_id": exp_id,
                    "source_name": exp_name,
                    "target_id": step_id,
                    "target_name": step_name,
                    "type": "contains",
                    "confidence": 0.85,
                    "source": "extraction"
                })
                stats["relations"] += 1
                rel_counter += 1
                
                # ========== 规则：步骤顺序关系 ==========
                if step_idx > 0:
                    prev_step = steps[step_idx - 1]
                    prev_name = prev_step[:50] if len(prev_step) > 50 else prev_step
                    curr_name = step_name
                    
                    rule_id = self._generate_id("rule", rule_counter)
                    rules.append({
                        "id": rule_id,
                        "horn_clause": f"完成({prev_name}) → 进行({curr_name})",
                        "conditions": [f"完成({prev_name})"],
                        "conclusion": f"进行({curr_name})",
                        "confidence": 0.8,
                        "concept": exp_name,
                        "source": "step_order"
                    })
                    stats["rules"] += 1
                    rule_counter += 1
            
            # ========== 规则：数据处理规则 ==========
            data_proc = exp_data.get("data_processing")
            if data_proc and len(data_proc) > 10:
                proc_short = data_proc[:60] if len(data_proc) > 60 else data_proc
                rule_id = self._generate_id("rule", rule_counter)
                rules.append({
                    "id": rule_id,
                    "horn_clause": f"实验完成 → 应用({proc_short})",
                    "conditions": ["实验完成"],
                    "conclusion": f"应用({proc_short})",
                    "confidence": 0.7,
                    "concept": exp_name,
                    "source": "data_processing"
                })
                stats["rules"] += 1
                rule_counter += 1
        
        # 生成唯一标识符
        data_hash = hashlib.md5(
            json.dumps(entities, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        return {
            "metadata": {
                "source": self.physics_json_path,
                "converted_at": self.physics_data.get("metadata", {}).get("generated_at", ""),
                "original_experiments": len(self.physics_data.get("experiments", [])),
                "conversion_version": "1.0",
                "kg_id": f"physics_kg_{data_hash}"
            },
            "statistics": stats,
            "entities": entities,
            "relations": relations,
            "rules": rules
        }
    
    def save_dkrf_format(self, output_path: str) -> None:
        """保存为DKRF格式JSON"""
        dkrf_kg = self.convert()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dkrf_kg, f, ensure_ascii=False, indent=2)
        print(f"DKRF格式知识图谱已保存至: {output_path}")
        print(f"统计: {dkrf_kg['statistics']}")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="物理实验KG转DKRF格式")
    parser.add_argument("input", help="输入的experiments.json文件路径")
    parser.add_argument("-o", "--output", default="dkrf_physics_kg.json", 
                        help="输出的DKRF格式JSON文件路径")
    args = parser.parse_args()
    
    converter = PhysicsToDKRFConverter(args.input)
    converter.save_dkrf_format(args.output)


if __name__ == "__main__":
    main()
