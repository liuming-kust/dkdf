#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
物理实验知识图谱提取器
从大学物理实验PDF中提取结构化实验信息

功能：
1. PDF加载与智能分块（按实验结构分割）
2. LLM提取实验信息（名称、目的、原理、仪器、步骤、数据处理）
3. 增量更新与去重合并
4. 输出标准JSON格式

与DKRF框架集成：
- 输出格式与HKG模块兼容
- 支持转换为DKRF标准知识图谱格式
"""

import json
import time
import re
import hashlib
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# LangChain依赖
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.llms import Ollama

# 进度条
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dkrf_physics_extractor.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PhysicsKGExtractor:
    """
    物理实验知识图谱提取器
    
    与DKRF框架对应关系：
    - 实体抽取：实验名称、仪器设备、实验步骤
    - 关系抽取：实验-仪器关联、实验-步骤关联
    - 增量更新：对应DRD内环的去重合并逻辑
    """
    
    def __init__(self, output_path: str = "experiments.json"):
        """
        初始化提取器
        
        Args:
            output_path: 输出JSON文件路径
        """
        self.output_path = output_path
        self.existing_experiments = self._load_existing_data()
        logger.info(f"初始化完成，已加载 {len(self.existing_experiments)} 个已有实验")
    
    def _load_existing_data(self) -> Dict[str, Any]:
        """加载已存在的实验数据"""
        try:
            with open(self.output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {exp["data"]["name"]: exp for exp in data.get("experiments", [])}
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("未找到已有数据文件，将创建新文件")
            return {}
    
    def _generate_experiment_hash(self, experiment: Dict[str, Any]) -> str:
        """生成实验数据的唯一哈希值"""
        hash_data = {
            "name": experiment["name"],
            "purpose": experiment["purpose"],
            "theory": experiment["theory"][:500] if experiment.get("theory") else "",  # 限制长度
            "steps": "|".join(experiment.get("steps", []))[:1000]
        }
        return hashlib.md5(json.dumps(hash_data, sort_keys=True).encode()).hexdigest()
    
    def _merge_experiments(self, existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """合并两个相似实验的数据"""
        merged = existing.copy()
        
        # 合并数据处理方法
        if new.get("data_processing"):
            existing_dp = existing.get("data_processing", "")
            if not existing_dp:
                merged["data_processing"] = new["data_processing"]
            elif new["data_processing"] not in existing_dp:
                merged["data_processing"] = f"{existing_dp}\n{new['data_processing']}"
        
        # 合并仪器列表（去重）
        existing_eq = {eq["name"]: eq for eq in existing.get("equipment", [])}
        for eq in new.get("equipment", []):
            if eq["name"] not in existing_eq:
                existing_eq[eq["name"]] = eq
        merged["equipment"] = list(existing_eq.values())
        
        # 合并步骤（保留顺序，去重）
        existing_steps = existing.get("steps", [])
        for step in new.get("steps", []):
            if step not in existing_steps:
                existing_steps.append(step)
        merged["steps"] = existing_steps
        
        return merged
    
    def pdf_to_chunks(self, file_path: str) -> List[Any]:
        """PDF转文本块（按实验结构智能分割）"""
        logger.info(f"加载PDF: {file_path}")
        
        loader = PyPDFLoader(file_path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            separators=[
                r'\n(?=\d+\.\d+\s+.+实验)',      # 匹配如"3.3 刚体转动惯量的测定"
                r'\n(?=\[实验目的\])',
                r'\n(?=\[实验原理\])',
                r'\n(?=\[实验仪器用具\])',
                r'\n(?=\[实验内容及步骤\])',
                r'\n(?=\[实验数据处理\])',
                "\n\n"
            ],
            keep_separator=True
        )
        
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        logger.info(f"PDF加载完成，共 {len(documents)} 页，分割为 {len(chunks)} 个文本块")
        return chunks
    
    def _build_prompt(self) -> ChatPromptTemplate:
        """构建提取提示模板"""
        return ChatPromptTemplate.from_messages([
            ("system", """
你是物理实验专家。请从文本中提取标准物理实验的结构化信息。

【输出JSON格式】
{{
  "experiment": {{
    "name": "实验编号和名称（如'3.3 刚体转动惯量的测定'）",
    "purpose": "实验目的",
    "theory": "实验原理（包含核心公式）",
    "equipment": [
      {{"name": "仪器名称", "quantity": 1}}
    ],
    "steps": ["步骤1", "步骤2", "步骤3"],
    "data_processing": "数据处理方法（可选）"
  }}
}}

【要求】
1. 实验名称必须包含编号（如"3.3"）
2. 仪器列表中每个仪器必须有name字段
3. 步骤按顺序列出
4. 如果某部分缺失，使用null或空列表
5. 只输出JSON，不要有其他内容
"""),
            ("human", "请从以下实验文本中提取结构化信息：\n{text}")
        ])
    
    def extract_experiments(self, chunks: List[Any]) -> List[Dict[str, Any]]:
        """从文本块中提取实验信息"""
        llm = Ollama(model="qwen2.5:14b", temperature=0.1)
        chain = self._build_prompt() | llm | JsonOutputParser()
        
        experiments = []
        
        for chunk in tqdm(chunks, desc="提取实验信息"):
            try:
                clean_text = re.sub(r'\n{3,}', '\n\n', chunk.page_content)
                result = chain.invoke({"text": clean_text})
                
                if self._validate_experiment(result):
                    experiments.append({
                        "metadata": {
                            "page": chunk.metadata.get("page", 0),
                            "source": "大学物理实验教程"
                        },
                        "data": result["experiment"]
                    })
                    logger.debug(f"成功提取: {result['experiment']['name']}")
            except Exception as e:
                logger.warning(f"提取失败: {str(e)[:100]}")
        
        logger.info(f"提取完成，共提取 {len(experiments)} 个实验")
        return experiments
    
    def _validate_experiment(self, data: Dict[str, Any]) -> bool:
        """验证实验数据格式"""
        required = ["name", "purpose", "theory", "equipment", "steps"]
        
        if "experiment" not in data:
            return False
        exp = data["experiment"]
        
        # 校验必填字段
        if not all(field in exp for field in required):
            return False
        
        # 校验实验名称格式（应包含编号）
        if not re.match(r'^\d+\.\d+\s+', exp["name"]):
            return False
        
        # 校验仪器格式
        if not isinstance(exp["equipment"], list):
            return False
        for item in exp["equipment"]:
            if not isinstance(item, dict) or "name" not in item:
                return False
            item["quantity"] = item.get("quantity", 1)
        
        # 校验步骤
        if not isinstance(exp["steps"], list) or len(exp["steps"]) == 0:
            return False
        
        return True
    
    def merge_with_existing(self, new_experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """与已有实验合并（去重和融合）"""
        merged = self.existing_experiments.copy()
        
        for exp in new_experiments:
            exp_data = exp["data"]
            exp_name = exp_data["name"]
            exp_hash = self._generate_experiment_hash(exp_data)
            
            if exp_name in merged:
                existing_hash = self._generate_experiment_hash(merged[exp_name]["data"])
                if exp_hash == existing_hash:
                    logger.debug(f"跳过重复实验: {exp_name}")
                    continue
                
                # 合并相似实验
                merged_data = self._merge_experiments(
                    merged[exp_name]["data"],
                    exp_data
                )
                merged[exp_name]["data"] = merged_data
                
                # 合并页面信息
                if "pages" not in merged[exp_name]["metadata"]:
                    merged[exp_name]["metadata"]["pages"] = [
                        merged[exp_name]["metadata"].get("page", 0)
                    ]
                merged[exp_name]["metadata"]["pages"].append(exp["metadata"]["page"])
                logger.info(f"合并实验: {exp_name}")
            else:
                merged[exp_name] = exp
                logger.info(f"新增实验: {exp_name}")
        
        return list(merged.values())
    
    def save_results(self, data: List[Dict[str, Any]], source_file: str, duration: float) -> None:
        """保存结果到JSON文件"""
        output = {
            "metadata": {
                "source": source_file,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "processing_time": f"{duration:.2f}s",
                "experiment_count": len(data),
                "version": "2.0"
            },
            "experiments": data
        }
        
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存至 {self.output_path}，共 {len(data)} 个实验")
    
    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """完整处理流程"""
        start_time = time.time()
        
        try:
            # 1. PDF转文本块
            chunks = self.pdf_to_chunks(file_path)
            
            # 2. 提取实验信息
            new_experiments = self.extract_experiments(chunks)
            
            # 3. 与已有实验合并
            merged_experiments = self.merge_with_existing(new_experiments)
            
            # 4. 保存结果
            self.save_results(merged_experiments, file_path, time.time() - start_time)
            
            logger.info(f"处理完成，总耗时 {time.time() - start_time:.2f}s")
            return merged_experiments
            
        except Exception as e:
            logger.error(f"处理失败: {str(e)}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_experiments": len(self.existing_experiments),
            "output_path": self.output_path,
            "version": "2.0"
        }


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="物理实验知识图谱提取器")
    parser.add_argument("pdf_path", help="PDF文件路径")
    parser.add_argument("-o", "--output", default="experiments.json", help="输出JSON文件路径")
    args = parser.parse_args()
    
    extractor = PhysicsKGExtractor(output_path=args.output)
    extractor.process_pdf(args.pdf_path)


if __name__ == "__main__":
    main()
