import numpy as np
from typing import Dict
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING
from utils.logging_utils import logger
from pathlib import Path
import faiss
import json
import os
import torch

class SemanticSearch:
    def __init__(self):
        """初始化语义搜索引擎"""
        self.embeddings_path = EMBEDDING
        self.faiss_index = self._load_index()
        logger.info("Semantic search engine initialized")


    def _load_index(self) -> None:
        """从.pt文件加载实体索引"""
        # try:
            
        #     # 加载.pt文件
        #     pt_path = SEMANTIC_SEARCH_CONFIG['pt_path']
        #     data = torch.load(pt_path)
        #     self.names = data['names']
        #     self.embeddings = data['embeddings'].numpy()

        """从.npy和.json文件加载实体索引"""
        try:
            # 加载嵌入向量(.npy)
            embeddings_path = self.embeddings_path['embeddings_path']  # 假设配置中已更新路径
            self.embeddings = np.load(embeddings_path)
            
            # 加载实体名称(.json)
            names_path = self.embeddings_path['names_path']
            with open(names_path, 'r', encoding='utf-8') as f:
                self.names = json.load(f)
            
            # 检查数据一致性
            if len(self.names) != len(self.embeddings):
                raise ValueError(
                    f"数据不一致: names长度({len(self.names)}) != embeddings长度({len(self.embeddings)})"
                )

            # 归一化嵌入向量
            faiss.normalize_L2(self.embeddings)
            
            # 构建FAISS索引
            dimension = self.embeddings.shape[1]
            if len(self.embeddings) > 100000:
                nlist = min(100, len(self.embeddings)//1000)
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 8, 8)
                index.train(self.embeddings)
                index.add(self.embeddings)
                index.nprobe = min(10, nlist)
            else:
                index = faiss.IndexFlatIP(dimension)
                index.add(self.embeddings)
            
            self.entity_data = {
                "names": self.names,
                "embeddings": self.embeddings,
                "index": index
            }
            # logger.info(f"Loaded {len(self.names)} entities from {pt_path}")
            logger.info(f"Loaded {len(self.names)} entities (embeddings: {embeddings_path}, names: {names_path})")
            return self.entity_data["index"]
            
        except Exception as e:
            logger.error(f"Index loading failed: {e}")
            raise
