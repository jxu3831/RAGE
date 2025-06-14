import os
import sys
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

class EntityIndexBuilder:
    """实体索引构建器，支持多种实体表示方式和索引类型"""
    
    def __init__(self, mode: str = "entity_minilm-L6", device: str = "cuda:0"):
        """
        初始化索引构建器
        :param mode: 运行模式，决定使用的模型类型
        :param device: 计算设备，如"cuda:0"或"cpu"
        """
        # 配置日志系统
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('entity_index_builder.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 切换到脚本所在目录
        os.chdir(sys.path[0])
        
        # 初始化配置
        self.mode = mode
        self.device = device
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
        
        # 模型路径配置
        self.MODEL_PATHS = {
            "entity_minilm-L12": "all-MiniLM-L12-v2",
            "entity_minilm-L6": "all-MiniLM-L6-v2"
        }
        
        # 加载模型
        self.model = self._load_model()
        
    def _load_model(self) -> SentenceTransformer:
        """加载Sentence Transformer模型"""
        model_path = self.MODEL_PATHS.get(self.mode, self.MODEL_PATHS[self.mode])
        try:
            self.logger.info(f"Loading {self.mode} model from: {model_path}")
            model = SentenceTransformer(model_path)
            model.eval()
            # model.to(self.device)
            self.logger.info(f"Model loaded successfully, device: {next(model.parameters()).device}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def load_entities(self, entity_file: str, use_triples: bool = False) -> Tuple[List[str], List[str]]:
        """
        加载实体数据
        :param entity_file: 实体文件路径
        :param use_triples: 是否使用三元组信息增强实体表示
        :return: (names列表, 文本列表)
        """
        names = []
        texts = []
        
        try:
            with open(entity_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Loading Entities"):
                    try:
                        if entity_file.endswith('.jsonl'):
                            # 处理JSONL格式（包含三元组）
                            obj = json.loads(line)
                            name = obj['name']
                            if use_triples:
                                triples = obj.get('triples', [])
                                components = [name]
                                for sub, pred, obj_ in triples[:3]:  # 最多取3个三元组
                                    components.append(f"{pred} {obj_}")
                                text = " [SEP] ".join(components)
                            else:
                                text = name
                        else:
                            # 处理filter entity.txt格式（每行一个实体名）
                            name = line.strip()
                            text = name
                            # if not line:  # 跳过空行
                            #     continue

                            # parts = line.split('\t')  # 使用制表符分割，因为示例中是制表符分隔
                            # if len(parts) >= 2:  # 确保至少有两列
                            #     name = parts[1].strip()
                            #     text = name

                        if name:  # 确保不是空行
                            names.append(name)
                            texts.append(text)
                    except (json.JSONDecodeError, KeyError) as e:
                        self.logger.warning(f"Skipping malformed line: {line.strip()}. Error: {e}")
                        continue
                        
            self.logger.info(f"Loaded {len(names)} entities")
            return names, texts
        except Exception as e:
            self.logger.error(f"Error loading entity file: {e}")
            raise

    def encode_entities(self, texts: List[str], batch_size: int = 512) -> np.ndarray:
        """
        批量编码实体文本为嵌入向量
        :param texts: 文本列表
        :param batch_size: 批处理大小
        :return: 嵌入矩阵 (n_entities, embedding_dim)
        """
        try:
            self.logger.info(f"Encoding {len(texts)} entities with batch_size={batch_size}")
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
                output_value="sentence_embedding"
            )
            self.logger.info(f"Embeddings shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            self.logger.error(f"Encoding failed: {e}")
            raise

    def build_index(self, 
                   entity_file: str, 
                   output_dir: str = "index",
                   use_triples: bool = False,
                   index_suffix: Optional[str] = None) -> None:
        """
        构建并保存实体索引
        :param entity_file: 实体文件路径
        :param output_dir: 输出目录
        :param use_triples: 是否使用三元组信息
        :param index_suffix: 索引文件后缀（如"e&n"表示实体和邻居）
        """
        try:
            # 加载实体数据
            names, texts = self.load_entities(entity_file, use_triples)
            
            # 编码实体
            embeddings = self.encode_entities(texts)
            
            # 保存结果
            self._save_index(output_dir, names, embeddings, index_suffix)
            
            self.logger.info(f"Entity index built successfully in {self.mode} mode")
        except Exception as e:
            self.logger.critical(f"Index building failed: {e}")
            raise

    def _save_index(self, 
                output_dir: str,
                names: List[str],
                embeddings: np.ndarray,
                suffix: Optional[str] = None) -> None:
        """
        保存索引文件
        :param output_dir: 输出目录
        :param names: 实体名称列表
        :param embeddings: 嵌入矩阵
        :param suffix: 文件后缀
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 构建文件名
            base_name = 'L6'
            if suffix:
                base_name = f"{base_name}_{suffix}"
            
            # 保存嵌入向量为 .npy 文件
            embeddings_path = os.path.join(output_dir, f"{base_name}_embeddings.npy")
            np.save(embeddings_path, embeddings)
            
            # 保存实体名称列表为 JSON 文件
            names_path = os.path.join(output_dir, f"{base_name}_names.json")
            with open(names_path, 'w', encoding='utf-8') as f:
                json.dump(names, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved embeddings to {embeddings_path}")
            self.logger.info(f"Saved names to {names_path}")
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
            raise

if __name__ == "__main__":
    try:
        # 配置参数
        config = {
            "mode": "entity_minilm-L6",  # 运行模式
            "device": "cuda:4",    # 计算设备
            "entity_file": "../Freebase/index/entity_names.txt",  # 实体文件
            "output_dir": "index",  # 输出目录
            "use_triples": False,    # 是否使用三元组
            "index_suffix": "dedup"      # 索引文件后缀
        }
        
        # 构建索引
        builder = EntityIndexBuilder(mode=config["mode"], device=config["device"])
        builder.build_index(
            entity_file=config["entity_file"],
            output_dir=config["output_dir"],
            use_triples=config["use_triples"],
            index_suffix=config["index_suffix"]
        )
        
    except Exception as e:
        logging.critical(f"Fatal error: {e}")
        sys.exit(1)