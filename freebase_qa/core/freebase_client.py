from SPARQLWrapper import SPARQLWrapper, JSON
from typing import List, Dict, Optional
import time
from functools import lru_cache
from config.settings import SPARQL_ENDPOINT, UNKNOWN_ENTITY, FINISH_ID
from config.sparql_templates import SPARQL_TEMPLATES
from utils.logging_utils import logger

class FreebaseClient:
    def __init__(self, endpoint: str = SPARQL_ENDPOINT):
        self.sparql = SPARQLWrapper(endpoint)
        self.sparql.setReturnFormat(JSON)
    
    def execute_query(self, query: str, max_retries: int = 5) -> List[Dict]:
        """执行SPARQL查询"""
        self.sparql.setQuery(query)
        
        for attempt in range(max_retries):
            try:
                results = self.sparql.query().convert()
                return results.get("results", {}).get("bindings", [])
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"SPARQL query failed after {max_retries} attempts: {e}")
                    return []
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(2 * (attempt + 1))
        return []
    
    @staticmethod
    def remove_prefix(value: str) -> str:
        """移除Freebase URI前缀"""
        return value.replace("http://rdf.freebase.com/ns/", "")
    
    @lru_cache(maxsize=1000)
    def get_entity_info(self, entity_id: str) -> str:
        """获取实体名称/类型(带缓存)"""
        if entity_id == FINISH_ID:
            return FINISH_ID
            
        query = SPARQL_TEMPLATES['entity_info'] % (entity_id, entity_id)
        results = self.execute_query(query)
        return self.remove_prefix(results[0]['tailEntity']['value']) if results else UNKNOWN_ENTITY
    
    def get_entity_id(self, entity_name: str) -> List[str]:
        query = SPARQL_TEMPLATES['name2id'] % entity_name
        results = self.execute_query(query)
        return [self.remove_prefix(r['entity']['value']) for r in results]

    def get_relations(self, entity_id: str, relation_type: str = 'head') -> List[str]:
        """获取实体的关系"""
        query_template = 'head_relations' if relation_type == 'head' else 'tail_relations'
        query = SPARQL_TEMPLATES[query_template] % entity_id
        results = self.execute_query(query)
        return [self.remove_prefix(r['relation']['value']) for r in results]
    
    def get_related_entities(self, entity_id: str, relation: str, head: bool = True) -> List[str]:
        """获取相关实体"""
        if head:
            query_template = 'tail_entities'
            query = SPARQL_TEMPLATES[query_template] % (entity_id, relation)
        else:
            query_template = 'head_entities'
            query = SPARQL_TEMPLATES[query_template] % (relation, entity_id)
        results = self.execute_query(query)
        return [self.remove_prefix(r['tailEntity']['value']) for r in results 
                if r['tailEntity']['value'].startswith("http://rdf.freebase.com/ns/m.")]

    def get_all_relations(self, id: str) -> List[str]:
        """获取实体的关系"""
        query = SPARQL_TEMPLATES["relations"] % (id, id)
        results = self.execute_query(query)
        return [self.remove_prefix(r['relation']['value']) for r in results]
