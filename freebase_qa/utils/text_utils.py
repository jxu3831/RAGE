import re
from typing import List
from typing import Tuple

class TextUtils:
    @staticmethod
    def extract_answer(text: str) -> str:
        """从文本中提取答案"""
        start = text.find("{")
        end = text.find("}")
        return text[start+1:end].strip() if start != -1 and end != -1 else ""
    
    @staticmethod
    def is_true(text: str) -> bool:
        """检查文本是否表示肯定"""
        return text.lower().strip().replace(" ", "") == "yes"
    
    @staticmethod
    def is_yes_in_response(response: str) -> bool:
        """判断响应中是否出现独立的 'Yes'（区分大小写）"""
        if not response:
            return False
        
        # 使用正则匹配独立的 "Yes"
        # \b 表示单词边界，确保匹配的是完整单词
        return bool(re.search(r'\bYes\b', response))

    @staticmethod
    def check_finish(entities: List[str]) -> Tuple[bool, List[str]]:
        """检查实体列表是否完成"""
        if all(e == "[FINISH_ID]" for e in entities):
            return True, []
        return False, [e for e in entities if e != "[FINISH_ID]"]
    
    @staticmethod
    def filter_unknown_entities(entities: List[str]) -> List[str]:
        """过滤未知实体"""
        if len(entities) == 1 and entities[0] == "UnName_Entity":
            return entities
        return [e for e in entities if e != "UnName_Entity"]
    
    @staticmethod
    def extract_json(text):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        raise ValueError("No JSON object found in text")