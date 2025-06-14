import json
from pathlib import Path
from typing import Dict, List, Any, Set
from utils.logging_utils import logger

class FileUtils:
    @staticmethod
    def save_to_jsonl(data: Dict, filename: str) -> bool:
        """保存数据到JSONL文件"""
        try:
            with open(filename, 'a') as f:
                json.dump(data, f)
                f.write('\n')
            return True
        except IOError as e:
            logger.error(f"Error saving to {filename}: {e}")
            return False
    
    @staticmethod
    def load_processed_questions(filename: str) -> Set[str]:
        """加载已处理的问题"""
        questions = set()
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if 'question' in entry:
                        questions.add(entry['question'])
        except FileNotFoundError:
            pass
        return questions
    
    @staticmethod
    def jsonl2json(jsonl_file: str, json_file: str) -> Set[str]:
        with open(jsonl_file, 'r') as infile:
            with open(json_file, 'a', encoding='utf-8') as outfile:
                json_lines = infile.readlines()
                json_list = [json.loads(line) for line in json_lines]
                json.dump(json_list, outfile, indent=4)
