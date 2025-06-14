import re
from tqdm import tqdm

# 输入文件（已过滤非英语）
input_file = "FilterFreebase"
output_file = "index/entity_names.txt"

# 正则匹配 type.object.name 关系的三元组
pattern = re.compile(r'<http://rdf.freebase.com/ns/(m\..*?)>\s+<http://rdf.freebase.com/ns/type.object.name>\s+"(.*?)"@en')

# 使用集合来存储唯一的实体名称
unique_names = set()

with open(input_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Processing lines"):
        match = pattern.search(line)
        if match:
            _, name = match.groups()  # 我们只需要name，不需要mid
            unique_names.add(name.strip())  # 使用集合自动去重

# 保存到 TXT 文件
with open(output_file, "w", encoding="utf-8") as f:
    for name in sorted(unique_names):  # 按字母顺序排序
        f.write(f"{name}\n")  # 只写入名称，每行一个

print(f"提取完成，共找到 {len(unique_names)} 个唯一且有意义的实体名称，保存在 {output_file}")
