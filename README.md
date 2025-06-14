## Project Structure
+ `data/`: 数据集
+ `Freebase/`: 知识图谱及索引文件保存位置
+ `freebase_qa/`: 源代码
+ `requirements.txt`: 项目运行环境

## 知识图谱数据下载与处理
本过程尽量保证全程在`Freebase/`文件夹下运行。
1. 运行以下[Freebase Setup](https://github.com/GasolSun36/ToG/tree/main/Freebase)中的数据下载与处理步骤。后续代码测试过程中要保证`virtuoso`处于运行状态，后台启动`virtuoso`数据库主要依赖`../bin/virtuoso-t`命令。
2. 运行`filter_entities.py`，过滤出所有的实体名称。
3. 运行`build_index.py`构建索引。

## Run
1. 运行源代码前先修改`llm_handler.py`中[`run_llm`](https://github.com/jxu3831/RAGE/blob/main/freebase_qa/core/llm_handler.py)大模型部署方式。
2. 按以下步骤运行
```
cd ../freebase_qa/
python main.py
```
