from utils.eval_utils import *
from config.settings import MODEL
from utils.file_utils import FileUtils
import os, sys
os.chdir(sys.path[0])  #使用文件所在目录

def eval_em(dataset, output_file, model_name, constraints_refuse=True):
    model_stem = model_name.split("/")[-1]
    ground_truth_datas, question_string, output_datas = prepare_dataset_for_eval(dataset, output_file)

    num_right = 0
    num_error = 0
    for data in output_datas:
        answers = align(dataset, question_string, data, ground_truth_datas)
        results = data['results']
        if check_string(results):
            response = clean_results(results)
            if exact_match(response, answers):
                num_right+=1
            else:
                num_error+=1
        else:
            response = results
            if constraints_refuse and check_string(response):
                continue
            if exact_match(response, answers):
                num_right+=1
            else:
                num_error+=1

    print("Exact Match: {}".format(float(num_right/len(output_datas))))
    print("right: {}, error: {}".format(num_right, num_error))

    save_result2json(dataset, num_right, num_error, len(output_datas), model=model_stem)


if __name__ == '__main__':
    dataset = 'webqsp'
    model = MODEL['qwen']
    model_stem = model.split("/")[-1]
    jsonl_file = "./RAGE_webqsp_llama.jsonl"
    json_file = "./RAGE_webqsp_llama.json"
    FileUtils.jsonl2json(jsonl_file, json_file)
    
    eval_em(dataset, json_file, model)
    