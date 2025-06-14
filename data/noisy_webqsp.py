import json
from openai import OpenAI
import time

def rewrite_question_with_gpt(question):
    """
    Rewrites the question using OpenAI's GPT model.
    The goal is to replace keywords with their common abbreviations or alternative names.
    If no such name exists, introduce small typos or paraphrasing.
    """
    prompt = f"""
You are a question rewriting assistant. Follow these rules:

1. Identify the main entity or keyword in the question (e.g., person name, place, organization).
2. Apply exactly one of the following transformations to that keyword:
   a. Slightly change the letter order (e.g., "John" -> "Jhon")
   b. Replace one letter with a similar-looking one (e.g., "capital" -> "capitol")
   c. Remove one letter from the word (e.g., "Einstein" -> "Einsten")
3. Keep the overall structure and intent of the sentence unchanged.

Return only the rewritten question, without any explanation.

Now process this question:

{question}
"""

    try:
        openai_api_base = "http://localhost:8000/v1"
        client = OpenAI(
            api_key="EMPTY",
            base_url=openai_api_base,
        )
        completion = client.chat.completions.create(
            model="Qwen/Qwen3-8B",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=100,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            stream=False,
        )
        rewritten = completion.choices[0].message.content
        return rewritten
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        return question  # Fallback to original question on failure

def process_webqsp_data(data):
    """
    Process each item in the dataset and add a noisy version of RawQuestion.
    """
    for idx, item in enumerate(data):
        raw_question = item.get("RawQuestion", "")
        if raw_question:
            print(f"[{idx + 1}/{len(data)}] Processing: {raw_question}")
            noisy_question = rewrite_question_with_gpt(raw_question)
            item["NoisyQuestion"] = noisy_question  # Add new noisy question field
        time.sleep(0.5)  # Throttle requests to avoid hitting rate limits
    return data

if __name__ == "__main__":
    INPUT_PATH = "WebQSP_sampled_600.json"
    OUTPUT_PATH = "noisy_WebQSP_sampled_600.json"

    # Load original data
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Rewrite questions
    updated_data = process_webqsp_data(data)

    # Save updated data
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)

    print(f"Processing complete. Results saved to {OUTPUT_PATH}")