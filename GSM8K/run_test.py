from tqdm import tqdm
import time
import json
import os
from guidance import models, gen, select, system, user, assistant, regex
import re
import guidance
from gpt_function import gsm8k_reasoning
from demo import demos

file = 'chain_of_thought_with_guidance/GSM8K/test_data.jsonl'
with open(file) as f:
    total_data = [json.loads(line) for line in f]

####################################
# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = ''
####################################

TRY_REASON_CNT = 5
correct = 0
total = 0
all_data = []
for idx in tqdm(range(len(total_data))):
    total += 1
    d = total_data[idx]
    query = d['question']
    true_ans = d['answer']

    try_reason_cnt = 0
    reason_success = False
    while try_reason_cnt < TRY_REASON_CNT:
        try:
            ##################################
            # select a gpt model
            gpt = models.OpenAI("gpt-3.5-turbo")
            ##################################
            gpt.echo = False
            gpt += gsm8k_reasoning(demos=demos, query=query)
            pred_reason = gpt['reason']
            matches = re.findall(r'\d+', gpt['ans'])
            if matches:
                reason_success = True
                pred_ans = int(matches[0])
                d['predicted_reason'] = pred_reason
                d['predicted_answer'] = pred_ans
                if pred_ans == true_ans:
                    correct += 1
                break
            else:
                print("Selection failed, try again... (No. {})".format(try_reason_cnt+1))
                try_reason_cnt += 1
                time.sleep(2)
                continue
        except Exception as e:
            print("Reasoning failed, try again... (No. {})".format(try_reason_cnt+1), "Error:", e)
            try_reason_cnt += 1
            time.sleep(2)
            continue
    
    all_data.append(d)

print("Correct:", correct)
print("Total:", total)
print("Accuracy:", correct/total)

with open('chain_of_thought_with_guidance/GSM8K/test_result.json', 'w') as f:
    json.dump(all_data, f)