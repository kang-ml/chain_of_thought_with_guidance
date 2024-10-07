from tqdm import tqdm
import time
import json
import os
from guidance import models, gen, select, system, user, assistant, regex
import re
import guidance
from gpt_function import proofwriter_reasoning
from demo import demos

total_data = json.load(open('chain_of_thought_with_guidance/ProofWriter/test_data.json'))

####################################
# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = ''
####################################

TRY_REASON_CNT = 5
correct = 0
total = 0
all_data = []

ans_dict = {'A': "True", 'B': "False", 'C': 'Unknown'}
for idx in tqdm(range(len(total_data))):
    total += 1
    d = total_data[idx]
    premise=d['context']
    q=d['question']
    true_ans = ans_dict[d['answer']]

    try_reason_cnt = 0
    reason_success = False
    while try_reason_cnt < TRY_REASON_CNT:
        try:
            #########################################
            # select a gpt model
            gpt = models.OpenAI("gpt-3.5-turbo")
            #########################################

            gpt.echo = False
            gpt += proofwriter_reasoning(demos=demos, premise=premise, q=q)
            predict_reason = gpt['reason']
            matches = re.findall(r"\b(true|false|unknown)\b", gpt['ans'], re.IGNORECASE)
            if matches:
                prediction = matches[0]
                reason_success = True
                d['predicted_reason'] = predict_reason
                d['predicted_answer'] = prediction
                if prediction.lower() == true_ans.lower():
                    correct += 1
                break
            else:
                print("Selection failed, try again... (No. {})".format(try_reason_cnt+1))
                try_reason_cnt += 1
                time.sleep(2)
                continue
        except Exception as e:
            print("Reasoning failed, try again... (No. {})".format(try_reason_cnt+1))
            try_reason_cnt += 1
            time.sleep(2)
            continue

    all_data.append(d)

print("Correct:", correct)
print("Total:", total)
print("Accuracy:", correct/total)

with open('chain_of_thought_with_guidance/ProofWriter/test_result.json', 'w') as f:
    json.dump(all_data, f)