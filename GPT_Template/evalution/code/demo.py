import time
import requests
import json
import os
import re
import glob

# The system prompt for this attribute
system_prompt = open('path/to/prompt_for_evaluation/non/non.txt').read()

# Using the LLM API, GPT-4 was used in the experiment.
url = "**********************************"

headers = {
    "Content-Type": "******************",
    "Authorization": "********************************"
}


# Return 10 ratings to measure the degree of alignment.
def get_score(user_prompt):
    data = {
        "model": "gpt-4o",
        "temperature": 1.5,
        "top_p": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "n": 10,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data).encode('utf-8'))
    time.sleep(0.2)
    result = response.content.decode("utf-8")
    result = json.loads(result)

    ignore = 0
    all_responses = []
    for i in range(len(result["choices"])):
        s = result["choices"][i]["message"]["content"].strip()
        s = s.replace('- Score: ', '').replace('Score: ', '').replace('Score:', '').replace('  ', ' ').strip()
        if s != '' and s.isdigit() == True and int(s) >= 1 and int(s) <= 5:
            all_responses.append(int(s))
        elif s != '' and s[0].isdigit() == True and int(s[0]) >= 1 and int(s[0]) <= 5:
            all_responses.append(int(s[0]))
        else:
            ignore += 1
    return all_responses


if __name__ == '__main__':
    # Get the user prompt with the dialoge history and model reply.
    base_prompt = '''
    Dialogue:
    [[[dialogue]]]

    Evaluation sentence:
    [[[sentence]]]
    
    Output Evaluation Form (score ONLY):
    '''
    dialoge_history = ''
    model_output = ''
    user_prompt = base_prompt.replace('[[[dialogue]]]', dialoge_history).replace('[[[sentence]]]', model_output)
    ans_list = get_score(user_prompt)
    print(ans_list)
    # Note: Some LLM responses may be incorrect, so performing anomaly detection and regeneration if necessary
