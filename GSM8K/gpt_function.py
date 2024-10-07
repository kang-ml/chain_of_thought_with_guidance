from guidance import models, gen, select, system, user, assistant, regex
import os
import re
import guidance

@guidance
def gsm8k_reasoning(lm, demos, query, temp=0.1):
    with system():
        lm += '''Suppose you are one of the greatest AI scientists, logicians and mathematicians. You will solve mathematical questions. Let us think step by step.'''

    #################################################
    for demo in demos:
        question = demo['question']
        reason = demo['reason']
        ans = demo['ans']

        with user():
            lm += f"Question:\n{question}"

        with assistant():
            lm += f"Answer:"

        with assistant():
            lm += f"{reason}"

        with assistant():
            lm += f"The final answer is:"

        with assistant():
            lm += f"{ans}"
    #################################################

    with user():
        lm += f"Question:\n{query}"

    with assistant():
        lm += f"Answer:"

    with assistant():
        lm += gen('reason', temperature=temp)

    with assistant():
        lm += f"The final answer is:"

    with assistant():
        lm += gen('ans', temperature=temp)

    return lm