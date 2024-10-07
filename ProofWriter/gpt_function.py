import guidance
from guidance import models, gen, select, system, user, assistant, regex

@guidance
def proofwriter_reasoning(lm, demos, premise, q, temp=0.1):
    with system():
        lm += '''Suppose you are one of the greatest AI scientists, logicians. Given some context as premise, the task is to answer if a logical reasoning question is True or False or Unknown. Let us think step by step.'''

    #################################################
    for demo in demos:
        context = demo['context']
        question = demo['question']
        reason = demo['reason']
        ans = demo['ans']

        with user():
            lm += f"Context:\n{context}"

        with user():
            lm += f"Question:\n{question}"

        with assistant():
            lm += f"Reasoning:"

        with assistant():
            lm += f"{reason}"

        with assistant():
            lm += f"The final answer is:"

        with assistant():
            lm += f"{ans}"
    #################################################

    with user():
        lm += f"Context:\n{premise}"

    with user():
        lm += f"Question:\n{q}"

    with assistant():
        lm += f"Reasoning:"

    with assistant():
        lm += gen('reason', temperature=temp, stop='The final answer is:')

    with assistant():
        lm += f"The final answer is:"

    with assistant():
        lm += gen('ans', temperature=temp)

    return lm