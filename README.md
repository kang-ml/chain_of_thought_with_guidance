### Implementation of CoT using guidance-ai
Implement CoT using [guidance](https://github.com/guidance-ai/guidance) on three reasoning dataset:
+ [GSM8K](https://arxiv.org/abs/2110.14168)
+ [ProntoQA](https://arxiv.org/abs/2210.01240)
+ [ProofWriter](https://arxiv.org/abs/2012.13048)

  
### Requirement
To run our code, please install all the packages by using the following command:
```
pip install -r requirement.txt
```

### Run inference
First obtain [openai api key](https://openai.com/index/openai-api/), and fill in the blank space in ```.../run_test.py```.
```
python GSM8K/run_test.py
```
```
python ProntoQA/run_test.py
```
```
python ProofWriter/run_test.py
```
The inference results will be save in ```.../test_result.json```.
