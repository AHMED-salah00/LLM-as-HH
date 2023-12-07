# Large Language Models as Meta-Optimizers


### Usage

```bash
# for tsp_constructive
python main.py problem=tsp_constructive problem_type=constructive model=gpt-3.5-turbo
python main.py problem=tsp_constructive problem_type=constructive model=gpt-4-1106-preview # using GPT-4-turbo

# for tsp_aco
python main.py problem=tsp_aco problem_type=aco model=gpt-3.5-turbo diversify=False


```
### Dependency

- Python >= 3.9
- openai >= 1.0.0
- hydra-core

### Acknowledgements

[Eureka: Human-Level Reward Design via Coding Large Language Models](https://github.com/eureka-research/Eureka)