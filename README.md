# SciCode 

[**Homepage**](https://scicode-bench.github.io/) | [**Paper**](https://arxiv.org/abs/2407.13168)


This repo contains the evaluation code for the paper "[SciCode: A Research Coding Benchmark Curated by Scientists](https://arxiv.org/abs/2407.13168)"

## 🔔News

**[2025-01-24]: SciCode has been integrated with [`inspect_ai`](https://inspect.ai-safety-institute.org.uk/) for easier and faster model evaluations.**

**[2024-11-04]: Leaderboard is on! Check [here](https://scicode-bench.github.io/leaderboard/). We have also added Claude Sonnet 3.5 (new) results.**

**[2024-10-01]: We have added OpenAI o1-mini and o1-preview results.**

**[2024-09-26]: SciCode is accepted at NeurIPS D&B Track 2024.**

**[2024-08-22]: The SciCode benchmark has been successfully integrated into [OpenCompass](https://github.com/open-compass/opencompass).**

**[2024-07-24]: We add the scientist-annotated background and support setup for w/ background evaluation.**

## Introduction
SciCode is a challenging benchmark designed to evaluate the capabilities of language models (LMs) in generating code for solving realistic scientific research problems. It has a diverse coverage of **16** subdomains from **6** domains: Physics, Math, Material Science, Biology, and Chemistry. Unlike previous benchmarks that consist of exam-like question-answer pairs, SciCode is converted from real research problems. SciCode problems naturally factorize into multiple subproblems, each involving knowledge recall, reasoning, and code synthesis. In total, SciCode contains **338** subproblems decomposed from **80** challenging main problems, and it offers optional descriptions specifying useful scientific background information and scientist-annotated gold-standard solutions and test cases for evaluation. OpenAI o1-preview, the best-performing model among those tested, can solve only **7.7%** of the problems in the most realistic setting. Broadly, SciCode demonstrates a realistic and scientists' everyday workflow of identifying critical science concepts and facts and then transforming them into computation and simulation code. We believe SciCode not only helps demonstrate contemporary LLMs' progress towards helpful assistant for scientists but also helps shed light on future building and evaluation of scientific AI.



## Dataset Creation
SciCode sources challenging and realistic research-level coding problems across 6 natural science disciplines, covering a total of 16 subfields. Scicode mainly focuses on 1. Numerical methods 2.Simulation of systems 3. Scientific calculation. These are the tasks we believe require intense scientific knowledge and reasoning to optimally test LM’s science capability.

## 🏆 Leaderboard

| Models                   | Main Problem Resolve Rate           | <span style="color:grey">Subproblem</span>            |
|--------------------------|-------------------------------------|-------------------------------------|
| 🥇 OpenAI o1-preview      | <div align="center">**7.7**</div>       | <div align="center" style="color:grey">28.5</div>     |
| 🥈 Claude3.5-Sonnet       | <div align="center">**4.6**</div>       | <div align="center" style="color:grey">26.0</div>     |
| 🥉 Claude3.5-Sonnet (new) | <div align="center">**4.6**</div>       | <div align="center" style="color:grey">25.3</div>     |
| Deepseek-Coder-v2        | <div align="center">**3.1**</div>       | <div align="center" style="color:grey">21.2</div>     |
| GPT-4o                   | <div align="center">**1.5**</div>       | <div align="center" style="color:grey">25.0</div>     |
| GPT-4-Turbo              | <div align="center">**1.5**</div>       | <div align="center" style="color:grey">22.9</div>     |
| OpenAI o1-mini           | <div align="center">**1.5**</div>       | <div align="center" style="color:grey">22.2</div>     |
| Gemini 1.5 Pro           | <div align="center">**1.5**</div>       | <div align="center" style="color:grey">21.9</div>     |
| Claude3-Opus             | <div align="center">**1.5**</div>       | <div align="center" style="color:grey">21.5</div>     |
| Llama-3.1-405B-Chat      | <div align="center">**1.5**</div>       | <div align="center" style="color:grey">19.8</div>     |
| Claude3-Sonnet           | <div align="center">**1.5**</div>       | <div align="center" style="color:grey">17.0</div>     |
| Qwen2-72B-Instruct       | <div align="center">**1.5**</div>       | <div align="center" style="color:grey">17.0</div>     |
| Llama-3.1-70B-Chat       | <div align="center">**0.0**</div>       | <div align="center" style="color:grey">17.0</div>     |
| Mixtral-8x22B-Instruct   | <div align="center">**0.0**</div>       | <div align="center" style="color:grey">16.3</div>     |
| Llama-3-70B-Chat         | <div align="center">**0.0**</div>       | <div align="center" style="color:grey">14.6</div>     |


## Instructions to evaluate a new model

1. Clone this repository `git clone git@github.com:scicode-bench/SciCode.git`
2. Install the `scicode` package with `pip install -e .`
3. Download the [numeric test results](https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR?usp=drive_link) and save them as `./eval/data/test_data.h5`
4. Run `eval/scripts/gencode_json.py` to generate new model outputs (see the [`eval/scripts` readme](eval/scripts/)) for more information
5. Run `eval/scripts/test_generated_code.py` to evaluate the unittests


## Instructions to evaluate a new model using `inspect_ai` (recommended)

Scicode has been integrated with `inspect_ai` for easier and faster model evaluation, compared with the methods above. You need to run the first three steps in the [above section](#instructions-to-evaluate-a-new-model), and then go to the `eval/inspect_ai` directory, setup correspoinding API key, and run the following command:

```bash
cd eval/inspect_ai
export OPENAI_API_KEY=your-openai-api-key
inspect eval scicode.py --model openai/gpt-4o --temperature 0
```

For more detailed information of using `inspect_ai`, see [`eval/inspect_ai` readme](eval/inspect_ai/)

## More information and FAQ

More information, including a [FAQ section](https://scicode-bench.github.io/faq/), is provided on our [website](https://scicode-bench.github.io/).
If you have trouble reaching the website, please find the markdown source in its [github repository](https://github.com/scicode-bench/scicode-bench.github.io/tree/main/docs).

## Contact
- Minyang Tian: mtian8@illinois.edu
- Eliu Huerta: elihu@anl.gov
- Hao Peng: haopeng@illinois.edu

## Citation
```bibtex
@misc{tian2024scicode,
    title={SciCode: A Research Coding Benchmark Curated by Scientists},
    author={Minyang Tian and Luyu Gao and Shizhuo Dylan Zhang and Xinan Chen and Cunwei Fan and Xuefei Guo and Roland Haas and Pan Ji and Kittithat Krongchon and Yao Li and Shengyan Liu and Di Luo and Yutao Ma and Hao Tong and Kha Trinh and Chenyu Tian and Zihan Wang and Bohao Wu and Yanyu Xiong and Shengzhu Yin and Minhui Zhu and Kilian Lieret and Yanxin Lu and Genglin Liu and Yufeng Du and Tianhua Tao and Ofir Press and Jamie Callan and Eliu Huerta and Hao Peng},
    year={2024},
    eprint={2407.13168},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```
