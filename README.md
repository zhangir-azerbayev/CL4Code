## 1. Introduction 
Goal: establish that 
1. Neural code generation outperforms previous MWP baselines 
2. Curriculum learning further improves neural code generation performance 
### Related works 
Code generation:
- [Google paper](https://aclanthology.org/2020.acl-main.362/) 
- [Codex paper](https://arxiv.org/abs/2107.03374) 

MWP datasets:
- [Google paper](https://aclanthology.org/2020.acl-main.362/) 
- [ASDiV](https://arxiv.org/abs/2106.15772)
- [SVAMP](https://arxiv.org/abs/2103.07191) 

MWP Baselines:
- [Graph2Tree](https://aclanthology.org/2020.acl-main.362/)
- [MWPBert](https://arxiv.org/abs/2107.13435)

Curriculum Learning:
- [On the power of curriculum learning](https://arxiv.org/abs/1904.03626)
- [Leraning to execute](https://arxiv.org/abs/1410.4615)
- [Curriculum learning for large language model](https://arxiv.org/abs/2108.06084)

## 2. Datasets and Models
Use MathQA-Python, SVAMP, and ASDiV. For each dataset, fine-tune on two versions of the dataset
1. Goal is predict code which executes to correct answer. 
2. Goal is predict arithmetic expression that evaluates to correct answer (e.g (240-17)\*57+4). 
Interested on Gpt-Neo performance. Use MWPBert and Graph2Tree as additional baselines. 

## 3. Curriculum Learning 
Also try curriculum learning

Scoring functions: sequence length, transfer scoring function, self-taught scoring function, grade-level (ASDiV and SVAMP only). 

Pacing function: varied exponential

Strategies: naive curriculum learning vs. combined strategy from Zarmeba & Sutskever. 

**Fig 1:** GPT-Neo 125M fine-tuned on MathQA-Python accuracy for different scoring functions and strategies. 

## 4. Evaluation 
**Fig 2:** Comparison of following models on MathQA-Python, ASDiV, SVAMP (+CL refers to using best curriculum from part 3) 
1. Gpt-Neo 125M equation generation
2. Gpt-Neo 125M code generation
3. Gpt-Neo 125M code generation + CL
4. Gpt-Neo 1.3B equation generation
5. Gpt-Neo 1.3B code generation
6. Gpt-Neo 1.3B code generation + CL
7. MWPBert
8. Graph2Tree

**Fig 3:** Also compare following models to MathQA-Python pass@80 in Austin et al.
1. Gpt-Neo 125M equation generation
2. Gpt-Neo 125M code generation
3. Gpt-Neo 125M code generation + CL
4. Gpt-Neo 1.3B equation generation
5. Gpt-Neo 1.3B code generation
6. Gpt-Neo 1.3B code generation + CL


