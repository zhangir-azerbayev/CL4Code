# CL4Code
Curriculum learning for math word problems. 

### Models
- GPT-Neo 125M and 1.3B code-generation
- Gpt-Neo 125M and 1.3B generate equation
- Graph2Tree (good MWP baseline).  

### Datasets
- MathQA-Python 
- ASDiV
- SVAMP
Include analysis of data quality, particularly overlap between training and test set. 

### Scoring Functions 
- Sequence Length
- Transfer scoring function 
- self-taught scoring function
- Grade level (ASDiV and SVAMP) only 
Create graphs of performance vs. scoring function. Question: is the correlation between train/test performance and a scoring function a good heuristic for the quality of the scoring function? 

### Curriculum strategies 
- Baseline (no curriculum)
- Exponential pacing 
- Combined strategy 

## Papers 
### Code generation 
- [Google paper](https://aclanthology.org/2020.acl-main.362/) 
- [Codex paper](https://arxiv.org/abs/2107.03374) 

### MWP datasets
- [Google paper](https://aclanthology.org/2020.acl-main.362/) 
- [ASDiV](https://arxiv.org/abs/2106.15772)
- [SVAMP](https://arxiv.org/abs/2103.07191) 

### MWP Baselines
- [Graph2Tree](https://aclanthology.org/2020.acl-main.362/)
- [MWPBert](https://arxiv.org/abs/2107.13435)

### Curriculum Learning
- [On the power of curriculum learning](https://arxiv.org/abs/1904.03626)
- [Leraning to execute](https://arxiv.org/abs/1410.4615)
