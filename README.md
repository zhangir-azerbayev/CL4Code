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
