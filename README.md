
This repository is a reproduction-oriented implementation inspired by the 2022 article：
“The role of population structure in computations through neural dynamics” 
studying how recurrent neural networks solve a context-dependent XOR decision rule in a color–coherence task. 

I trained both a 512-unit full-rank RNN and a rank-2 low-rank RNN under the same task, 
and extended the reference setup by systematically comparing three different context-cue input encodings 
(implemented as separate variants for each model). 

The repo includes code and results that show how full-rank and low-rank networks can reach similarly high behavioral accuracy while 
relying on markedly different internal computational structures and interpretability.
## Code Attribution

The following files in this repository are reproduced from the publicly available code accompanying Dubreuil et al. (2022), *The role of population structure in computations through neural dynamics*, Nature Neuroscience:

- `FullRank.py`
- `LowRank (Reference).py`

**Original source:** [populations_paper_code](https://github.com/adrian-valente/populations_paper_code), developed by Adrian Valente and Alexis Dubreuil.

These files are included for reference purposes and are **not original implementations by the author of this repository**. All credit for the original code belongs to its respective authors.

The full abstract is also included in the repository root (XOR_TASK Abstract (1).md).



