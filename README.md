
This repository is a reproduction-oriented implementation inspired by the 2022 article：
“The role of population structure in computations through neural dynamics” 
studying how recurrent neural networks solve a context-dependent XOR decision rule in a color–coherence task. 

I trained both a 512-unit full-rank RNN and a rank-2 low-rank RNN under the same task, 
and extended the reference setup by systematically comparing three different context-cue input encodings 
(implemented as separate variants for each model). 

The repo includes code and results that show how full-rank and low-rank networks can reach similarly high behavioral accuracy while 
relying on markedly different internal computational structures and interpretability.

Full abstract: [XOR_TASK Abstract (1)].mdm(./XOR_TASK Abstract (1).md)



