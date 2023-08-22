### ETLD
encoder-transformation layer-decoder
You can get more understanding through this article (https://doi.org/10.1093/bib/bbad290).

Models require PyTorch. We tested on `v1.12.1`, and `1.13.1`. 

## Requirements

matplotlib, numba, numpy, scipy, torch, pandas

## Installation

```
$ cd source
$ python ./setup.py install

```

## Example

# run_example_1

Use mathematical derivation to predict contacts from "dhfr.fa".

"dhfr.fa" from PLMDCA software. [1]
[1] Ekeberg M, Lovkvist C, Lan Y H, et al. Improved contact prediction in proteins: using pseudolikelihoods to infer Potts models [J]. Phys Rev E, 2013, 87(1): 012707.
```
$ cd example/run_example_1
$ python run_example.py
$ python contprec_example.py

```

# run_example_2

Predicting the mutation effects of "BLAT_ECOLX" using "BLAT_ECOLX_1_b0.5.aln".

"BLAT_ECOLX_1_b0.5.aln" and "BLAT_ECOLX_Ranganathan2015.csv" from Riesselman et al [2]. 
[2] Riesselman A J, Ingraham J B, Marks D S. Deep generative models of genetic variation capture the effects of mutations [J]. Nat Methods, 2018, 15(10): 816-22.

```
$ cd example/run_example_2
$ python run_example.py
$ python mutprec_example.py

```

# run_example_3

Use ResNet to predict contacts from "dhfr.fa".

```
$ cd example/run_example_3
$ python contprec_example.py

```

