[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14019655.svg)](https://doi.org/10.5281/zenodo.14019655)
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://jupyter.org/try)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


# Focused learning by antibody language models using preferential masking of non-templated regions

While existing antibody language models (AbLMs) excel at predicting germline residues, they often struggle with mutated and non-templated residues, which concentrate in the complementarity-determining regions (CDRs) and are crucial for determining antigen-binding specificity. Many of these models are trained using a masked language modeling (MLM) objective with uniform masking probabilities; however, antibody recombination is modular in nature, creating relatively distinct regions of high and low complexity (non-templated and templated, respectively). We sought to determine whether and to what extent AbLMs can improve when trained using an alternative masking strategy based on this observation.

We developed a variation on MLM called ***Preferential Masking***, which alters masking probabilities to amplify training signals from the CDR3. We pre-trained two AbLMs using either uniform or preferential masking and observed that the latter improves pre-training efficiency and residue prediction accuracy in the highly variable CDR3. Preferential masking also improves antibody classification by native chain pairing and binding specificity, suggesting improved CDR3 understanding and indicating that non-random, learnable patterns help govern antibody chain pairing. We further show that specificity classification is largely informed by residues in the CDRs, demonstrating that AbLMs learn meaningful patterns that align with immunological understanding.

The Python scripts and Jupyter Notebooks in this repository contain all code necessary to re-train these AbLMs from scratch and replicate our downstream analyses.


## pre-training

Base models can be trained from scratch by running `AbLM_pretraining.py` with an associated `train-config.yaml`, as described [here](https://github.com/brineylab/deepspeed/tree/main).

Weights for the pre-trained model checkpoints used in the paper can also be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.14019655).



## how should I cite this?

The Preferential Masking paper has been published as a [preprint on biorxiv](https://doi.org/10.1101/2024.10.23.619908), and can be cited as:

```
Ng, K., & Briney, B. (2024). Focused learning by antibody language models using preferential masking of non-templated regions (p. 2024.10.23.619908). bioRxiv. https://doi.org/10.1101/2024.10.23.619908
```

The current version of the datasets used for pre-training and classifier head fine-tuning (v2024.10.31) can be cited as:

```
Ng, K., & Briney, B. (2024). Focused learning by antibody language models using preferential masking of non-templated regions (v2024.10.31) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14019655
```
