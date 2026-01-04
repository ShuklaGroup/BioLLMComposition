## Learning Physical Interactions to Compose Biological LLMs [[Preprint](https://arxiv.org/abs/2503.21017)]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShuklaGroup/BioLLMComposition/blob/main/BioLLMComposition.ipynb)

![TOC](toc.jpg) 

<div align="justify">
Large language models (LLMs) trained on biochemical sequences learn feature vectors that guide drug discovery through virtual screening. However, LLMs do not capture the molecular interactions important for binding affinity and specificity prediction. We compare a variety of methods to combine representations from distinct biological modalities to effectively represent molecular complexes. We demonstrate that learning to merge the representations from the internal layers of domain specific biological language models outperforms standard molecular interactions representations despite having significantly fewer features. 
</div>

### Quick Start
Our [Google Colab Notebook](https://colab.research.google.com/github/ShuklaGroup/BioLLMComposition/blob/main/BioLLMComposition.ipynb) compares and visualizes embeddings from four multimodal representation strategies.

### Python Scripts
You can also run each experiment and generate the corresponding plots using a standalone python script. We reccomend using `Python 3.10.18` and `PyTorch 2.5.1`:
```shell
git clone https://github.com/ShuklaGroup/BioLLMComposition
cd BioLLMComposition
python BioLLMComposition_peptide_mhc.py
python BioLLMComposition_protein_ligand.py
```

### Citation:
```bibtex
@misc{BioLLMComposition.
      title={Two for the Price of One: Integrating Large Language Models to Learn Biophysical Interactions}, 
      author={Joseph D. Clark and Tanner J. Dean and Diwakar Shukla},
      year={2025},
      eprint={2503.21017},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2503.21017}, 
}
```
