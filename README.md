## Learning Physical Interactions to Compose Biological LLMs [[Paper](https://www.nature.com/articles/s42004-025-01883-7)]

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
@article{Clark2026,
      title = {Learning physical interactions to compose biological large language models},
      ISSN = {2399-3669},
      url = {http://dx.doi.org/10.1038/s42004-025-01883-7},
      DOI = {10.1038/s42004-025-01883-7},
      journal = {Communications Chemistry},
      publisher = {Springer Science and Business Media LLC},
      author = {Clark,  Joseph D. and Dean,  Tanner J. and Shukla,  Diwakar},
      year = {2026},
      month = jan 
}
```
