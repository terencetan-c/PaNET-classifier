# PaNET classifier

## Introduction
The Photon and Neutron Experimental Techniques (PaNET) ontology [1], [2] that provides a standardised taxonomy of experimental techniques used across the photon and neutron (PaN) community. The ontology was developed as part of the European Open Science Cloud Photon and Neutron Data Service (ExPaNDS) project. It receives continuous updates and development thanks to researchers from several PaN facilities across Europe. The GitHub repository can be found here:
https://github.com/pan-ontologies/PaNET

PaNET provides a controlled vocabulary that can be used to tag datasets with semantically richer metadata. However, such annotations require the judgement of domain experts and often have to be done manually, thereby making it a time-consuming process for PaN facilities. Hence, it would be ideal to automate the annotation workflow using machine learning.

## Methodology

This repository investigates the use several machine learning models to tag publications with the relevant PaNET terms.

PaNET_mapping.xlsx maps the DLS technique terms to PaNET terms. Note that the map for the DLS technique terms is not publicly available, so it can't be shared directly in this repository. However, it can be partially reverse engineered by looking at all the DLS technique terms in the 'Discipline/Technical Tags' column of the publications, and then mapped to PaNET terms. Hence, while PaNET_mapping.xlsx was originally created using the internal Diamond map, it can also be reproduced using publicly available information.

## Models

1. Baseline model
   This is a simple SciBERT + MLP model that serves as a baseline. The GitHub repository for SciBERT [3] can be found here:
   https://github.com/allenai/scibert/
   
2. HGCLR
   This is a Hierarchy-guided Contrastive Learning (HGCLR) approach to hierarchical text classification [4]. The original repository can be found here:
   https://github.com/wzh9969/contrastive-htc

## Data preprocessing
Download the list of publications from Diamond Light Source. 

Download the PaNET ontology. I have provided a version in .xrdf format, but it may be outdated at this point.

Run it through code/data_prep_initial.ipynb. This notebook includes a section using OpenAlex API to get the abstract for the publications. It is recommended to get an OpenAlex API key for faster retrieval. 

Run baseline_model/code/data_prep_baseline.ipynb to get the dataset for the baseline model (SciBERT + MLP).

Run HGCLR/code/data_prep_hgclr.ipynb to get the dataset for the HGCLR model.

Note that this steps have to be done sequentially, as the code in data_prep_hgclr.ipynb depends on data produced by data_prep_baseline.ipynb.


## Results
The models were trained and tested using NVIDIA A100 GPUs hosted at Diamond Light Source.

The seeds used are: [1,2,3,4,42].

The models were evaluated using F1-micro, F1-macro, hierarchical F1-micro, and hierarchical F1-macro scores. The mean and standard deviation for each metric were calculated using the five seeds:

| Model | F1-micro (mean ± SD) | F1-macro (mean ± SD) | hF1-micro (mean ± SD) | hF1-macro (mean ± SD) |
|-------|---------:|----------:|---------:|----------:|
| Baseline model | 0.840 ± 0.001 | 0.376 ± 0.005 | 0.840 ± 0.001 | 0.392 ± 0.005 |
| HGCLR | 0.847 ± 0.002 | 0.574 ± 0.021 | 0.847 ± 0.002 | 0.592 ± 0.022 |

\
Per-seed results:

| Model / Seed | F1-micro | F1-macro | hF1-micro | hF1-macro |
|---|---:|---:|---:|---:|
| **Baseline model** | | | | |
| ↳ Seed 1 | 0.841288 | 0.379940 | 0.840957 | 0.393864 |
| ↳ Seed 2 | 0.839700 | 0.371216 | 0.841283 | 0.386215 |
| ↳ Seed 3 | 0.839801 | 0.381900 | 0.839817 | 0.399754 |
| ↳ Seed 4 | 0.838929 | 0.373937 | 0.838862 | 0.388453 |
| ↳ Seed 42 | 0.840737 | 0.371884 | 0.840711 | 0.389872 |
| **HGCLR** | | | | |
| ↳ Seed 1 | 0.845648 | 0.596046 | 0.845179 | 0.612860 |
| ↳ Seed 2 | 0.850296 | 0.544567 | 0.849745 | 0.559794 |
| ↳ Seed 3 | 0.845124 | 0.559876 | 0.844972 | 0.578736 |
| ↳ Seed 4 | 0.849400 | 0.585958 | 0.848215 | 0.604367 |
| ↳ Seed 42 | 0.846993 | 0.585445 | 0.846593 | 0.604059 |

## References
[1] Collins, Steve P., da Graça Ramos, Silvia, Iyayi, Daniel, Görzig, Heike, González Beltrán, Alejandra, Ashton, Alun, Egli, Stefan, and Minotti, Carlo. “Expands Ontologies V1.0”. Zenodo, June 4, 2021. doi:10.5281/zenodo.4806026.


[2] Tan, T., Bago, B., Busch, S., Duyme, R., Gaisne, G., Gonzalez Beltran, A. N., Gorzig, H., Koumoutsos, G., Krahl, R., Millar, P., Minotti, C., Nentwich, M., Schrettner, L., Syder, K., Rocca-Serra, P., Sansone, S.-A. & Collins, S. P. (2025). J. Synchrotron Rad. 32, 1361-1369.

[3] Beltagy, Iz, Kyle Lo, and Arman Cohan. ‘SciBERT: Pretrained Language Model for Scientific Text’. In EMNLP, 2019.

[4] Zihan Wang, Peiyi Wang, Lianzhe Huang, Xin Sun, and Houfeng Wang. 2022. Incorporating Hierarchy into Text Encoder: a Contrastive Learning Approach for Hierarchical Text Classification. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7109–7119, Dublin, Ireland. Association for Computational Linguistics.
