# PaNET classifier

## Introduction
The Photon and Neutron Experimental Techniques (PaNET) ontology [1], [2] that provides a standardised taxonomy of experimental techniques used across the photon and neutron (PaN) community. The ontology was developed as part of the European Open Science Cloud Photon and Neutron Data Service (ExPaNDS) project. It receives continuous updates and development thanks to researchers from several PaN facilities across Europe. The GitHub repository can be found here:
https://github.com/pan-ontologies/PaNET

PaNET provides a controlled vocabulary that can be used to tag datasets with semantically richer metadata. However, such annotations require the judgement of domain experts and often have to be done manually, thereby making it a time-consuming process for PaN facilities. Hence, it would be ideal to automate the annotation workflow using machine learning.

## Methodology

This repository investigates the use several machine learning models to tag publications with the relevant PaNET terms.

PaNET_mapping.xlsx maps the DLS technique terms to PaNET terms. Note that the map for the DLS technique terms is not publicly available, so it can't be shared directly in this repository. However, it can be partially reverse engineered by looking at all the DLS technique terms in the 'Discipline/Technical Tags' column of the publications, and then mapped to PaNET terms. Hence, while PaNET_mapping.xlsx was originally created using the internal Diamond map, it can also be reproduced using publicly available information.

## Data preprocessing
Download the list of publications from Diamond Light Source. 

Download the PaNET ontology. I have provided a version in .xrdf format, but it may be outdated at this point.

Run it through code/data_prep_initial.ipynb. This notebook includes a section using OpenAlex API to get the abstract for the publications. It is recommended to get an OpenAlex API key for faster retrieval. 

Run baseline_model/code/data_prep_baseline.ipynb to get the dataset for the baseline model (SciBERT + MLP).

Run HGCLR/code/data_prep_hgclr.ipynb to get the dataset for the HGCLR model.

Note that this steps have to be done sequentially, as the code in data_prep_hgclr.ipynb depends on data produced by data_prep_baseline.ipynb.



## References
[1] Collins, Steve P., da Graça Ramos, Silvia, Iyayi, Daniel, Görzig, Heike, González Beltrán, Alejandra, Ashton, Alun, Egli, Stefan, and Minotti, Carlo. “Expands Ontologies V1.0”. Zenodo, June 4, 2021. doi:10.5281/zenodo.4806026.


[2] Tan, T., Bago, B., Busch, S., Duyme, R., Gaisne, G., Gonzalez Beltran, A. N., Gorzig, H., Koumoutsos, G., Krahl, R., Millar, P., Minotti, C., Nentwich, M., Schrettner, L., Syder, K., Rocca-Serra, P., Sansone, S.-A. & Collins, S. P. (2025). J. Synchrotron Rad. 32, 1361-1369.
