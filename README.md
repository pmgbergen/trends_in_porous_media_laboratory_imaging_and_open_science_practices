# Scripts for paper and data availability for article "Na Liu et al., Trends in porous media laboratory imaging and open science practices"

Three analyses are provided, as presented in Section 5 of the accompanied publication:
"Trends in porous media laboratory imaging and open science practices"
by Na Liu, Jakub Wiktor Both, Geir Ersland, Jan Martin Nordbotten, Martin Fernø.

1. Paper and data availability of the works cited in the paper.
2. Paper and data availability in the Springer journal Transport in Porous Media.
3. Paper and data availability in the Springer journal Computational Geosciences.

The repository collects all data and scripts required to reproduce the figures and results of the analysis. For this, run the jupyter notebooks (tested with Python 3.12.10):
- notebooks/cited_works_analysis.ipynb
- notebooks/tipm_analysis.ipynb
- notebooks/compgeo_analysis.ipynbs

For the cited works, the notebook illustrates the manually analyzed reference database. The manual analysis is provided as Excel file under database/cited_works.xlsx (for the preprint) and database/cited_works_v2 (for the revised article).

The notebooks will both perform the download of 1000 articles form TiPM and Computational Geosciences usin the stored URLs in database/tipm.csv and database/compgeo.csv, classify them into different categories and determine the availability of the paper itself and accompanying data. The results will be stored in csv files under results. The notebooks also generate figures, displayed in the article, and also displays these in the notebook.

The notebooks call generic Python scripts in the scripts folder. 

As explained in the publication, statistical robustness checks have been performed to assess the accuracy of the framework. For this the two notebooks notebooks/tipm_robustness_check.ipynb and notebooks/compgeo_robustness_check.ipynb provide guidance and the routines used.

The database of the cited works has been curated by Na Liu. The scripts for the analysis of the journal submissions and the plotting scripts have been developed by Jakub Both.

The associated preprint:
Liu, N., Both, J. W., Ersland, G., Nordbotten, J. M., & Fernø, M. (2025). Trends in porous media laboratory imaging and open science practices. arXiv preprint arXiv:2510.05190.
https://arxiv.org/abs/2510.05190