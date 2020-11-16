This repo contains the code and data to accompany a working paper: *Ratcheting Research Productivity -- the Case of Computer Science*.

### Requirements

This code uses python3, along with the following packages

- numpy
- pandas
- matplotlib
- statsmodels

### Data

The file `data/matched_data.csv` contains an anonymized copy of the publication data used for the main analyses in the paper. Each record indicates the number of papers published by a single person in a single year, along with relevant information about that year, the person and/or their institution. The file `data/cra.csv` contains information extracted from CRA annual reports. Information on obtaining additional data sources used in the paper can be found in the materials and methods section.

### Code

The `code` directory contains the code use to run the main age-period-cohort analyses. Specifically, from the `code` directory run `python run_all.py --only-faculty` and `python run_all.py --no-faculty` to fit a series of models. The script `export_effects.py` can then be used to extract the relevant effect sizes, as reported in Table 1 in the main paper.


