# HARMES
Official Code and Support Repository for the HARMES dataset

(This repository is anonymized while our paper is still under review. After acceptance, it will be fully de-anonymized.)

HARMES is a multi-modal dataset for human activity recognition (HAR), comprised of 20 participants conducting 15 activities of daily living (ADLs). 61 hours of it are fully labeled, and we include 1 additional hour of other, free-form, mostly labeled activities on top (total: 20h).

#### Download

Download the dataset here(removed for review) and the paper here(removed for review) for additional details


---

### Structure of this repository:
The folder `data_collection` contains all code and tools we used for collecting the data for our dataset, including the source code for our WearOS data collection app, our experiment and live-annotation tool, and the Puck.js firmware we used.
In `validation`, all code for loading, processing, and plotting the data are made available. Additionally, in the `machine_learning` folder, the entire source code for running our validation experiments is contained. Each of these folders contains ReadMe files, where applicable, to ensure full reproduction with ease.


<details>
  <summary>Folder and file structure</summary>
  
```
.
└── data_collection
│   └── PuckJs
│   │   └── puckBME.png
│   │   └── README.md
│   │   └── wearPuck.jpg
│   │   └── firmware
│   │   │   └── puckBTService.js
│   │   │   └── beacon.js
│   └── WearOS
│   │   └── README.md
│   └── Experiment
│   │   └── README.md
└── validation
│   └── ReadMe.md
│   └── requirements.txt
│   └── notebooks
│   └── machine_learning
│   │   └── main.py
│   │   └── preprocessing
│   │   │   │   └── load_and_process_raw_data-checkpoint.py
│   │   │   │   └── extract_all_features-checkpoint.py
│   │   │   └── visualization.py
│   │   │   └── extract_all_features.py
│   │   │   └── __init__.py
│   │   │   └── load_and_process_raw_data.py
│   │   └── config.py
│   │   └── training
│   │   │   └── train.py
│   │   │   └── __init__.py
│   │   └── models
│   │   │   └── DeepConvLSTM.py
│   │   │   └── AudioModel.py
│   │   │   └── __init__.py
│   │   │   └── MultiModalModel.py
│   │   └── __init__.py
│   │   └── evaluation
│   │   │   └── evaluation.py
│   │   │   └── __init__.py
│   │   │   └── result_analysis.py
```
</details>

### Reproduction and using the dataset
This repository is available under the MIT License. The data is under CC-BY 4.0. See the ReadMe files in the subfolders for detailed instructions on reproducing our results!

If you use our dataset or code from this repository, please cite the dataset and our paper:

```(citation removed for review)```

