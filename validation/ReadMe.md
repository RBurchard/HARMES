This folder contains all our experiment code, which we used for validation of the collected dataset as well as for our first benchmark machine learning experiments and the ablation study.

### Instructions on running the code:
Prerequisites: [conda](https://anaconda.org/) must be installed. To run the machine learning experiments, a recent NVIDIA GPU and its drivers must be present on the system. 

0. Download the raw dataset (anonymized link), and place it in any folder - remember its relative or full path!
1. Create a fresh conda environment: `conda create -n "harmes" python=3.11` and activate the new environment `conda activate harmes`.
2. Install all requirements: `pip install -r requirements.txt`
3. Edit `machine_learning/config.py`, by adding the dataset path (to the folder containing all participant folders). If you want to run certain sensor combinations only (or try additional combinations), edit the `sensor_configs` list accordingly.
4. To run the experiments: `python -m machine_learning.main`. This will take several days, depending on the hardware and configuration. If you want to reproduce the export into the preprocessed dataset only, use `python -m machine_learning.main --no-eval --export`. 
5. The results of the experiments will be saved in a Python shelve called `cache` (which also caches pre-computed, windowed data for future runs).
6. To reproduce our plots or data and results, as well as all metric calculations, change into the `notebooks` folder and run `jupyter notebook`. In the opened browser window, open all notebooks, and run all cells. The cell outputs and plot files will now contain all the results we used for the publication.
