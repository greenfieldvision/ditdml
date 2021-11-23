# Data Interfaces for Triplet-based Distance Metric Learning

This repository contains code to interface with the data in the THINGS dataset (<a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0223792">1</a> <a href="https://www.nature.com/articles/s41562-020-00951-3">2</a>) that is relevant for distance metric learning. More datasets hopefully coming soon!

The code is close to production grade and provides an effective way to access triplet labeled datasets for distance metric learning.

## Requirements

* Python 3.8+
* more_itertools
* numpy
* scipy
* Pillow
* Tkinter

## Instructions for Use

### THINGS

1. Navigate to the <a href="https://osf.io/jum2f/">main THINGS dataset page on OSF</a> and download the Main folder as a zip archive.
2. Unzip the archive and its subarchives under folder {THINGS_ROOT}/Main.
3. Navigate to the "Revealing the multidimensional mental representations..." <a href="https://osf.io/z2784/">page on OSF</a> and download both the "data" and the "variables" folder as zip archives.
4. Unzip the two archives under folder {THINGS_ROOT}/Revealing.
5. Ask the corresponding author of <a href="https://www.nature.com/articles/s41562-020-00951-3">THINGS dataset</a> for the labeled triplet data.
6. Place the files under {THINGS_ROOT}/Revealing/triplets.

Take a look at the scripts in the tools/ directory, eg ```report_data_statistics.py```, and the ```ThingsDataInterface``` and ```ThingsReader``` classes. Hopefully it should be clear how to implement a PyTorch ```Dataset``` or a tool to write TensorFlow records.

## Dataset Splits

The code includes functionality to split the triplets into training, validation and test subsets - unit testing included. If desired, new splits can be implemented in the ```ThingsDataInterface``` class.

## Tools

All tools must be run from the parent folder of ditdml.

To see statistics like the number of images etc:

```
python ditdml/tools/report_data_statistics.py --data-directory-name {THINGS_ROOT} --split-type quasi_original --seed 13
python ditdml/tools/report_data_statistics.py --data-directory-name {THINGS_ROOT} --split-type by_class --seed 14
```

To interactively visualize labeled triplets:

```
python ditdml/tools/visualize_triplets.py --data-directory-name {THINGS_ROOT} --split-type quasi_original --seed 15
python ditdml/tools/visualize_triplets.py --data-directory-name {THINGS_ROOT} --split-type quasi_original --seed 16 --subset-name test --initial-triplet-index 200
```

(press left, right arrows)

To interactively visualize the similarity matrix together with image pairs:

```
python ditdml/tools/visualize_similarity_matrix.py --data-directory-name {THINGS_ROOT}
```

(click on matrix elements)
