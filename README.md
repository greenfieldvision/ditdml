# Data Interfaces for Triplet-based Distance Metric Learning

Interfaces to the THINGS (<a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0223792">1</a> <a href="https://www.nature.com/articles/s41562-020-00951-3">2</a>), IHSJ (<a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Roads_Enriching_ImageNet_With_Human_Similarity_Judgments_and_Psychological_Embeddings_CVPR_2021_paper.pdf">3</a> <a href="https://osf.io/cn2s3/">4</a>) and Yummly (<a href="file:///home/rdondera/Downloads/13152-Article%20Text-16669-1-2-20201228.pdf">5</a> <a href="https://vision.cornell.edu/se3/projects/cost-effective-hits/">6</a>) datasets that produce data for triplet-based distance metric learning.

The code is close to production grade and provides an effective way to access triplet labeled datasets for distance metric learning.

## Requirements

* Python 3.8+
* more_itertools
* numpy
* scipy
* scikit-learn
* Pillow
* Tkinter

## Instructions for Use

### THINGS

1. Navigate to the <a href="https://osf.io/jum2f/">main THINGS dataset page on OSF</a> and download the Main folder as a zip archive.
2. Unzip the archive and its subarchives to folder {THINGS_ROOT}/Main.
3. Navigate to the "Revealing the multidimensional mental representations..." <a href="https://osf.io/z2784/">page on OSF</a> and download both the "data" and the "variables" folder as zip archives.
4. Unzip the two archives to folder {THINGS_ROOT}/Revealing.
5. Ask the corresponding author of the <a href="https://www.nature.com/articles/s41562-020-00951-3">THINGS dataset</a> for the labeled triplet data.
6. Place the files in {THINGS_ROOT}/Revealing/triplets.

### IHSJC

1. Download the ImageNet dataset to folder {IHSJ_ROOT}/imagenet.
2. Navigate to the <a href="https://osf.io/7f96y/">IHSJ dataset page on OSF</a> and download the file data/deprecated/psiz0.4.1/catalog.hdf5 to folder {IHSJ_ROOT}/val/catalogs/psiz0.4.1 and data/deprecated/psiz0.4.1/obs-195.hd5 to folder {IHSJ_ROOT}/val/obs/psiz0.4.1.

### Yummly

1. Download the zip archive http://vision.cornell.edu/se3/wp-content/uploads/2014/09/food100-dataset.zip.
2. Unzip the archive to {YUMMLY_ROOT}.

Take a look at the scripts in the tools/ directory, eg ```report_data_statistics.py```, and the ```*DataInterface``` classes in order to understand how to implement a PyTorch ```Dataset``` / write TensorFlow records using on the data interfaces.

## Dataset Splits

The code includes functionality to split the triplets into training, validation and test subsets - unit testing included. If desired, new splits can be implemented in the ```ThingsDataInterface``` class.

## Tools

All tools must be run from the parent folder of ditdml.

To see statistics like the number of images etc:

```
python ditdml/tools/report_data_statistics.py --dataset-name things --data-directory-name {THINGS_ROOT} --split-type quasi_original --seed 13
python ditdml/tools/report_data_statistics.py --dataset-name things --data-directory-name {THINGS_ROOT} --split-type by_class --class-triplet-conversion-type all_instances --seed 13
python ditdml/tools/report_data_statistics.py --dataset-name things --data-directory-name {THINGS_ROOT} --split-type by_class --class-triplet-conversion-type prototypes --seed 13
python ditdml/tools/report_data_statistics.py --dataset-name things --data-directory-name {THINGS_ROOT} --split-type by_class_same_training_validation --class-triplet-conversion-type all_instances --seed 13
python ditdml/tools/report_data_statistics.py --dataset-name things --data-directory-name {THINGS_ROOT} --split-type by_class_same_training_validation --class-triplet-conversion-type prototypes --seed 13

```

To interactively visualize labeled triplets:

```
python ditdml/tools/visualize_triplets.py --dataset-name things --data-directory-name {THINGS_ROOT} --split-type quasi_original --seed 23 --subset-name test --initial-triplet-index 200
python ditdml/tools/visualize_triplets.py --dataset-name things --data-directory-name {THINGS_ROOT} --split-type by_class --class-triplet-conversion-type all_instances --seed 23 --subset-name training --initial-triplet-index 315715
python ditdml/tools/visualize_triplets.py --dataset-name things --data-directory-name {THINGS_ROOT} --split-type by_class --class-triplet-conversion-type prototypes --seed 23 --subset-name validation --initial-triplet-index 42
python ditdml/tools/visualize_triplets.py --dataset-name things --data-directory-name {THINGS_ROOT} --split-type by_class_same_training_validation --class-triplet-conversion-type all_instances --seed 23 --subset-name test --initial-triplet-index 101
python ditdml/tools/visualize_triplets.py --dataset-name things --data-directory-name {THINGS_ROOT} --split-type by_class_same_training_validation --class-triplet-conversion-type prototypes --seed 23 --subset-name training --initial-triplet-index 22
```

(press left, right arrows)

To interactively visualize neighbors according to the provided embedding:

```
python ditdml/tools/visualize_neighbors.py --data-directory-name {THINGS_ROOT} --num-neighbors 4 --initial-class-index 1854
```

(press left, right arrows)

To interactively visualize the similarity matrix together with image pairs:

```
python ditdml/tools/visualize_similarity_matrix.py --data-directory-name {THINGS_ROOT}
```

(click on matrix elements in left pane)
