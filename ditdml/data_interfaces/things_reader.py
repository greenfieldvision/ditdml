import csv
import os
import time

import numpy as np
import scipy.io


def _load_class_names(file_name):
    """Load the class names."""

    # Open the TSV file and skip its header.
    with open(file_name, "rt") as f:
        csv_reader = csv.reader(f, delimiter="\t")
        next(csv_reader)

        # The class name is on the second column.
        class_names = [row[1] for row in csv_reader]

    return class_names


def _load_class_names_test(file_name):
    """Load the test class names."""

    # Load the entire file contents.
    data = scipy.io.loadmat(file_name)

    # Convert the class names to list of strings, replacing ambiguous names camera and file with camera1 and file1.
    class_names_test = [
        str(w[0][0].tolist()).replace(" ", "_").replace("camera", "camera1").replace("file", "file1")
        for w in data["words48"]
    ]

    return class_names_test


def _load_pairwise_similarity(file_name):
    """Load the similarity matrix for the test classes."""

    # Load the entire file contents.
    data = scipy.io.loadmat(file_name)

    # Convert to similarity values.
    pairwise_similarity = 1.0 - data["RDM48_triplet"]

    return pairwise_similarity


def _load_index_remapping(file_name):
    """Load the mapping from arbitrarily sorted class indexes to original class indexes.

    The class indexes in the triplet data refer to an arbitrary sorting of the original classes and must be converted
    back. Original order in the "things_concepts.tsv" file, arbitrary order in the file passed as a parameter.
    """

    # Load the entire file contents and then select the sorted class indexes.
    data = scipy.io.loadmat(file_name)
    sorted_indexes = data["sortind"].flatten()

    # Invert the sorted indexes to obtain the mapping back to the original class indexes. Take into account the fact
    # that the sorted indexes are 1 based.
    inverse_sorted_indexes = np.zeros((sorted_indexes.shape[0] + 1,), dtype=np.int32)
    inverse_sorted_indexes[sorted_indexes] = np.arange(start=1, stop=sorted_indexes.shape[0] + 1, dtype=np.int32)
    inverse_sorted_indexes = inverse_sorted_indexes[1:] - 1

    return inverse_sorted_indexes.tolist()


def _load_and_remap_triplets(file_name, index_remapping):
    """Load class triplets and remap the class indexes."""

    # Read triplets line by line and remap their indexes.
    with open(file_name, "rt") as f:
        triplets = [[index_remapping[int(i)] for i in line.split(" ")] for line in f.readlines()]

    return triplets


def _load_triplets(remapping_file_name, training_triplets_file_name, test_triplets_file_name):
    """Load the class triplets for the training and test subsets and concatenate them."""

    # Load the mapping from order of classes used in triplets to original order of classes.
    index_remapping = _load_index_remapping(remapping_file_name)

    # Load the triplets for training and test, remap them and put them together.
    triplets_training = _load_and_remap_triplets(training_triplets_file_name, index_remapping)
    triplets_test = _load_and_remap_triplets(test_triplets_file_name, index_remapping)
    triplets = triplets_training + triplets_test

    return triplets


def _load_embeddings(file_name):
    """Load the sample class embeddings."""

    # Read the embedding vectors line by line.
    with open(file_name, "rt") as f:
        embeddings = [[float(x) for x in line.rstrip().split(" ")] for line in f.readlines()]

    return np.array(embeddings)


class ThingsReader:
    """Basic loading functionality for data in the THINGS dataset.

    Provides access to image file names, class names, class triplets, test classes, similarity matrix for test classes,
    sample class embeddings.
    """

    RESOURCE_PATHS_RELATIVE = {
        "image_directory": ("Main",),
        "concepts": ("Main", "things_concepts.tsv"),
        "triplets_index_remapping": ("Revealing", "variables", "sortind.mat"),
        "triplets_training": ("Revealing", "triplets", "data1854_baseline_train90.txt"),
        "triplets_test": ("Revealing", "triplets", "data1854_baseline_test10.txt"),
        "RDM48": ("Revealing", "data", "RDM48_triplet.mat"),
        "words48": ("Revealing", "variables", "words48.mat"),
        "embeddings": ("Revealing", "data", "spose_embedding_49d_sorted.txt"),
    }

    def __init__(self, directory_name):
        # Copy directory name to class attribute and make full resource paths.
        self._directory_name = directory_name
        self._resource_paths = {
            k: os.path.join(self._directory_name, *v) for k, v in self.RESOURCE_PATHS_RELATIVE.items()
        }

        # Report times for loading operations.
        print("loading THINGS data...")
        start_time = time.time()

        # Load resource data and image file data.
        self._load_nonimage_info()
        self._load_image_file_info()

        print("done ({:.2f} s)".format(time.time() - start_time))

    @property
    def image_records(self):
        return self._image_records

    @property
    def num_images(self):
        return len(self._image_records)

    @property
    def class_names(self):
        return self._class_names

    @property
    def num_classes(self):
        return len(self._class_names)

    @property
    def class_triplets(self):
        return self._class_triplets

    @property
    def pairwise_similarity_original_test(self):
        return self._pairwise_similarity_original_test

    @property
    def classes_original_test(self):
        return self._classes_original_test

    @property
    def class_embeddings(self):
        return self._class_embeddings

    def _load_nonimage_info(self):
        """Load all non-image resources: class names, triplets, similarity matrix, sample embeddings."""

        # Load the class names and make the mapping from class names to class indexes.
        self._class_names = _load_class_names(self._resource_paths["concepts"])
        self._class_name_to_index = {name: index for index, name in enumerate(self._class_names)}

        # Load the triplets defining relations between classes.
        self._class_triplets = _load_triplets(
            self._resource_paths["triplets_index_remapping"],
            self._resource_paths["triplets_training"],
            self._resource_paths["triplets_test"],
        )

        # Load the similarity matrix for the 48 classes. Load their names and convert to indexes.
        self._pairwise_similarity_original_test = _load_pairwise_similarity(self._resource_paths["RDM48"])
        class_names_original_test = _load_class_names_test(self._resource_paths["words48"])
        self._classes_original_test = [self._class_name_to_index[name] for name in class_names_original_test]

        # Load the embeddings for each class.
        self._class_embeddings = _load_embeddings(self._resource_paths["embeddings"])

    def _load_image_file_info(self):
        """Load all the image information: file names and classes."""

        # Initialize the image records to an empty list.
        self._image_records = []

        # Go through the subdirectories of the image directory (each subdirectory corresponds to a class).
        image_directory_name = os.path.join(self._directory_name, self._resource_paths["image_directory"])
        for subdirectory_name in os.listdir(image_directory_name):
            full_subdirectory_name = os.path.join(image_directory_name, subdirectory_name)

            # Go through the files in each valid subdirectory.
            if os.path.isdir(full_subdirectory_name) and (subdirectory_name != ".") and (subdirectory_name != ".."):
                class_index = self._class_name_to_index[subdirectory_name]
                for file_name in os.listdir(full_subdirectory_name):
                    full_file_name = os.path.join(full_subdirectory_name, file_name)

                    # If image file, make new image record.
                    if os.path.isfile(full_file_name) and file_name.endswith(".jpg"):
                        self._image_records.append((full_file_name, class_index))

        # Sort the image records according to the file name.
        self._image_records = sorted(self._image_records, key=lambda image_record: image_record[0])
