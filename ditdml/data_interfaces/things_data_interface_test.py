import os
import random
import tempfile

import numpy as np
import scipy.io

from parameterized import parameterized

from data_interface_test import DataInterfaceTest
from things_data_interface import ThingsDataInterface


NUM_CLASSES = 20
NUM_TEST_CLASSES = 5
NUM_IMAGES_PER_CLASS = 2
NUM_TRIPLETS = 1000


def _class_name(class_index):
    return "class{:02d}".format(class_index)


class ThingsDataInterfaceTest(DataInterfaceTest):
    def _make_dataset(self, data_directory_name):
        # Create resource files.
        os.makedirs(os.path.join(data_directory_name, "Main"))

        with open(os.path.join(data_directory_name, "Main", "things_concepts.tsv"), "wt") as f:
            f.write("Word\tuniqueID\n")
            for i in range(NUM_CLASSES):
                class_name = _class_name(i)
                f.write(f"{class_name}\t{class_name}\n")

        for i in range(NUM_CLASSES):
            subdirectory_name = os.path.join(data_directory_name, "Main", _class_name(i))
            os.makedirs(subdirectory_name)
            for j in range(NUM_IMAGES_PER_CLASS):
                open(os.path.join(subdirectory_name, f"image_{j}.jpg"), "wt")

        random.seed(1)
        triplets = list(set([tuple(random.sample(range(NUM_CLASSES), 3)) for _ in range(2 * NUM_TRIPLETS)]))

        subdirectory_name = os.path.join(data_directory_name, "Revealing", "triplets")
        os.makedirs(subdirectory_name)
        with open(os.path.join(subdirectory_name, "data1854_baseline_train90.txt"), "wt") as f:
            for t in range(NUM_TRIPLETS):
                i, j, k = triplets[t]
                f.write(f"{i} {j} {k}\n")
        with open(os.path.join(subdirectory_name, "data1854_baseline_test10.txt"), "wt") as f:
            for t in range(NUM_TRIPLETS, 3 * NUM_TRIPLETS // 2):
                i, j, k = triplets[t]
                f.write(f"{i} {j} {k}\n")

        subdirectory_name = os.path.join(data_directory_name, "Revealing", "data")
        os.makedirs(subdirectory_name)
        scipy.io.savemat(
            os.path.join(subdirectory_name, "RDM48_triplet.mat"),
            {"RDM48_triplet": [[0.0 for _ in range(NUM_TEST_CLASSES)] for _ in range(NUM_TEST_CLASSES)]},
        )
        with open(os.path.join(subdirectory_name, "spose_embedding_49d_sorted.txt"), "wt") as f:
            embeddings = [[0.0] for _ in range(NUM_TEST_CLASSES)]
            for e in embeddings:
                f.write(" ".join([str(x) for x in e]) + "\n")

        subdirectory_name = os.path.join(data_directory_name, "Revealing", "variables")
        os.makedirs(subdirectory_name)
        scipy.io.savemat(os.path.join(subdirectory_name, "sortind.mat"), {"sortind": list(range(1, NUM_CLASSES + 1))})
        scipy.io.savemat(
            os.path.join(subdirectory_name, "words48.mat"),
            {"words48": [[[np.array(_class_name(i))]] for i in range(NUM_CLASSES - NUM_TEST_CLASSES, NUM_CLASSES)]},
        )

    def test_quasi_original_split(self):
        with tempfile.TemporaryDirectory() as data_directory_name:
            self._make_dataset(data_directory_name)

            # Make data interface object and check its triplets, instances and prototypes.
            data_interface = ThingsDataInterface(data_directory_name, "quasi_original", 42)

            self.assertCountEqual(data_interface.triplets_by_subset.keys(), {"training", "validation", "test"})
            for triplets in data_interface.triplets_by_subset.values():
                self.assertGreaterEqual(len(triplets), 100)

            num_triplets_training = len(data_interface.triplets_by_subset["training"])
            num_triplets_validation = len(data_interface.triplets_by_subset["validation"])
            self.assertAlmostEqual(float(num_triplets_training) / num_triplets_validation, 4.0, delta=0.1)

            start_index_test_images = NUM_IMAGES_PER_CLASS * (NUM_CLASSES - NUM_TEST_CLASSES)
            for t in data_interface.triplets_by_subset["training"]:
                self.assertGreaterEqual(len([c for c in t if c < start_index_test_images]), 2)
            for t in data_interface.triplets_by_subset["validation"]:
                self.assertGreaterEqual(len([c for c in t if c < start_index_test_images]), 2)
            for t in data_interface.triplets_by_subset["test"]:
                self.assertGreaterEqual(len([c for c in t if c >= start_index_test_images]), 2)

            self.assertEqual(len(data_interface.prototypes_per_class), data_interface.reader.num_classes)
            for c, p in enumerate(data_interface.prototypes_per_class):
                self.assertEqual(data_interface.reader.image_records[p][1], c)

            # Make another data interface object and check that its triplets and prototypes are different from the first
            # object's.
            data_interface2 = ThingsDataInterface(data_directory_name, "quasi_original", 24)
            self._check_different_interfaces(
                data_interface, data_interface2, ["training", "validation"], has_classes=True
            )

    @parameterized.expand(["all_instances", "prototypes"])
    def test_by_class_split(self, class_triplet_conversion_type):
        with tempfile.TemporaryDirectory() as data_directory_name:
            self._make_dataset(data_directory_name)

            # Make data interface object and check its triplets, instances and prototypes.
            data_interface = ThingsDataInterface(
                data_directory_name, "by_class", 42, class_triplet_conversion_type=class_triplet_conversion_type
            )

            # Check that the triplets are disjoint.
            self._check_triplets(data_interface)

            # Check that the triplets respect the instance split.
            self._check_instances_triplets(data_interface)

            # Check that the prototypes are valid.
            self._check_prototypes(data_interface)

            # Make another data interface object and check that its triplets and prototypes are different from the first
            # object's.
            data_interface2 = ThingsDataInterface(
                data_directory_name, "by_class", 24, class_triplet_conversion_type=class_triplet_conversion_type
            )
            self._check_different_interfaces(
                data_interface, data_interface2, ["training", "validation", "test"], has_classes=True
            )

    @parameterized.expand(["all_instances", "prototypes"])
    def test_by_class_same_training_validation_split(self, class_triplet_conversion_type):
        with tempfile.TemporaryDirectory() as data_directory_name:
            self._make_dataset(data_directory_name)

            # Make data interface object.
            data_interface = ThingsDataInterface(
                data_directory_name,
                "by_class_same_training_validation",
                42,
                class_triplet_conversion_type=class_triplet_conversion_type,
            )

            # Check that the triplets are disjoint.
            self._check_triplets(data_interface)

            # Check that the triplets respect the instance split.
            self._check_instances_triplets(data_interface)

            # Check that the prototypes are valid.
            self._check_prototypes(data_interface)

            # Make another data interface object and check that its triplets and prototypes are different from the first
            # object's.
            data_interface2 = ThingsDataInterface(
                data_directory_name,
                "by_class_same_training_validation",
                24,
                class_triplet_conversion_type=class_triplet_conversion_type,
            )
            self._check_different_interfaces(
                data_interface, data_interface2, ["training", "validation", "test"], has_classes=True
            )

    def test_by_class_consistency(self):
        with tempfile.TemporaryDirectory() as data_directory_name:
            self._make_dataset(data_directory_name)

            # Make two data interface objects that partition by class: one with a standard split and one with the same
            # classes in training and validation.
            data_interface1 = ThingsDataInterface(
                data_directory_name, "by_class", 42, class_triplet_conversion_type="prototypes"
            )
            data_interface2 = ThingsDataInterface(
                data_directory_name, "by_class_same_training_validation", 42, class_triplet_conversion_type="prototypes"
            )

            # Check that the test triplet subsets are the same.
            self.assertEqual(data_interface1.triplets_by_subset["test"], data_interface2.triplets_by_subset["test"])
