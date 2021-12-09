import os
import random
import tempfile
import unittest

import numpy as np
import scipy.io

from things_data_interface import ThingsDataInterface


NUM_CLASSES = 20
NUM_TEST_CLASSES = 5
NUM_IMAGES_PER_CLASS = 2
NUM_TRIPLETS = 1000


def class_name(class_index):
    return "class{:02d}".format(class_index)


class ThingsDataInterfaceTest(unittest.TestCase):
    def test_quasi_original_split(self):
        with tempfile.TemporaryDirectory() as data_directory_name:
            # Create resource files.
            os.makedirs(os.path.join(data_directory_name, "Main"))

            with open(os.path.join(data_directory_name, "Main", "things_concepts.tsv"), "wt") as f:
                f.write("Word\tuniqueID\n")
                for i in range(NUM_CLASSES):
                    f.write("{class_name}\t{class_name}\n".format(class_name=class_name(i)))

            for i in range(NUM_CLASSES):
                subdirectory_name = os.path.join(data_directory_name, "Main", class_name(i))
                os.makedirs(subdirectory_name)
                for j in range(NUM_IMAGES_PER_CLASS):
                    open(os.path.join(subdirectory_name, f"image_{j}.jpg"), "wt")

            subdirectory_name = os.path.join(data_directory_name, "Revealing", "triplets")
            os.makedirs(subdirectory_name)
            with open(os.path.join(subdirectory_name, "data1854_baseline_train90.txt"), "wt") as f:
                for _ in range(NUM_TRIPLETS):
                    i, j, k = random.sample(range(NUM_CLASSES), 3)
                    f.write(f"{i} {j} {k}\n")
            with open(os.path.join(subdirectory_name, "data1854_baseline_test10.txt"), "wt") as f:
                for _ in range(NUM_TRIPLETS // 2):
                    i, j, k = random.sample(range(NUM_CLASSES), 3)
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
            scipy.io.savemat(
                os.path.join(subdirectory_name, "sortind.mat"), {"sortind": list(range(1, NUM_CLASSES + 1))}
            )
            scipy.io.savemat(
                os.path.join(subdirectory_name, "words48.mat"),
                {"words48": [[[np.array(class_name(i))]] for i in range(NUM_CLASSES - NUM_TEST_CLASSES, NUM_CLASSES)]},
            )

            # Make data interface object and check its triplets and prototypes.
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

            self.assertNotEqual(data_interface.prototypes_per_class, data_interface2.prototypes_per_class)
            classes_per_prototype = {p: c for c, p in enumerate(data_interface.prototypes_per_class)}
            classes_per_prototype2 = {p: c for c, p in enumerate(data_interface2.prototypes_per_class)}

            for subset_name in ["training", "validation"]:

                def to_class_triplet_strings(p2c, ts):
                    return set([" ".join([str(p2c[p]) for p in t]) for t in ts])

                triplets = to_class_triplet_strings(
                    classes_per_prototype, data_interface.triplets_by_subset[subset_name]
                )
                triplets2 = to_class_triplet_strings(
                    classes_per_prototype2, data_interface2.triplets_by_subset[subset_name]
                )
                self.assertNotEqual(triplets, triplets2)

    def test_by_class_split(self):
        with tempfile.TemporaryDirectory() as data_directory_name:
            # Create resource files.
            os.makedirs(os.path.join(data_directory_name, "Main"))

            with open(os.path.join(data_directory_name, "Main", "things_concepts.tsv"), "wt") as f:
                f.write("Word\tuniqueID\n")
                for i in range(NUM_CLASSES):
                    f.write("{class_name}\t{class_name}\n".format(class_name=class_name(i)))

            for i in range(NUM_CLASSES):
                subdirectory_name = os.path.join(data_directory_name, "Main", class_name(i))
                os.makedirs(subdirectory_name)
                for j in range(NUM_IMAGES_PER_CLASS):
                    open(os.path.join(subdirectory_name, f"image_{j}.jpg"), "wt")

            subdirectory_name = os.path.join(data_directory_name, "Revealing", "triplets")
            os.makedirs(subdirectory_name)
            with open(os.path.join(subdirectory_name, "data1854_baseline_train90.txt"), "wt") as f:
                for _ in range(NUM_TRIPLETS):
                    i, j, k = random.sample(range(NUM_CLASSES), 3)
                    f.write(f"{i} {j} {k}\n")
            with open(os.path.join(subdirectory_name, "data1854_baseline_test10.txt"), "wt") as f:
                for _ in range(NUM_TRIPLETS // 2):
                    i, j, k = random.sample(range(NUM_CLASSES), 3)
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
            scipy.io.savemat(
                os.path.join(subdirectory_name, "sortind.mat"), {"sortind": list(range(1, NUM_CLASSES + 1))}
            )
            scipy.io.savemat(
                os.path.join(subdirectory_name, "words48.mat"),
                {"words48": [[[np.array(class_name(i))]] for i in range(NUM_CLASSES - NUM_TEST_CLASSES, NUM_CLASSES)]},
            )

            # Make data interface object and check its triplets and prototypes.
            data_interface = ThingsDataInterface(data_directory_name, "by_class", 42)

            classes_by_subset = {"training": set(), "validation": set(), "test": set()}
            for subset_name, triplets in data_interface.triplets_by_subset.items():
                self.assertIn(subset_name, {"training", "validation", "test"})
                self.assertGreaterEqual(len(triplets), 1)
                for t in triplets:
                    classes_by_subset[subset_name].update(t)

            self.assertTrue(classes_by_subset["training"].isdisjoint(classes_by_subset["validation"]))
            self.assertTrue(classes_by_subset["validation"].isdisjoint(classes_by_subset["test"]))
            self.assertTrue(classes_by_subset["test"].isdisjoint(classes_by_subset["training"]))

            self.assertEqual(len(data_interface.prototypes_per_class), data_interface.reader.num_classes)
            for c, p in enumerate(data_interface.prototypes_per_class):
                self.assertEqual(data_interface.reader.image_records[p][1], c)

            # Make another data interface object and check that its triplets and prototypes are different from the first
            # object's.
            data_interface2 = ThingsDataInterface(data_directory_name, "by_class", 24)

            self.assertNotEqual(data_interface.prototypes_per_class, data_interface2.prototypes_per_class)
            classes_per_prototype = {p: c for c, p in enumerate(data_interface.prototypes_per_class)}
            classes_per_prototype2 = {p: c for c, p in enumerate(data_interface2.prototypes_per_class)}

            for subset_name in ["training", "validation", "test"]:

                def to_class_triplet_strings(p2c, ts):
                    return set([" ".join([str(p2c[p]) for p in t]) for t in ts])

                triplets = to_class_triplet_strings(
                    classes_per_prototype, data_interface.triplets_by_subset[subset_name]
                )
                triplets2 = to_class_triplet_strings(
                    classes_per_prototype2, data_interface2.triplets_by_subset[subset_name]
                )
                self.assertNotEqual(triplets, triplets2)
