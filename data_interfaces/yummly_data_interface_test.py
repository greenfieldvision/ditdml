import json
import os
import random
import tempfile

from data_interface_test import DataInterfaceTest
from yummly_data_interface import YummlyDataInterface


NUM_INSTANCES = 100
NUM_TRIPLETS = 1000


class YummlyDataInterfaceTest(DataInterfaceTest):
    def _make_dataset(self, data_directory_name):
        os.makedirs(os.path.join(data_directory_name, "images"))

        image_directory_name = os.path.join(data_directory_name, "images")
        for i in range(NUM_INSTANCES):
            open(os.path.join(image_directory_name, "image_{}.jpg".format(i)), "wt")

        random.seed(1)
        triplets = list(set([tuple(random.sample(range(NUM_INSTANCES), 3)) for _ in range(NUM_TRIPLETS)]))
        self.num_triplets = len(triplets)

        with open(os.path.join(data_directory_name, "all-triplets.csv"), "wt") as f:
            for i, j, k in triplets:
                f.write("images/image_{}.jpg; images/image_{}.jpg; images/image_{}.jpg\n".format(i, j, k))

        with open(os.path.join(data_directory_name, "ingredients.json"), "wt") as f:
            names = {"image_{}.jpg".format(i): {"name": "name_".format(i)} for i in range(NUM_INSTANCES)}
            json.dump(names, f)

    def test_same_training_validation_test_split(self):
        with tempfile.TemporaryDirectory() as data_directory_name:
            # Create resource files.
            self._make_dataset(data_directory_name)

            # Make data interface object and check its triplets and instances.
            data_interface = YummlyDataInterface(data_directory_name, "same_training_validation_test", 42)

            # Check that the triplet subsets are non-empty, disjoint and come from the original triplet set.
            self._check_triplets(data_interface)

            # Check that the triplet subsets partition the original triplet set.
            triplets_by_subset = data_interface.triplets_by_subset
            self.assertEqual(
                len(triplets_by_subset["training"])
                + len(triplets_by_subset["validation"])
                + len(triplets_by_subset["test"]),
                self.num_triplets,
            )

            # Check that the triplet sets respect the instance split.
            self._check_instances_triplets(data_interface)

            # Make another data interface object and check that its triplets are different from the first object's.
            data_interface2 = YummlyDataInterface(data_directory_name, "same_training_validation_test", 24)
            self._check_different_interfaces(
                data_interface, data_interface2, ["training", "validation", "test"], has_different_instances=False
            )

    def test_by_instance_split(self):
        with tempfile.TemporaryDirectory() as data_directory_name:
            # Create resource files.
            self._make_dataset(data_directory_name)

            # Make data interface object and check its triplets and instances.
            data_interface = YummlyDataInterface(data_directory_name, "by_instance", 42)

            # Check that the triplets are non-empty, disjoint and come from the original triplets.
            self._check_triplets(data_interface)

            # Check that the triplets respect the instance split.
            self._check_instances_triplets(data_interface)

            # Make another data interface object and check that its triplets are different from the first object's.
            data_interface2 = YummlyDataInterface(data_directory_name, "by_instance", 24)
            self._check_different_interfaces(data_interface, data_interface2, ["training", "validation", "test"])

    def test_by_instance_same_training_validation_split(self):
        with tempfile.TemporaryDirectory() as data_directory_name:
            # Create resource files.
            self._make_dataset(data_directory_name)

            # Make data interface object and check its triplets and instances.
            data_interface = YummlyDataInterface(data_directory_name, "by_instance_same_training_validation", 42)

            # Check that the triplet subsets partition the original triplet set.
            self._check_triplets(data_interface)

            # Check that the triplet sets respect the instance split.
            self._check_instances_triplets(data_interface)

            # Make another data interface object and check that its triplets and instances are different from the first object's.
            data_interface2 = YummlyDataInterface(data_directory_name, "by_instance_same_training_validation", 24)
            self._check_different_interfaces(data_interface, data_interface2, ["training", "validation", "test"])
