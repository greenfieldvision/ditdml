import os
import random
import tempfile

import numpy as np

from data_interface_test import DataInterfaceTest
from ihsj_data_interface import IHSJDataInterface


NUM_CLASSES = 20
NUM_IMAGES_PER_CLASS = 5
NUM_TRIPLETS = 1000


def _class_name_id(class_index):
    return "n{:02d}".format(class_index), "class{:02d}".format(class_index)


class IHSJDataInterfaceTest(DataInterfaceTest):
    def _make_dataset(self, data_directory_name):
        os.makedirs(os.path.join(data_directory_name, "imagenet", "validation"))

        with open(os.path.join(data_directory_name, "imagenet", "LOC_synset_mapping.txt"), "wt") as f:
            for i in range(NUM_CLASSES):
                class_name, class_id = _class_name_id(i)
                f.write("{class_id} {class_name}\n".format(class_id=class_id, class_name=class_name))

        with open(
            os.path.join(data_directory_name, "imagenet", "validation", "imagenet_2012_validation_synset_labels.txt"),
            "wt",
        ) as f:
            for i in range(NUM_CLASSES):
                for j in range(NUM_IMAGES_PER_CLASS):
                    f.write("{class_id}\n".format(class_id=_class_name_id(i)[1]))

        image_directory_name = os.path.join(data_directory_name, "imagenet", "validation")
        for i in range(NUM_CLASSES):
            for j in range(NUM_IMAGES_PER_CLASS):
                open(
                    os.path.join(image_directory_name, "image_{n}.JPEG".format(n=i * NUM_IMAGES_PER_CLASS + j)),
                    "wt",
                )

        random.seed(1)
        triplets = list(
            set([tuple(random.sample(range(NUM_CLASSES * NUM_IMAGES_PER_CLASS), 3)) for _ in range(NUM_TRIPLETS)])
        )

        np.save(os.path.join(data_directory_name, "triplets.npy"), triplets)
        np.save(os.path.join(data_directory_name, "ninelets.npy"), [])

    def test_by_instance_split(self):
        with tempfile.TemporaryDirectory() as data_directory_name:
            # Create resource files.
            self._make_dataset(data_directory_name)

            # Make data interface object and check its triplets and instances.
            data_interface = IHSJDataInterface(data_directory_name, "by_instance", 42)

            # Check that the triplets are non-empty, disjoint and come from the original triplets.
            self._check_triplets(data_interface)

            # Check that the triplets respect the instance split.
            self._check_instances_triplets(data_interface)

            # Make another data interface object and check that its triplets are different from the first object's.
            data_interface2 = IHSJDataInterface(data_directory_name, "by_instance", 24)
            self._check_different_interfaces(data_interface, data_interface2, ["training", "validation", "test"])

    def test_by_instance_same_training_validation_split(self):
        with tempfile.TemporaryDirectory() as data_directory_name:
            # Create resource files.
            self._make_dataset(data_directory_name)

            # Make data interface object and check its triplets and instances.
            data_interface = IHSJDataInterface(data_directory_name, "by_instance_same_training_validation", 42)

            # Check that the triplets are disjoint.
            self._check_triplets(data_interface)

            # Check that the triplets respect the instance split.
            self._check_instances_triplets(data_interface)

            # Make another data interface object and check that its triplets and instances are different from the first object's.
            data_interface2 = IHSJDataInterface(data_directory_name, "by_instance_same_training_validation", 24)
            self._check_different_interfaces(data_interface, data_interface2, ["training", "validation", "test"])

    def test_by_class_split(self):
        with tempfile.TemporaryDirectory() as data_directory_name:
            # Create resource files.
            self._make_dataset(data_directory_name)

            # Make data interface object and check its triplets and instances.
            data_interface = IHSJDataInterface(data_directory_name, "by_class", 42)

            # Check that the triplets are disjoint.
            self._check_triplets(data_interface)

            # Check that the triplets respect the instance split.
            self._check_instances_triplets(data_interface)

            # Make another data interface object and check that its triplets are different from the first object's.
            data_interface2 = IHSJDataInterface(data_directory_name, "by_class", 24)
            self._check_different_interfaces(data_interface, data_interface2, ["training", "validation", "test"])

    def test_by_class_same_training_validation_split(self):
        with tempfile.TemporaryDirectory() as data_directory_name:
            # Create resource files.
            self._make_dataset(data_directory_name)

            # Make data interface object and check its triplets and instances.
            data_interface = IHSJDataInterface(data_directory_name, "by_class_same_training_validation", 42)

            # Check that the triplets are disjoint.
            self._check_triplets(data_interface)

            # Check that the triplets respect the instance split.
            self._check_instances_triplets(data_interface)

            # Make another data interface object and check that its triplets and instances are different from the first object's.
            data_interface2 = IHSJDataInterface(data_directory_name, "by_class_same_training_validation", 24)
            self._check_different_interfaces(data_interface, data_interface2, ["training", "validation", "test"])

    def test_by_class_consistency(self):
        with tempfile.TemporaryDirectory() as data_directory_name:
            self._make_dataset(data_directory_name)

            # Make two data interface objects that partition by class: one with a standard split and one with the same
            # classes in training and validation.
            data_interface1 = IHSJDataInterface(
                data_directory_name, "by_class", 42, class_triplet_conversion_type="prototypes"
            )
            data_interface2 = IHSJDataInterface(
                data_directory_name, "by_class_same_training_validation", 42, class_triplet_conversion_type="prototypes"
            )

            # Check that the test triplet subsets are the same.
            self.assertEqual(data_interface1.triplets_by_subset["test"], data_interface2.triplets_by_subset["test"])
