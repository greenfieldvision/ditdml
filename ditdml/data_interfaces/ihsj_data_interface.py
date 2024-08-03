import copy
import random
import time

from ditdml.data_interfaces.data_interface import DataInterfaceWithClasses
from ditdml.data_interfaces.data_interface_detail import (
    split_groups_by_instance,
    split_groups_by_instance_same_training_validation,
    split_groups_by_class,
    split_groups_by_class_same_training_validation,
)
from ditdml.data_interfaces.ihsj_reader import IHSJReader


class IHSJDataInterface(DataInterfaceWithClasses):
    """Interface to data in the IHSJ dataset: image file names and triplets split into training, validation and test."""

    def __init__(self, directory_name, split_type, seed, **kwargs):
        # Make reader object that loads the data from the specified directory.
        self._reader = IHSJReader(directory_name)

        # Make list of image indexes per class index.
        self._instances_per_class = [[] for _ in range(self._reader.num_classes)]
        for i, r in enumerate(self._reader.image_records):
            self._instances_per_class[r[1]].append(i)

        # Report times for splitting operations.
        print("splitting IHSJ data...")
        start_time = time.time()

        # Split the raw set of triplets into training, validation and test based on the specified split type.
        random.seed(seed)
        super()._split_dataset(split_type)

        # Split the raw set of ninelets into training, validation and test based on the specified split type.
        random.seed(seed)
        self._split_dataset(split_type)

        print("done ({:.2f} s)".format(time.time() - start_time))

    @property
    def ninelets_by_subset(self):
        """Ninelets of instance indexes for training, validation and test."""

        return self._ninelets_by_subset

    @property
    def raw_triplets(self):
        return self._reader.triplets

    @property
    def raw_triplet_type(self):
        return "instance"

    def _split_dataset(self, split_type):
        """Split the set of triplets/ninelets and instances/classes into subsets for training, validation and test."""

        # Get the ninelets, the instances and the classes.
        ninelets = copy.deepcopy(self._reader.ninelets)
        instances = list(range(self._reader.num_images))
        classes = list(range(self._reader.num_classes))

        if split_type == "by_instance":
            # Split ninelets by instance.
            self._ninelets_by_subset, _ = split_groups_by_instance(ninelets, instances)
        elif split_type == "by_instance_same_training_validation":
            # Split ninelets by instance, making the training and validation instance sets the same.
            self._ninelets_by_subset, _ = split_groups_by_instance_same_training_validation(ninelets, instances)
        elif split_type == "by_class":
            self._ninelets_by_subset, _ = split_groups_by_class(ninelets, classes, self._instances_per_class)
        elif split_type == "by_class_same_training_validation":
            self._ninelets_by_subset, _ = split_groups_by_class_same_training_validation(
                ninelets, classes, self._instances_per_class
            )
