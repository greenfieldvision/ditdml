import copy
import random

import ditdml.data_interfaces.data_interface_detail as data_interface_detail

from abc import ABC, abstractmethod, abstractproperty
from collections import Counter


class DataInterface(ABC):
    @abstractmethod
    def __init__(self, directory_name, split_type, seed, **kwargs):
        raise NotImplementedError()

    @property
    def reader(self):
        """Object with basic data loading functionality for dataset."""

        return self._reader

    @property
    def triplets_by_subset(self):
        """Triplets of instance indexes for training, validation and test."""

        return self._triplets_by_subset

    @property
    def instances_by_subset(self):
        """Instance indexes for training, validation and test."""

        return self._instances_by_subset

    @abstractproperty
    def raw_triplets(self):
        """Raw triplet data from the reader (eg class triplets)."""

        raise NotImplementedError()

    def calculate_num_triplets_per_instance(self):
        """Calculate for each instance the number of triplets it appears in."""

        occurrences = Counter([i for ts in self._triplets_by_subset.values() for t in ts for i in t])
        num_triplets_per_instance = [occurrences.get(i, 0) for i in range(self._reader.num_images)]

        return num_triplets_per_instance

    def _split_dataset(self, split_type):
        """Split the original sets of triplets and instances into subsets for training, validation and test."""

        # Get the triplets and the instances.
        triplets = copy.deepcopy(self.raw_triplets)
        instances = list(range(self._reader.num_images))

        # Switch on split type.
        if split_type == "by_instance":
            # Split triplets by instance.
            self._triplets_by_subset, self._instances_by_subset = data_interface_detail.split_groups_by_instance(
                triplets, instances
            )

        elif split_type == "by_instance_same_training_validation":
            # Split triplets by instance, making the training and validation instance sets the same.
            (
                self._triplets_by_subset,
                self._instances_by_subset,
            ) = data_interface_detail.split_groups_by_instance_same_training_validation(triplets, instances)

        else:
            # Unrecognized split type. All subsets are empty.
            self._triplets_by_subset = {"training": [], "validation": [], "test": []}
            self._instances_by_subset = {"training": [], "validation": [], "test": []}


class DataInterfaceWithClasses(DataInterface):
    @property
    def prototypes_per_class(self):
        """The index of the prototype instance for each class."""

        return self._prototypes_per_class

    @abstractproperty
    def raw_triplet_type(self):
        """The type of raw triplets: instance or class based."""

        return NotImplementedError()

    def _choose_prototypes(self):
        """Selects a single image for each class randomly."""

        # Randomly choose an image index for each class index.
        self._prototypes_per_class = [
            random.choice(self._instances_per_class[c]) for c in range(self._reader.num_classes)
        ]

    def _split_dataset(self, split_type, class_triplet_conversion_type=None):
        """Split the original sets of triplets, classes and instances into subsets for training, validation and test."""

        # Get the instance triplets/class triplets and the classes.
        triplets = copy.deepcopy(self.raw_triplets)
        classes = list(range(self._reader.num_classes))

        # Switch on split type.
        if split_type == "by_class":
            # Split triplets by class.

            # Switch on raw triplet type.
            if self.raw_triplet_type == "instance":
                self._triplets_by_subset, self._instances_by_subset = data_interface_detail.split_groups_by_class(
                    triplets, classes, self._instances_per_class
                )
            elif self.raw_triplet_type == "class":
                class_triplets_by_subset, classes_by_subset = data_interface_detail.split_class_groups_by_class(
                    triplets, classes
                )
                self._make_triplets_from_class_triplets(
                    class_triplets_by_subset, classes_by_subset, class_triplet_conversion_type
                )
            else:
                raise NotImplementedError("Could not recognize type of raw triplets.")

        elif split_type == "by_class_same_training_validation":
            # Split triplets by class, making the training and validation class sets the same.

            # Switch on raw triplet type.
            if self.raw_triplet_type == "instance":
                (
                    self._triplets_by_subset,
                    self._instances_by_subset,
                ) = data_interface_detail.split_groups_by_class_same_training_validation(
                    triplets, classes, self._instances_per_class
                )
            elif self.raw_triplet_type == "class":
                (
                    class_triplets_by_subset,
                    classes_by_subset,
                ) = data_interface_detail.split_class_groups_by_class_same_training_validation(triplets, classes)
                self._make_triplets_from_class_triplets(
                    class_triplets_by_subset, classes_by_subset, class_triplet_conversion_type
                )
            else:
                raise NotImplementedError("Could not recognize type of raw triplets.")

        else:
            super()._split_dataset(split_type)

    def _make_triplets_from_class_triplets(
        self, class_triplets_by_subset, classes_by_subset, class_triplet_conversion_type
    ):
        if class_triplet_conversion_type == "all_instances":
            # Get the image indexes for each class.
            self._instances_by_subset = {
                subset_name: [i for c in classes for i in self._instances_per_class[c]]
                for subset_name, classes in classes_by_subset.items()
            }

            # For each triplet, sample an image index from each class (instead of choosing the prototype). The number
            # of samples per class triplet is set so each image in the three classes will likely appear at least once
            # in the image triplets.
            self._triplets_by_subset = {subset_name: [] for subset_name in class_triplets_by_subset}
            for _ in range(self.NUM_INSTANCE_SAMPLES_FOR_CLASS_TRIPLET):
                for subset_name, class_triplets in class_triplets_by_subset.items():
                    current_triplets = [
                        [random.choice(self._instances_per_class[c]) for c in ct] for ct in class_triplets
                    ]
                    self._triplets_by_subset[subset_name].extend(current_triplets)

        elif class_triplet_conversion_type == "prototypes":
            # Replace class indexes with prototype image indexes for subsets.
            self._instances_by_subset = {
                subset_name: [self._prototypes_per_class[c] for c in classes]
                for subset_name, classes in classes_by_subset.items()
            }

            # Replace class indexes with prototype image indexes for triplets.
            self._triplets_by_subset = {
                subset_name: [[self._prototypes_per_class[c] for c in ct] for ct in class_triplets]
                for subset_name, class_triplets in class_triplets_by_subset.items()
            }

        else:
            raise NotImplementedError("Could not recognize type of class triplet conversion.")
