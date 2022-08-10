import copy
import random
import time

from more_itertools import random_permutation

from ditdml.data_interfaces.yummly_reader import YummlyReader


def _index_subset_fully_in(s, *instance_sets):
    for i, instance_set in enumerate(instance_sets):
        if all([j in instance_set for j in s]):
            return i

    return -1


class YummlyDataInterface:
    """Interface to data the Yummly dataset: image file names and triplets split into training, validation and test."""

    TRAINING_FRACTION1, VALIDATION_FRACTION1 = 0.6, 0.2
    TRAINING_FRACTION2 = 0.8

    def __init__(self, directory_name, split_type, seed):
        # Make reader object that loads the data from the specified directory.
        self._reader = YummlyReader(directory_name)

        # Report times for splitting operations.
        print("splitting Yummly data...")
        start_time = time.time()

        random.seed(seed)
        self._split_triplets(split_type)

        print("done ({:.2f} s)".format(time.time() - start_time))

    @property
    def reader(self):
        return self._reader

    @property
    def triplets_by_subset(self):
        """Triplets of instance indexes for training, validation and test."""

        return self._triplets_by_subset

    @property
    def instances_by_subset(self):
        """Instance indexes for training, validation and test."""

        return self._instances_by_subset

    def _split_triplets(self, split_type):
        triplets = copy.deepcopy(self._reader.triplets)

        # Make the sets of triplet indexes for training, validation and test.
        if split_type == "by_instance":
            # Randomly permute image indexes.
            instances = random_permutation(range(self._reader.num_images))

            # Split image indexes into 60% training, 20% validation and 20% test.
            i = int(self.TRAINING_FRACTION1 * len(instances))
            j = int((self.TRAINING_FRACTION1 + self.VALIDATION_FRACTION1) * len(instances))
            training_instances = set(instances[:i])
            validation_instances = set(instances[i:j])
            test_instances = set(instances[j:])

            # Split triplets into training, validation and test.
            subset_indexes = [
                _index_subset_fully_in(t, training_instances, validation_instances, test_instances) for t in triplets
            ]
            self._triplets_by_subset = {
                "training": [t for i, t in enumerate(triplets) if subset_indexes[i] == 0],
                "validation": [t for i, t in enumerate(triplets) if subset_indexes[i] == 1],
                "test": [t for i, t in enumerate(triplets) if subset_indexes[i] == 2],
            }

            self._instances_by_subset = {
                "training": training_instances,
                "validation": validation_instances,
                "test": test_instances,
            }

        elif split_type == "by_instance_same_training_validation":
            # Randomly permute image indexes.
            instances = random_permutation(range(self._reader.num_images))

            # Split image indexes into 80% training+validation and 20% test.
            i = int((self.TRAINING_FRACTION1 + self.VALIDATION_FRACTION1) * len(instances))
            training_validation_instances = set(instances[:i])
            test_instances = set(instances[i:])

            # Split triplets into training+validation and test according to the image split.
            subset_indexes = [
                _index_subset_fully_in(t, training_validation_instances, test_instances) for t in triplets
            ]
            training_validation_triplets = [t for i, t in enumerate(triplets) if subset_indexes[i] == 0]
            test_triplets = [t for i, t in enumerate(triplets) if subset_indexes[i] == 1]

            # Randomly split the triplets in training+validation into 80% for training and 20% for validation.
            random.shuffle(training_validation_triplets)
            j = int(self.TRAINING_FRACTION2 * len(training_validation_triplets))
            training_triplets = training_validation_triplets[:j]
            validation_triplets = training_validation_triplets[j:]

            # Make final triplet sets.
            self._triplets_by_subset = {
                "training": training_triplets,
                "validation": validation_triplets,
                "test": test_triplets,
            }

            self._instances_by_subset = {
                "training": training_validation_instances,
                "validation": training_validation_instances,
                "test": test_instances,
            }

        elif split_type == "same_training_validation_test_instances":
            # Randomly split triplets into 60% training, 20% validation and 20% test.
            random.shuffle(triplets)
            i = int(self.TRAINING_FRACTION1 * len(triplets))
            j = int((self.TRAINING_FRACTION1 + self.VALIDATION_FRACTION1) * len(triplets))
            self._triplets_by_subset = {"training": triplets[:i], "validation": triplets[i:j], "test": triplets[j:]}

            instances = set(range(self._reader.num_images))
            self._instances_by_subset = {"training": instances, "validation": instances, "test": instances}

        else:
            # Unrecognized split type. All subsets are empty.
            self._triplets_by_subset = {"training": [], "validation": [], "test": []}
            self._instances_by_subset = {"training": [], "validation": [], "test": []}
