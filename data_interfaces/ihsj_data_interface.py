import random
import time

from more_itertools import random_permutation

from ditdml.data_interfaces.ihsj_reader import IHSJReader


def _index_subset_fully_in(s, *instance_sets):
    for i, instance_set in enumerate(instance_sets):
        if all([j in instance_set for j in s]):
            return i

    return -1


class IHSJDataInterface:
    """Interface to data the IHSJ dataset: image file names and triplets split into training, validation and test."""

    TRAINING_FRACTION1, VALIDATION_FRACTION1 = 0.6, 0.2
    TRAINING_FRACTION2 = 0.8
    TRAINING_FRACTION3, VALIDATION_FRACTION3 = 0.6, 0.2
    TRAINING_FRACTION4 = 0.8

    def __init__(self, directory_name, split_type, seed):
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
        self._split_triplets(split_type)

        # Split the raw set of ninelets into training, validation and test based on the specified split type.
        random.seed(seed)
        self._split_ninelets(split_type)

        print("done ({:.2f} s)".format(time.time() - start_time))

    @property
    def reader(self):
        return self._reader

    @property
    def triplets_by_subset(self):
        """Triplets of instance indexes for training, validation and test."""

        return self._triplets_by_subset

    @property
    def ninelets_by_subset(self):
        """Ninelets of instance indexes for training, validation and test."""

        return self._ninelets_by_subset

    @property
    def instances_by_subset(self):
        """Instance indexes for training, validation and test."""

        return self._instances_by_subset

    def _split_triplets(self, split_type):
        triplets = self._reader.triplets

        # Switch on split type.
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

        elif split_type == "by_class":
            # Randomly permute class indexes.
            classes = random_permutation(range(self._reader.num_classes))

            # Split class indexes into 60% training, 20% validation and 20% test.
            i = int(self.TRAINING_FRACTION3 * len(classes))
            j = int((self.TRAINING_FRACTION3 + self.VALIDATION_FRACTION3) * len(classes))
            training_classes = classes[:i]
            validation_classes = classes[i:j]
            test_classes = classes[j:]

            # Split instance indexes into training, validation and test according to the class split.
            training_instances = set([i for c in training_classes for i in self._instances_per_class[c]])
            validation_instances = set([i for c in validation_classes for i in self._instances_per_class[c]])
            test_instances = set([i for c in test_classes for i in self._instances_per_class[c]])

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

        elif split_type == "by_class_same_training_validation":
            # Randomly permute class indexes.
            classes = random_permutation(range(self._reader.num_classes))

            # Split class indexes into 80% training+validation and 20% test.
            i = int((self.TRAINING_FRACTION3 + self.VALIDATION_FRACTION3) * len(classes))
            training_validation_classes = set(classes[:i])
            test_classes = set(classes[i:])

            # Split instance indexes into training+validation and test according to the class split.
            training_validation_instances = set(
                [i for c in training_validation_classes for i in self._instances_per_class[c]]
            )
            test_instances = set([i for c in test_classes for i in self._instances_per_class[c]])

            # Split triplets into training+validation and test according to the image split.
            subset_indexes = [
                _index_subset_fully_in(t, training_validation_instances, test_instances) for t in triplets
            ]
            training_validation_triplets = [t for i, t in enumerate(triplets) if subset_indexes[i] == 0]
            test_triplets = [t for i, t in enumerate(triplets) if subset_indexes[i] == 1]

            # Randomly split the triplets in training+validation into 80% for training and 20% for validation.
            random.shuffle(list(training_validation_triplets))
            j = int(self.TRAINING_FRACTION4 * len(training_validation_triplets))
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

        else:
            # Unrecognized split type. All subsets are empty.
            self._triplets_by_subset = {"training": [], "validation": [], "test": []}
            self._instances_by_subset = {"training": [], "validation": [], "test": []}

    def _split_ninelets(self, split_type):
        ninelets = self._reader.ninelets

        # Switch on split type.
        if split_type == "by_instance":
            # Randomly permute image indexes.
            instances = random_permutation(range(self._reader.num_images))

            # Split image indexes into 60% training, 20% validation and 20% test.
            i = int(self.TRAINING_FRACTION1 * len(instances))
            j = int((self.TRAINING_FRACTION1 + self.VALIDATION_FRACTION1) * len(instances))
            training_instances = set(instances[:i])
            validation_instances = set(instances[i:j])
            test_instances = set(instances[j:])

            # Split ninelets into training, validation and test.
            subset_indexes = [
                _index_subset_fully_in(n, training_instances, validation_instances, test_instances) for n in ninelets
            ]
            self._ninelets_by_subset = {
                "training": [t for i, t in enumerate(ninelets) if subset_indexes[i] == 0],
                "validation": [t for i, t in enumerate(ninelets) if subset_indexes[i] == 1],
                "test": [t for i, t in enumerate(ninelets) if subset_indexes[i] == 2],
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

            # Split ninelets into training+validation and test according to the image split.
            subset_indexes = [
                _index_subset_fully_in(n, training_validation_instances, test_instances) for n in ninelets
            ]
            training_validation_ninelets = [n for i, n in enumerate(ninelets) if subset_indexes[i] == 0]
            test_ninelets = [n for i, n in enumerate(ninelets) if subset_indexes[i] == 1]

            # Randomly split the ninelets in training+validation into 80% for training and 20% for validation.
            random.shuffle(training_validation_ninelets)
            j = int(self.TRAINING_FRACTION2 * len(training_validation_ninelets))
            training_ninelets = training_validation_ninelets[:j]
            validation_ninelets = training_validation_ninelets[j:]

            # Make final ninelet sets.
            self._ninelets_by_subset = {
                "training": training_ninelets,
                "validation": validation_ninelets,
                "test": test_ninelets,
            }

            self._instances_by_subset = {
                "training": training_validation_instances,
                "validation": training_validation_instances,
                "test": test_instances,
            }

        elif split_type == "by_class":
            # Randomly permute class indexes.
            classes = random_permutation(range(self._reader.num_classes))

            # Split class indexes into 60% training, 20% validation and 20% test.
            i = int(self.TRAINING_FRACTION3 * len(classes))
            j = int((self.TRAINING_FRACTION3 + self.VALIDATION_FRACTION3) * len(classes))
            training_classes = classes[:i]
            validation_classes = classes[i:j]
            test_classes = classes[j:]

            # Split instance indexes into training, validation and test according to the class split.
            training_instances = set([i for c in training_classes for i in self._instances_per_class[c]])
            validation_instances = set([i for c in validation_classes for i in self._instances_per_class[c]])
            test_instances = set([i for c in test_classes for i in self._instances_per_class[c]])

            # Split ninelets into training, validation and test.
            subset_indexes = [
                _index_subset_fully_in(n, training_instances, validation_instances, test_instances) for n in ninelets
            ]
            self._ninelets_by_subset = {
                "training": [t for i, t in enumerate(ninelets) if subset_indexes[i] == 0],
                "validation": [t for i, t in enumerate(ninelets) if subset_indexes[i] == 1],
                "test": [t for i, t in enumerate(ninelets) if subset_indexes[i] == 2],
            }

            self._instances_by_subset = {
                "training": training_instances,
                "validation": validation_instances,
                "test": test_instances,
            }

        elif split_type == "by_class_same_training_validation":
            # Randomly permute class indexes.
            classes = random_permutation(range(self._reader.num_classes))

            # Split class indexes into 80% training+validation and 20% test.
            i = int((self.TRAINING_FRACTION3 + self.VALIDATION_FRACTION3) * len(classes))
            training_validation_classes = set(classes[:i])
            test_classes = set(classes[i:])

            # Split instance indexes into training+validation and test according to the class split.
            training_validation_instances = set(
                [i for c in training_validation_classes for i in self._instances_per_class[c]]
            )
            test_instances = set([i for c in test_classes for i in self._instances_per_class[c]])

            # Split ninelets into training+validation and test according to the image split.
            subset_indexes = [
                _index_subset_fully_in(n, training_validation_instances, test_instances) for n in ninelets
            ]
            training_validation_ninelets = [t for i, t in enumerate(ninelets) if subset_indexes[i] == 0]
            test_ninelets = [n for i, n in enumerate(ninelets) if subset_indexes[i] == 1]

            # Randomly split the ninelets in training+validation into 80% for training and 20% for validation.
            random.shuffle(list(training_validation_ninelets))
            j = int(self.TRAINING_FRACTION4 * len(training_validation_ninelets))
            training_ninelets = training_validation_ninelets[:j]
            validation_ninelets = training_validation_ninelets[j:]

            # Make final ninelet sets.
            self._ninelets_by_subset = {
                "training": training_ninelets,
                "validation": validation_ninelets,
                "test": test_ninelets,
            }

            self._instances_by_subset = {
                "training": training_validation_instances,
                "validation": training_validation_instances,
                "test": test_instances,
            }

        else:
            # Unrecognized split type. All subsets are empty.
            self._ninelets_by_subset = {"training": [], "validation": [], "test": []}
            self._instances_by_subset = {"training": [], "validation": [], "test": []}
