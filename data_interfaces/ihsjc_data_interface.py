import random
import time

from collections import defaultdict
from more_itertools import random_permutation

from ditdml.data_interfaces.ihsj_reader import IHSJReader
from ditdml.data_interfaces.ihsj_data_interface import IHSJDataInterface


def triplets_fully_in(class_triplets, classes):
    return list(filter(lambda t: len(classes.intersection(t)) == 3, class_triplets))


class IHSJCDataInterface(IHSJDataInterface):
    """Interface to data the IHSJC dataset: image file names and triplets split into training, validation and test."""

    TRAINING_FRACTION1 = 0.8
    TRAINING_FRACTION2, VALIDATION_FRACTION2 = 0.6, 0.2
    TRAINING_FRACTION3 = 0.8

    NUM_INSTANCE_SAMPLES_FOR_CLASS_TRIPLET = 17

    def __init__(self, directory_name, split_type, seed):
        # Make reader object that loads the data from the specified directory.
        self._reader = IHSJReader(directory_name)

        # Make list of image indexes per class index.
        self._instances_per_class = [[] for _ in range(self._reader.num_classes)]
        for i, r in enumerate(self._reader.image_records):
            self._instances_per_class[r[1]].append(i)

        # Report times for splitting operations.
        print("splitting IHSJC data...")
        start_time = time.time()

        # Choose a random representative image per class.
        random.seed(seed)
        self._choose_prototypes()

        # Split the raw set of triplets into training, validation and test based on the specified split type.
        random.seed(seed)
        self._split_triplets(split_type)

        print("done ({:.2f} s)".format(time.time() - start_time))

    @property
    def triplets(self):
        n, n1 = 0, 0
        class_triplets_by_id = defaultdict(list)
        for t in self._reader.triplets:
            ct = [self._reader.image_records[i][1] for i in t]

            # Check if the classes of the images in the triplet are all different.
            if len(set(ct)) == 3:
                key = "_".join([str(c) for c in sorted(ct)])

                # Discard duplicate class triplets.
                if not any([ct == ect for ect in class_triplets_by_id[key]]):
                    class_triplets_by_id[key].append(ct)
                    n += 1
                else:
                    n1 += 1

        # Discard class triplets contradicting others.
        n2, n3 = 0, 0
        for key, cts in class_triplets_by_id.items():
            kept = [True for _ in range(len(cts))]
            for i, ct_i in enumerate(cts):
                for j, ct_j in enumerate(cts):
                    if (j > i) and (ct_j == [ct_i[0], ct_i[2], ct_i[1]]):
                        kept[i], kept[j] = False, False
            cts = [ct for ct, k in zip(cts, kept) if k]
            n2 += sum([0 if k else 1 for k in kept])

            if len(cts) == 3:
                ct_i, ct_j, ct_k = cts
                if (ct_j == [ct_i[1], ct_i[2], ct_i[0]]) and (ct_k == [ct_i[2], ct_i[0], ct_i[1]]):
                    class_triplets_by_id[key] = []
                    n3 += 3
                elif (ct_j == [ct_i[2], ct_i[0], ct_i[1]]) and (ct_k == [ct_i[1], ct_i[0], ct_i[2]]):
                    class_triplets_by_id[key] = []
                    n3 += 3
                else:
                    class_triplets_by_id[key] = cts

        class_triplets = [ct for cts in class_triplets_by_id.values() for ct in cts]
        return class_triplets

    @property
    def prototypes_per_class(self):
        return self._prototypes_per_class

    def _choose_prototypes(self):
        """Selects a single image for each class randomly."""

        # Randomly choose an image index for each class index.
        self._prototypes_per_class = [
            random.choice(self._instances_per_class[c]) for c in range(self._reader.num_classes)
        ]

    def _split_triplets(self, split_type):
        """Splits the original set of triplets into subsets for training, validation and test.

        Transforms class triplets into image triplets using the class prototypes.
        """

        class_triplets = self.triplets

        # Switch on split type.
        if split_type == "by_class":
            # Split type based on classes. First, classes are split into training, validation and test (60%, 20%, 20%)
            # and then triplets are split into these three subsets according to the class split. Triplets that do not
            # have all classes in the same subset (ie either all training, all validation or all test) are discarded.

            # Randomly permute class indexes.
            classes = random_permutation(range(self._reader.num_classes))

            # Split class indexes into 60% training, 20% validation and 20% testing.
            i = int(self.TRAINING_FRACTION2 * len(classes))
            j = int((self.TRAINING_FRACTION2 + self.VALIDATION_FRACTION2) * len(classes))
            training_classes = set(classes[:i])
            validation_classes = set(classes[i:j])
            test_classes = set(classes[j:])

            # Assign each triplet to one of training, validation and test if all the triplet classes belong to that
            # subset.
            class_triplets_by_subset = {
                "training": triplets_fully_in(class_triplets, training_classes),
                "validation": triplets_fully_in(class_triplets, validation_classes),
                "test": triplets_fully_in(class_triplets, test_classes),
            }

            # Save the class assignments.
            classes_by_subset = {
                "training": training_classes,
                "validation": validation_classes,
                "test": test_classes,
            }

        elif split_type == "by_class_same_training_validation":
            # Split type based on classes. First, classes are split into training+validation and test (80%, 20%), then
            # triplets are split into these two subsets according to the class split. Finally, the training+validation
            # triplets are split randomly into 80% training and 20% validation.

            # Randomly permute class indexes.
            classes = random_permutation(range(self._reader.num_classes))

            # Split class indexes into 80% training+validation and 20% testing.
            i = int((self.TRAINING_FRACTION2 + self.VALIDATION_FRACTION2) * len(classes))
            training_validation_classes = set(classes[:i])
            test_classes = set(classes[i:])

            # Split triplets into training+validation and test according to the class split.
            training_validation_triplets = triplets_fully_in(class_triplets, training_validation_classes)
            test_triplets = triplets_fully_in(class_triplets, test_classes)

            # Randomly split the triplets in training+validation into 80% for training and 20% for validation.
            random.shuffle(training_validation_triplets)
            j = int(self.TRAINING_FRACTION3 * len(training_validation_triplets))
            training_triplets = training_validation_triplets[:j]
            validation_triplets = training_validation_triplets[j:]

            # Save the triplet and class assignments.
            class_triplets_by_subset = {
                "training": training_triplets,
                "validation": validation_triplets,
                "test": test_triplets,
            }
            classes_by_subset = {
                "training": training_validation_classes,
                "validation": training_validation_classes,
                "test": test_classes,
            }

        else:
            # Unrecognized split type. All subsets are empty.
            class_triplets_by_subset = {"training": [], "validation": [], "test": []}
            classes_by_subset = {"training": [], "validation": [], "test": []}

        # Switch on split type.
        if split_type == "by_class_same_training_validation":
            # Replace class indexes with prototype image indexes for triplets.
            self._triplets_by_subset = {
                subset_name: [[self._prototypes_per_class[c] for c in ct] for ct in class_triplets]
                for subset_name, class_triplets in class_triplets_by_subset.items()
            }

            # Replace class indexes with prototype image indexes for subsets.
            self._instances_by_subset = {
                subset_name: [self._prototypes_per_class[c] for c in classes]
                for subset_name, classes in classes_by_subset.items()
            }

        elif split_type == "by_class":
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

            # Get the image indexes for each class.
            self._instances_by_subset = {
                subset_name: [i for c in classes for i in self._instances_per_class[c]]
                for subset_name, classes in classes_by_subset.items()
            }

        else:
            # Unrecognized split type. All subsets are empty.
            self._instances_by_subset = {"training": [], "validation": [], "test": []}
