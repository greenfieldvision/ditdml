import random
import time

from more_itertools import partition, random_permutation

from ditdml.data_interfaces.things_reader import ThingsReader


def triplets_fully_in(class_triplets, classes):
    return list(filter(lambda t: len(classes.intersection(t)) == 3, class_triplets))


class ThingsDataInterface:
    """Interface to data the THINGS dataset: image file names and triplets split into training, validation and test.

    Converts class triplets to image triplets by choosing a prototype image per class randomly. Provides access to
    additional data via the encapsulated reader object; see comments for `ThingsReader` for fields.
    """

    TRAINING_FRACTION1 = 0.8
    TRAINING_FRACTION2, VALIDATION_FRACTION2 = 0.6, 0.2
    TRAINING_FRACTION3 = 0.8

    NUM_INSTANCE_SAMPLES_FOR_CLASS_TRIPLET = 15

    def __init__(self, directory_name, split_type, seed):
        # Make reader object that loads the data from the specified directory.
        self._reader = ThingsReader(directory_name)

        # Make list of image indexes per class index.
        self._instances_per_class = [[] for _ in range(self._reader.num_classes)]
        for i, r in enumerate(self._reader.image_records):
            self._instances_per_class[r[1]].append(i)

        # Report times for splitting operations.
        print("splitting THINGS data...")
        start_time = time.time()

        # Choose a random representative image per class.
        random.seed(seed)
        self._choose_prototypes()

        # Split the raw set of triplets into training, validation and test based on the specified split type.
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

        class_triplets = self._reader.class_triplets

        # Switch on split type.
        if split_type == "quasi_original":
            # Split type close to that described in the "Revealing interpretable object representations..." paper: the
            # training and validation triplets have at most 1 class from a set of 48 test classes. The training and
            # validation follow a 80%-20% split.

            # Partition original triplet set into two subsets, according to the number of test classes in a triplet
            # (<=1 or >=2).
            test_classes = set(self._reader.classes_original_test)
            high_overlap_test_triplets, low_overlap_test_triplets = partition(
                lambda t: len(test_classes.intersection(t)) <= 1, class_triplets
            )
            low_overlap_test_triplets = list(low_overlap_test_triplets)
            high_overlap_test_triplets = list(high_overlap_test_triplets)

            # Randomly split the triplets with <=1 test classes into 80% training and 20% validation.
            random.shuffle(low_overlap_test_triplets)
            i = int(self.TRAINING_FRACTION1 * len(low_overlap_test_triplets))
            class_triplets_by_subset = {
                "training": low_overlap_test_triplets[:i],
                "validation": low_overlap_test_triplets[i:],
                "test": high_overlap_test_triplets,
            }

            # Save the class assignments.
            training_validation_classes = set(range(self._reader.num_classes)).difference(test_classes)
            classes_by_subset = {
                "training": training_validation_classes,
                "validation": training_validation_classes,
                "test": test_classes,
            }

        elif split_type == "by_class":
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
        if split_type in {"quasi_original", "by_class_same_training_validation"}:
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
