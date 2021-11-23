import random
import time

from more_itertools import partition, random_permutation

from ditdml.data_interfaces.things_reader import ThingsReader


class ThingsDataInterface:
    """Interface to data the THINGS dataset: image file names and triplets split into training, validation and test.

    Converts class triplets to image triplets by choosing a prototype image per class randomly. Provides access to
    additional data via the encapsulated reader object; see comments for `ThingsReader` for fields.
    """

    TRAINING_FRACTION1 = 0.8
    TRAINING_FRACTION2, VALIDATION_FRACTION2 = 0.6, 0.2

    def __init__(self, directory_name, split_type, seed):
        # Make reader object that loads the data from the specified directory.
        self._reader = ThingsReader(directory_name)

        # Report times for splitting operations.
        print("splitting THINGS data...")
        start_time = time.time()

        # Choose a random representative image per class.
        random.seed(seed)
        self._choose_prototypes()

        # Split the raw set of triplets into training, validation and test based on the specified split type.
        random.seed(seed + 1)
        self._split_triplets(split_type)

        print("done ({:.2f} s)".format(time.time() - start_time))

    @property
    def reader(self):
        return self._reader

    @property
    def triplets_by_subset(self):
        return self._triplets_by_subset

    @property
    def prototypes_per_class(self):
        return self._prototypes_per_class

    def _choose_prototypes(self):
        """Selects a single image for each class randomly."""

        num_classes = self._reader.num_classes

        # Make list of image indexes per class index.
        images_per_class = [[] for _ in range(num_classes)]
        for i, r in enumerate(self._reader.image_records):
            images_per_class[r[1]].append(i)

        # Randomly choose an image index for each class index.
        self._prototypes_per_class = [random.choice(images_per_class[c]) for c in range(num_classes)]

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
            classes_test = set(self._reader.classes_original_test)
            high_overlap_test_triplets, low_overlap_test_triplets = partition(
                lambda t: len(classes_test.intersection(t)) <= 1, class_triplets
            )
            low_overlap_test_triplets = list(low_overlap_test_triplets)
            high_overlap_test_triplets = list(high_overlap_test_triplets)

            # Randomly split the triplets with <=1 test classes into 80% training and 20% validation.
            # random.shuffle(low_overlap_test_triplets)
            i = int(self.TRAINING_FRACTION1 * len(low_overlap_test_triplets))
            class_triplets_by_subset = {
                "training": low_overlap_test_triplets[:i],
                "validation": low_overlap_test_triplets[i:],
                "test": high_overlap_test_triplets,
            }

        elif split_type == "by_class":
            # Split type based on classes. Classes are first split into training, validation and test and then the
            # triplets that have all the classes in one of the three subsets are retained.

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
            def triplet_fully_in(classes):
                return filter(lambda t: len(classes.intersection(t)) == 3, class_triplets)

            class_triplets_by_subset = {
                "training": triplet_fully_in(training_classes),
                "validation": triplet_fully_in(validation_classes),
                "test": triplet_fully_in(test_classes),
            }

        else:
            # Unrecognized split type. All subsets are empty.
            class_triplets_by_subset = {"training": [], "validation": [], "test": []}

        # Replace class indexes with prototype image indexes.
        self._triplets_by_subset = {
            subset_name: [[self._prototypes_per_class[c] for c in t] for t in triplets]
            for subset_name, triplets in class_triplets_by_subset.items()
        }
