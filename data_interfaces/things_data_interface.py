import copy
import random
import time

from ditdml.data_interfaces.data_interface import DataInterfaceWithClasses
from ditdml.data_interfaces.data_interface_detail import split_class_triplets_things_style
from ditdml.data_interfaces.things_reader import ThingsReader


class ThingsDataInterface(DataInterfaceWithClasses):
    """Interface to data the THINGS dataset: image file names and triplets split into training, validation and test.

    Converts class triplets to image triplets by choosing a prototype image per class randomly. Provides access to
    additional data via the encapsulated reader object; see comments for `ThingsReader` for fields.
    """

    NUM_INSTANCE_SAMPLES_FOR_CLASS_TRIPLET = 5

    def __init__(self, directory_name, split_type, seed, **kwargs):
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
        self._split_dataset(split_type, kwargs.get("class_triplet_conversion_type"))

        print("done ({:.2f} s)".format(time.time() - start_time))

    @property
    def raw_triplets(self):
        return self._reader.class_triplets

    @property
    def raw_triplet_type(self):
        return "class"

    def _split_dataset(self, split_type, class_triplet_conversion_type=None):
        """Splits the original set of triplets and classes into subsets for training, validation and test."""

        # Get the class triplets and the classes.
        class_triplets = copy.deepcopy(self.raw_triplets)
        classes = list(range(self._reader.num_classes))

        if split_type == "quasi_original":
            class_triplets_by_subset, classes_by_subset = split_class_triplets_things_style(
                class_triplets, classes, self._reader.classes_original_test
            )
            self._make_triplets_from_class_triplets(class_triplets_by_subset, classes_by_subset, "prototypes")

        else:
            super()._split_dataset(split_type, class_triplet_conversion_type)
