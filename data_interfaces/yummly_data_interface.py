import random
import time

from ditdml.data_interfaces.data_interface import DataInterface
from ditdml.data_interfaces.data_interface_detail import split_groups
from ditdml.data_interfaces.yummly_reader import YummlyReader


class YummlyDataInterface(DataInterface):
    """Interface to data the Yummly dataset: image file names and triplets split into training, validation and test."""

    TRAINING_FRACTION1, VALIDATION_FRACTION1 = 0.6, 0.2
    TRAINING_FRACTION2 = 0.8

    def __init__(self, directory_name, split_type, seed, **kwargs):
        # Make reader object that loads the data from the specified directory.
        self._reader = YummlyReader(directory_name)

        # Report times for splitting operations.
        print("splitting Yummly data...")
        start_time = time.time()

        random.seed(seed)
        self._split_dataset(split_type)

        print("done ({:.2f} s)".format(time.time() - start_time))

    @property
    def raw_triplets(self):
        return self._reader.triplets

    def _split_dataset(self, split_type):
        """Splits the original set of triplets and instances into subsets for training, validation and test."""

        # Get the class triplets and the classes.
        triplets = self.raw_triplets
        instances = list(range(self._reader.num_images))

        if split_type == "same_training_validation_test":
            self._triplets_by_subset, self._instances_by_subset = split_groups(triplets, instances)

        else:
            super()._split_dataset(split_type)
