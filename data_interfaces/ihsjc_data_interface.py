import random
import time

from collections import defaultdict

from ditdml.data_interfaces.data_interface import DataInterfaceWithClasses
from ditdml.data_interfaces.ihsj_reader import IHSJReader


class IHSJCDataInterface(DataInterfaceWithClasses):
    """Interface to data in the IHSJC dataset: image file names and triplets split into training, validation and test."""

    NUM_INSTANCE_SAMPLES_FOR_CLASS_TRIPLET = 3

    def __init__(self, directory_name, split_type, seed, **kwargs):
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
        self._split_dataset(split_type, kwargs.get("class_triplet_conversion_type"))

        print("done ({:.2f} s)".format(time.time() - start_time))

    @property
    def raw_triplets(self):
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
    def raw_triplet_type(self):
        return "class"
