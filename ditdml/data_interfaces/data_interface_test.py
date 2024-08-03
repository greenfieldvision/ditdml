import random
import unittest

from collections import namedtuple

from data_interface import DataInterface


class DataInterfaceMock(DataInterface):
    def __init__(self, num_images, raw_triplets):
        self._reader = namedtuple("Reader", "num_images")(num_images)
        self._raw_triplets = raw_triplets

        self._triplets_by_subset = {None: raw_triplets}
        self._instances_by_subset = {None: list(range(num_images))}

    def raw_triplets(self):
        return self._raw_triplets


class DataInterfaceTest(unittest.TestCase):
    def _check_triplets(self, data_interface):
        for triplets in data_interface.triplets_by_subset.values():
            self.assertGreater(len(triplets), 0)

        if (not hasattr(data_interface, "raw_triplet_type")) or (data_interface.raw_triplet_type == "instance"):
            original_triplet_set = set([tuple(t) for t in data_interface.raw_triplets])
            for triplets in data_interface.triplets_by_subset.values():
                for triplet in triplets:
                    self.assertIn(tuple(triplet), original_triplet_set)

        triplets_by_subset = {
            subset_name: set([tuple(t) for t in triplets])
            for subset_name, triplets in data_interface.triplets_by_subset.items()
        }

        for subset_name1, triplets1 in triplets_by_subset.items():
            for subset_name2, triplets2 in triplets_by_subset.items():
                if subset_name1 < subset_name2:
                    self.assertTrue(triplets1.isdisjoint(triplets2))

    def _check_instances_triplets(self, data_interface):
        self.assertEqual(data_interface.triplets_by_subset.keys(), data_interface.instances_by_subset.keys())

        for subset_name in data_interface.triplets_by_subset:
            instance_set = set(data_interface.instances_by_subset[subset_name])
            for t in data_interface.triplets_by_subset[subset_name]:
                for i in t:
                    self.assertIn(i, instance_set)

    def _check_prototypes(self, data_interface):
        self.assertEqual(len(data_interface.prototypes_per_class), data_interface.reader.num_classes)

        for c, p in enumerate(data_interface.prototypes_per_class):
            self.assertEqual(data_interface.reader.image_records[p][1], c)

    def _check_different_interfaces(
        self, data_interface1, data_interface2, subset_names, has_classes=False, has_different_instances=True
    ):
        if has_classes:
            self.assertNotEqual(data_interface1.prototypes_per_class, data_interface2.prototypes_per_class)

        for subset_name in subset_names:
            triplets1_set = set([tuple(t) for t in data_interface1.triplets_by_subset[subset_name]])
            triplets2_set = set([tuple(t) for t in data_interface2.triplets_by_subset[subset_name]])
            self.assertNotEqual(triplets1_set, triplets2_set)

            if has_different_instances:
                instances1_set = set(data_interface1.instances_by_subset[subset_name])
                instances2_set = set(data_interface2.instances_by_subset[subset_name])
                self.assertNotEqual(instances1_set, instances2_set)

    def test_calculate_num_triplets_per_instance_small(self):
        data_interface = DataInterfaceMock(10, [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 7, 8], [0, 4, 8]])

        self.assertEqual(data_interface.calculate_num_triplets_per_instance(), [3, 1, 2, 1, 3, 1, 1, 1, 2, 0])

    def test_calculate_num_triplets_per_instance_random_small_coverage(self):
        random.seed(42)

        num_images = 1000
        data_interface = DataInterfaceMock(num_images, [tuple(random.sample(range(num_images), 3)) for _ in range(100)])

        num_triplets_per_instance = data_interface.calculate_num_triplets_per_instance()
        self.assertEqual(min(num_triplets_per_instance), 0)
        self.assertLessEqual(max(num_triplets_per_instance), 30)
        self.assertAlmostEqual(
            sum(num_triplets_per_instance) / (len(num_triplets_per_instance) + 1e-9), 0.3, delta=0.001
        )

    def test_calculate_num_triplets_per_instance_random_large_coverage(self):
        random.seed(42)

        num_images = 1000
        data_interface = DataInterfaceMock(
            num_images, [tuple(random.sample(range(num_images), 3)) for _ in range(10000)]
        )

        num_triplets_per_instance = data_interface.calculate_num_triplets_per_instance()
        self.assertGreater(min(num_triplets_per_instance), 0)
        self.assertLessEqual(max(num_triplets_per_instance), 60)
        self.assertAlmostEqual(
            sum(num_triplets_per_instance) / (len(num_triplets_per_instance) + 1e-9), 30.0, delta=0.001
        )
