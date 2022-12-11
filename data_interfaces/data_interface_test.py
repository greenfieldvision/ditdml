import unittest


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
