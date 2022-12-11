import random
import unittest

import data_interface_detail


class DataInterfaceDetailTest(unittest.TestCase):
    def setUp(self):
        random.seed(42)

    def _check_groups_respect_subsets(self, elements_by_subset, groups_by_subset):
        """Check that the instances/classes referenced by the groups of each subset belong to that subset."""

        self.assertCountEqual(elements_by_subset.keys(), groups_by_subset.keys())

        for subset_name, groups in groups_by_subset.items():
            element_set = set(elements_by_subset[subset_name])
            self.assertTrue(element_set.issuperset(set([instance for group in groups for instance in group])))

    def test_split_groups(self):
        instances = list(range(30))
        groups = [(i, j, k) for i in range(0, 10) for j in range(10, 20) for k in range(20, 30)]
        groups_by_subset, instances_by_subset = data_interface_detail.split_groups(groups, instances)

        self._check_groups_respect_subsets(instances_by_subset, groups_by_subset)

        self.assertEqual(instances_by_subset, {"training": instances, "validation": instances, "test": instances})

        self.assertCountEqual(groups_by_subset.keys(), ["training", "validation", "test"])
        for g in groups_by_subset.values():
            self.assertTrue(set(groups).issuperset(set([tuple(i) for i in g])))

        self.assertAlmostEqual(float(len(groups_by_subset["training"])) / len(groups), 0.6)
        self.assertAlmostEqual(float(len(groups_by_subset["validation"])) / len(groups), 0.2)
        self.assertAlmostEqual(float(len(groups_by_subset["test"])) / len(groups), 0.2)

    def test_split_groups_by_instance(self):
        instances = list(range(30))
        groups = [(i, j, k) for i in range(0, 10) for j in range(10, 20) for k in range(20, 30)]
        groups_by_subset, instances_by_subset = data_interface_detail.split_groups_by_instance(groups, instances)

        self._check_groups_respect_subsets(instances_by_subset, groups_by_subset)

        self.assertCountEqual(instances_by_subset.keys(), ["training", "validation", "test"])
        for i in instances_by_subset.values():
            self.assertTrue(set(instances).issuperset(set(i)))

        self.assertAlmostEqual(float(len(instances_by_subset["training"])) / len(instances), 0.6)
        self.assertAlmostEqual(float(len(instances_by_subset["validation"])) / len(instances), 0.2)
        self.assertAlmostEqual(float(len(instances_by_subset["test"])) / len(instances), 0.2)

        self.assertCountEqual(groups_by_subset.keys(), ["training", "validation", "test"])
        for g in groups_by_subset.values():
            self.assertTrue(set(groups).issuperset(set([tuple(i) for i in g])))

        self.assertGreater(len(groups_by_subset["training"]), 0)
        self.assertGreater(len(groups_by_subset["validation"]), 0)
        self.assertGreater(len(groups_by_subset["test"]), 0)

    def test_split_groups_by_instance_same_training_validation(self):
        instances = list(range(30))
        groups = [(i, j, k) for i in range(0, 10) for j in range(10, 20) for k in range(20, 30)]
        groups_by_subset, instances_by_subset = data_interface_detail.split_groups_by_instance_same_training_validation(
            groups, instances
        )

        self._check_groups_respect_subsets(instances_by_subset, groups_by_subset)

        self.assertCountEqual(instances_by_subset.keys(), ["training", "validation", "test"])
        for i in instances_by_subset.values():
            self.assertTrue(set(instances).issuperset(set(i)))

        self.assertCountEqual(instances_by_subset["training"], instances_by_subset["validation"])
        self.assertAlmostEqual(float(len(instances_by_subset["training"])) / len(instances), 0.8)
        self.assertAlmostEqual(float(len(instances_by_subset["test"])) / len(instances), 0.2)

        self.assertCountEqual(groups_by_subset.keys(), ["training", "validation", "test"])
        for g in groups_by_subset.values():
            self.assertTrue(set(groups).issuperset(set([tuple(i) for i in g])))

        self.assertAlmostEqual(
            float(len(groups_by_subset["training"])) / float(len(groups_by_subset["validation"])), 4.0, delta=0.01
        )

    def test_split_groups_by_class(self):
        classes = list(range(10))
        instances_per_class = [(3 * i + 0, 3 * i + 1, 3 * i + 2) for i in range(10)]
        groups = [(random.randint(0, 29), random.randint(0, 29), random.randint(0, 29)) for _ in range(2000)]
        groups_by_subset, instances_by_subset = data_interface_detail.split_groups_by_class(
            groups, classes, instances_per_class
        )

        class_per_instance = {j: c for c, i in enumerate(instances_per_class) for j in i}
        classes_by_subset = {s: set([class_per_instance[j] for j in i]) for s, i in instances_by_subset.items()}

        self._check_groups_respect_subsets(instances_by_subset, groups_by_subset)

        self.assertCountEqual(instances_by_subset.keys(), ["training", "validation", "test"])
        for instances in instances_by_subset.values():
            self.assertTrue(set(classes).issuperset(set([class_per_instance[i] for i in instances])))

        self.assertAlmostEqual(float(len(classes_by_subset["training"])) / len(classes), 0.6)
        self.assertAlmostEqual(float(len(classes_by_subset["validation"])) / len(classes), 0.2)
        self.assertAlmostEqual(float(len(classes_by_subset["test"])) / len(classes), 0.2)

        self.assertCountEqual(groups_by_subset.keys(), ["training", "validation", "test"])
        for g in groups_by_subset.values():
            self.assertTrue(set(groups).issuperset(set([tuple(i) for i in g])))

        self.assertGreater(len(groups_by_subset["training"]), 0)
        self.assertGreater(len(groups_by_subset["validation"]), 0)
        self.assertGreater(len(groups_by_subset["test"]), 0)

    def test_split_groups_by_class_same_training_validation(self):
        classes = list(range(10))
        instances_per_class = [(3 * i + 0, 3 * i + 1, 3 * i + 2) for i in range(10)]
        groups = [(random.randint(0, 29), random.randint(0, 29), random.randint(0, 29)) for _ in range(2000)]
        groups_by_subset, instances_by_subset = data_interface_detail.split_groups_by_class_same_training_validation(
            groups, classes, instances_per_class
        )

        class_per_instance = {j: c for c, i in enumerate(instances_per_class) for j in i}
        classes_by_subset = {s: set([class_per_instance[j] for j in i]) for s, i in instances_by_subset.items()}

        self._check_groups_respect_subsets(instances_by_subset, groups_by_subset)

        self.assertCountEqual(instances_by_subset.keys(), ["training", "validation", "test"])
        for i in instances_by_subset.values():
            self.assertTrue(set(classes).issuperset(set([class_per_instance[j] for j in i])))

        self.assertCountEqual(instances_by_subset["training"], instances_by_subset["validation"])

        self.assertCountEqual(classes_by_subset["training"], classes_by_subset["validation"])
        self.assertAlmostEqual(float(len(classes_by_subset["training"])) / len(classes), 0.8)
        self.assertAlmostEqual(float(len(classes_by_subset["test"])) / len(classes), 0.2)

        self.assertCountEqual(groups_by_subset.keys(), ["training", "validation", "test"])
        for g in groups_by_subset.values():
            self.assertTrue(set(groups).issuperset(set([tuple(i) for i in g])))

        self.assertAlmostEqual(
            float(len(groups_by_subset["training"])) / float(len(groups_by_subset["validation"])), 4.0, delta=0.01
        )

        self.assertGreater(len(groups_by_subset["training"]), 0)
        self.assertGreater(len(groups_by_subset["validation"]), 0)
        self.assertGreater(len(groups_by_subset["test"]), 0)

    def test_split_class_groups_by_class(self):
        classes = list(range(100))
        class_groups = [(random.randint(0, 99), random.randint(0, 99), random.randint(0, 99)) for _ in range(1000)]
        class_groups_by_subset, classes_by_subset = data_interface_detail.split_class_groups_by_class(
            class_groups, classes
        )

        self._check_groups_respect_subsets(classes_by_subset, class_groups_by_subset)

        self.assertCountEqual(classes_by_subset.keys(), ["training", "validation", "test"])
        for c in classes_by_subset.values():
            self.assertTrue(set(classes).issuperset(set(c)))

        self.assertAlmostEqual(float(len(classes_by_subset["training"])) / len(classes), 0.6)
        self.assertAlmostEqual(float(len(classes_by_subset["validation"])) / len(classes), 0.2)
        self.assertAlmostEqual(float(len(classes_by_subset["test"])) / len(classes), 0.2)

        self.assertCountEqual(class_groups_by_subset.keys(), ["training", "validation", "test"])
        for g in class_groups_by_subset.values():
            self.assertTrue(set(class_groups).issuperset(set([tuple(i) for i in g])))

        self.assertGreater(len(class_groups_by_subset["training"]), 0)
        self.assertGreater(len(class_groups_by_subset["validation"]), 0)
        self.assertGreater(len(class_groups_by_subset["test"]), 0)

    def test_split_class_groups_by_class_same_training_validation(self):
        classes = list(range(100))
        class_groups = [(random.randint(0, 99), random.randint(0, 99), random.randint(0, 99)) for _ in range(1000)]
        (
            class_groups_by_subset,
            classes_by_subset,
        ) = data_interface_detail.split_class_groups_by_class_same_training_validation(class_groups, classes)

        self._check_groups_respect_subsets(classes_by_subset, class_groups_by_subset)

        self.assertCountEqual(classes_by_subset.keys(), ["training", "validation", "test"])
        for c in classes_by_subset.values():
            self.assertTrue(set(classes).issuperset(set(c)))

        self.assertCountEqual(classes_by_subset["training"], classes_by_subset["validation"])
        self.assertAlmostEqual(float(len(classes_by_subset["training"])) / len(classes), 0.8)
        self.assertAlmostEqual(float(len(classes_by_subset["test"])) / len(classes), 0.2)

        self.assertCountEqual(class_groups_by_subset.keys(), ["training", "validation", "test"])
        for g in class_groups_by_subset.values():
            self.assertTrue(set(class_groups).issuperset(set([tuple(i) for i in g])))

        self.assertGreater(len(class_groups_by_subset["training"]), 0)
        self.assertGreater(len(class_groups_by_subset["validation"]), 0)
        self.assertGreater(len(class_groups_by_subset["test"]), 0)

    def test_split_class_triplets_things_style(self):
        classes = list(range(100))
        test_classes = list(range(75, 100))
        class_triplets = [(random.randint(0, 99), random.randint(0, 99), random.randint(0, 99)) for _ in range(1000)]
        class_triplets_by_subset, classes_by_subset = data_interface_detail.split_class_triplets_things_style(
            class_triplets, classes, test_classes
        )

        self.assertCountEqual(classes_by_subset.keys(), class_triplets_by_subset.keys())

        test_classes_set = set(test_classes)
        for s, ct in class_triplets_by_subset.items():
            if s == "test":
                for ct_ in ct:
                    self.assertGreaterEqual(len(test_classes_set.intersection(ct_)), 2)
            else:
                for ct_ in ct:
                    self.assertLessEqual(len(test_classes_set.intersection(ct_)), 1)

        self.assertCountEqual(classes_by_subset.keys(), ["training", "validation", "test"])
        for c in classes_by_subset.values():
            self.assertTrue(set(classes).issuperset(set(c)))

        self.assertCountEqual(classes_by_subset["training"], classes_by_subset["validation"])
        self.assertCountEqual(classes_by_subset["test"], test_classes)

        self.assertCountEqual(class_triplets_by_subset.keys(), ["training", "validation", "test"])
        for s, ct in class_triplets_by_subset.items():
            self.assertTrue(set(class_triplets).issuperset(set([tuple(c) for c in ct])))

        self.assertGreater(len(class_triplets_by_subset["training"]), 0)
        self.assertGreater(len(class_triplets_by_subset["validation"]), 0)
        self.assertGreater(len(class_triplets_by_subset["test"]), 0)

        self.assertAlmostEqual(
            float(len(class_triplets_by_subset["training"])) / float(len(class_triplets_by_subset["validation"])),
            4.0,
            delta=0.02,
        )
