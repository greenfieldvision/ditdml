import random

from more_itertools import partition, random_permutation


TRAINING_FRACTION_SPLIT, VALIDATION_FRACTION_SPLIT = 0.6, 0.2
TRAINING_FRACTION_SPLIT_BY_INSTANCE, VALIDATION_FRACTION_SPLIT_BY_INSTANCE = 0.6, 0.2
TRAINING_FRACTION_SPLIT_BY_INSTANCE_SAME_TRAINING_VALIDATION = 0.8
TRAINING_FRACTION_SPLIT_BY_CLASS, VALIDATION_FRACTION_SPLIT_BY_CLASS = 0.6, 0.2
TRAINING_FRACTION_SPLIT_BY_CLASS_SAME_TRAINING_VALIDATION = 0.8
TRAINING_FRACTION_SPLIT_THINGS = 0.8


def _groups_fully_in_set(groups, element_set):
    return list(filter(lambda g: len(element_set.intersection(g)) == len(g), groups))


def split_groups(groups, instances):
    # Randomly split the groups into 60% training, 20% validation and 20% test, allowing them to share instances.
    random.shuffle(groups)
    i = int(TRAINING_FRACTION_SPLIT * len(groups))
    j = int((TRAINING_FRACTION_SPLIT + VALIDATION_FRACTION_SPLIT) * len(groups))
    groups_by_subset = {"training": groups[:i], "validation": groups[i:j], "test": groups[j:]}

    # For completeness, copy the instances to training, validation and test.
    instances_by_subset = {"training": instances, "validation": instances, "test": instances}

    return groups_by_subset, instances_by_subset


def split_groups_by_instance(groups, instances):
    # Randomly split the instances into 60% training, 20% validation and 20% test.
    instances = random_permutation(instances)
    i = int(TRAINING_FRACTION_SPLIT_BY_INSTANCE * len(instances))
    j = int((TRAINING_FRACTION_SPLIT_BY_INSTANCE + VALIDATION_FRACTION_SPLIT_BY_INSTANCE) * len(instances))
    instances_by_subset = {
        "training": set(instances[:i]),
        "validation": set(instances[i:j]),
        "test": set(instances[j:]),
    }

    # Split groups into training, validation and test according to the instance split.
    groups_by_subset = {
        subset_name: _groups_fully_in_set(groups, instances) for subset_name, instances in instances_by_subset.items()
    }

    return groups_by_subset, instances_by_subset


def split_groups_by_instance_same_training_validation(groups, instances):
    # Randomly split the instances into 80% training+validation and 20% test.
    instances = random_permutation(instances)
    i = int((TRAINING_FRACTION_SPLIT_BY_INSTANCE + VALIDATION_FRACTION_SPLIT_BY_INSTANCE) * len(instances))
    instances_by_subset = {"training_validation": set(instances[:i]), "test": set(instances[i:])}

    # Split groups into training+validation and test according to the instance split.
    groups_by_subset = {
        subset_name: _groups_fully_in_set(groups, instances) for subset_name, instances in instances_by_subset.items()
    }

    # Randomly split the groups in training+validation into 80% for training and 20% for validation.
    training_validation_groups = groups_by_subset["training_validation"]
    random.shuffle(training_validation_groups)
    j = int(TRAINING_FRACTION_SPLIT_BY_INSTANCE_SAME_TRAINING_VALIDATION * len(training_validation_groups))
    groups_by_subset = {
        "training": training_validation_groups[:j],
        "validation": training_validation_groups[j:],
        "test": groups_by_subset["test"],
    }

    # For completeness, copy the instances in training+validation to both training and validation.
    instances_by_subset = {
        "training": instances_by_subset["training_validation"],
        "validation": instances_by_subset["training_validation"],
        "test": instances_by_subset["test"],
    }

    return groups_by_subset, instances_by_subset


def split_groups_by_class(groups, classes, instances_per_class):
    # Randomly split the classes into 60% training, 20% validation and 20% testing.
    classes = random_permutation(classes)
    i = int(TRAINING_FRACTION_SPLIT_BY_CLASS * len(classes))
    j = int((TRAINING_FRACTION_SPLIT_BY_CLASS + VALIDATION_FRACTION_SPLIT_BY_CLASS) * len(classes))
    classes_by_subset = {"training": set(classes[:i]), "validation": set(classes[i:j]), "test": set(classes[j:])}

    # Split instances into training, validation and test according to the class split.
    instances_by_subset = {
        subset_name: set([i for c in classes for i in instances_per_class[c]])
        for subset_name, classes in classes_by_subset.items()
    }

    # Split groups into training, validation and test.
    groups_by_subset = {
        subset_name: _groups_fully_in_set(groups, instances) for subset_name, instances in instances_by_subset.items()
    }

    return groups_by_subset, instances_by_subset


def split_groups_by_class_same_training_validation(groups, classes, instances_per_class):
    # Randomly split the classes into 80% training+validation and 20% testing.
    classes = random_permutation(classes)
    i = int((TRAINING_FRACTION_SPLIT_BY_CLASS + VALIDATION_FRACTION_SPLIT_BY_CLASS) * len(classes))
    classes_by_subset = {"training_validation": set(classes[:i]), "test": set(classes[i:])}

    # Split instances into training, validation and test according to the class split.
    instances_by_subset = {
        subset_name: set([i for c in classes for i in instances_per_class[c]])
        for subset_name, classes in classes_by_subset.items()
    }

    # Split groups into training+validation and test according to the instance split.
    groups_by_subset = {
        subset_name: _groups_fully_in_set(groups, instances) for subset_name, instances in instances_by_subset.items()
    }

    # Randomly split the groups in training+validation into 80% for training and 20% for validation.
    training_validation_groups = groups_by_subset["training_validation"]
    random.shuffle(training_validation_groups)
    j = int(TRAINING_FRACTION_SPLIT_BY_CLASS_SAME_TRAINING_VALIDATION * len(training_validation_groups))
    groups_by_subset = {
        "training": training_validation_groups[:j],
        "validation": training_validation_groups[j:],
        "test": groups_by_subset["test"],
    }

    # For completeness, copy the instances in training+validation to both training and validation.
    instances_by_subset = {
        "training": instances_by_subset["training_validation"],
        "validation": instances_by_subset["training_validation"],
        "test": instances_by_subset["test"],
    }

    return groups_by_subset, instances_by_subset


def split_class_groups_by_class(class_groups, classes):
    # Split based on classes. First, classes are split into training, validation and test (60%, 20%, 20%)
    # and then groups are split into these three subsets according to the class split. Triplets that do not
    # have all classes in the same subset (ie either all training, all validation or all test) are discarded.

    # Randomly split the classes into 60% training, 20% validation and 20% testing.
    classes = random_permutation(classes)
    i = int(TRAINING_FRACTION_SPLIT_BY_CLASS * len(classes))
    j = int((TRAINING_FRACTION_SPLIT_BY_CLASS + VALIDATION_FRACTION_SPLIT_BY_CLASS) * len(classes))
    classes_by_subset = {"training": set(classes[:i]), "validation": set(classes[i:j]), "test": set(classes[j:])}

    # Assign each group to one of training, validation or test if all the classes in the group belong to that subset.
    class_groups_by_subset = {
        subset_name: _groups_fully_in_set(class_groups, classes) for subset_name, classes in classes_by_subset.items()
    }

    return class_groups_by_subset, classes_by_subset


def split_class_groups_by_class_same_training_validation(class_groups, classes):
    # Split based on classes, with the same set for training and validation. First, classes are split into
    # training+validation and test (80%, 20%), then groups are split into these two subsets according to the class
    # split. Finally, the training+validation groups are split randomly into 80% training and 20% validation.

    # Randomly split the classes into 80% training+validation and 20% testing.
    classes = random_permutation(classes)
    i = int((TRAINING_FRACTION_SPLIT_BY_CLASS + VALIDATION_FRACTION_SPLIT_BY_CLASS) * len(classes))
    classes_by_subset = {"training_validation": set(classes[:i]), "test": set(classes[i:])}

    # Split the groups into training+validation and test according to the class split.
    class_groups_by_subset = {
        subset_name: _groups_fully_in_set(class_groups, classes) for subset_name, classes in classes_by_subset.items()
    }

    # Randomly split the groups in training+validation into 80% for training and 20% for validation.
    class_groups_training_validation = class_groups_by_subset["training_validation"]
    random.shuffle(class_groups_training_validation)
    j = int(TRAINING_FRACTION_SPLIT_BY_CLASS_SAME_TRAINING_VALIDATION * len(class_groups_training_validation))
    class_groups_by_subset = {
        "training": class_groups_training_validation[:j],
        "validation": class_groups_training_validation[j:],
        "test": class_groups_by_subset["test"],
    }

    # For completeness, copy the classes in training+validation to both training and validation.
    classes_by_subset = {
        "training": classes_by_subset["training_validation"],
        "validation": classes_by_subset["training_validation"],
        "test": classes_by_subset["test"],
    }

    return class_groups_by_subset, classes_by_subset


def split_class_triplets_things_style(class_triplets, classes, test_classes):
    # Split similar to that described in the "Revealing interpretable object representations..." paper: the
    # training and validation triplets have at most 1 class from a set of 48 test classes. The training and
    # validation follow a 80%-20% split.

    # Partition original triplet set into two subsets, according to the number of test classes in a triplet
    # (<=1 or >=2).
    test_classes = set(test_classes)
    high_overlap_test_triplets, low_overlap_test_triplets = partition(
        lambda t: len(test_classes.intersection(t)) <= 1, class_triplets
    )
    low_overlap_test_triplets = list(low_overlap_test_triplets)
    high_overlap_test_triplets = list(high_overlap_test_triplets)

    # Randomly split the triplets with <=1 test classes into 80% training and 20% validation.
    random.shuffle(low_overlap_test_triplets)
    i = int(TRAINING_FRACTION_SPLIT_THINGS * len(low_overlap_test_triplets))
    class_triplets_by_subset = {
        "training": low_overlap_test_triplets[:i],
        "validation": low_overlap_test_triplets[i:],
        "test": high_overlap_test_triplets,
    }

    # Copy the non-test classes to both training and validation.
    training_validation_classes = set(classes).difference(test_classes)
    classes_by_subset = {
        "training": training_validation_classes,
        "validation": training_validation_classes,
        "test": test_classes,
    }

    return class_triplets_by_subset, classes_by_subset
