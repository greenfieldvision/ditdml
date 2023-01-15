"""Report data statistics: number of images, classes, triplets in each subset and triplet coverage for instances."""

import argparse

from ditdml.data_interfaces.things_data_interface import ThingsDataInterface
from ditdml.data_interfaces.ihsj_data_interface import IHSJDataInterface
from ditdml.data_interfaces.ihsjc_data_interface import IHSJCDataInterface
from ditdml.data_interfaces.yummly_data_interface import YummlyDataInterface


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name", help="Name of dataset.", required=True, choices=["things", "ihsj", "ihsjc", "yummly"]
    )
    parser.add_argument("--data-directory-name", help="Root folder for the raw data.", required=True)
    parser.add_argument("--split-type", help="Dataset split type.", required=True)
    parser.add_argument(
        "--class-triplet-conversion-type", help="Class triplet conversion type.", required=False, default=None
    )
    parser.add_argument("--seed", help="Seed for random number generator.", type=int, required=True)
    args = parser.parse_args()

    return args


def make_data_interface(args):
    if args.dataset_name == "things":
        data_interface = ThingsDataInterface(
            args.data_directory_name,
            args.split_type,
            args.seed,
            class_triplet_conversion_type=args.class_triplet_conversion_type,
        )
    elif args.dataset_name == "ihsj":
        data_interface = IHSJDataInterface(args.data_directory_name, args.split_type, args.seed)
    elif args.dataset_name == "ihsjc":
        data_interface = IHSJCDataInterface(
            args.data_directory_name,
            args.split_type,
            args.seed,
            class_triplet_conversion_type=args.class_triplet_conversion_type,
        )
    elif args.dataset_name == "yummly":
        data_interface = YummlyDataInterface(args.data_directory_name, args.split_type, args.seed)
    else:
        data_interface = None

    return data_interface


def report_statistics(data_interface):
    #  Get the reader and the triplets for training, test and validation.
    reader = data_interface.reader
    triplets_by_subset = data_interface.triplets_by_subset

    # Print basic data statistics.
    print()
    print("number of images: {}".format(reader.num_images))
    print("number of classes: {}".format(reader.num_classes))
    print("number of raw triplets: {}".format(len(data_interface.raw_triplets)))
    print(
        "number of triplets by subset: training {} validation {} test {}".format(
            len(triplets_by_subset["training"]), len(triplets_by_subset["validation"]), len(triplets_by_subset["test"])
        )
    )

    # Print detailed data statistics.
    num_triplets_per_instance = data_interface.calculate_num_triplets_per_instance()
    print()
    print(
        "number of triplets per instance: min {} max {} avg {:.2f}".format(
            min(num_triplets_per_instance),
            max(num_triplets_per_instance),
            sum(num_triplets_per_instance) / (len(num_triplets_per_instance) + 1e-9),
        )
    )
    print(
        "number of instances not covered by any triplets: {}".format(sum([i == 0 for i in num_triplets_per_instance]))
    )


if __name__ == "__main__":
    # Parse command line arguments.
    args = parse_arguments()

    # Make the data interface object.
    data_interface = make_data_interface(args)

    # Report the data statistics.
    report_statistics(data_interface)
