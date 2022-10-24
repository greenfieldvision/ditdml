"""Data statistics: number of images, classes and triplets in each subset."""

import argparse

from ditdml.data_interfaces.things_data_interface import ThingsDataInterface
from ditdml.data_interfaces.ihsj_data_interface import IHSJDataInterface
from ditdml.data_interfaces.ihsjc_data_interface import IHSJCDataInterface
from ditdml.data_interfaces.yummly_data_interface import YummlyDataInterface


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name", help="Name of dataset.", required=True, choices=["things", "ihsj", "ihsjc", "yummly"]
    )
    parser.add_argument("--data-directory-name", help="Root folder for the raw data.", required=True)
    parser.add_argument("--split-type", help="Dataset split type.", required=True)
    parser.add_argument("--seed", help="Seed for random number generator.", type=int, required=True)
    args = parser.parse_args()

    # Make the data interface object.
    if args.dataset_name == "things":
        interface = ThingsDataInterface(args.data_directory_name, args.split_type, args.seed)
    elif args.dataset_name == "ihsj":
        interface = IHSJDataInterface(args.data_directory_name, args.split_type, args.seed)
    elif args.dataset_name == "ihsjc":
        interface = IHSJCDataInterface(args.data_directory_name, args.split_type, args.seed)
    elif args.dataset_name == "yummly":
        interface = YummlyDataInterface(args.data_directory_name, args.split_type, args.seed)
    else:
        interface = None

    #  Get the reader and the triplets for training, test and validation.
    reader = interface.reader
    triplets_by_subset = interface.triplets_by_subset

    # Print data statistics.
    print("number of images: {}".format(reader.num_images))
    print("number of classes: {}".format(reader.num_classes))
    print(
        "number of triplets by subset: training {} validation {} test {}".format(
            len(triplets_by_subset["training"]), len(triplets_by_subset["validation"]), len(triplets_by_subset["test"])
        )
    )
