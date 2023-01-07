import os
import time

import h5py
import numpy as np
import psiz


def _load_catalog(file_name):
    h5_file = h5py.File(file_name, "r")
    stimulus_id = h5_file["stimulus_id"][()]
    # pylint: disable=no-member
    stimulus_filepath = h5_file["stimulus_filepath"][:]
    class_id = h5_file["class_id"][()]

    try:
        class_map_class_id = h5_file["class_map_class_id"][()]
        class_map_label = h5_file["class_map_label"][()]
        class_label_dict = {}
        for idx in np.arange(len(class_map_class_id)):
            class_label_dict[class_map_class_id[idx]] = class_map_label[idx].decode("ascii")
    except KeyError:
        class_label_dict = None

    h5_file.close()

    return stimulus_id, stimulus_filepath, class_id, class_label_dict


def _load_triplets(catalog_file_name, trials_file_name):
    stimulus_id, stimulus_filepath, _, _ = _load_catalog(catalog_file_name)

    index_remapping = {}
    for old_index, file_name in zip(stimulus_id, stimulus_filepath):
        file_name = file_name.decode("utf-8")
        i, j = file_name.rfind("_"), file_name.rfind(".")
        new_index = int(file_name[(i + 1) : j]) - 1
        index_remapping[old_index] = new_index

    obs = psiz.trials.load_trials(trials_file_name)

    n = len(obs.stimulus_set)
    triplets = np.zeros((12 * n, 3), dtype=np.int32)

    triplet_index = 0
    for image_indexes, selected_mask in zip(obs.stimulus_set, obs.is_select()):
        selected_mask[0] = True
        indexes_1 = [image_indexes[i] for i, s in enumerate(selected_mask) if (i > 0) and s]
        indexes_0 = [image_indexes[i] for i, s in enumerate(selected_mask) if not s]

        i = image_indexes[0]
        for j in indexes_1:
            for k in indexes_0:
                triplets[triplet_index, :] = [index_remapping[i], index_remapping[j], index_remapping[k]]
                triplet_index += 1

    return triplets


def _load_ninelets(catalog_file_name, trials_file_name):
    stimulus_id, stimulus_filepath, _, _ = _load_catalog(catalog_file_name)

    index_remapping = {}
    for old_index, file_name in zip(stimulus_id, stimulus_filepath):
        file_name = file_name.decode("utf-8")
        i, j = file_name.rfind("_"), file_name.rfind(".")
        new_index = int(file_name[(i + 1) : j]) - 1
        index_remapping[old_index] = new_index

    obs = psiz.trials.load_trials(trials_file_name)

    n = len(obs.stimulus_set)
    ninelets = np.zeros((n, 9), dtype=np.int32)

    for i, (image_indexes, selected_mask) in enumerate(zip(obs.stimulus_set, obs.is_select())):
        canonical_image_indexes = (
            [image_indexes[0]]
            + [image_indexes[j] for j, s in enumerate(selected_mask) if s]
            + [image_indexes[j] for j, s in enumerate(selected_mask) if (j > 0) and (not s)]
        )
        ninelets[i, :] = [index_remapping[j] for j in canonical_image_indexes]

    return ninelets


class IHSJReader:
    """Basic loading functionality for data in the IHSJ dataset."""

    RESOURCE_PATHS_RELATIVE = {
        "image_directory": ("imagenet", "validation"),
        "image_class_ids": ("imagenet", "validation", "imagenet_2012_validation_synset_labels.txt"),
        "class_mapping": ("imagenet", "LOC_synset_mapping.txt"),
        "catalog": ("val", "catalogs", "psiz0.4.1", "catalog.hdf5"),
        "trials": ("val", "obs", "psiz0.4.1", "obs-195.hdf5"),
        "cached_triplets": ("triplets.npy",),
        "cached_ninelets": ("ninelets.npy",),
    }

    def __init__(self, directory_name):
        # Copy directory name to class attribute.
        self._directory_name = directory_name
        self._resource_paths = {
            k: os.path.join(self._directory_name, *v) for k, v in self.RESOURCE_PATHS_RELATIVE.items()
        }

        # Report times for loading operations.
        print("loading IHSJ data...")
        start_time = time.time()

        # Load image and triplet + ninelet data.
        self._load_class_mapping()
        self._load_image_info()
        self._load_triplets_cached()
        self._load_ninelets_cached()

        print("done ({:.2f} s)".format(time.time() - start_time))

    @property
    def image_records(self):
        return self._image_records

    @property
    def num_images(self):
        return len(self._image_records)

    @property
    def class_names(self):
        return self._class_names

    @property
    def num_classes(self):
        return len(self._class_names)

    @property
    def triplets(self):
        return self._triplets

    @property
    def ninelets(self):
        return self._ninelets

    def _load_class_mapping(self):
        # Load the class ids and names.
        with open(self._resource_paths["class_mapping"], "rt") as f:
            class_ids_names = [l.split(" ") for l in f.readlines()]

        self._class_index_by_id = {id_name[0]: index for index, id_name in enumerate(class_ids_names)}
        self._class_names = [" ".join(id_name[1:]) for id_name in class_ids_names]

    def _load_image_info(self):
        """Load the file names and the class ids of the images."""

        # Load the class ids.
        with open(self._resource_paths["image_class_ids"], "rt") as f:
            class_id_per_image = [l.rstrip() for l in f.readlines()]

        # Initialize the image records to an empty list.
        self._image_records = []

        # Make the image records based on the image file names.
        image_directory_name = self._resource_paths["image_directory"]
        for file_name in sorted(os.listdir(image_directory_name)):
            full_file_name = os.path.join(image_directory_name, file_name)
            if os.path.isfile(full_file_name) and full_file_name.endswith(".JPEG"):
                image_index = int(file_name[file_name.rfind("_") + 1 : file_name.rfind(".")]) - 1
                class_id = class_id_per_image[image_index]
                class_index = self._class_index_by_id[class_id]
                self._image_records.append((full_file_name, class_index))

        # Sort the image records according to the file name.
        self._image_records = sorted(self._image_records, key=lambda image_record: image_record[0])

    def _load_triplets_cached(self):
        file_name = self._resource_paths["cached_triplets"]
        if not os.path.exists(file_name):
            self._triplets = _load_triplets(self._resource_paths["catalog"], self._resource_paths["trials"])
            np.save(file_name, self._triplets)

        else:
            self._triplets = np.load(file_name)

        self._triplets = self._triplets.tolist()

    def _load_ninelets_cached(self):
        file_name = self._resource_paths["cached_ninelets"]
        if not os.path.exists(file_name):
            self._ninelets = _load_ninelets(self._resource_paths["catalog"], self._resource_paths["trials"])
            np.save(file_name, self._ninelets)

        else:
            self._ninelets = np.load(file_name)

        self._ninelets = self._ninelets.tolist()
