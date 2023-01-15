import csv
import json
import os


def _load_triplets(file_name):
    # Load the UUID triplets.
    uuid_triplets = []
    with open(file_name, "rt") as f:
        for row in csv.reader(f, delimiter=";"):
            t = [c.replace(" ", "").replace("images/", "").replace(".jpg", "") for c in row]
            uuid_triplets.append(t)

    return uuid_triplets


def _load_names(file_name):
    """Load the names from the ingredients file."""

    with open(file_name, "rt") as f:
        ingredients = json.load(f)

    names_by_uuid = {
        file_name[:-4]: culinary_information["name"] if culinary_information is not None else file_name[:-4]
        for file_name, culinary_information in ingredients.items()
    }
    return names_by_uuid


class YummlyReader:
    """Basic loading functionality for data in the Yummly dataset."""

    RESOURCE_PATHS_RELATIVE = {
        "image_directory": ("images",),
        "triplets": ("all-triplets.csv",),
        "ingredients": ("ingredients.json",),
    }

    def __init__(self, directory_name):
        self._resource_paths = {k: os.path.join(directory_name, *v) for k, v in self.RESOURCE_PATHS_RELATIVE.items()}

        self._load_image_info()
        self._load_non_image_info()

    @property
    def image_records(self):
        return self._image_records

    @property
    def num_images(self):
        return len(self._image_records)

    @property
    def triplets(self):
        return self._triplets

    @property
    def num_classes(self):
        return None

    @property
    def class_names(self):
        return self._class_names

    def _load_image_info(self):
        self._image_records = []

        directory_name = self._resource_paths["image_directory"]
        for file_name in os.listdir(directory_name):
            full_file_name = os.path.join(directory_name, file_name)

            if os.path.isfile(full_file_name) and full_file_name.endswith(".jpg"):
                self._image_records.append((full_file_name,))

        # Sort the image records according to the file name and assign a class to each image.
        self._image_records = sorted(self._image_records, key=lambda image_record: image_record[0])
        self._image_records = [(r[0], i) for i, r in enumerate(self._image_records)]

    def _load_non_image_info(self):
        # Get the mapping from image index to UUID.
        u2i = {r[0][(r[0].rfind("/") + 1) : (r[0].find("jpg") - 1)]: i for i, r in enumerate(self._image_records)}

        # Read the UUID triplets and convert them into index triplets.
        uuid_triplets = _load_triplets(self._resource_paths["triplets"])
        self._triplets = [[u2i[t[0]], u2i[t[1]], u2i[t[2]]] for t in uuid_triplets]

        # Read the name associated with each UUID.
        names_by_uuid = _load_names(self._resource_paths["ingredients"])

        # Store the names as class names in the order of the UUIDs.
        self._class_names = [0] * len(self._image_records)
        for uuid, name in names_by_uuid.items():
            self._class_names[u2i[uuid]] = name
