import os
import tempfile
import unittest

import numpy as np
import scipy.io

from things_reader import ThingsReader


class ThingsReaderTest(unittest.TestCase):
    def test_toy_data(self):
        with tempfile.TemporaryDirectory() as data_directory_name:
            # Create toy version of THINGS dataset.
            os.makedirs(os.path.join(data_directory_name, "Main"))

            with open(os.path.join(data_directory_name, "Main", "things_concepts.tsv"), "wt") as f:
                f.write("Word\tuniqueID\n")
                f.write("animal\tanimal\n")
                f.write("place\tplace1\n")
                f.write("plant\tplant1\n")
                f.write("tool\ttool\n")

            subdirectory_name = os.path.join(data_directory_name, "Main", "animal")
            os.makedirs(subdirectory_name)
            open(os.path.join(subdirectory_name, "animal_001.jpg"), "wt")
            open(os.path.join(subdirectory_name, "animal_002.jpg"), "wt")

            subdirectory_name = os.path.join(data_directory_name, "Main", "place1")
            os.makedirs(subdirectory_name)
            open(os.path.join(subdirectory_name, "place_001.jpg"), "wt")
            open(os.path.join(subdirectory_name, "place_002.jpg"), "wt")
            open(os.path.join(subdirectory_name, "place_003.jpg"), "wt")

            subdirectory_name = os.path.join(data_directory_name, "Main", "plant1")
            os.makedirs(subdirectory_name)
            open(os.path.join(subdirectory_name, "plant_a.jpg"), "wt")
            open(os.path.join(subdirectory_name, "plant_b.jpg"), "wt")

            subdirectory_name = os.path.join(data_directory_name, "Main", "tool")
            os.makedirs(subdirectory_name)
            open(os.path.join(subdirectory_name, "tool_004.jpg"), "wt")

            subdirectory_name = os.path.join(data_directory_name, "Revealing", "triplets")
            os.makedirs(subdirectory_name)
            with open(os.path.join(subdirectory_name, "data1854_baseline_train90.txt"), "wt") as f:
                f.write("0 1 2\n0 1 3\n")
            with open(os.path.join(subdirectory_name, "data1854_baseline_test10.txt"), "wt") as f:
                f.write("1 2 3\n")

            subdirectory_name = os.path.join(data_directory_name, "Revealing", "data")
            os.makedirs(subdirectory_name)
            scipy.io.savemat(
                os.path.join(subdirectory_name, "RDM48_triplet.mat"), {"RDM48_triplet": [[0.0, 0.75], [0.75, 0.0]]}
            )
            with open(os.path.join(subdirectory_name, "spose_embedding_49d_sorted.txt"), "wt") as f:
                embeddings = [
                    [0.023, 0.224, 2.634, 0.003, 0.005],
                    [1.216, 0.212, 0.030, 0.086, 0.936],
                    [0.415, 0.155, 0.122, 0.189, 0.339],
                    [0.705, 0.843, 0.149, 0.067, 0.497],
                ]
                for e in embeddings:
                    f.write(" ".join([str(x) for x in e]) + "\n")

            subdirectory_name = os.path.join(data_directory_name, "Revealing", "variables")
            os.makedirs(subdirectory_name)
            scipy.io.savemat(os.path.join(subdirectory_name, "sortind.mat"), {"sortind": [1, 3, 2, 4]})
            scipy.io.savemat(
                os.path.join(subdirectory_name, "words48.mat"),
                {"words48": [[[np.array("animal")]], [[np.array("plant1")]]]},
            )

            # Make reader object and do sanity checks on its data.
            reader = ThingsReader(data_directory_name)

            self.assertEqual(reader.num_images, 8)
            self.assertEqual(reader.num_classes, 4)

            self.assertEqual(len(reader.class_triplets), 3)

            s = reader.pairwise_similarity_original_test.shape
            self.assertEqual(s[0], 2)
            self.assertEqual(s[1], 2)
            self.assertEqual(len(reader.classes_original_test), 2)

            s = reader.class_embeddings.shape
            self.assertEqual(s[0], 4)
            self.assertEqual(s[1], 5)
