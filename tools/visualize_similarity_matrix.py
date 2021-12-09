"""GUI showing the similarity matrix for the test classes."""

import argparse
import sys

import numpy as np
import tkinter as tk

from PIL import Image, ImageTk

from ditdml.data_interfaces.things_data_interface import ThingsDataInterface
from ditdml.tools.visualization_utils import draw_text


VISUALIZATION_WIDTH, VISUALIZATION_HEIGHT = 512, 512


class SimilarityMatrixVisualization:
    """Tk-based interactive visualization of similarity matrix of the 48 test classes.

    Shows the similarity matrix and two class images side-by-side, when the user clicks on an element of the matrix.
    """

    def __init__(self, data_interface):
        # Extract the image records, class names, class prototypes and similarity matrix from the data interface object.
        self.image_records = data_interface.reader.image_records
        self.class_names = data_interface.reader.class_names
        self.prototypes_per_class = data_interface.prototypes_per_class
        self.similarity_matrix = data_interface.reader.pairwise_similarity_original_test
        self.classes_similarity_matrix = data_interface.reader.classes_original_test

        # Make the similarity image from the similarity matrix (shades of grey with white similar and black dissimilar).
        self.similarity_image = Image.fromarray((255.0 * self.similarity_matrix).astype(np.uint8))
        self.similarity_image = self.similarity_image.resize((VISUALIZATION_WIDTH, VISUALIZATION_HEIGHT), Image.NEAREST)

        # Set the class indexes and similarity value to invalid.
        self.class_i, self.class_j, self.similarity_ij = None, None, None

        # Set up Tk to a window that accommodates three images side-by-side.
        root = tk.Tk()
        root.title("Similarity Matrix Visualization")
        frame = tk.Frame(root, width=3 * VISUALIZATION_WIDTH, height=VISUALIZATION_HEIGHT)
        frame.pack()
        self.label = tk.Label(frame)

        # Set the click to select a pair of classes and ESC to quit.
        root.bind("<Button-1>", self.click_fn)
        root.bind("<Escape>", self.quit_fn)

        # Visualize and start the Tk loop.
        self.visualize()
        root.mainloop()

    def visualize(self):
        # Make visualization image with three subimages and fill the first with the similarity image.
        visualization_image = Image.new("RGB", (3 * VISUALIZATION_WIDTH, VISUALIZATION_HEIGHT))
        visualization_image.paste(self.similarity_image, (0, 0))

        # Check if the class indexes and similarity values are valid.
        if (self.class_i is not None) and (self.class_j is not None) and (self.similarity_ij is not None):
            for k, c in enumerate([self.class_i, self.class_j]):
                # Load the image of the prototype for this class and resize it.
                record = self.image_records[self.prototypes_per_class[c]]
                image = Image.open(record[0])
                resized_image = image.resize((VISUALIZATION_WIDTH, VISUALIZATION_HEIGHT))

                # Write class name at top left of image.
                draw_text(resized_image, self.class_names[c], (0, 0))

                # Paste on visualization image.
                visualization_image.paste(resized_image, ((k + 1) * VISUALIZATION_WIDTH, 0))

            # Write similarity value in middle of two horizontally concatenated prototype images.
            similarity_str = "{:.2f}".format(self.similarity_ij)
            middle_of_2_and_three = (2 * VISUALIZATION_WIDTH, VISUALIZATION_HEIGHT // 2)
            draw_text(visualization_image, similarity_str, middle_of_2_and_three)

        # Show the visualization image.
        photo_image = ImageTk.PhotoImage(visualization_image)
        self.label.configure(image=photo_image)
        self.label.image = photo_image
        self.label.pack()

    def click_fn(self, event):
        # Make sure the clicked point coordinates are valid.
        if (event.x >= 0) and (event.x < VISUALIZATION_WIDTH) and (event.y >= 0) and (event.y < VISUALIZATION_HEIGHT):

            def relative_index(z, max_z):
                return int((float(z) / max_z) * len(self.classes_similarity_matrix))

            # Scale clicked point to [0,1] then to [0,num_test_classes-1].
            i, j = relative_index(event.y, VISUALIZATION_HEIGHT), relative_index(event.x, VISUALIZATION_WIDTH)

            # Get the actual class indexes and similarity value.
            self.class_i, self.class_j = self.classes_similarity_matrix[i], self.classes_similarity_matrix[j]
            self.similarity_ij = self.similarity_matrix[i, j]

        # Update the visualization in response to the mouse click.
        self.visualize()

    def quit_fn(self, _):
        sys.exit(0)


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory-name", help="Root folder for the raw data.", required=True)
    args = parser.parse_args()

    # Make the data data_interface object (note: no split needed) and start the visualization.
    data_interface = ThingsDataInterface(args.data_directory_name, None, 0)
    SimilarityMatrixVisualization(data_interface)
