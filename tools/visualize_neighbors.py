import argparse
import sys

import sklearn.neighbors
import tkinter as tk

from PIL import Image, ImageTk

from ditdml.data_interfaces.things_data_interface import ThingsDataInterface
from ditdml.tools.visualization_utils import draw_text


VISUALIZATION_WIDTH, VISUALIZATION_HEIGHT = 256, 256


class NeighborVisualization:
    """Tk-based interactive visualization of the nearest classes according to the embedding.

    Shows class and neighbors images side-by-side and allows navigating through the classes.
    """

    def __init__(self, data_interface, num_neighbors, initial_class_index):
        # Extract the image records, class names, class embeddings and class prototypes from the data interface object.
        self.image_records = data_interface.reader.image_records
        self.class_names = data_interface.reader.class_names
        self.class_embeddings = data_interface.reader.class_embeddings
        self.prototypes_per_class = data_interface.prototypes_per_class

        # Compute the class neighbors based on the class embeddings. The first neighbor is the original class.
        k_nearest_neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=num_neighbors + 1, n_jobs=-1)
        k_nearest_neighbors.fit(self.class_embeddings)
        self.distances, self.neighbors = k_nearest_neighbors.kneighbors(self.class_embeddings, return_distance=True)

        # Set the number of neighbors and initialize the class index being visualized.
        self.num_neighbors = num_neighbors
        self.class_index = initial_class_index

        # Set up Tk to a window that accommodates num_neighbors + 1 images side-by-side.
        root = tk.Tk()
        root.title("Class Embedding Visualization via Neighbors")
        frame = tk.Frame(root, width=(self.num_neighbors + 1) * VISUALIZATION_WIDTH, height=VISUALIZATION_HEIGHT)
        frame.pack()
        self.label = tk.Label(frame)

        # Set the left/right arrow to decrease/increase the index of the class being visualized and ESC to quit.
        root.bind("<Left>", self.left_fn)
        root.bind("<Right>", self.right_fn)
        root.bind("<Escape>", self.quit_fn)

        # Visualize and start the Tk loop.
        self.visualize()
        root.mainloop()

    def visualize(self):
        # Print class index.
        print("class {} of {}".format(self.class_index + 1, len(self.class_names)))

        # Load the images of the current class and of its neighbors. (`neighbors` includes the original class as the
        # first element)
        neighbors = self.neighbors[self.class_index]
        records = [self.image_records[self.prototypes_per_class[c]] for c in neighbors]
        images = [Image.open(record[0]) for record in records]

        # Get the names and distances of the class neighbors.
        class_names = [self.class_names[c] for c in neighbors]
        distances = self.distances[self.class_index]

        # Make the visualization image by resizing the class and neighbor images and placing them side-by-side.
        visualization_image = Image.new("RGB", ((self.num_neighbors + 1) * VISUALIZATION_WIDTH, VISUALIZATION_HEIGHT))
        for i, image in enumerate(images):
            # Resize image and write class name at top left corner.
            resized_image = image.resize((VISUALIZATION_WIDTH, VISUALIZATION_HEIGHT))
            text = class_names[i] + (" {:.2f}".format(distances[i]) if i > 0 else "")
            draw_text(resized_image, text, (0, 0))

            # Paste on visualization image.
            visualization_image.paste(resized_image, (i * VISUALIZATION_WIDTH, 0))

        # Show the visualization image.
        photo_image = ImageTk.PhotoImage(visualization_image)
        self.label.configure(image=photo_image)
        self.label.image = photo_image
        self.label.pack()

    def left_fn(self, _):
        # Move cursor left if within valid range.
        if self.class_index > 0:
            self.class_index -= 1

        # Update the visualization in response to the keypress.
        self.visualize()

    def right_fn(self, _):
        # Move cursor right if within valid range.
        if self.class_index < len(self.class_names) - 1:
            self.class_index += 1

        # Update the visualization in response to the keypress.
        self.visualize()

    def quit_fn(self, _):
        sys.exit(0)


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory-name", help="Root folder for the raw data.", required=True)
    parser.add_argument("--num-neighbors", help="Number of neighbors.", type=int, required=False, default=8)
    parser.add_argument(
        "--initial-class-index", help="Class whose neighbors to show first.", type=int, required=False, default=1
    )
    args = parser.parse_args()

    # Make the data interface object and start the visualization.
    data_interface = ThingsDataInterface(args.data_directory_name, None, 0)
    NeighborVisualization(data_interface, args.num_neighbors, args.initial_class_index - 1)
