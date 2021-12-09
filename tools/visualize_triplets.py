import argparse
import sys

import tkinter as tk

from PIL import Image, ImageTk

from ditdml.data_interfaces.things_data_interface import ThingsDataInterface
from ditdml.tools.visualization_utils import draw_text


VISUALIZATION_WIDTH, VISUALIZATION_HEIGHT = 512, 512


class TripletVisualization:
    """Tk-based interactive visualization of labeled triplets.

    Shows triplet images side-by-side and allows navigating through the triplets in a given subset.
    """

    def __init__(self, data_interface, subset_name, initial_triplet_index):
        # Extract the image records, class names and triplets from the data interface object.
        self.image_records = data_interface.reader.image_records
        self.class_names = data_interface.reader.class_names
        self.triplets = data_interface.triplets_by_subset[subset_name]

        # Initialize the triplet index being visualized.
        self.triplet_index = initial_triplet_index

        # Set up Tk to a window that accommodates three images side-by-side.
        root = tk.Tk()
        root.title("Labeled Triplet Visualization (3=odd one out)")
        frame = tk.Frame(root, width=3 * VISUALIZATION_WIDTH, height=VISUALIZATION_HEIGHT)
        frame.pack()
        self.label = tk.Label(frame)

        # Set the left/right arrow to decrease/increase the index of the triplet being visualized and ESC to quit.
        root.bind("<Left>", self.left_fn)
        root.bind("<Right>", self.right_fn)
        root.bind("<Escape>", self.quit_fn)

        # Visualize and start the Tk loop.
        self.visualize()
        root.mainloop()

    def visualize(self):
        # Print triplet index.
        print("triplet {} of {}".format(self.triplet_index + 1, len(self.triplets)))

        # Load the images of the current triplet.
        triplet = self.triplets[self.triplet_index]
        records = [self.image_records[image_index] for image_index in triplet]
        images = [Image.open(record[0]) for record in records]
        class_names = [self.class_names[record[1]] for record in records]

        # Make the visualization image by resizing the three images and placing them side-by-side.
        visualization_image = Image.new("RGB", (3 * VISUALIZATION_WIDTH, VISUALIZATION_HEIGHT))
        for i, image in enumerate(images):
            # Resize image and write class name at top left corner.
            resized_image = image.resize((VISUALIZATION_WIDTH, VISUALIZATION_HEIGHT))
            draw_text(resized_image, class_names[i], (0, 0))

            # Paste on visualization image.
            visualization_image.paste(resized_image, (i * VISUALIZATION_WIDTH, 0))

        # Show the visualization image.
        photo_image = ImageTk.PhotoImage(visualization_image)
        self.label.configure(image=photo_image)
        self.label.image = photo_image
        self.label.pack()

    def left_fn(self, _):
        # Move cursor left if within valid range.
        if self.triplet_index > 0:
            self.triplet_index -= 1

        # Update the visualization in response to the keypress.
        self.visualize()

    def right_fn(self, _):
        # Move cursor right if within valid range.
        if self.triplet_index < len(self.triplets) - 1:
            self.triplet_index += 1

        # Update the visualization in response to the keypress.
        self.visualize()

    def quit_fn(self, _):
        sys.exit(0)


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory-name", help="Root folder for the raw data.", required=True)
    parser.add_argument("--split-type", help="Dataset split type.", required=True)
    parser.add_argument("--seed", help="Seed for random number generator.", type=int, required=True)
    parser.add_argument("--subset-name", help="Subset to visualize.", required=False, default="training")
    parser.add_argument("--initial-triplet-index", help="Triplet to show first.", type=int, required=False, default=1)
    args = parser.parse_args()

    # Make the data interface object and start the visualization.
    data_interface = ThingsDataInterface(args.data_directory_name, args.split_type, args.seed)
    TripletVisualization(data_interface, args.subset_name, args.initial_triplet_index - 1)
