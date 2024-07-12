import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def perform_bilinear_interpolation(input_x, input_y, input_image, output_x, output_y):
    output_shape = (output_y.size, output_x.size, input_image.shape[2])
    output_image = np.zeros(output_shape, dtype=input_image.dtype)

    for i, x in enumerate(output_x):
        x_index = np.searchsorted(input_x, x)
        x_index = max(1, x_index)
        x1, x2 = input_x[x_index - 1], input_x[x_index]

        for j, y in enumerate(output_y):
            y_index = np.searchsorted(input_y, y)
            y_index = max(1, y_index)
            y1, y2 = input_y[y_index - 1], input_y[y_index]

            for channel in range(input_image.shape[2]):
                f11 = input_image[y_index - 1, x_index - 1, channel]
                f21 = input_image[y_index - 1, x_index, channel]
                f12 = input_image[y_index, x_index - 1, channel]
                f22 = input_image[y_index, x_index, channel]

                interpolated_value = (
                    f11 * (x2 - x) * (y2 - y) +
                    f21 * (x - x1) * (y2 - y) +
                    f12 * (x2 - x) * (y - y1) +
                    f22 * (x - x1) * (y - y1)
                ) / ((x2 - x1) * (y2 - y1))

                output_image[j, i, channel] = interpolated_value

    return output_image

def resize_image(image_path, target_width=1000):
    original_image = np.array(Image.open(image_path))
    height, width = original_image.shape[:2]

    input_x = np.linspace(0, width - 1, width)
    input_y = np.linspace(0, height - 1, height)

    target_height = int((target_width / width) * height)
    output_x = np.linspace(0, width - 1, target_width)
    output_y = np.linspace(0, height - 1, target_height)

    resized_image = perform_bilinear_interpolation(input_x, input_y, original_image, output_x, output_y)
    return original_image, resized_image.astype(np.uint8)

def display_images(original_image, resized_image):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    ax1.imshow(original_image)
    ax1.set_title("Original Image")

    ax2.imshow(resized_image)
    ax2.set_title("Resized Image (Bilinear Interpolation)")

    plt.tight_layout()
    plt.show()

def main():
    image_filename = r"C:\Users\OWNER\Downloads\capybara.jpg"
    script_dir = os.path.dirname(__file__)
    image_path = os.path.join(script_dir, image_filename)

    if not os.path.exists(image_path):
        print(f"File {image_path} not found.")
        return

    original_image, resized_image = resize_image(image_path)
    display_images(original_image, resized_image)

if __name__ == "__main__":
    main()