#For transparency, this is a program written by ChatGPT according to my prompts.

import os
import numpy as np
from PIL import Image
from PIL import ExifTags
import math
import cv2

# Function to create output folder if it doesn't exist
def create_output_folder(output_folder):
    """
    Create output folder if it doesn't exist.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder '{output_folder}' created.")
    else:
        print(f"Output folder '{output_folder}' already exists.")

# Function to extract EXIF metadata from an image
def extract_exif_metadata(image_path):
    """
    Extract EXIF metadata from the image. Returns as a dictionary.
    """
    image = Image.open(image_path)
    exif_data = image._getexif()

    if not exif_data:
        print("No EXIF metadata found in this image.")
        return None

    # Convert EXIF data to a dictionary with human-readable keys
    metadata = {}
    for tag, value in exif_data.items():
        tag_name = ExifTags.TAGS.get(tag, tag)
        metadata[tag_name] = value

    return metadata

# Function to calculate Shannon entropy for a given block
def calculate_shannon_entropy(block):
    """
    Calculate Shannon entropy for a given block of pixel values.
    """
    values, counts = np.unique(block, return_counts=True)
    probabilities = counts / len(block)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Add small value to avoid log(0)
    return entropy

# Function to calculate edge intensity using Sobel operator (Perceptual Masking)
def calculate_edge_intensity(block):
    """
    Calculate the edge intensity in a block using the Sobel operator.
    """
    block = block.astype(np.float32)
    # Apply Sobel operator to detect edges
    sobel_x = cv2.Sobel(block, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(block, cv2.CV_32F, 0, 1, ksize=3)

    # Calculate the magnitude of the gradient
    magnitude = cv2.magnitude(sobel_x, sobel_y)

    # Calculate the edge intensity (mean gradient magnitude)
    edge_intensity = np.mean(magnitude)
    return edge_intensity

# Function to convert metadata to binary
def convert_metadata_to_bin(metadata):
    """
    Convert metadata string to a binary string.
    """
    return ''.join(format(ord(char), '08b') for char in metadata)

# Function to embed metadata adaptively using 2nd LSB, entropy masking, and perceptual masking (edge detection)
def embed_metadata_adaptive_lsb_with_masking_to_folder(image_path, metadata, output_folder, entropy_threshold=5.0, edge_threshold=10.0):
    """
    Embed metadata adaptively using 2nd LSB, entropy masking, and perceptual masking (edge detection).
    Save the stego image into a specified output folder.
    """
    # Create the output folder if it doesn't exist
    create_output_folder(output_folder)

    # Load the image
    image = Image.open(image_path)
    image = image.convert('RGB')
    pixels = np.array(image)

    # Convert metadata to binary
    metadata_bin = convert_metadata_to_bin(metadata)
    metadata_index = 0
    metadata_length = len(metadata_bin)

    # Block size for entropy and masking calculation
    block_size = 8
    embedding_bit = 2  # Embedding in the 2nd LSB

    # Iterate over the image in blocks
    for i in range(0, pixels.shape[0], block_size):
        for j in range(0, pixels.shape[1], block_size):
            # Extract a block
            block = pixels[i:i+block_size, j:j+block_size, 0]
            flattened_block = block.flatten()

            # Calculate entropy and edge intensity
            entropy = calculate_shannon_entropy(flattened_block)
            edge_intensity = calculate_edge_intensity(block)

            # Embed data only if the block meets the entropy and edge thresholds
            if entropy > entropy_threshold and edge_intensity > edge_threshold:
                for x in range(block.size):
                    if metadata_index >= metadata_length:
                        break
                    # Embed one bit of metadata in the 2nd LSB of each pixel
                    pixel_value = pixels[i + (x // block_size), j + (x % block_size), 0]
                    mask = ~(1 << embedding_bit)
                    new_pixel_value = (pixel_value & mask) | (int(metadata_bin[metadata_index]) << embedding_bit)
                    pixels[i + (x // block_size), j + (x % block_size), 0] = new_pixel_value
                    metadata_index += 1
                if metadata_index >= metadata_length:
                    break
        if metadata_index >= metadata_length:
            break

    # Generate a unique output file name for the stego image
    output_file_name = os.path.join(output_folder, f"stego_image.png")

    # Save the modified image to the folder
    stego_image = Image.fromarray(pixels.astype('uint8'), 'RGB')
    stego_image.save(output_file_name)

    if metadata_index < metadata_length:
        print(f"Warning: Not all metadata was embedded. {metadata_length - metadata_index} bits left.")
    else:
        print(f"Metadata embedded successfully and saved as {output_file_name}")

# Example Usage
if __name__ == "__main__":
    # Input image
    input_image_path = "input_image.png"  # Replace with your image path
    output_folder_path = "output_folder"  # Folder to save stego images

    # Extract EXIF metadata
    extracted_metadata = extract_exif_metadata(input_image_path)
    if extracted_metadata:
        print("Extracted EXIF Metadata:", extracted_metadata)

        # Embed the metadata into the stego image and save to the folder
        embed_metadata_adaptive_lsb_with_masking_to_folder(input_image_path, str(extracted_metadata), output_folder_path)
    else:
        print("No EXIF metadata found in the image.")
