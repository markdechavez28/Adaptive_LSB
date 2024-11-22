#For transparency, this is a program rewritten by researchers according to lsb2.py.

import cv2
import numpy as np
from PIL import Image
import piexif
import math
import random

# Step 1: EXIF Metadata Extraction
def extract_exif(image_path):
    img = Image.open(image_path)
    exif_data = piexif.load(img.info["exif"])
    metadata = str(exif_data)
    binary_metadata = ''.join(format(ord(char), '08b') for char in metadata)
    return binary_metadata

# Step 2: Adaptive Entropy-based Region Selection
def compute_entropy(block):
    # Calculate the Shannon entropy of an image block
    hist = cv2.calcHist([block], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))  # Add epsilon to avoid log(0)
    return entropy

def select_high_entropy_regions(image, block_size=8, threshold=4.5):
    height, width = image.shape
    selected_regions = []

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y+block_size, x:x+block_size]
            entropy = compute_entropy(block)
            if entropy > threshold:
                selected_regions.append((y, x))
    
    return selected_regions

# Step 3: Second Least Significant Bit (2nd LSB) Embedding
def embed_2nd_lsb(pixel, bit):
    pixel = pixel & 0b11111101  # Clear the second least significant bit
    return pixel | (bit << 1)  # Set the second least significant bit

# Step 4: Perceptual Masking (using edge detection)
def perceptual_masking(image):
    # Apply edge detection using Sobel filter for perceptual masking
    edges = cv2.Sobel(image, cv2.CV_8U, 1, 1, ksize=3)
    return edges

# Step 5: Embedding Metadata
def embed_metadata(image, metadata):
    height, width = image.shape
    metadata_index = 0
    selected_regions = select_high_entropy_regions(image)

    for (y, x) in selected_regions:
        if metadata_index >= len(metadata):
            break
        
        block = image[y:y+8, x:x+8]
        for i in range(8):
            for j in range(8):
                if metadata_index >= len(metadata):
                    break
                pixel = block[i, j]
                bit = int(metadata[metadata_index])
                block[i, j] = embed_2nd_lsb(pixel, bit)
                metadata_index += 1

        image[y:y+8, x:x+8] = block  # Update the block in the image
    
    return image

# Step 6: EXIF Metadata Extraction and Embedding
def extract_and_embed(image_path, output_path):
    # Extract EXIF metadata from the image
    metadata = extract_exif(image_path)
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Perform perceptual masking to get regions where data is less perceptible
    image_with_masking = perceptual_masking(image)
    
    # Embed metadata
    stego_image = embed_metadata(image_with_masking, metadata)
    
    # Save the stego image
    cv2.imwrite(output_path, stego_image)
    print("Stego image saved at:", output_path)

# Example Usage
image_path = "input_image.jpg"
output_path = "stego_image.jpg"
extract_and_embed(image_path, output_path)
