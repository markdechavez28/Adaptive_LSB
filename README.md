The 2nd LSB with Entropy and Edge Detection Technique for Secure Metadata Embedding is a novel steganographic algorithm that enhances digital image security while preserving image quality.

Problem:
While the LSB technique has been proven effective as a steganographic method, several researchers argue that the metadata it hides can still be detected.

Solution:
Utilize the Shannon Entropy formula to identify regions in the digital image that are highly complex. Evaluate these regions using perceptual masking to determine which are less sensitive to changes. Then, embed the metadata using the second least significant bit (2nd LSB) technique, as the first least significant bit (1st LSB) technique is easier to detect.

Methodology:
Obtain EXIF metadata, as this format contains privacy-related information.
Perform adaptive entropy-based region selection.
Embed metadata using the 2nd LSB technique.
Apply perceptual masking using edge detection.
Embed the metadata.

Limitation:
The algorithm is only applicable to images standardized with EXIF metadata.
