# Image Steganography for UAV using Edge-based Technique: Arithmetic Encoding

## Overview
This project implements an edge-based image steganography technique using arithmetic encoding for UAV images.

## Process
1. **User Input:** Receives text and an image from the user.
2. **Key Generation:** Generates an AES encryption key.
3. **Text Encryption:** Encrypts the input text using AES encryption.
4. **Edge Detection:** Applies Canny Edge Detection to identify edge pixels in the image.
5. **Text Embedding:**
   - Compresses the encrypted text using Arithmetic Encoding, obtaining a single floating-point value.
   - Embeds the compressed text into the LSB of the detected edge pixels.
6. **Text Extraction:**
   - Extracts the embedded floating-point value from the LSB of edge pixels.
   - Decompresses it using Arithmetic Decoding to retrieve the encrypted text.
7. **Text Decryption:** Decrypts the extracted text using the AES key.

## Prerequisites
- Python 3.7 or above
- Python virtual environment (optional but recommended)

## Setup
### 1. Clone the repository:
```bash
git clone git@github.com:rishithaisb/uav_stego.git
cd uav_stego
```
### 2. Install dependencies:
```bash
pip install opencv-python numpy pycryptodome
```
### 3. Run the file:
```bash
python stego.py
```
### 4. Input Text File & Image*
\* your image file can be either in jpg/png.

