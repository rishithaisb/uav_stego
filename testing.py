import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def calculate_bpp(image_path, text_path):
    """ Calculate BPP (Bits Per Pixel) based on text file size and image dimensions. """
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image '{image_path}'. Check file path.")
        return None, None

    height, width, _ = image.shape
    total_pixels = height * width

    if not os.path.exists(text_path):
        print(f"❌ Error: Could not find text file '{text_path}'.")
        return None, None

    text_size_bytes = os.path.getsize(text_path)
    print(f"📄 Debug: Text file '{text_path}' size = {text_size_bytes} bytes") 

    text_size_bits = text_size_bytes * 8 
    bpp = text_size_bits / total_pixels

    return bpp, text_size_bytes

def calculate_psnr(img1, img2):
    """ Compute PSNR between two images. """
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    mse = np.mean((img1_gray - img2_gray) ** 2)
    if mse == 0:
        return float('inf'), 0 

    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr, mse

def process_stego_images(image_path, text_files):
    """ Process text files, compute BPP & PSNR, and display results in visualizations. """
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"❌ Error: Could not load original image '{image_path}'.")
        return

    results = [] 

    for i, text_file in enumerate(text_files):
        print(f"📂 Processing: {text_file}...")

        bpp, text_size = calculate_bpp(image_path, text_file)
        if bpp is None:
            continue  

        if i == 0:
            encrypted_image_path = "encrypted_image.png"  
        else:
            encrypted_image_path = f"encrypted_image{i+1}.png"  
        encrypted_image = cv2.imread(encrypted_image_path)
        if encrypted_image is None:
            print(f"❌ Error: Could not load encrypted image '{encrypted_image_path}'.")
            continue

        psnr, mse = calculate_psnr(original_image, encrypted_image)

        results.append({
            "Text Size (KB)": text_size / 1024, 
            "BPP": bpp,
            "MSE": mse,
            "PSNR (dB)": psnr
        })

    text_sizes = [result["Text Size (KB)"] for result in results]
    bpp_values = [result["BPP"] for result in results]
    mse_values = [result["MSE"] for result in results]
    psnr_values = [result["PSNR (dB)"] for result in results]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.bar(range(len(text_sizes)), bpp_values, color='blue', alpha=0.7)
    plt.xticks(range(len(text_sizes)), [f"Text {i+1}" for i in range(len(text_sizes))])
    plt.xlabel("Text File")
    plt.ylabel("BPP (Bits Per Pixel)")
    plt.title("Text Size vs BPP")

    plt.subplot(1, 3, 2)
    plt.bar(range(len(text_sizes)), mse_values, color='green', alpha=0.7)
    plt.xticks(range(len(text_sizes)), [f"Text {i+1}" for i in range(len(text_sizes))])
    plt.xlabel("Text File")
    plt.ylabel("MSE")
    plt.title("Text Size vs MSE")

    plt.subplot(1, 3, 3)
    plt.bar(range(len(text_sizes)), psnr_values, color='red', alpha=0.7)
    plt.xticks(range(len(text_sizes)), [f"Text {i+1}" for i in range(len(text_sizes))])
    plt.xlabel("Text File")
    plt.ylabel("PSNR (dB)")
    plt.title("Text Size vs PSNR")

    plt.tight_layout()
    plt.savefig("results_visualization.png", dpi=300)
    print("✅ Visualization saved as 'results_visualization.png'.")

image_path = "image2.jpg" 
text_files = sorted(glob.glob("text*.txt")) 

process_stego_images(image_path, text_files)