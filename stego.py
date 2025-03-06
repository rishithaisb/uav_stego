import cv2
import numpy as np
import pickle  
import os

class ArithmeticCoding:
    def __init__(self):
        self.prob_table = {}
        self.sorted_chars = [] 

    def build_probability_table(self, text):
        """Build a probability table for the text."""
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
        total = len(text)
        self.prob_table = {char: freq[char] / total for char in freq}
        self.sorted_chars = sorted(self.prob_table.keys())  

    def encode(self, text):
        """Encode text using arithmetic encoding."""
        self.build_probability_table(text)
        low, high = 0.0, 1.0
        
        for char in text:
            range_size = high - low
            char_index = self.sorted_chars.index(char)
            low = low + range_size * sum(self.prob_table[c] for c in self.sorted_chars[:char_index])
            high = low + range_size * self.prob_table[char]
        
        return (low + high) / 2  

    def decode(self, value, length):
        """Decode value to text using arithmetic decoding."""
        text = ""
        for _ in range(length):
            for char in self.sorted_chars:
                low = sum(self.prob_table[c] for c in self.sorted_chars if c < char)
                high = low + self.prob_table[char]
                if low <= value < high:
                    text += char
                    value = (value - low) / (high - low)
                    break
        return text

def detect_edges(image):
    """Detect edges in the image using Canny edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def float_to_binary(value):
    """Convert a floating-point value to a binary string."""
    packed = np.float64(value).tobytes()
    binary = ''.join(f"{byte:08b}" for byte in packed)
    return binary

def binary_to_float(binary):
    """Convert a binary string back to a floating-point value."""
    if len(binary) != 64:
        raise ValueError("Binary string must be 64 bits long.")
    bytes_list = [int(binary[i:i + 8], 2) for i in range(0, 64, 8)]
    value = np.frombuffer(bytes(bytes_list), dtype=np.float64)[0]
    return value

def embed_text_in_edges(image, edges, text, chunk_size=10):
    """Embed text into the edges of the image using chunk-based arithmetic encoding."""
    ac = ArithmeticCoding()
    edge_pixels = np.argwhere(edges > 0)
    total_pixels = len(edge_pixels)
    
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    if not os.path.exists("chunk_files"):
        os.makedirs("chunk_files")
    
    with open("chunk_files/chunk_info.pkl", "wb") as file:
        pickle.dump((len(chunks), chunk_size, len(text)), file)
    
    for i, chunk in enumerate(chunks):
        encoded_value = ac.encode(chunk)
        binary_value = float_to_binary(encoded_value)
        
        print(f"Chunk {i + 1} Binary Value to Embed:", binary_value) 
        
        with open(f"chunk_files/prob_table_chunk_{i + 1}.pkl", "wb") as file:
            pickle.dump((ac.prob_table, ac.sorted_chars), file)
        
        start = i * 64
        end = start + 64
        
        if end > total_pixels:
            raise ValueError("Text is too large to embed in the image edges.")
        
        for j, bit in enumerate(binary_value):
            x, y = edge_pixels[start + j]
            image[x, y, 0] = (image[x, y, 0] & 0xFE) | int(bit)
    
    return image

def extract_text_from_edges(image, edges):
    """Extract text from the edges of the image using chunk-based arithmetic decoding."""
    edge_pixels = np.argwhere(edges > 0)
    total_pixels = len(edge_pixels)
    
    with open("chunk_files/chunk_info.pkl", "rb") as file:
        num_chunks, chunk_size, text_length = pickle.load(file)
    
    decoded_text = ""
    
    for i in range(num_chunks):
        binary_value = []
        
        start = i * 64
        end = start + 64
        
        if end > total_pixels:
            raise ValueError("Invalid chunk size or edge pixels.")
        
        for j in range(64):
            x, y = edge_pixels[start + j]
            bit = image[x, y, 0] & 1
            binary_value.append(str(bit))
        
        binary_value = ''.join(binary_value)
        print(f"Chunk {i + 1} Binary Value Extracted:", binary_value) 
        
        encoded_value = binary_to_float(binary_value)
        
        with open(f"chunk_files/prob_table_chunk_{i + 1}.pkl", "rb") as file:
            prob_table, sorted_chars = pickle.load(file)
        
        ac = ArithmeticCoding()
        ac.prob_table = prob_table
        ac.sorted_chars = sorted_chars
        decoded_chunk = ac.decode(encoded_value, chunk_size)
        decoded_text += decoded_chunk
    
    decoded_text = decoded_text[:text_length]
    
    print("Decoded Text:", decoded_text)  
    return decoded_text

def main():
    image_path = input("Enter the path to the image file: ")
    if not os.path.exists(image_path):
        print("Image file not found!")
        return

    text_path = input("Enter the path to the text file: ")
    if not os.path.exists(text_path):
        print("Text file not found!")
        return
    
    image = cv2.imread(image_path)
    with open(text_path, "r") as file:
        text = file.read()
    
    print("Input Text:", text) 
    
    edges = detect_edges(image)
    cv2.imwrite("edges.png", edges)
    
    encrypted_image = embed_text_in_edges(image, edges, text, chunk_size=10)
    cv2.imwrite("encrypted_image.png", encrypted_image)
    
    decrypted_text = extract_text_from_edges(encrypted_image, edges)
    
    with open("decrypted_text.txt", "w") as file:
        file.write(decrypted_text)
    
    print("Encryption and decryption completed successfully!")

if __name__ == "__main__":
    main()