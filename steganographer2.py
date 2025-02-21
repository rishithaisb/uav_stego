import cv2  # for reading and writing image data
import heapq  # for heap data structure
import os  # to work with files
import random  # for randomisation
from algos.AES import *
from algos.arithmetic_encoding import ArithmeticEncoder, ArithmeticDecoder


compTextRef = ""
side0Mode = 0
side1Mode = 0
side2Mode = 0
side3Mode = 0
embeddedBits = 0

retrievedBits = 0
retrievedTextRef = ""
retrievedTextLength = 0
lengthRetrieved = False


def pad_encoded_text(encoded_text):
    extra_padding = 120 - len(encoded_text) % 128
    for i1 in range(extra_padding):
        encoded_text += "0"

    padded_info = "{0:08b}".format(extra_padding)
    encoded_text = padded_info + encoded_text
    return encoded_text


def get_byte_array(padded_encoded_text):
    if len(padded_encoded_text) % 8 != 0:
        print("Error occurred while padding")
        exit(0)
    b = bytearray()
    for i1 in range(0, len(padded_encoded_text), 8):
        byte1 = padded_encoded_text[i1:i1 + 8]
        b.append(int(byte1, 2))
    return b


def remove_padding(padded_encoded_text):
    padded_info = padded_encoded_text[:8]
    extra_padding = int(padded_info, 2)
    padded_encoded_text = padded_encoded_text[8:]
    encoded_text = padded_encoded_text[:-1 * extra_padding]

    return encoded_text


def compress_with_arithmetic(input_path):
    global compTextRef
    filename, _ = os.path.splitext(input_path)
    output_path = filename + ".bin"

    with open(input_path, 'r', encoding="utf-8") as file1, open(output_path, 'wb') as output:
        text = file1.read().strip()
        print("\n🔹 Original Input Text:", text)

        encoder = ArithmeticEncoder()
        encoded_value, symbol_probs = encoder.encode(text)

        # Convert encoded float to binary string
        encoded_binary = format(int(encoded_value * (2**64)), '064b')
        print("\n🔹 Encoded Binary:", encoded_binary)

        compTextRef = encoded_binary
        b = get_byte_array(encoded_binary)
        output.write(bytes(b))

        with open(filename + "_probs.txt", 'w', encoding="utf-8") as prob_file:
            for symbol, prob in symbol_probs.items():
                prob_file.write(f"{symbol}:{prob}\n")

    return output_path




def decompress_with_arithmetic(input_path):
    filename, _ = os.path.splitext(input_path)
    output_path = filename + "_decompressed.txt"

    with open(input_path, 'rb') as file1, open(output_path, 'w', encoding="utf-8", errors="replace") as output:
        bit_string = ""
        byte1 = file1.read(1)
        while len(byte1) > 0:
            byte1 = ord(byte1)
            bits1 = bin(byte1)[2:].rjust(8, '0')
            bit_string += bits1
            byte1 = file1.read(1)

        encoded_text = remove_padding(bit_string)

        # Convert back to float
        encoded_value = int(encoded_text, 2) / (2**64)
        print("\n🔹 Retrieved Encoded Value:", encoded_value)

        # Load symbol probabilities
        symbol_probs = {}
        with open(filename + "_probs.txt", 'r', encoding="utf-8", errors="replace") as prob_file:
            for line in prob_file:
                if ':' in line:
                    symbol, prob = line.strip().split(':')
                    symbol_probs[symbol] = float(prob)

        print("\n🔹 Loaded Symbol Probabilities:", symbol_probs)

        # Ensure correct text length
        with open(input_path, 'r', encoding="utf-8", errors="replace") as text_file:
            text_length = len(text_file.read().strip())
        print("\n🔹 Expected Text Length:", text_length)

        # Decode
        decoder = ArithmeticDecoder()
        decoded_text = decoder.decode(encoded_value, text_length, symbol_probs)
        print("\n🔹 Decoded Text:", decoded_text)

        output.write(decoded_text)

    return output_path





def randomiseSides():
    n = []
    count1 = 0
    keepLooping = True
    sideData = []
    while keepLooping:  # randomise 3 sides
        temp1 = random.randint(0, 3)
        if temp1 not in n:
            n.append(temp1)
            count1 += 1
        if count1 == 3:
            keepLooping = False
    for i1 in n:
        mode = random.randint(0, 1)
        if i1 == 0:
            sideData.append([0, 0, mode])
        elif i1 == 1:
            sideData.append([0, 1, mode])
        elif i1 == 2:
            sideData.append([1, 0, mode])
        else:
            sideData.append([1, 1, mode])
    return sideData


def getBinary(x1):
    s = bin(x1)[2:]
    while len(s) < 8:
        s = '0' + s
    return s


def getDecimal(s):
    a = (int(s[0], 2), int(s[1], 2), int(s[2], 2))
    return a


def changePixel(row, col, data):
    global img
    (b, g, r) = img[row][col]
    s = [getBinary(b), getBinary(g), getBinary(r)]
    for i1 in range(3):
        temp1 = list(s[i1])
        temp1[7] = str(data[i1])
        s[i1] = "".join(temp1)
    img[row][col] = getDecimal(s)


def assignSideInfo(currRow, sideData):
    global rows, columns
    changePixel(currRow, currRow, sideData[0])
    changePixel(currRow, columns - currRow - 1, sideData[1])
    changePixel(rows - currRow - 1, columns - currRow - 1, sideData[2])


def assignMode(side, mode):
    global side0Mode, side1Mode, side2Mode, side3Mode
    if side == 0:
        side0Mode = mode
    elif side == 1:
        side1Mode = mode
    elif side == 2:
        side2Mode = mode
    else:
        side3Mode = mode


def side0(pixel, channel, data, currRow):
    global columns, img
    tRow = currRow
    if side0Mode == 0:
        tCol = currRow + 1 + pixel
    else:
        tCol = columns - currRow - 2 - pixel
    temp1 = img[tRow][tCol][channel] % 2
    if temp1 != int(data):
        if temp1 == 0:
            img[tRow][tCol][channel] += 1
        else:
            img[tRow][tCol][channel] -= 1


def side1(pixel, channel, data, currRow):
    global rows, columns, img
    tCol = columns - currRow - 1
    if side1Mode == 0:
        tRow = currRow + 1 + pixel
    else:
        tRow = rows - currRow - 2 - pixel
    temp1 = img[tRow][tCol][channel] % 2
    if temp1 != int(data):
        if temp1 == 0:
            img[tRow][tCol][channel] += 1
        else:
            img[tRow][tCol][channel] -= 1


def side2(pixel, channel, data, currRow):
    global rows, columns, img
    tRow = rows - currRow - 1
    if side2Mode == 0:
        tCol = columns - currRow - 2 - pixel
    else:
        tCol = currRow + 1 + pixel
    temp1 = img[tRow][tCol][channel] % 2
    if temp1 != int(data):
        if temp1 == 0:
            img[tRow][tCol][channel] += 1
        else:
            img[tRow][tCol][channel] -= 1


def side3(pixel, channel, data, currRow):
    global rows, img
    tCol = currRow
    if side3Mode == 0:
        tRow = rows - currRow - 2 - pixel
    else:
        tRow = currRow + 1 + pixel
    temp1 = img[tRow][tCol][channel] % 2
    if temp1 != int(data):
        if temp1 == 0:
            img[tRow][tCol][channel] += 1
        else:
            img[tRow][tCol][channel] -= 1


def embedEdge(currRow):
    global compTextRef, embeddedBits, rows, img

    sides = []
    bitsThisIter = 0

    (b, g, r) = img[currRow, currRow]
    s = [getBinary(b), getBinary(g), getBinary(r)]
    side = int(s[0][7] + s[1][7], 2)
    sides.append(side)
    assignMode(side, int(s[2][7]))

    (b, g, r) = img[currRow, columns - currRow - 1]
    s = [getBinary(b), getBinary(g), getBinary(r)]
    side = int(s[0][7] + s[1][7], 2)
    sides.append(side)
    assignMode(side, int(s[2][7]))

    (b, g, r) = img[rows - currRow - 1, columns - currRow - 1]
    s = [getBinary(b), getBinary(g), getBinary(r)]
    side = int(s[0][7] + s[1][7], 2)
    sides.append(side)
    assignMode(side, int(s[2][7]))

    for i1 in range(4):
        if i1 not in sides:
            sides.append(i1)
            break

    print("Sides embedding order:", sides)
    print("Modes of side 0 to 3:", side0Mode, side1Mode, side2Mode, side3Mode)

    spaceAvailable = (rows - (currRow + 1) * 2) * 3 * 4
    while True:
        if spaceAvailable < 64:
            if spaceAvailable >= len(compTextRef) - embeddedBits:
                n = spaceAvailable // 4
            else:
                return True
        else:
            n = 16
        for i1 in range(n):
            for j1 in range(4):
                if embeddedBits < len(compTextRef):
                    temp1 = bitsThisIter // 4
                    if sides[j1] == 0:
                        side0(temp1 // 3, temp1 % 3, compTextRef[embeddedBits], currRow)
                    elif sides[j1] == 1:
                        side1(temp1 // 3, temp1 % 3, compTextRef[embeddedBits], currRow)
                    elif sides[j1] == 2:
                        side2(temp1 // 3, temp1 % 3, compTextRef[embeddedBits], currRow)
                    else:
                        side3(temp1 // 3, temp1 % 3, compTextRef[embeddedBits], currRow)
                    embeddedBits += 1
                    bitsThisIter += 1
                    spaceAvailable -= 1
                else:
                    return False


def embed():
    currRow = 0
    keepLooping = True
    while keepLooping:
        sideData = randomiseSides()
        assignSideInfo(currRow, sideData)
        keepLooping = embedEdge(currRow)
        if embeddedBits >= len(compTextRef):
            keepLooping = False
        currRow += 1
        print('Edge', currRow, 'filled')
    print('Bits embedded:', embeddedBits, 'bits')


def getSide(a, b):
    if a == 0:
        if b == 0:
            return 0
        else:
            return 1
    else:
        if b == 0:
            return 2
        else:
            return 3


def getSideData(currRow):
    global img, side0Mode, side1Mode, side2Mode, side3Mode, columns, rows
    sides = []

    (b, g, r) = img[currRow][currRow]
    side = getSide(b % 2, g % 2)
    sides.append(side)
    assignMode(side, r % 2)
    (b, g, r) = img[currRow][columns - currRow - 1]
    side = getSide(b % 2, g % 2)
    sides.append(side)
    assignMode(side, r % 2)
    (b, g, r) = img[rows - currRow - 1][columns - currRow - 1]
    side = getSide(b % 2, g % 2)
    sides.append(side)
    assignMode(side, r % 2)

    for i1 in range(4):
        if i1 not in sides:
            sides.append(i1)
            assignMode(i1, 0)

    return sides
def clamp(value, min_value, max_value):
    """ Ensure the value stays within the given min and max range. """
    return max(min_value, min(value, max_value))

def getSide0(currRow, pixel, channel):
    global img, columns
    tRow = currRow
    tCol = currRow + pixel + 1 if side0Mode == 0 else columns - currRow - 2 - pixel

    # Clamp tCol to stay within valid range
    tCol = clamp(tCol, 0, columns - 1)

    return img[tRow][tCol][channel] % 2

def getSide1(currRow, pixel, channel):
    global img, columns, rows
    tCol = columns - currRow - 1
    tRow = currRow + 1 + pixel if side1Mode == 0 else rows - currRow - 2 - pixel

    # Clamp tRow to stay within valid range
    tRow = clamp(tRow, 0, rows - 1)

    return img[tRow][tCol][channel] % 2

def getSide2(currRow, pixel, channel):
    global img, rows, columns
    tRow = rows - currRow - 1
    tCol = columns - currRow - 2 - pixel if side2Mode == 0 else currRow + 1 + pixel

    # Clamp tCol to stay within valid range
    tCol = clamp(tCol, 0, columns - 1)

    return img[tRow][tCol][channel] % 2

def getSide3(currRow, pixel, channel):
    global img, rows
    tCol = currRow
    tRow = rows - currRow - 2 - pixel if side3Mode == 0 else currRow + 1 + pixel

    # Clamp tRow to stay within valid range
    tRow = clamp(tRow, 0, rows - 1)

    return img[tRow][tCol][channel] % 2


def retrieveData(currRow, sides):
    global img, side0Mode, side1Mode, side2Mode, side3Mode,\
        retrievedTextRef, lengthRetrieved, retrievedTextLength, retrievedBits

    totalBits = (rows - (currRow + 1) * 2) * 3 * 4
    bitsThisIter = 0

    while True:
        for i1 in range(4):
            for j1 in range(4):
                temp1 = bitsThisIter // 4
                row_idx = temp1 // 3
                col_idx = temp1 % 3

                # Ensure valid indices before retrieval
                if 0 <= row_idx < rows and 0 <= col_idx < columns:
                    if sides[j1] == 0:
                        retrievedTextRef += str(getSide0(currRow, row_idx, col_idx))
                    elif sides[j1] == 1:
                        retrievedTextRef += str(getSide1(currRow, row_idx, col_idx))
                    elif sides[j1] == 2:
                        retrievedTextRef += str(getSide2(currRow, row_idx, col_idx))
                    else:
                        retrievedTextRef += str(getSide3(currRow, row_idx, col_idx))
                else:
                    print(f"⚠️ Skipping invalid retrieval at temp1={temp1}, row_idx={row_idx}, col_idx={col_idx}")

                retrievedBits += 1
                bitsThisIter += 1
                totalBits -= 1

                if retrievedBits == 16 and retrievedTextLength == 0:
                    retrievedTextLength = int(retrievedTextRef, 2)
                    retrievedTextRef = ""

                if retrievedTextLength != 0 and retrievedBits - 16 == retrievedTextLength:
                    return False
                if totalBits == 0:
                    return True



def retrieve():
    global img
    currRow = 0
    maxRows = min(rows // 2, rows - 1)  # Ensure within valid range
    keepLooping = True
    while keepLooping and currRow < maxRows:
        sides = getSideData(currRow)
        print("Sides retrieval order:", sides)
        print("Modes of side 0 to 3:", side0Mode, side1Mode, side2Mode, side3Mode)
        keepLooping = retrieveData(currRow, sides)
        currRow += 1
        print('Edge', currRow, 'retrieved')
    print("Retrieved bits:", retrievedBits, "bits")



print("Image Steganography Using an Edge Based Embedding Technique: Arithmetic Encoding.")
ipPath = input("\nEnter the name of file containing input text: ")
imPath = input("Enter the name of file containing the image: ")
img = cv2.imread(imPath, 1)
if img is None:
    print(f"❌ Error: Could not load image '{imPath}'. Check file path.")
    exit(1)
print("\nIn image:\nRows:", len(img), "\nColumns:", len(img[0]))


print("\nStep 1: Text compression using Arithmetic Encoding")
output_path = compress_with_arithmetic(ipPath)
print("Compressed file path: " + output_path)
print("Compressed text in binary: " + compTextRef)
print("Bits in compressed text:", len(compTextRef), "bits")
print("Text compression completed")


print("\nStep 2: Encrypting using AES")
encrypted = ""
plainValues = []
with open(output_path, 'rb') as file:
    encryptionIp = ""
    byte = file.read(1)
    while len(byte) > 0:
        byte = ord(byte)
        bits = bin(byte)[2:].rjust(8, '0')
        encryptionIp += bits
        byte = file.read(1)
file.close()
len1 = len(encryptionIp) / 4
encryptionIpCopy = encryptionIp
encryptionIp = hex(int(encryptionIp, 2))[2:].zfill(int(len1))
for j in range(int(len(encryptionIp) / 32)):
    input_plain = encryptionIp[j * 32:(j + 1) * 32]  # Extract 32-character segment
    input_plain = input_plain.rstrip().replace(" ", "").lower()  # Clean and format
    plain = split_string(8, input_plain)  # Split into 8-character segments
    plain = [split_string(2, word) for word in plain]  # Further split each segment into 2-character pairs
    plainValues.append(plain)
    
    rounds = [apply_round_key(keys[0], plain)]
    for i in range(1, 11):  # Loop from 1 to 10
        rounds.append(aes_round(rounds[i - 1], keys[i]))

    paddingLen = len("".join(int_to_hex(flatten(rounds[10])))) * 4  # Fix misplaced parenthesis
    temp = bin(int("".join(int_to_hex(flatten(rounds[10]))), 16))[2:].zfill(paddingLen)  # Correct parenthesis alignment
    
    encrypted += temp

print("After encryption (in binary): " + encrypted)
print("Encryption completed")

print("\nStep 3: Embedding data into image")
rows, columns, _ = img.shape
compTextRef = encrypted
compTextLen = bin(len(compTextRef))
compTextLen = compTextLen[2:]
while len(compTextLen) < 16:
    compTextLen = '0' + compTextLen
compTextRef = compTextLen + compTextRef
print("Size of data to be embedded after padding 16-bit length:", len(compTextRef), "bits")
embed()
cv2.imwrite('encrypted.PNG', img)
cv2.imshow('encrypted image', img)
print("Data embedded successfully")


print("\nStep 4: Retrieving data from image")
img = cv2.imread('encrypted.PNG', 1)
if img is None:
    print("❌ Error: Could not reload the encrypted image.")
    exit(1)
rows, columns, _ = img.shape
retrieve()
print("Bits in retrieved text after removing 16-bit padding:", len(retrievedTextRef), "bits")
print("Retrieved text: " + retrievedTextRef)
print("Successfully retrieved")


print("\nStep 5: Decrypting retrieved data")
decrypted = ""
len1 = len(retrievedTextRef) / 4
retrievedTextRef = hex(int(retrievedTextRef, 2))[2:].zfill(int(len1))
for j in range(int(len(retrievedTextRef) / 32)):
    if j < len(plainValues):
        plain = plainValues[j]
else:
    print(f"❌ Error: Out of range access in plainValues at index {j}")
    exit(1)

    for i in range(10, 0, -1):
        rounds.append(inverse_aes_round(rounds[i - 1], keys[i]))
    paddingLen = len("{}".format(''.join(flatten(plainValues[j])))) * 4
    temp = bin(int("{}".format(''.join(flatten(plainValues[j]))), 16))[2:].zfill(paddingLen)
    decrypted += temp
print("Number of bits in decrypted data:", len(decrypted), "bits")
print("Decrypted data: " + decrypted)
print("Decryption completed")


print("\nStep 6: Decompression of data")
print("Decompressed file path: " + decompress_with_arithmetic(output_path))
print("Decompression completed")
cv2.waitKey(0)