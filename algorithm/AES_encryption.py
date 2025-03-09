from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

#Encrypt text using AES.
def encrypt_aes(text, key):
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(text.encode(), AES.block_size))
    return cipher.iv + ct_bytes  

#Decrypt text using AES.
def decrypt_aes(encrypted_data, key):
    iv = encrypted_data[:AES.block_size] 
    ct = encrypted_data[AES.block_size:]  
    cipher = AES.new(key, AES.MODE_CBC, iv)
    pt = unpad(cipher.decrypt(ct), AES.block_size)
    return pt.decode()


    