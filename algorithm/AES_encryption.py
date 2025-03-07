from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

def encrypt_aes(text, key):
    """Encrypt text using AES."""
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(text.encode(), AES.block_size))
    return cipher.iv + ct_bytes  

def decrypt_aes(encrypted_data, key):
    """Decrypt text using AES."""
    iv = encrypted_data[:AES.block_size] 
    ct = encrypted_data[AES.block_size:]  
    cipher = AES.new(key, AES.MODE_CBC, iv)
    pt = unpad(cipher.decrypt(ct), AES.block_size)
    return pt.decode()