import rsa
import sys
import os

def encrypt_submission(file_path):
    # 1. Load the Public Key
    # We assume the key is in the same folder as this script
    key_path = os.path.join(os.path.dirname(__file__), 'public_key.pem')
    
    if not os.path.exists(key_path):
        print(f"❌ Error: public_key.pem not found at {key_path}")
        return

    with open(key_path, 'rb') as f:
        public_key = rsa.PublicKey.load_pkcs1(f.read())

    # 2. Read the submission file
    if not os.path.exists(file_path):
        print(f"❌ Error: Submission file not found at {file_path}")
        return

    with open(file_path, 'rb') as f:
        file_data = f.read()

    print(f"🔒 Encrypting {file_path}...")

    # 3. Encrypt in chunks (RSA can only encrypt small amounts of data at a time)
    encrypted_data = b""
    chunk_size = 245  # Safe limit for 2048-bit keys
    
    try:
        for i in range(0, len(file_data), chunk_size):
            chunk = file_data[i:i+chunk_size]
            encrypted_chunk = rsa.encrypt(chunk, public_key)
            encrypted_data += encrypted_chunk
            
        # 4. Save the .enc file
        output_path = file_path + ".enc"
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
        
        print(f"✅ Success! Created: {output_path}")
        print(f"👉 Please submit ONLY the .enc file to the 'submissions/' folder.")

    except Exception as e:
        print(f"❌ Encryption failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python encrypt.py <path_to_submission.csv>")
    else:
        encrypt_submission(sys.argv[1])
