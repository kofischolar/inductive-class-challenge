import rsa
import sys
import os
import glob

# 1. Get Private Key from GitHub Secrets
private_key_data = os.environ.get("PRIVATE_KEY")
if not private_key_data:
    print("❌ Error: PRIVATE_KEY secret is missing!")
    sys.exit(1)

private_key = rsa.PrivateKey.load_pkcs1(private_key_data.encode('utf8'))

# 2. Find the .enc file in submissions/
enc_files = glob.glob('submissions/*.enc')
if not enc_files:
    print("ℹ️ No .enc file found. Skipping decryption.")
    sys.exit(0)

enc_file_path = enc_files[0] # Take the first one found
print(f"🔓 Decrypting: {enc_file_path}")

# 3. Decrypt
with open(enc_file_path, 'rb') as f:
    encrypted_data = f.read()

decrypted_data = b""
chunk_size = 256 # Standard block size for 2048-bit key decryption

try:
    for i in range(0, len(encrypted_data), chunk_size):
        chunk = encrypted_data[i:i+chunk_size]
        decrypted_chunk = rsa.decrypt(chunk, private_key)
        decrypted_data += decrypted_chunk

    # 4. Save as 'submission.csv' for the scoring script to find
    # We save it in the root or wherever your scoring script expects it
    output_path = "submissions/submission.csv" 
    with open(output_path, 'wb') as f:
        f.write(decrypted_data)
    
    print(f"✅ Decrypted to: {output_path}")

except Exception as e:
    print(f"❌ Decryption failed! {e}")
    sys.exit(1)
