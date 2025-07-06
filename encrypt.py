from Crypto import Random
from tqdm import tqdm
import base64
from Crypto.PublicKey import RSA 
from Crypto.Signature import PKCS1_v1_5 as PKCS1_signature
from Crypto.Cipher import PKCS1_v1_5 as PKCS1_cipher
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Hash import SHA256


# 生成RSA密钥对
def generate_key():
    # 创建随机数生成器
    random_generator = Random.new().read
    # 生成2048位的RSA密钥
    rsa = RSA.generate(2048, random_generator)
    # 导出公钥
    public_key = rsa.publickey().exportKey()
    # 导出私钥
    private_key = rsa.exportKey()
    
    # 将私钥保存到文件
    with open('rsa_private_key.pem', 'wb')as f:
        f.write(private_key)
    
    # 将公钥保存到文件
    with open('rsa_public_key.pem', 'wb')as f:
        f.write(public_key)
    

# 从文件读取密钥
def get_key(key_file):
    with open(key_file) as f:
        data = f.read()
        key = RSA.importKey(data)
    return key    

# 数字签名函数
def sign(msg):
    # 读取私钥
    private_key = get_key('rsa_private_key.pem')
    # 创建签名对象
    signer = PKCS1_signature.new(private_key)
    # 创建哈希对象
    digest = SHA256.new()
    # 更新哈希值
    digest.update(bytes(msg.encode("utf8")))
    # 返回签名
    return signer.sign(digest)

# 验证签名函数
def verify(msg, signature):
    # 使用签名，因为RSA加密库默认添加盐值
    # 读取公钥
    pub_key = get_key('rsa_public_key.pem')
    # 创建签名验证对象
    signer = PKCS1_signature.new(pub_key)
    # 创建哈希对象
    digest = SHA256.new()
    # 更新哈希值
    digest.update(bytes(msg.encode("utf8")))
    # 验证签名
    return signer.verify(digest, signature)
    
# 加密数据函数
def encrypt_data(msg): 
    # 读取公钥
    pub_key = get_key('rsa_public_key.pem')
    # 创建加密器
    cipher = PKCS1_OAEP.new(pub_key)
    # 加密数据并进行base64编码
    encrypt_text = base64.b64encode(cipher.encrypt(bytes(msg.encode("utf8"))))
    return encrypt_text.decode('utf-8')

# 解密数据函数
def decrypt_data(encrypt_msg): 
    # 读取私钥
    private_key = get_key('rsa_private_key.pem')
    # 创建解密器
    cipher = PKCS1_OAEP.new(private_key)
    # 解密数据
    back_text = cipher.decrypt(base64.b64decode(encrypt_msg))
    return back_text.decode('utf-8')