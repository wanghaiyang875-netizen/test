import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

base_url = os.getenv('GPT_URL')
api_key = os.getenv('GPT_API_KEY')

print(f"1. base_url = {base_url}")
print(f"2. api_key  = {api_key[:8]}...{api_key[-4:] if api_key else 'None'}")

# 拼接完整 URL（注意：有些中转站的 base_url 已经带 /v1，有些没有，需要判断）
if base_url.rstrip('/').endswith('/v1'):
    full_url = f"{base_url.rstrip('/')}/chat/completions"
else:
    full_url = f"{base_url.rstrip('/')}/v1/chat/completions"

print(f"3. 实际请求 URL = {full_url}\n")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": "gpt-5.6-sol",
    "messages": [{"role": "user", "content": "你好"}]
}

try:
    response = requests.post(full_url, headers=headers, json=data, timeout=30)
    print(f"4. HTTP 状态码: {response.status_code}")
    print(f"5. Content-Type: {response.headers.get('Content-Type')}")
    print("---- 服务器返回内容（前 800 字符） ----")
    print(response.text[:800])
except Exception as e:
    print(f"请求异常: {e}")