from openai import OpenAI

# Your API credentials
API_KEY = "sk-ws-H.DDYMMRY.Ukej.MEQCIA3kuaZEdq-mUA8ppUuO6V3BgFvFIzxcpJU_cWeW4-rSAiB2wEid9XJSPbwbk7Rjw14p5tdjoH1Djr_TzGCTsr3_Iw"  # Replace with your real key
BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

print("🔄 Testing Qwen API connection...")

try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "Say 'API test successful!' in exactly these words."}],
        temperature=0.0
    )
    
    print("✅ SUCCESS!")
    print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print("❌ FAILED!")
    print(f"Error: {e}")