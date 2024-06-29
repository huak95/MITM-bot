import requests
import json

def get_response(
        system_prompt, 
        user_prompt, 
        model="llama3-8b-typhoon",
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        max_tokens_to_generate=4096,
        base_url="https://kjddazcq2e2wzvzv.snova.ai/api/v1/chat/completion",
        api_key="<api-key>",
        ):
    """
    Model List
    - llama3-8b-typhoon
    - llama3-70b-typhoon
    """

    headers = {
        "Authorization": f"Basic {api_key}",
        "Content-Type": "application/json"
    }

    # bugs: cannot un-stream.
    data = {
        "inputs": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # "max_tokens": 4096,
        "max_tokens_to_generate": max_tokens_to_generate,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "model": model,
        "stop": ["<|eot_id|>", "<|end_of_text|>"],
    }

    response = requests.post(base_url, headers=headers, data=json.dumps(data))

    lines_result = response.text.strip().split("\n")
    text_result = lines_result[-1].replace("data: ", '')
    out = json.loads(text_result)

    return out['completion']
