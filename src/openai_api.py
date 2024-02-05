import openai
openai.api_key = "<your_openai_key_here>"
import time, sys
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff



@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(60))
def openai_completion(prompt, engine="gpt-3.5-turbo", max_tokens=700, temperature=0, api_key=None):
    if api_key is not None:
        openai.api_key = api_key
    resp =  openai.ChatCompletion.create(
        model=engine,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        timeout=10,
        request_timeout=10,
        temperature=temperature,
        stop=["\n\n", "<|endoftext|>"],
        )
    
    return resp['choices'][0]['message']['content']