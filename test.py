import requests

files = {
'file': open('samples/q1.wav', 'rb'),
'language': (None, 'bengali'),
'vtt': (None, 'true'),
}

response = requests.post('https://asr.iitm.ac.in/internal/asr/decode', files=files)
print(response.json())
