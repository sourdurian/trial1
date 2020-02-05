import requests
import json
with open("x_sample.json", "r") as read_file:
    x_sample_loaded = json.load(read_file)
url = 'http://localhost:8080/predict'
r = requests.post(url,json=x_sample_loaded)
print(r.json())
