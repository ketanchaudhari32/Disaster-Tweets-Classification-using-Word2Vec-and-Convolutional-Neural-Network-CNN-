import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'tweet':''})

print(r.json())