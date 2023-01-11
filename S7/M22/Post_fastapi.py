import requests

pload = {'username':'TEst123','password':'Test'}
response = requests.post('http://localhost:8000/login/', params = pload)

print(response)
print(response.json())