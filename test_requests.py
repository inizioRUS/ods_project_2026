import requests

print(requests.post("http://127.0.0.1:8000/v1/search", json={"query": "Что считается крупным ущербом"}).json())


#print(requests.post("http://127.0.0.1:8000/v1/generate", json={"query": "Проверка"}).json())
