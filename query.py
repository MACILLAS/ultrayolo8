import requests
url = "http://172.17.0.2:5000/predict"


data = {'img': open('bus.jpg', 'rb')} # do this way because YOLO is fast
r = requests.post(url, files=data)
