import requests
from PIL import Image
from io import BytesIO
response = requests.get('https://imgs.xkcd.com/comics/making_progress.png')
#print(response.json())

with open(r'img.png','wb') as f:
   f.write(response.content)

img = Image.open(BytesIO(response.content))
img.show()