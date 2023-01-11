import requests
response = requests.get(
   'https://api.github.com/search/repositories',
   params={'q': 'requests+language:python'},
)

if response.status_code == 200:
   print('Success!')
elif response.status_code == 404:
   print('Not Found.')
print(type(response.content))

print(response.json())