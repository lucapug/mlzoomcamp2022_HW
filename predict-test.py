import requests


url = 'http://localhost:9696/predict'

client_id = 'xyz-123'
client = {"reports": 0, 
    "share": 0.245, 
    "expenditure": 3.438, 
    "owner": "yes"
}


response = requests.post(url, json=client).json()
print(response)

if response['card_given'] == True:
    print('Credit card given to %s' % client_id)
else:
    print('not emitted card to %s' % client_id)