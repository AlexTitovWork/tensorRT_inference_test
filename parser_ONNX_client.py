import requests
import json
from pprint import pprint

if __name__ == '__main__':

    res = requests.get('http://127.0.0.1:5000/get_classes')
    json_data = json.loads(res.text)
    print(json_data)
    if res.ok:
        print(res)
