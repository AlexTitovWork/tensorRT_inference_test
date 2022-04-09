import requests
import json
from pprint import pprint

if __name__ == '__main__':

    res = requests.post(
        'http://127.0.0.1:5000/get_classes', data)
    if res.ok:
        print(res)
