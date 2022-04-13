import requests
import json
from pprint import pprint
import cv2

'''
usage curl command instead client request
time curl -X GET -H "Content-type: application/json" -d "{""}" "http://127.0.0.1:5000/get_classes"
Code example for server-client image request (link)
 https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594
'''

def get_classes_test():
    res = requests.get('http://127.0.0.1:5000/get_classes')
    json_data = json.loads(res.text)
    print(json_data)
    if res.ok:
        print(res)

'''Sending image to for classify'''
def load_image_test():
    
    # prepare headers for http request
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}
    # img = cv2.imread('data/coffee_cup1.jpg')
    # img = cv2.imread('data/coffee_cup2.jpg')
    img = cv2.imread('data/restaurant.jpg')

    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', img)
    # send http request with image and receive response
    response = requests.post('http://127.0.0.1:5000/load_image', data=img_encoded.tostring(), headers=headers)
    # decode response
    print(json.loads(response.text))

if __name__ == '__main__':

    # get_classes_test()
    load_image_test()
