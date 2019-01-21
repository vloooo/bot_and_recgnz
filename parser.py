from bs4 import BeautifulSoup
from urllib.request import urlopen
import numpy as np
import cv2

cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

ur =['https://www.autotrader.co.uk/classified/advert/201901113839180?onesearchad=Used&onesearchad=Nearly%20New&onesearchad=New&advertising-location=at_cars&radius=1500&postcode=sr12rh&sort=sponsored&page=3',
     'https://www.autotrader.co.uk/classified/advert/201812053023853?onesearchad=Used&onesearchad=Nearly%20New&onesearchad=New&advertising-location=at_cars&radius=1500&postcode=sr12rh&sort=sponsored&page=3',
     'https://www.autotrader.co.uk/classified/advert/201901083737916?onesearchad=Used&onesearchad=Nearly%20New&onesearchad=New&advertising-location=at_cars&radius=1500&postcode=sr12rh&sort=sponsored&page=3',
     'https://www.autotrader.co.uk/classified/advert/201812263484151?onesearchad=Used&onesearchad=Nearly%20New&onesearchad=New&advertising-location=at_cars&radius=1500&postcode=sr12rh&sort=sponsored&page=3',
     'https://www.autotrader.co.uk/classified/advert/201811122367859?onesearchad=Used&onesearchad=Nearly%20New&onesearchad=New&advertising-location=at_cars&radius=1500&postcode=sr12rh&sort=sponsored&page=3',
     'https://www.autotrader.co.uk/classified/advert/201811072222011?onesearchad=Used&onesearchad=Nearly%20New&onesearchad=New&advertising-location=at_cars&radius=1500&postcode=sr12rh&sort=sponsored&page=3']

# for j in ur:
#     while True:
#         try:
#             url = j
#             counter = 0
#             soup = BeautifulSoup(urlopen(url), "lxml")
#             div = soup.find("div", {"class": "fpaImages__mainImage"})
#             fig = div.find_all("img", {"class": "tracking-standard-link"})
#
#             urls = [fig[0]['src']]
#             for i in range(1, len(fig)):
#                     urls.append(fig[i]['data-src'])
#
#             for i in urls:
#                 resp = urlopen(i)
#                 image_url = np.asarray(bytearray(resp.read()), dtype="uint8")
#                 im_org = cv2.imdecode(image_url, cv2.IMREAD_COLOR)
#                 h, w = im_org.shape[:2]
#                 im_org = im_org[int(h / 100 * 35): h - 20, 40: w - 40]
#                 gray = cv2.cvtColor(im_org, cv2.COLOR_BGR2GRAY)
#                 lower = 0.4
#                 uppper = 0.6
#                 plate_area = [gray.shape[1] * lower, gray.shape[1] * uppper]
#                 plates = cascade.detectMultiScale(gray, scaleFactor=1.1)
#                 plates = [plates for x, y, w, h in plates if plate_area[0] < (x + x + w) / 2 < plate_area[1]]
#                 if len(plates):
#                     # cv2.imshow('ll', im_org)
#                     # cv2.waitKey(0)
#                     counter += 1
#                 # else:
#                     # print(i)
#             print(counter, 'next')
#             break
#         except AttributeError:
#             continue

import requests