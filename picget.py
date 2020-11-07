import requests
import urllib.request
from bs4 import BeautifulSoup
import os
import time

url  = 'https://www.shutterstock.com/zh/search/'
headers = {'User-Agent': ''}
response = requests.get(url, headers=headers)  # 使用headers避免訪問受限
soup = BeautifulSoup(response.content, 'html.parser')
items = soup.find_all('img')
banana3_path = './photo/'
if os.path.exists(banana3_path) == False:  # 判斷資料夾是否已經存在
    os.makedirs(banana3_path)  # 建立資料夾

for index,item in enumerate(items):
	if item:
		html = requests.get(item.get('src'))   # get函式獲取圖片連結地址，requests傳送訪問請求
		img_name = banana3_path + str(index + 1) +'.png'
		with open(img_name, 'wb') as file:  # 以byte形式將圖片資料寫入
			file.write(html.content)
			file.flush()
		file.close()  # 關閉檔案
		print('第%d張圖片下載完成' %(index+1))
		time.sleep(1)  # 自定義延時
print('抓取完成')