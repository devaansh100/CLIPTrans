import requests
from tqdm import tqdm
import sys
import os
import hashlib
import time

NAME = ''
EMAIL = ''

assert len(NAME) > 0 and len(EMAIL) > 0, 'Please set your name and email in the code. This will be used when you call the API for images and will help the hosting service to manage server load.'

with open(f'mmt/{sys.argv[1]}/{sys.argv[2]}_url.txt') as f:
    urls = f.read().splitlines()
save_at = f'mmt/{sys.argv[1]}/images'
os.makedirs(save_at, exist_ok = True)
names = []
ignore_indices = []
for i, url in enumerate(tqdm(urls)):
  while True:
    try:
        img_bytes = requests.get(url, headers = {'User-Agent': f'{NAME}/0.0 ({EMAIL})'})
    except requests.exceptions.ConnectionError as e:
        code = 403 # Max retries hit
        break
    code = img_bytes.status_code
    if code in [429, 502]: # period quota limit, bad gateway
      time.sleep(1)
    else:
      break
  names.append('\n' + str(hashlib.md5(url.encode("utf-8")).hexdigest()) + '.jpg')
  if code != 200:
    print(f'{url} | {code}')
    ignore_indices.append(str(i))
  with open(f'{save_at}/{names[-1][1:]}', 'wb') as f:
    f.write(img_bytes.content)

with open(f'mmt/{sys.argv[1]}/{sys.argv[2]}_image.txt', 'w') as f:
   f.writelines([','.join(ignore_indices)] + names) 
