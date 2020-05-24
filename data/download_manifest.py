import boto3
import json
import os
import requests
import io as file_io
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

s3client = boto3.client('s3')
data, raw_data = [], []
with open('data/snowman_crn_train_20200521.man', 'r') as fd:
  raw_data = fd.readlines()
for d in raw_data:
  pair = json.loads(d)
  data.append(pair)

from shutil import copyfile
for i in range(10):
    d = data[i]
    image_id = d['camera']
    image_path = os.path.join('/home/ec2-user/CRN/hover_data/image', f'{image_id}.jpg')
    mask_path = os.path.join('/home/ec2-user/CRN/hover_data/mask', f'{image_id}.jpg')
    occlusion_path = os.path.join('/home/ec2-user/CRN/hover_data/occlusion', f'{image_id}.npz')
    copyfile(image_path, os.path.join('data/hover_small/image', f'{image_id}.jpg'))
    copyfile(mask_path, os.path.join('data/hover_small/mask', f'{image_id}.jpg'))
    copyfile(occlusion_path, os.path.join('data/hover_small/occlusion', f'{image_id}.jpg'))

if False: #download image
  for d in data:
    image_id = d['camera']
    image_s3 = d['source-ref']
    bucket = 'files.production.hover.to'
    file_name = image_s3.partition(f's3://{bucket}/')[2]
    print(image_id, bucket, file_name)
    s3client.download_file(bucket, file_name, os.path.join('/home/ec2-user/CRN/hover_data/image', f'{image_id}.jpg'))

if False: #download mask
  for d in data:
    image_id = d['camera']
    order_id = d['order']
    if image_id == 313171:
      a = 5
    image_s3 = d['artifact-ref']
    bucket = 'sagemaker-hover'
    file_name = image_s3.partition(f's3://{bucket}/')[2]
    print(image_id, order_id, bucket, file_name)
    s3client.download_file(bucket, file_name, os.path.join('/home/ec2-user/CRN/hover_data/mask', f'{image_id}.jpg'))
