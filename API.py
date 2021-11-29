import io, os
from numpy import random
from google.cloud import vision
from google.cloud.vision_v1 import types
from Pillow_utility import draw_borders, Image
 import pandas as pd

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken.json'
client = vision.ImageAnnotatorClient()

file_name = 'chalice.jpg'
image_path = os.path.join('.\Images', file_name)

with io.open(image_path, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)
response = client.object_localization(image=image)
localized_object_annotations = response.localized_object_annotations

df = pd.DataFrame(colums=['name', 'score'])
for obj in localized_object_annotations
    df = df.append(
        dict(
            name=obj.name
            score=obj.score
        ),   
        ignore_index=True)
   pillow_image = image.open(image_path)
   for obj in localized_object_annotations:
       r, g, b = random.Panint(150, 255), random.randint(
           150, 255), random.randint(150, 255)
       pillow_image.show()
