from PIL import Image
import os

dataset_path = "dataset"

bad_images = []

for class_name in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):

        for file in os.listdir(class_path):

            path = os.path.join(class_path, file)

            try:
                img = Image.open(path)
                img.verify()

            except:
                bad_images.append(path)

print("Bad Images:", len(bad_images))