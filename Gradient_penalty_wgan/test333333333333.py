import os
from PIL import Image




current_dir = os.getcwd()
# parent = os.path.dirname(current_dir)
pokemon_dir = os.path.join(current_dir, 'data/')
i = 1
for each in os.listdir(pokemon_dir):
    print(each)
    img = Image.open(pokemon_dir+each)
    img = img.convert("RGB")
    img.save('./jpg_data/'+str(i)+".jpg", format="jpeg")
    i += 1