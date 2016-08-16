from PIL import Image

im = Image.open("test.jpg").load()

print(im)