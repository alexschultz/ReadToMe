from PIL import Image
import pytesseract

im = Image.open("textBlock.PNG")
text = pytesseract.image_to_string(im)
print(text)