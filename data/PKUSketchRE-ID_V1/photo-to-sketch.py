from PIL import Image
import os


# image = "basketballGround.jpg"
# img = Image.open(image)
# img_all = "素描" + image
# new = Image.new("L", img.size, 255)
# width, height = img.size
# img = img.convert("L")

# 定义画笔的大小
Pen_size = 3
# 色差扩散器
Color_Diff = 6

path = '/omnisky3/public-datasets/ccq/PKUSketchRE-ID_V1/photo/'
for image in os.listdir(path):
    img = Image.open(path+image)
    img_all = '/omnisky3/public-datasets/ccq/PKUSketchRE-ID_V1/photo-to-sketch/' + image
    new = Image.new("L", img.size, 255)
    width, height = img.size
    img = img.convert("L")

    for i in range(Pen_size + 1, width - Pen_size - 1):
        for j in range(Pen_size + 1, height - Pen_size - 1):
            # 原始的颜色
            originalColor = 255
            lcolor = sum([img.getpixel((i - r, j)) for r in range(Pen_size)]) // Pen_size
            rcolor = sum([img.getpixel((i + r, j)) for r in range(Pen_size)]) // Pen_size

            # 通道----颜料
            if abs(lcolor - rcolor) > Color_Diff:
                originalColor -= (255 - img.getpixel((i, j))) // 4
                new.putpixel((i, j), originalColor)

            ucolor = sum([img.getpixel((i, j - r)) for r in range(Pen_size)]) // Pen_size
            dcolor = sum([img.getpixel((i, j + r)) for r in range(Pen_size)]) // Pen_size

            # 通道----颜料
            if abs(ucolor - dcolor) > Color_Diff:
                originalColor -= (255 - img.getpixel((i, j))) // 4
                new.putpixel((i, j), originalColor)

            acolor = sum([img.getpixel((i - r, j - r)) for r in range(Pen_size)]) // Pen_size
            bcolor = sum([img.getpixel((i + r, j + r)) for r in range(Pen_size)]) // Pen_size

            # 通道----颜料
            if abs(acolor - bcolor) > Color_Diff:
                originalColor -= (255 - img.getpixel((i, j))) // 4
                new.putpixel((i, j), originalColor)

            qcolor = sum([img.getpixel((i + r, j - r)) for r in range(Pen_size)]) // Pen_size
            wcolor = sum([img.getpixel((i - r, j + r)) for r in range(Pen_size)]) // Pen_size

            # 通道----颜料
            if abs(qcolor - wcolor) > Color_Diff:
                originalColor -= (255 - img.getpixel((i, j))) // 4
                new.putpixel((i, j), originalColor)

    new.save(img_all)

i = os.system('mshta vbscript createobject("sapi.spvoice").speak("%s")(window.close)' % '您的图片转换好了')
os.system(img_all)
