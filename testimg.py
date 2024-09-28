#用于可视化图像增强的效果
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from ChannelAug import ChannelRandomErasing,ChannelExchange
from clustercontrast.utils.data import transforms as T
import config as cfg


# Load an image
img = Image.open('/data2/liuweiqi/home/project1/data/PKUSketchRE-ID_V1/rgb_modify/1/bounding_box_test/2_c1_2_05_355.jpg')  # Replace 'path/to/your/image.jpg' with your image path

color_aug = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
height, width = 224, 224
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# 定义完整的变换流程
# train_transformer_rgb = transforms.Compose([
#     color_aug,
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ToTensor(),
#     normalizer,
#     ChannelRandomErasing(probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]),
#     ChannelExchange(gray=2)
# ])

train_transformer_rgb = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.Grayscale(num_output_channels=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),

        ])

img_transformed = train_transformer_rgb(img)

# 将变换后的Tensor转回PIL Image以便显示
img_transformed= transforms.ToPILImage()(img_transformed)

# Transform the image
# transform = ChannelRandomErasing(probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465])
# img_tensor = transforms.ToTensor()(img)
# img_tensor = transform(img_tensor)
# img_transformed = transforms.ToPILImage()(img_tensor)

# Plot the images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(img_transformed)
ax[1].set_title('Transformed Image')
ax[1].axis('off')

# 保存图片
img_transformed.save('./transformed_image.jpg') 

plt.show()

# import cv2
# import numpy as np

# def extract_high_frequency(image_path):
#     # 读取图像
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     if img is None:
#         print("Image not found")
#         return
    
#     # 转换为float32类型
#     img = np.float32(img)
    
#     # 定义拉普拉斯算子核
#     kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    
#     # 应用拉普拉斯算子
#     laplacian = cv2.filter2D(img, -1, kernel)
    
#     # 将负值设置为0
#     laplacian = np.clip(laplacian, 0, 255)
    
#     # 转换回uint8类型
#     laplacian = np.uint8(laplacian)
    
#     # 显示原图和处理后的图像
#     cv2.imshow('Original Image', img)
#     cv2.imshow('High Frequency Image', laplacian)
    
#     # 等待按键后关闭窗口
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # 使用函数
# extract_high_frequency('/data2/liuweiqi/home/project1/data/PKUSketchRE-ID_V1/rgb_modify/1/bounding_box_test/2_c1_2_05_355.jpg')