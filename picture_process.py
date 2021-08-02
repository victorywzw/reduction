import PIL.Image as Image
import os
 
image_load_path = './pic/all_species_frac20210724_124722/'  # 图片集地址
image_save_path = './pic/all_species_frac20210724_124722/final.png'  # 图片转换后的地址
image_format = ['.png']  # 图片格式
image_size   = 512       # 每张小图片的大小
row = 7     # 图片间隔，也就是合并成一张图后，一共有几行
column = 9  # 图片间隔，也就是合并成一张图后，一共有几列
 
# # 获取图片集地址下的所有图片名称
# image_names = [name for name in os.listdir(image_load_path) for item in image_format if
#                os.path.splitext(name)[1] == item]
arg1 = [0.10, 0.22, 0.46, 1.00, 2.15, 4.64, 10.00]
arg2 = [1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600]
image_names = []
for p in arg1:
    for T in arg2:
        image_name = '%sT_%sp.png' % ('%.2f'%T, '%.2f'%p)
        image_names.append(image_name)


# 简单的对于参数的设定和实际图片集的大小进行数量判断
if len(image_names) != row * column:
    raise ValueError("合成图片的参数和要求的数量不能匹配！")
 
# 定义图像拼接函数
def image_compose():
    to_image = Image.new('RGB', (column * image_size, row * image_size)) # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(row):
        for x in range(column):
            from_image = Image.open(image_load_path + image_names[column * y + x]).resize(
                (image_size, image_size), Image.ANTIALIAS)
            to_image.paste(from_image, (x * image_size, y * image_size))
    return to_image.save(image_save_path) # 保存新图

image_compose()