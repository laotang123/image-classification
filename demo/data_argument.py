# -*- coding: utf-8 -*-
# @Time    : 2019/4/15 11:16
# @Author  : ljf


"""数据增强
   1. 翻转变换 flip
   2. 随机修剪 random crop
   3. 色彩抖动 color jittering
   4. 平移变换 shift
   5. 尺度变换 scale
   6. 对比度变换 contrast
   7. 噪声扰动 noise
   8. 旋转变换/反射变换 Rotation/reflection
   author: XiJun.Gong
   date:2016-11-29
"""

from PIL import Image, ImageEnhance, ImageOps, ImageFile, ImageChops
import numpy as np
import random
import cv2
import os
import time
import logging
import glob
import argparse
import time


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# basic
parser = argparse.ArgumentParser(description="data argumentation of images")
parser.add_argument("--x0", type=int, default=155, help="top left coordinate ")
parser.add_argument("--y0", type=int, default=115, help="top left coordinate ")
parser.add_argument("--w", type=int, default=480, help="width of crop image ")
parser.add_argument("--h", type=int, default=480, help="height of crop image")
parser.add_argument("--excursion", type=int, default=10,
                    help="random excursion of x0 and y0")
parser.add_argument(
    "--path",
    type=str,
    default="./data/classification-8-729",
    help="original image path ")
parser.add_argument(
    "--new_path",
    type=str,
    default="./data/enhance-classification-8-729",
    help="")
parser.add_argument("--max_num", type=int, default=2499,
                    help="the maximum number of images")
parser.add_argument("--min_num", type=int, default=2499,
                    help="the minimum number of images")
parser.add_argument("--img_format", type=str, default="tif",
                    help="the minimum number of images")
parser.add_argument(
    "--offset",
    type=int,
    default=30,
    help="offset of translation")

# extra argumentation
parser.add_argument(
    "--is_crop",
    type=str2bool,
    default="False",
    help="rotate image ")
parser.add_argument(
    "--is_translation",
    type=str2bool,
    default="True",
    help="rotate image ")
parser.add_argument(
    "--is_rotate",
    type=str2bool,
    default="True",
    help="rotate image ")
parser.add_argument(
    "--is_noise",
    type=str2bool,
    default="False",
    help="add noise on image")
parser.add_argument(
    "--is_saturability",
    type=str2bool,
    default="False",
    help="")
parser.add_argument("--is_contrast", type=str2bool, default="False", help="")
parser.add_argument("--is_bright", type=str2bool, default="False", help="")
parser.add_argument("--minimum_images", type=int, default=10, help="")
args = parser.parse_args()

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataAugmentation:
    """
    包含数据增强的八种方式
    """

    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")

    @staticmethod
    def randomRotation(image, mode=Image.BICUBIC):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode)

    @staticmethod
    def randomCrop(image):
        """
        对图像随意剪切,考虑到图像大小范围(68,68),使用一个一个大于(36*36)的窗口进行截图
        :param image: PIL的图像image
        :return: 剪切之后的图像
        """
        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_size = np.random.randint(40, 68)
        random_region = (
            (image_width - crop_win_size) >> 1,
            (image_height - crop_win_size) >> 1,
            (image_width + crop_win_size) >> 1,
            (image_height + crop_win_size) >> 1)
        return image.crop(random_region)

    @staticmethod
    def randomColor(image):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(
            image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(
            color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(
            brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        return ImageEnhance.Sharpness(
            contrast_image).enhance(random_factor)  # 调整图像锐度

    @staticmethod
    def randomGaussian(image, mean=0.2, sigma=0.3):
        """
         对图像进行高斯噪声处理
        :param image:
        :return:
        """

        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        # 将图像转化成数组
        img = np.asarray(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img))

    @staticmethod
    def saveImage(image, path):
        image.save(path)


class DataArgument(object):
    def __init__(self, args):
        self.x0 = args.x0
        self.y0 = args.y0
        self.w = args.w
        self.h = args.h
        self.excursion = args.excursion
        self.offset = args.offset

    def RandomCrop(self, image):
        dx, dy = np.random.randint(-self.excursion, +self.excursion, 2)
        new_img = image.crop(
            [self.x0 + dx, self.y0 + dy, self.x0 + dx + self.w, self.y0 + dy + self.h])
        return new_img

    def ImgeRotate(self, img, angle, mode=Image.BICUBIC):
        """
        mode : Image.NEAREST,Image.BILINEAR,Image.BICUBIC,Image.ANTIALIAS
        """
        return img.rotate(angle, mode)

    def ImageTranslation(self, img):
        xoff, yoff = np.random.randint(0, self.offset, 2)
        width, height = img.size
        offset_img = ImageChops.offset(img, xoff, yoff)
        if np.random.randint(0, 2, 1):
            offset_img.paste((0), (0, 0, xoff, height))
            offset_img.paste((0), (0, 0, width, yoff))
        else:
            offset_img.paste((0), (width - xoff, 0, width, height))
            offset_img.paste((0), (0, 0, width, yoff))
        return offset_img

    def ImageNoise(self):
        pass

    def ImageColor(self):
        pass

    def SaveImage(self, img, save_path):
        img.save(save_path)


def main(args):
    """
    完成主要功能：
    1.限制范围对图像进行上采样和下采样
    2.可以对图像进行随机剪切翻转等多种数据增强手段
    3.图像数量低于阈值则进行删除该类别
    4.在原有的图像基础上增加三倍，如果没有达到最低图像数据，则进行随机平移直到达到最低图像数量
    先保存原图像，然后进行图像增强
    """
    # 生成对应新文件的文件夹
    argument = DataArgument(args)
    # 原始文件夹列表
    old_folder_list = os.listdir(args.path)

    # cn2en = {"点":"dot","粉尘":"dust","划伤":"scratch","良品":"good","丝印":"silk","毛丝":"filament","点粉":"ddust","断胶":"glue","脏污":"smudge"}
    if args.is_crop:
        print("开始进行图像剪切并存储图片")
        time.sleep(2)
    else:
        print("读取图像到新文件夹")
        time.sleep(2)
    for folder in old_folder_list:
        folder_path = os.path.join(args.path, folder)
        assert os.path.isdir(folder_path), "文件夹读取错误"
        print("正在处理{}图像".format(folder))
        new_folder_path = os.path.join(args.new_path, folder)
        img_paths = glob.glob(os.path.join(folder_path, "*.{}".format(args.img_format)))
        img_num = len(img_paths)
        if not os.path.exists(new_folder_path):
            print(new_folder_path)
            os.mkdir(new_folder_path)
        time.sleep(2)
        print("{}图像数量：{}".format(folder, img_num))

        id = 0
        for img_path in img_paths:
            # 进行图像文件重命名和数据增强
            img_name_list = img_path.split("\\")[-1].split(".")
            img = Image.open(img_path, mode="r")
            # print(img_name)
            # img_name = "{}-".format(folder) +"0" * (6 - len(str({}))) + str({}) + ".{}".format(args.img_format)
            save_path = os.path.join(args.new_path, folder, folder+str(id)+"-{}."+img_name_list[1])
            print(save_path)
            # print("正在处理第{}张图片".format(id))
            if args.is_crop:
            # 如果处理的图像是需要剪切的，则需要先剪切存储到新的文件夹
                print("第{}张".format(id))
                img = argument.RandomCrop(img)
                argument.SaveImage(img, save_path.format("crop"))
                id += 1
            else:
                print("第{}张".format(id))
                # 如果处理的图像是不需要剪切的，则先存储所有的原始图像，再对原始图像做处理，然后存储
                argument.SaveImage(img, save_path.format(""))
                id += 1

    # 增强后的文件夹列表
    new_folder_list = os.listdir(args.new_path)
    print("开始从新文件夹中进行图像增强...")
    time.sleep(2)
    for folder in new_folder_list:
        new_folder_path = os.path.join(args.new_path, folder)
        assert os.path.isdir(new_folder_path), "文件夹读取错误"
        print("正在处理{}图像".format(folder))
        time.sleep(3)
        new_folder_path = os.path.join(args.new_path, folder)
        img_paths = glob.glob(os.path.join(new_folder_path, "*.{}".format(args.img_format)))
        img_num = len(img_paths)
        print("{}图像数量：{}".format(folder, img_num))

        id = img_num
        for img_path in img_paths:
            # 进行图像文件重命名和数据增强
            img_name_list = img_path.split("\\")[-1].split(".")
            img = Image.open(img_path, mode="r")
            # img_name = "{}-".format(folder)+"0" * (6 - len(str({}))) + str({}) + ".{}".format(args.img_format)
            save_path = os.path.join(args.new_path, folder, folder+str(id)+"-{}."+img_name_list[1])
            # print("正在处理第{}张图片".format(id))

            if id > args.min_num:
                break
            else:
                if args.is_rotate:
                    for angle in [90,180,270]:
                        if id > args.min_num:
                            break
                        else:
                            print("图像翻转中，已有图像{}张".format(id))
                            rotate_img = argument.ImgeRotate(img, angle)
                            argument.SaveImage(rotate_img, save_path.format("rotate{}".format(angle)))
                            id += 1
        print(id)
        img_paths = glob.glob(os.path.join(new_folder_path, "*.{}".format(args.img_format)))
        #img_num = len(img_paths)
        # 通过上述处理后数量仍达不到要求，则通过随机偏移增加图像数量
        while id < args.max_num:

            # time.sleep(2)
            for img_path in img_paths:
                print("已有图像{}张".format(id))
                img_name_list = img_path.split("\\")[-1].split(".")
                img = Image.open(img_path, mode="r")
                # img_name = "{}-".format(folder)+"0" * (6 - len(str({}))) + str({}) + ".{}".format(args.img_format)
                save_path = os.path.join(args.new_path, folder, folder+str(id)+"-{}."+img_name_list[1])
                if id > args.min_num:
                    break
                else:
                    translation_img = argument.ImageTranslation(img)
                    print(save_path.format("translation",id))
                    print("正在进行图像偏移中...")
                    argument.SaveImage(translation_img,save_path.format("translation"))
                    id +=1
        print("{}图像处理完成".format(folder))
        time.sleep(1)


if __name__ == '__main__':
    main(args)
    # path = "data/binary"
    # folders = os.listdir(path)
    # print(folders)
    # w = []
    # h = []
    # for folder in folders:
    #
    #     img_paths = glob.glob(os.path.join(path,folder, "*.bmp"))
    #     for id ,img_path in enumerate(img_paths):
    #         img = Image.open(img_path)
    #         w.append(img.size[0])
    #         h.append(img.size[1])
    #         resize_img = img.resize((480,480),Image.ANTIALIAS)
    #         resize_img.show()
    #         folder_path = "data/binary-two/{}".format(folder)
    #         new_img_path = folder_path+"/"+str(id)+".tif"
    #         if not os.path.exists(folder_path):
    #             os.mkdir(folder_path)
    #         # print(new_img_path)
    #         # resize_img.save(new_img_path)
    #         print(resize_img.size)
    # print(sum(w)/len(w),sum(h)/len(h))
            # print(img.shape)
            # cv2.imshow("img", img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
    # argument = DataArgument(args)
    # folder_list = os.listdir(args.path)
    #
    # for folder in folder_list:
    #     if folder == "良品":
    #         n = 0
    #         for img_path in glob.glob(os.path.join(args.path,folder,"*.tif")):
    #             if n >= 250:
    #                 break
    #             else:
    #                 # print(img_path
    #                 img_name = img_path.split("\\")[-1]
    #                 img = Image.open(img_path)
    #                 img = argument.RandomCrop(img)
    #                 img.save(os.path.join(args.new_path,folder,img_name))
    #                 n +=1
    #         print("{}图片共{}张".format(folder,n))
    #     elif folder == "点":
    #         n = 0
    #         for img_path in glob.glob(os.path.join(args.path,folder,"*.tif")):
    #             if n >= 100:
    #                 break
    #             else:
    #                 img_name = img_path.split("\\")[-1]
    #                 img = Image.open(img_path)
    #                 img = argument.RandomCrop(img)
    #                 img.save(os.path.join(args.new_path, folder, img_name))
    #                 n += 1
    #         print("{}图片共{}张".format(folder, n))
    #     elif folder == "划伤":
    #         n = 0
    #         for img_path in glob.glob(os.path.join(args.path,folder,"*.tif")):
    #             if n >= 300:
    #                 break
    #             else:
    #                 img_name = img_path.split("\\")[-1]
    #                 img = Image.open(img_path)
    #                 img = argument.RandomCrop(img)
    #                 img.save(os.path.join(args.new_path, folder, img_name))
    #                 n += 1
    #         print("{}图片共{}张".format(folder, n))
    #     elif folder == "丝印不良":
    #         n = 0
    #         for img_path in glob.glob(os.path.join(args.path,folder,"*.tif")):
    #             if n >= 300:
    #                 break
    #             else:
    #                 img_name = img_path.split("\\")[-1]
    #                 img = Image.open(img_path)
    #                 img = argument.RandomCrop(img)
    #                 img.save(os.path.join(args.new_path, "丝印", img_name))
    #                 n += 1
    #         print("{}图片共{}张".format(folder, n))
    #     elif folder == "毛丝":
    #         n = 0
    #         for img_path in glob.glob(os.path.join(args.path,folder,"*.tif")):
    #             if n >= 300:
    #                 break
    #             else:
    #                 img_name = img_path.split("\\")[-1]
    #                 img = Image.open(img_path)
    #                 img = argument.RandomCrop(img)
    #                 img.save(os.path.join(args.new_path, folder, img_name))
    #                 n += 1
    #         print("{}图片共{}张".format(folder, n))
    #     elif folder == "脏污":
    #         n = 0
    #         for img_path in glob.glob(os.path.join(args.path,folder,"*.tif")):
    #             if n >= 100:
    #                 break
    #             else:
    #                 img_name = img_path.split("\\")[-1]
    #                 img = Image.open(img_path)
    #                 img = argument.RandomCrop(img)
    #                 img.save(os.path.join(args.new_path, folder, img_name))
    #                 n += 1
    #         print("{}图片共{}张".format(folder, n))
    # path = "./data/SaveChip/*.tif"
    # for img_path in glob.glob(path):
    #     img_name = img_path.split("\\")[-1]
    #     if img_name[0] == "点":
    #         img = Image.open(img_path)
    #         img.save("./data/classification-7-424/点/{}".format(img_name))
    #     elif img_name[:2] == "丝印":
    #         img = Image.open(img_path)
    #         img.save("./data/classification-7-424/丝印/{}".format(img_name))
    #     elif img_name[:2] == "划伤":
    #         img = Image.open(img_path)
    #         img.save("./data/classification-7-424/划伤/{}".format(img_name))
    #     elif img_name[:2] == "断胶":
    #         img = Image.open(img_path)
    #         img.save("./data/classification-7-424/断胶/{}".format(img_name))
    #     elif img_name[:2] == "毛丝":
    #         img = Image.open(img_path)
    #         img.save("./data/classification-7-424/毛丝/{}".format(img_name))
    #     elif img_name[:2] == "粉尘":
    #         img = Image.open(img_path)
    #         img.save("./data/classification-7-424/粉尘/{}".format(img_name))
    #     elif img_name[:2] == "脏污":
    #         img = Image.open(img_path)
    #         img.save("./data/classification-7-424/脏污/{}".format(img_name))
