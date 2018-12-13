import glob
import cv2
import os

dir = '/Users/glennjocher/downloads/app/screenshots/'  # directory of screenshots to resize


def resize_pad(img, height=416, width=416, color=(255, 255, 255)):  # resizes a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio_h, ratio_w = float(height) / shape[0], float(width) / shape[1]
    ratio = min(ratio_h, ratio_w)

    new_shape = [round(shape[0] * ratio), round(shape[1] * ratio)]
    dw = width - new_shape[1]  # width padding
    dh = height - new_shape[0]  # height padding
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def main():
    images = glob.glob(dir + '/*.*')

    formats = ['5_5', '5_8', '12_9']  # (inches) iPhone 8, iPhone XS, iPad Pro 12.9
    size_x = [1242, 1125, 2048]
    size_y = [2208, 2436, 2732]

    os.system('rm -rf ' + dir + 'output')
    os.system('mkdir ' + dir + 'output')
    for image in images:
        img = cv2.imread(image)
        print(image)

        for i, format in enumerate(formats):
            img_resized = resize_pad(img, height=size_y[i], width=size_x[i])
            cv2.imwrite(dir + 'output/' + format + '_' + image.split('/')[-1], img_resized)


if __name__ == '__main__':
    main()
