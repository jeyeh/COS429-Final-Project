import os, cv2

def get_sorted_img_names(folder_path):
    image_list = []
    for img in os.listdir(folder_path):
        if not img.endswith('.jpg'):
            continue
        image_list.append(img)

    image_list.sort()
    return image_list

def rename_imgs(path, image_list, name_prefix, extension):
    for i, img in enumerate(image_list):
        if not img.endswith(extension):
            continue
        num = f"{i:03d}"
        os.rename(path + r'/' + img, path + r'/' + name_prefix + '_' + num + extension)
    
    
def get_image_shapes(path, image_list):
    shapes = []
    for img in image_list:
        p = path + r'/' + img
        i = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        shapes.append(i.shape)

    return shapes

def get_images(path, image_list):
    imgs = []

    for img in image_list:
        p = path + r'/' + img
        i = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        imgs.append(i)

    return get_images