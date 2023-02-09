from PIL import Image
import json
import os
import numpy as np
import cv2
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='make new images')
    parser.add_argument('images_path', help='images path')
    parser.add_argument('json_path', help='json path')
    parser.add_argument('dst_path', help='destination path')
    parser.add_argument('version', help='version')
    parser.add_argument('num_patches', help='maximum number of patches')
    parser.add_argument('seed', help='random seed')
    args = parser.parse_args()

    return args


# 사각형 겹침 여부 확인
def overlap(bbox, patch):
    l1, t1, r1, b1 = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
    l2, t2, r2, b2 = patch[0], patch[1], patch[0]+patch[2], patch[1]+patch[3]

    overlap = not (r1 <= l2 or 
                  l1 >= r2 or 
                  b1 <= t2 or 
                  t1 >= b2)
    return overlap # True면 겹침


# 점이 사각형 내부에 있는지 여부 확인
def dot_in_sq(bbox, dot):
    rect_x, rect_y, rect_width, rect_height = bbox[0], bbox[1], bbox[2], bbox[3]
    dot_x, dot_y = dot[0], dot[1]
    if rect_x <= dot_x <= rect_x + rect_width:
        if rect_y <= dot_y <= rect_y + rect_height:
            return True
    return False


# 체크 표시 합성
def mix_check():
    print('mix check marks...')
    json_path = args.json_path
    with open(json_path, 'r') as f:
        annos = json.load(f)

    images_list = os.listdir(args.images_path)

    for path in tqdm(images_list):
        img = cv2.imread(args.images_path + '/' + path, cv2.IMREAD_GRAYSCALE)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                                        param1=50, param2=20, minRadius=12, maxRadius=14) # 이미지에서 원 찾기
        if circles is None:
            img = Image.fromarray(img)
            img.save(args.dst_path + '/' + path)
            continue
        c = []
        for circle in circles[0]:
            c.append([int(circle[0]), int(circle[1])])

        for x in annos['images']:
            if x['file_name'] == path.split('/')[-1]:
                num = x['id']
                break

        bboxes = dict()

        for x in annos['annotations']:
            if x['image_id'] == num and x['category_id'] < 7:
                bboxes[x['id']] = x['bbox']
        

        bboxes_keys = bboxes.keys()
        cc = dict()
        for key in bboxes_keys:
            cc[key] = []

        for x in c:
            for key in bboxes_keys:
                if dot_in_sq(bboxes[key], x):
                    cc[key].append(x)
        for x in cc:
            m, M = 100000, -1
            for xx in cc[x]:
                if m > xx[1]:
                    m = xx[1]
                if M < xx[1]:
                    M = xx[1]
            if M - m < 10:
                cc[x] = sorted(cc[x], key=lambda x: x[0])
            elif M - m < 200:
                cc1, cc2 = [], []
                for xx in cc[x]:
                    if abs(m - xx[1]) < abs(M - xx[1]):
                        cc1.append(xx)
                    else:
                        cc2.append(xx)
                cc1 = sorted(cc1, key=lambda x: x[0])
                cc2 = sorted(cc2, key=lambda x: x[0])
                cc[x] = cc1 + cc2
            else:
                cc[x] = sorted(cc[x], key=lambda x: x[1])
        
        result = Image.new("RGBA", img.shape)

        img = Image.fromarray(img)
        result.paste(img, (0, 0))

        for key in bboxes_keys:
            num = np.random.randint(0, 5)
            if cc[key]:
                if len(cc[key]) % 5 == 0:
                    coordinate = cc[key][num]
                    annos['annotations'][key]['category_id'] = num + 2

                    png_path = '/opt/ml/input/data/hwdata/check/'
                    png_list = os.listdir(png_path)
                    png_im = Image.open(png_path + png_list[np.random.randint(0, len(png_list))])
                    
                    coordinate = [coordinate[0] - png_im.size[0] // 2 + np.random.randint(-10, 10), coordinate[1] - png_im.size[1] + np.random.randint(0, 24)]

                    img.paste(png_im, tuple(coordinate), png_im)
        
        img.save(args.dst_path + '/' + path)
    
    for x in annos['images']:
        x['file_name'] = x['file_name'][:9] + str(args.version) + x['file_name'][10:]
    
    
    with open(args.dst_path + '/' + f'output{args.version}.json', 'w') as f:
        json.dump(annos, f)

# 필기 합성
def mix_patch():
    print('mix patches...')
    images_list = os.listdir(args.dst_path)
    for path in tqdm(images_list):
        if 'json' in path:
            continue
        image = Image.open(args.dst_path + '/' + path)

        json_path = args.json_path
        with open(json_path, 'r') as f:
            annos = json.load(f)

        for x in annos['images']:
            if x['file_name'] == path.split('/')[-1]:
                num = x['id']
                break

        bboxes = []
        for x in annos['annotations']:
            if x['image_id'] == num and x['category_id'] < 7:
                bboxes.append(x['bbox'])

        
        png_path = '/opt/ml/input/data/hwdata/formula/'
        png_list = os.listdir(png_path)
        cnt = 0
        n_patches = np.random.randint(0, int(args.num_patches) + 1)
        while True:
            if cnt == n_patches:
                break
            idx = np.random.randint(0, len(png_list))
            png_im = Image.open(png_path + png_list[idx])

            x = np.random.randint(0, image.size[0] - png_im.size[0])
            y = np.random.randint(0, image.size[1] - png_im.size[1])
            patch = (x, y, png_im.size[0], png_im.size[1])

            tmp = 0
            for bbox in bboxes:
                if overlap(bbox, patch):
                    tmp = 1
                    break
            if tmp == 1:
                continue
            image.paste(png_im, (x, y), png_im)
            cnt += 1

        image = image.convert('RGB')
        image.save(args.dst_path + '/' + path)


if __name__ == '__main__':
    args = parse_args()

    np.random.seed(seed=int(args.seed))
    if not os.path.exists(args.dst_path):
        os.mkdir(args.dst_path)
    mix_check()
    mix_patch()

    # 최종 파일 이름 변경
    images_list = os.listdir(args.dst_path)
    for path in images_list:
        if 'json' in path:
            continue
        new_image_name = path[:9] + str(args.version) + path[10:]
        os.rename(args.dst_path + '/' + path, args.dst_path + '/' + new_image_name)
