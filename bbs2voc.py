import os, glob
import cv2
# from scipy.io import loadmat
from collections import defaultdict
import numpy as np
from lxml import etree, objectify

# TEST  = ['V003']

TRAIN = False

def bbs_anno2dict(bbs_file, person_types=None):
    """
    Parse caltech bbs annotation file to dict
    Args:
        bbs_file: input bbs file path
        person_types: list of person type that will be used (total 4 types: person, person-fa, person?, people).
            If None, all will be used:
    Return:
        Annotation info dict with filename as key and anno info as value
    """
    frame_name = os.path.splitext(os.path.basename(bbs_file))[0]+'.png'
    annos = defaultdict(dict)
    annos[frame_name] = defaultdict(list)
    annos[frame_name]["id"] = frame_name

    # print(bbs_file)

    f = open(bbs_file, 'r')
    lines = f.readlines()
    f.close


    for i in range(1,len(lines)):
        line = lines[i]
        args = line.split(" ")

        if args[0] == 'person':
            annos[frame_name]["label"] = args[0]
            bbox = [int(args[j]) for j in range(1, 5)]
            diff = 0

            if bbox[3] < 80: # reasonable height data
                diff = 1

        elif args[0] == 'ignore':  # and frame_name.split('_')[1] in TEST:
            # print(frame_name.split('_')[1])
            if TRAIN == False: annos[frame_name]["label"] = 'person'
            if TRAIN == False: bbox = [int(args[j]) for j in range(1, 5)]
            if TRAIN == False: diff = 1

        elif args[0] == 'rear':
            annos[frame_name]["label"] = 'rear'
            bbox = [int(args[j]) for j in range(1, 5)]
            diff = 0

        else:
            continue


        annos[frame_name]["bbox"].append(bbox)
        annos[frame_name]["difficult"].append(diff)


    if len(lines) < 2 and TRAIN:
        print('************** delete anno_1')
        del annos[frame_name]
        # if frame_name.split('_')[1] in TEST:
        #     print("Not delete for test")
        # else:
        #     del annos[frame_name]
    else:
        if len(annos[frame_name]["label"]) and TRAIN:
            print('************** delete anno_2')
            del annos[frame_name]

        else:
            if annos[frame_name]["label"] == 'rear':
                print("********************************************* delete rear!!")
                del annos[frame_name]
            # elif frame_name.split('_')[1] in TEST:
            #     print("None")
            elif annos[frame_name]["label"] == 'person':
                sum = 0
                for value in annos[frame_name]["difficult"]:
                    sum += value

                if len(annos[frame_name]["difficult"]) - sum == 0:
                    # print(annos[frame_name]["difficult"], sum)
                    if TRAIN:
                        del annos[frame_name]

    # print(len(lines), annos)

    return annos

def instance2xml_base(anno, img_size, bbox_type='xyxy'):
    """
    Parse annotation data to VOC XML format
    Args:
        anno: annotation info returned by bbs_anno2dict function
        img_size: camera captured image size
        bbox_type: bbox coordinate record format: xyxy (xmin, ymin, xmax, ymax); xywh (xmin, ymin, width, height)
    Returns:
        Annotation xml info tree
    """
    assert bbox_type in ['xyxy', 'xywh']
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('pos'),
        E.filename(anno['id']),
        E.source(
            E.database('KFTEC Person'),
            E.annotation('KFTEC Person'),
            E.image('KFTEC Person'),
            E.url('None')
        ),
        E.size(
            E.width(img_size[0]),
            E.height(img_size[1]),
            E.depth(3)
        ),
        E.segmented(0),
    )
    for index, bbox in enumerate(anno['bbox']):
        # bbox = [float(x) for x in bbox]
        if bbox_type == 'xyxy':
            xmin, ymin, w, h = bbox
            xmax = xmin+w
            ymax = ymin+h
        else:
            xmin, ymin, xmax, ymax = bbox
        # if h < 50:
        #     diff = 1
        # else:
        #     diff = 0
        E = objectify.ElementMaker(annotate=False)
        anno_tree.append(
            E.object(
            E.name(anno['label']),
            E.bndbox(
                E.xmin(xmin),
                E.ymin(ymin),
                E.xmax(xmax),
                E.ymax(ymax)
            ),
            E.difficult(anno['difficult'][index]),
            E.occlusion(0)
            )
        )
    return anno_tree

def parse_anno_file(bbs_inputdir, bbs_outputdir, person_types=None):
    """
    Parse Caltech data stored in txt files to VOC xml format
    Args:
        bbs_inputdir: bbs file saved pth
        bbs_outputdir: bbs data converted xml file saved path
        person_types: list of person type that will be used (total 4 types: person, person-fa, person?, people).
            If None, all will be used:
    """
    # annotation sub-directories in hda annotation input directory
    assert os.path.exists(bbs_inputdir)
    print("Parsing annotations ...")
    bbs_files = glob.glob(os.path.join(bbs_inputdir, "*.txt")) #glob enable to use a wild char.
    print("finish to read!!")

    for bbs_file in bbs_files:
        # print(bbs_file.split('/')[6][:])

        # # set04 에 해당하는 것들만 변경
        # if bbs_file.split('/')[6][:] != 'set04_V005_I00263.txt':
        #     continue
        # else:
        #     print("found that!")

        annos = bbs_anno2dict(bbs_file, person_types=person_types)
        if annos:
            if not os.path.exists(bbs_outputdir):
                os.makedirs(bbs_outputdir)

            for filename, anno in sorted(annos.items(), key=lambda x: x[0]):
                anno_tree = instance2xml_base(anno, (1280, 720))
                outfile = os.path.join(bbs_outputdir, os.path.splitext(filename)[0]+".xml")
                print("Generating annotation xml file of picture: ", filename)
                etree.ElementTree(anno_tree).write(outfile, pretty_print=True)

def visualize_bbox(xml_file, img_file):
    import cv2
    tree = etree.parse(xml_file)
    # load image
    image = cv2.imread(img_file)
    # get bbox
    for bbox in tree.xpath('//bndbox'):
        coord = []
        for corner in bbox.getchildren():
            coord.append(int(float(corner.text)))
        # draw rectangle
        # coord = [int(x) for x in coord]
        image = cv2.rectangle(image, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255), 2)
    # visualize image
    cv2.imshow("test", image)
    cv2.waitKey(0)


def main():
    if TRAIN:
        bbs_inputdir = "/home/JunYS/data/SKND2/annotations"
    else:
        bbs_inputdir = "/home/JunYS/data/SKND3/annotations"

    if TRAIN:
        bbs_outputdir = "/home/JunYS/data/VOCKftec/Annotations"

        if not os.path.exists('/home/JunYS/data/VOCKftec/Annotations'):
            os.mkdir('/home/JunYS/data/VOCKftec/Annotations')
    else:
        bbs_outputdir = "/home/JunYS/data/VOCKftec/Annotations_TEST"

        if not os.path.exists('/home/JunYS/data/VOCKftec/Annotations_TEST'):
            os.mkdir('/home/JunYS/data/VOCKftec/Annotations_TEST')

    person_types = ["person"]
    parse_anno_file(bbs_inputdir, bbs_outputdir, person_types=person_types)
    # img_file = "/home/JunYS/data/VOCKftec/JPEGIMages/set04_V005_I00014.png"
    if TRAIN:
        img_file = '/nas/MATLAB/01.SKND/exp_script/20180511_skip1_gtTest_testSet/train/images/set04_V005_I00014.png'
        xml_file = "/home/JunYS/data/VOCKftec/Annotations/set04_V005_I00014.xml"
        visualize_bbox(xml_file, img_file)

    print('finished!!')

if __name__ == "__main__":
    main()
