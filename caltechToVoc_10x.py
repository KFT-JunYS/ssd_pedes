import os, glob
import cv2
from scipy.io import loadmat
from collections import defaultdict
import numpy as np
from lxml import etree, objectify

CAM_ID = ['set00', 'set01', 'set02', 'set03', 'set04', 'set05', ]

def vbb_anno2dict(vbb_file, cam_id):
    filename = os.path.splitext(os.path.basename(vbb_file))[0]
    annos = defaultdict(dict)
    vbb = loadmat(vbb_file)

    # object info in each frame: id, pos, occlusion, lock, posv
    objLists = vbb['A'][0][0][1][0]
    objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]

    # person index
    person_index_list = np.where(np.array(objLbl) == "person")[0]

    for frame_id, obj in enumerate(objLists):

        if len(obj) > 0 and obj != []:
            frame_name = str(cam_id) + "_" + str(filename) + "_" + str(frame_id+1) + ".jpg"

            annos[frame_name] = defaultdict(list)
            annos[frame_name]["id"] = frame_name
            annos[frame_name]["label"] = "person"

            for id, pos, occl in zip(obj['id'][0], obj['pos'][0], obj['occl'][0]):
                id = int(id[0][0]) - 1

                # occlusion 객체들은 넣지 않음
                if int(occl) == 1:
                    continue

                if not id in person_index_list:  # only use bbox whose label is person
                    continue

                pos = pos[0].tolist()
                occl = int(occl[0][0])

                annos[frame_name]["occlusion"].append(occl)
                annos[frame_name]["bbox"].append(pos)

            if not annos[frame_name]["bbox"]:
                del annos[frame_name]

    return annos


def seq2img(annos, seq_file, outdir, cam_id):
    cap = cv2.VideoCapture(seq_file)
    index = 1
    # captured frame list
    v_id = os.path.splitext(os.path.basename(seq_file))[0]
    cap_frames_index = np.sort([int(os.path.splitext(id)[0].split("_")[2]) for id in annos.keys()])

    if cam_id in CAM_ID:
        cap_frames_index = 3 * cap_frames_index - 1

    print('cap_frames_index = ', cap_frames_index)

    width   = 0
    height  = 0

    while True:
        ret, frame = cap.read()
        if ret:
            if not index in cap_frames_index:
                index += 1
                continue
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            if cam_id in CAM_ID:
                outname = os.path.join(outdir, str(cam_id) + "_" + v_id + "_" + str(int((index + 1) / 3)) + ".jpg")
            else:
                outname = os.path.join(outdir, str(cam_id)+"_"+v_id+"_"+str(index)+".jpg")

            # print ("Current frame: ", v_id, str(index))

            height, width, _ = frame.shape
            cv2.imwrite(outname, frame)

        else:
            break
        index += 1
    img_size = (int(width), int(height))
    return img_size


def instance2xml_base(anno, img_size, bbox_type='xyxy'):
    """bbox_type: xyxy (xmin, ymin, xmax, ymax); xywh (xmin, ymin, width, height)"""
    assert bbox_type in ['xyxy', 'xywh']

    # print(anno)

    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('VOC2014_instance/person'),
        E.filename(anno['id']),
        E.source(
            E.database('Caltech pedestrian'),
            E.annotation('Caltech pedestrian'),
            E.image('Caltech pedestrian'),
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
        bbox = [float(x) for x in bbox]
        if bbox_type == 'xyxy':
            xmin, ymin, w, h = bbox
            xmax = xmin+w
            ymax = ymin+h
        else:
            xmin, ymin, xmax, ymax = bbox

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
            E.difficult(0),
            E.occlusion(anno["occlusion"][index])
            )
        )

    return anno_tree


def parse_anno_file(vbb_inputdir, seq_inputdir, vbb_outputdir, seq_outputdir):
    # annotation sub-directories in hda annotation input directory
    assert os.path.exists(vbb_inputdir)
    sub_dirs = os.listdir(vbb_inputdir)
    for sub_dir in sub_dirs:
        print ("Parsing annotations of camera: ", sub_dir)
        cam_id = sub_dir
        vbb_files = glob.glob(os.path.join(vbb_inputdir, sub_dir, "*.vbb"))
        for vbb_file in vbb_files:
            annos = vbb_anno2dict(vbb_file, cam_id)
            if annos:
                vbb_outdir = os.path.join(vbb_outputdir, "annotations", sub_dir, "bbox")

                # extract frames from seq
                seq_file = os.path.join(seq_inputdir, sub_dir, os.path.splitext(os.path.basename(vbb_file))[0]+".seq")
                seq_outdir = os.path.join(seq_outputdir, sub_dir, "frame")

                if not os.path.exists(vbb_outdir):
                    os.makedirs(vbb_outdir)
                if not os.path.exists(seq_outdir):
                    os.makedirs(seq_outdir)

                img_size = seq2img(annos, seq_file, seq_outdir, cam_id)

                for filename, anno in sorted(annos.items(), key=lambda x: x[0]):
                    if "bbox" in anno:
                        anno_tree = instance2xml_base(anno, img_size)

                        outfile = os.path.join(vbb_outdir, os.path.splitext(filename)[0]+".xml")
                        print ("Generating annotation xml file of picture: ", filename)
                        etree.ElementTree(anno_tree).write(outfile, pretty_print=True)


def main():
    seq_inputdir = "/home/JunYS/data/Caltech_pedestrian"
    # vbb_inputdir = "/home/JunYS/data/Caltech_pedestrian/annotations"
    vbb_inputdir = "/home/JunYS/data/Caltech_pedestrian/annotations_correct"
    seq_outputdir = "/home/JunYS/data/Caltech_pedestrian/VOC_type"
    vbb_outputdir = "/home/JunYS/data/Caltech_pedestrian/VOC_type"
    parse_anno_file(vbb_inputdir, seq_inputdir, vbb_outputdir, seq_outputdir)

    print("Finished!!")

if __name__ == "__main__":
    main()
