from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import time
from imutils.video import FPS, WebcamVideoStream
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='../weights/K_mAP90_TESTpth_AllData.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

THRESHOLD = 0.6

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 60.0, (480,272))

def cv2_demo(net, transform):
    def predict(frame):
        height, width = np.array(frame).shape[:2]

        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        x = x.cuda()

        y = net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= THRESHOLD:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]),
                                                                int(pt[3])), COLORS[i % 3], 2)
                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), FONT,
                            2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        return frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")

    # stream = WebcamVideoStream(src=0).start()  # default camera
    # stream = cv2.VideoCapture('Uber dashcam footage shows lead up to fatal self-driving crash.avi')
    # stream = cv2.VideoCapture('real_test_edit_720p.avi')
    stream = cv2.VideoCapture('daytime_sample.avi')

    print(stream.isOpened())

    time.sleep(1.0)
    # start fps timer
    # loop over frames from the video file stream
    while True:
        # grab next frame
        # frame = stream.read()
        ret, frame = stream.read()

        # print(frame.shape)

        key = cv2.waitKey(1) & 0xFF

        # update FPS counter
        fps.update()

        if ret == False:
            print('None None !!')
            break
        else:
            frame = predict(frame)

        # keybindings for display
        if key == ord('p'):  # pause
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('frame', frame)
                if key2 == ord('p'):  # resume
                    break
        cv2.imshow('frame', frame)
        out.write(frame)

        if key == 27:  # exit
            break

    out.release()
    stream.release()
    cv2.destroyAllWindows()
    return stream


if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, VOC_CLASSES as labelmap
    from ssd import build_ssd

    net = build_ssd('test', 300, 2)    # initialize SSD

    net = net.cuda()

    net.load_state_dict(torch.load(args.weights))
    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

    fps = FPS().start()
    # stop the timer and display FPS information
    stream = cv2_demo(net.eval(), transform)
    fps.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup
    cv2.destroyAllWindows()
