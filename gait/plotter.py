'''
Made to verify if Gait cycle detection is working properly.
'''
import subprocess
import cv2
import numpy as np

gait_num = 3
x = np.load(f"data/gait_cycles/0/{gait_num}.npy")
for i, frame in enumerate(x):
    blank = np.zeros((1080, 1920, 3), np.uint8)
    for keypoint in frame:
        cv2.circle(blank, (int(keypoint[0]*540), int(keypoint[1]*960)), 3, (255, 255, 255), -1)
    cv2.imwrite("plots/visualize/" + str(i) + ".png", blank)
    cv2.destroyAllWindows()

# create a video of this
img_array = []
for filename in range(x.shape[0]):
    img = cv2.imread("plots/visualize/" + str(filename) + ".png")
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter(f'plots/visualize/gait_cycle_{gait_num}.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, size)
[out.write(frame) for frame in img_array]
out.release()

# clear the folder for the next video
subprocess.run("rm plots/visualize/*.png", shell=True)