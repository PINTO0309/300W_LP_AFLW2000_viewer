import cv2
import glob
import numpy as np
import scipy.io as sio
from math import cos, sin
from typing import Optional
from natsort import natsorted


def draw_axis(
    img: np.ndarray,
    pitch: float,
    yaw: float,
    roll: float,
    tdx: Optional[int]=None,
    tdy: Optional[int]=None,
    size: Optional[int]=100,
):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2
    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy
    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)), (0,0,255), 2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)), (0,255,0), 2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)), (255,0,0), 2)
    return img


def main():
    image_files = glob.glob("datasets/*.jpg")
    mat_files = glob.glob("datasets/*.mat")

    for image_file, mat_file in zip(natsorted(image_files), natsorted(mat_files)):
        # Load .jpg
        img = cv2.imread(image_file)
        # Load .mat
        mat = sio.loadmat(mat_file)

        # Crop Image
        pt2d = mat['pt2d']
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])
        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img[int(y_min):int(y_max), int(x_min):int(x_max)]

        # [pitch yaw roll tdx tdy tdz scale_factor]
        pre_pose_params = mat['Pose_Para'][0]
        # Get [pitch, yaw, roll]
        pose_params = pre_pose_params[:3]

        # Convert to degrees
        pitch = pose_params[0]# * 180 / np.pi
        yaw = pose_params[1]  # * 180 / np.pi
        roll = pose_params[2] # * 180 / np.pi

        # Covert to R
        x = pitch
        y = yaw
        z = roll
        # x
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(x), -np.sin(x)],
                [0, np.sin(x), np.cos(x)]
            ]
        )
        # y
        Ry = np.array(
            [
                [np.cos(y), 0, np.sin(y)],
                [0, 1, 0],
                [-np.sin(y), 0, np.cos(y)]
            ]
        )
        # z
        Rz = np.array(
            [
                [np.cos(z), -np.sin(z), 0],
                [np.sin(z), np.cos(z), 0],
                [0, 0, 1]
            ]
        )
        R = Rz.dot(Ry.dot(Rx))

        # Convert to degrees
        pitch = pose_params[0] * 180 / np.pi
        yaw = pose_params[1]   * 180 / np.pi
        roll = pose_params[2]  * 180 / np.pi

        # Draw Axis
        cx = img.shape[1]//2
        cy = img.shape[0]//2
        draw_axis(
            img,
            pitch,
            yaw,
            roll,
            tdx=cx,
            tdy=cy,
            size=cy,
        )

        # View
        cv2.imshow("View", img)

        print(f'pitch: {pitch}, Yaw: {yaw}, Roll: {roll}, R: {R}')

        key = cv2.waitKey(0)
        if key == 27:  # ESC
            break

if __name__ == "__main__":
    main()