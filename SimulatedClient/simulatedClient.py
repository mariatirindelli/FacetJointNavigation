from receiver_server import *
from os import path
import sys
import pygtlink
import os
import imfusion
import numpy as np
import time
from scipy.signal import butter, filtfilt
import pandas as pd
from SimulatedClient.DataReader import ForceUSSimulator

SERVER_ADDRESS = "127.0.0.1"
PORT = 9003

v_data_dir = "/media/maria/Elements/Maria/DataBases/SpineIFL/SpinousProcessDb/SecondAcquisitionSession/MariaT_F10.imf"

USE_FORCE = False


def find_valid_indexes(y):
    smoothed_vel = np.squeeze(smooth(np.diff(y), 10))

    max_vel_pos = np.argmax(smoothed_vel)
    max_vel = smoothed_vel[max_vel_pos]
    first_half = np.flip(smoothed_vel[0:max_vel_pos])

    try:
        p1 = max_vel_pos - np.where(first_half<(0.1*max_vel))[0][0]

        second_half = smoothed_vel[max_vel_pos::]
        p2 = max_vel_pos + np.where(second_half < (0.1 * max_vel))[0][0]
    except:
        return 0, len(smoothed_vel) - 1, smoothed_vel

    return p1, p2, smoothed_vel


def smooth(signal, N=30):
    N = 2  # Filter order
    Wn = 0.05  # Cutoff frequency
    B, A = butter(N, Wn, output='ba')
    # Second, apply the filter
    filt = filtfilt(B, A, signal)
    return filt


def send_handshake(streamer_client, command_client):
    status_msg_1 = pygtlink.StatusMessage()
    status_msg_1.setDeviceName("StreamerSocket")
    status_msg_1.pack()
    streamer_client.connectToServer(SERVER_ADDRESS, PORT)
    streamer_client.send(status_msg_1.header + status_msg_1.body)

    status_msg_2 = pygtlink.StatusMessage()
    status_msg_2.setDeviceName("CommandSocket")
    status_msg_2.pack()
    command_client.connectToServer(SERVER_ADDRESS, PORT)
    command_client.send(status_msg_2.header + status_msg_2.body)


def get_data(sweep_path, traj_path):

    if not os.path.exists(traj_path):
        raise RuntimeError("Trajectory path not found: not existing dir: {}".format(traj_path))

    if not os.path.exists(sweep_path):
        raise RuntimeError("Trajectory path not found: not existing dir: {}".format(sweep_path))

    logging.info("\n**************\nSweep File: {} \nTrajectory File: {}".format(sweep_path, traj_path))
    imageset = imfusion.open(sweep_path)

    sweep = np.squeeze(np.array(imageset[0]))

    csv_dframe = pd.read_csv(traj_path)

    # if there is only one column probably the delimiter is ;
    if len(csv_dframe.columns) < 2:
        csv_dframe = pd.read_csv(traj_path, delimiter=";")

    y_robot = np.array(csv_dframe["y"])
    p_i, p_e, vel = find_valid_indexes(y_robot)

    # if force data are present, load them as well
    if "y_force" in csv_dframe.columns:
        y_force = np.array(csv_dframe["y_force"])
        logging.info("Force data present and correctly loaded")
    else:
        y_force = np.zeros(y_robot.shape)
        logging.info("Force data not present - zeros vector loaded")

    sweep = sweep[p_i:p_e, :, :]
    y_robot = y_robot[p_i:p_e]
    y_force = y_force[p_i:p_e]
    return sweep, y_robot, y_force


def send_data(streamer_client, sweep, y_pos, force_data = None, use_force = False):
    for (frame, y, force) in zip(sweep, y_pos, force_data):

        frame = np.squeeze(frame)

        matrix = np.eye(4)
        matrix[1, 3] = y

        img_msg = pygtlink.ImageMessage2()
        img_msg.setData(frame)
        img_msg.setSpacing([1, 1, 1])
        img_msg.setMatrix(matrix)

        sensor_msg = pygtlink.SensorMessage()
        sensor_msg.setData([0, force, 0])
        sensor_msg.setLength(3)

        img_msg.pack()
        streamer_client.send(img_msg.header + img_msg.body)

        sensor_msg.pack()
        streamer_client.send(sensor_msg.header + sensor_msg.body)
        time.sleep(0.03)

    # todo: send step done message

def do_smt(reader):
    """
    :param ForceUSSimulator reader: the reader object
    """
    while True:
        ts_img, img = reader.get_img_data()
        ts_force, force = reader.get_force_data()
        ts_pose, pose = reader.get_pose_data()

        if ts_img == -1:
            break

        # TODO: send data over openigtlink

def main():

    streamer_client = pygtlink.ClientSocket()
    command_client = pygtlink.ClientSocket()
    imfusion.init()

    while True:
        a = input("Chose step: SACRUM DETECTION [s], VERTEBRAL COUNTING [v], FACET DETECTION [f]")

        if "s" in a:
            print("Streaming Sacrum Data")

        elif "v" in a:
            print("Streaming Spinous Process Data")
            sweep_path = os.path.join(v_data_dir, v_subject_name + ".imf")
            traj_path = os.path.join(v_data_dir, v_subject_name + ".csv")
            sweep, y_pos, force_data = get_data(sweep_path, traj_path)

            status_msg = pygtlink.StatusMessage()
            # status_msg.setMessa


            sensor_msg.setData([0, force, 0])
            sensor_msg.setLength(3)

        elif "f" in a:
            print("Streaming Facet Detection Data")


if __name__ == "__main__":
    main()