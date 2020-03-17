from receiver_server import IGTLinkServer
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from models import ResNet18, MSTCN
import torch
from scipy.signal import butter, filtfilt
import time
from scipy.interpolate import interp1d

ckpt_path = "/media/maria/Elements/Maria/Submissions/IROS2020/Experiments/models/ResNet18.pt"
mstcn_ckpt_path = "/media/maria/Elements/Maria/Submissions/IROS2020/Experiments/models/TCN_imgProb.ckpt"
DESIRED_LEVEL = 4


def smooth(signal, N=30):
    N = 2  # Filter order
    Wn = 0.05  # Cutoff frequency
    B, A = butter(N, Wn, output='ba')
    # Second, apply the filter
    filt = filtfilt(B, A, signal)
    return filt


def undrift(signal):
    N = 2  # Filter order
    Wn = 0.01  # Cutoff frequency
    B, A = butter(N, Wn, output='ba')
    # Second, apply the filter
    avg = filtfilt(B, A, signal)
    undrifted = signal - avg

    return undrifted


def getPosition(matrix):
    pos = matrix[0:3, 3]
    rot = R.from_matrix(matrix[0:3, 0:3])
    quat = rot.as_quat()

    return pos, quat


def run_detection(batch, model):

    model.model.eval() # batch norm screws with eval check this
    with torch.no_grad():
        out_detections = model.model.forward(batch.to("cuda"))
        prob = torch.sigmoid(out_detections)

    return prob


def find_valid_indexes(y):
    smoothed_vel = np.squeeze(smooth(np.diff(y), 10))

    max_vel_pos = np.argmax(smoothed_vel)
    max_vel = smoothed_vel[max_vel_pos]
    first_half = np.flip(smoothed_vel[0:max_vel_pos])

    try:
        p1 = max_vel_pos - np.where(first_half<(0.2*max_vel))[0][0]

        second_half = smoothed_vel[max_vel_pos::]
        p2 = max_vel_pos + np.where(second_half < (0.2 * max_vel))[0][0]
    except:
        return 0, len(smoothed_vel) - 1, smoothed_vel

    return p1, p2, smoothed_vel


def prepare_data(y, probs, forces):

    # only considering samples where the robot is already moving
    p_i, p_e, vel = find_valid_indexes(y)
    y_arr = y[p_i: p_e]
    prob_np = probs[p_i: p_e]
    force_array = forces[p_i: p_e]

    # re-sampling in an equally spaced space grid
    y_eq = np.linspace(y_arr[0], y_arr[-1], y_arr.size)

    f1 = interp1d(y_arr, prob_np)
    prob_np = f1(y_eq)

    f2 = interp1d(y_arr, force_array)
    force_array = f2(y_eq)

    # smoothing and normalizing force and probabilities
    undrifted_force = undrift(force_array)
    smoothed_force = smooth(undrifted_force)

    normalized_force = smoothed_force - np.min(smoothed_force)
    if np.max(normalized_force) != 0:
        normalized_force = normalized_force / np.max(normalized_force)

    prob_np = smooth(prob_np)

    # run the tcn model on the input data
    tensor_input = torch.from_numpy(np.flip(prob_np, axis=0).copy()).float().to("cuda")
    tensor_input = tensor_input.unsqueeze(0)
    tensor_input = tensor_input.unsqueeze(0)
    return tensor_input, [y_eq, prob_np, normalized_force]


def check_step(server_handler, cnn_model, tcn_model):
    outputs = None
    tcn_in = None

    pose_list = []
    force_list = []
    y = []
    i = 0

    # cnn_model.model.to("cuda")
    # tcn_model.model.to("cuda")

    while True:
        i += 1

        # getting image and force data
        data = server_handler.get_data()
        # force_data = server_handler.get_force_data()

        if data == -1:
            cnn_model.model.to("cpu")
            tcn_model.model.to("cpu")
            return -1, None

        # discarding data if you don't have the position cause it screws interpolation
        if data is not None and data.matrix[1, 3] == 0:
            continue

        if data is None:
            return None, None

        image = np.squeeze(data.image)
        plt.clf()
        plt.imshow(image)
        plt.pause(0.0001)


def process_step1(server_handler, cnn_model, tcn_model):
    outputs = None
    tcn_in = None

    pose_list = []
    force_list = []
    y = []
    i = 0

    # cnn_model.model.to("cuda")
    # tcn_model.model.to("cuda")

    while True:
        i += 1

        # getting image and force data
        data = server_handler.get_data()
        force_data = server_handler.get_force_data()

        if data == -1:
            cnn_model.model.to("cpu")
            tcn_model.model.to("cpu")
            return -1, None

        # discarding data if you don't have the position cause it screws interpolation
        if data is not None and data.matrix[1, 3] == 0:
            continue

        if force_data is not None:
            force_list.append(-force_data)
        else:
            print("Force queue is empty")

        if data is not None:
            y.append(data.matrix[1, 3])
            pose_list.append(data.matrix)

            img_tensor = cnn_model.transform_image(np.squeeze(data.image))
            if outputs is None:
                outputs = img_tensor
            else:
                outputs = torch.cat((outputs, img_tensor), dim=0)

        if outputs is not None and (outputs.size()[0] == 34 or data is None):
            probs = run_detection(outputs, cnn_model)
            outputs = None

            if tcn_in is None:
                tcn_in = probs[:, 1]
            else:
                tcn_in = torch.cat((tcn_in, probs[:, 1]), dim=0)

        if data is None:
            print("Image queue is empty - proceeding with processing")
            # data None means that the image stream is finished and therefore the method finalize the processing and
            # returns the result
            if tcn_in is None:
                return 0, None

            # converting the input to the tcn to an array so that pre-processing operations can be performed
            prob_np = np.squeeze(tcn_in.to("cpu").detach().numpy())
            y_arr = np.array(y)
            force_array = np.array(force_list)

            # preparing the input data
            # prepared_data = [y_eq, prob_np, normalized_force]
            input_tensor, [y_eq, smoothed_probs, smoothed_force] = prepare_data(y_arr, prob_np, force_array)

            # extract the labels
            predictions = tcn_model.model.forward(input_tensor)
            p = predictions[-1].squeeze()  # only last layer -> -1
            res = torch.argmax(p, dim=0)

            vertebral_levels = np.squeeze(res.to("cpu").detach().numpy())
            plt.subplot(3, 1, 1)
            plt.plot(smoothed_probs)
            plt.subplot(3, 1, 2)
            plt.plot(force_array)
            plt.subplot(3, 1, 3)
            plt.plot(smoothed_force)
            plt.show()

            vert_2 = np.where(vertebral_levels == DESIRED_LEVEL)[0]

            if vert_2.size > 0:
                avg_2 = int(np.mean(vert_2))
                position = pose_list[avg_2]
                y_pos = y_eq[avg_2]
                position[1, 3] = y_pos
                return 1, position

            else:
                return 2, None


def main():

    receiving_server = IGTLinkServer()

    receiving_server.set_address(address="127.0.0.1",
                                 port=9004)

    receiving_server.start()

    model_handler = ResNet18 (ckpt_path)
    mstcn_model = MSTCN(mstcn_ckpt_path)

    model_handler.model.to("cuda")
    mstcn_model.model.to("cuda")

    receiving_server.listen_for_connection()
    receiving_server.start_receiver()

    while True:

        is_cuda_cnn = next(model_handler.model.parameters()).is_cuda
        is_cuda_tcn = next(mstcn_model.model.parameters()).is_cuda

        if not is_cuda_cnn:
            model_handler.model.to("cuda")

        if not is_cuda_tcn:
            mstcn_model.model.to("cuda")

        print("cnn is cuda: ", is_cuda_cnn)
        print("tcn is cuda: ", is_cuda_tcn)

        res, vertebral_position = check_step(receiving_server, model_handler, mstcn_model)
        if res == -1:
            print("The client connection got lost, restarting the server and waiting for new connections")

            # killing actual connections
            receiving_server.stop_receiver()
            receiving_server.kill_client_sockets()
            receiving_server.init()

            # restarting and listening for new connections
            receiving_server.listen_for_connection()
            receiving_server.start_receiver()
            continue

        elif res == 0:
            print("No data are streamed: keep waiting for data")
            plt.close()
            continue

        elif res == 1:
            print("vertebral level correctly detected - sending position to client")
            pos, quat = getPosition(vertebral_position)
            receiving_server.send_position(pos, quat)
            #receiving_server.init()

        elif res == 2:
            print("no vertebra was found, not sending anything - Press a button to restart the same step")
            #receiving_server.init()


main()
