from step_processors import *
from models import *
from receiver_server import *
import torch
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
import numpy as np

__all__ = ['VertebraeCounting']


class VertebraeCounting(GenericStep):
    def __init__(self, cnn_model_ckpt="", tcn_model_ckpt="", cnn_batch_size=34, use_force=True):
        super(VertebraeCounting, self).__init__()

        self.name = "VertebraCountingStep"

        self.cnn_model_ckpt = cnn_model_ckpt
        self.tcn_model_ckpt = tcn_model_ckpt

        self.cnn_model = None
        self.tcn_model = None

        self.data_dict = None
        self.current_batch = None
        self.batch_size = cnn_batch_size
        self.use_force = use_force

    def load_models(self, cnn_model_ckpt=None, tcn_model_ckpt=None):

        if cnn_model_ckpt is not None:
            self.cnn_model_ckpt = cnn_model_ckpt

        if tcn_model_ckpt is not None:
            self.tcn_model_ckpt = tcn_model_ckpt

        self.cnn_model = ResNet18(self.cnn_model_ckpt)
        self.cnn_model = MSTCN(self.tcn_model_ckpt)

        self._load_models_to_device(self.device)

    def release_models(self):
        self._load_models_to_device("cpu")
        self.cnn_model = None
        self.tcn_model = None

    def _load_models_to_device(self, device):
        self.cnn_model.model.to(device)
        self.tcn_model.model.to(device)

    def init_data_dict(self):
        self.data_dict = dict()

        self.data_dict["time_force"] = []
        self.data_dict["force"] = []

        self.data_dict["time_image"] = []
        self.data_dict["y_image"] = []
        self.data_dict["poses"] = []
        self.data_dict["probs"] = []

    def process_batch(self):

        batch_timestamps = [data.timestamp for data in self.current_batch]
        batch_poses = [data.matrix for data in self.current_batch]
        batch_y = [data.matrix[1, 3] for data in self.current_batch]
        batch_tensor = torch.cat([self.cnn_model.transform_image(np.squeeze(data.image))
                                  for data in self.current_batch], dim=0)

        self.cnn_model.eval()  # batch norm screws with eval check this
        with torch.no_grad():
            out_detections = self.cnn_model.model.forward(batch_tensor.to("cuda"))
            probs = torch.sigmoid(out_detections)

        np_probs = probs[:, 1].to("cpu").numpy().to_list()

        self.data_dict["time_image"].extend(batch_timestamps)
        self.data_dict["y_image"].extend(batch_y)
        self.data_dict["poses"].extend(batch_poses)
        self.data_dict["probs"].extend(np_probs)

        self.current_batch = []

        return

    def preprocess_probs(self, y, probs, y_eq):
        f = interp1d(y, probs)
        probs_eq = f(y_eq)

        # smoothing probabilities
        smoothed_probs = self.smooth(probs_eq)
        smoothed_probs = np.expand_dims(smoothed_probs, axis=0)
        return smoothed_probs

    def preprocess_force(self, y, force, y_eq):
        f = interp1d(y, force)
        force_eq = f(y_eq)

        # smoothing and normalizing force
        undrifted_force = self.undrift(force_eq)
        smoothed_force = self.smooth(undrifted_force)
        normalized_force = smoothed_force - np.min(smoothed_force)

        if np.max(normalized_force) != 0:
            normalized_force = normalized_force / np.max(normalized_force)
        normalized_force = np.expand_dims(normalized_force, axis=0)
        return normalized_force

    def tcn_preprocessing(self, y, probs, force, t_probs=None, t_force=None):

        # if timestamps are provided, also resample the force signal in the probabilities time grid
        if self.use_force and t_probs is not None and t_force is not None:
            f = interp1d(t_force, force)
            force = f(t_probs)

        # generating an equally sampled time grid
        y_eq = np.linspace(y[0], y[-1], y.size)

        # pre-process probabilities
        prep_probs = self.preprocess_probs(y, probs, y_eq)

        # pre-process force
        if self.use_force:
            prep_force = self.preprocess_force(y, force, y_eq)
            input_np = np.concatenate([prep_probs, prep_force], axis=0)
        else:
            input_np = prep_probs

        # converting to tensor and unsqueezing first dimension, cause batch size is 1
        input_tensor = torch.from_numpy(input_np).float()
        input_tensor = input_tensor.unsqueeze(0)
        return input_tensor

    def run_tcn(self):

        data = self.tcn_preprocessing(y=np.array(self.data_dict["y"]),
                                      probs=np.array(self.data_dict["probs"]),
                                      force=np.array(self.data_dict["force"]))

        predictions = self.tcn_model.model.forward(data.to("gpu"))
        p = predictions[-1].squeeze()  # only last layer -> -1
        res = torch.argmax(p, dim=0)

        vertebral_levels = np.squeeze(res.to("cpu").detach().numpy())
        return vertebral_levels

    def process(self, server_handler):

        self.init_data_dict()
        self.current_batch = []

        while True:

            # getting image and force data
            img_data = server_handler.get_image_data()

            if self.use_force:
                force_data = server_handler.get_sensor_data()
            else:
                force_data = Data(valid=False)

            # If the client got disconnected, clean the buffers and return
            if img_data.status == DISCONNECTED or force_data.status == DISCONNECTED:
                return Data(status=DISCONNECTED)

            if img_data.status == STEP_ENDED or force_data.status == STEP_ENDED:
                self.process_batch()
                vertebral_levels = self.run_tcn()
                # todo: send position to server
                return Data(status=STEP_ENDED)

            if force_data.is_valid and force_data.status == PROCESS:
                self.data_dict["time_force"].append(force_data.timestamp)
                self.data_dict["force"].append(force_data.sensor_data)

            if img_data.is_valid and img_data.status == PROCESS:
                self.current_batch.append(img_data)

            if len(self.current_batch) >= self.batch_size:
                self.process_batch()

    @staticmethod
    def smooth(signal, N=30):
        N = 2  # Filter order
        Wn = 0.05  # Cutoff frequency
        B, A = butter(N, Wn, output='ba')
        # Second, apply the filter
        filt = filtfilt(B, A, signal)
        return filt

    @staticmethod
    def undrift(signal):
        N = 2  # Filter order
        Wn = 0.01  # Cutoff frequency
        B, A = butter(N, Wn, output='ba')
        # Second, apply the filter
        avg = filtfilt(B, A, signal)
        undrifted = signal - avg
        return undrifted

    @staticmethod
    def getPosition(matrix):
        pos = matrix[0:3, 3]
        rot = R.from_matrix(matrix[0:3, 0:3])
        quat = rot.as_quat()
        return pos, quat