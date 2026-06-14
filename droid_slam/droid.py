from collections import OrderedDict

import torch
from depth_video import DepthVideo
from droid_backend import DroidBackend
from droid_frontend import DroidFrontend
from droid_net import DroidNet
from motion_filter import MotionFilter
from timing_stats import TimingStats
from torch.multiprocessing import Process
from trajectory_filler import PoseTrajectoryFiller


class Droid:
    def __init__(self, args, timing: TimingStats | None = None):
        super(Droid, self).__init__()
        self.timing = timing if timing is not None else TimingStats()
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(
            self.net, self.video, thresh=args.filter_thresh, timing=self.timing
        )

        # frontend process
        self.frontend = DroidFrontend(
            self.net, self.video, self.args, timing=self.timing
        )

        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            from visualizer.droid_visualizer import visualization_fn

            self.visualizer = Process(target=visualization_fn, args=(self.video, None))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)

    def load_weights(self, weights):
        """load trained model weights"""

        print(weights)
        self.net = DroidNet()
        state_dict = OrderedDict(
            [(k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()]
        )

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def track(
        self,
        tstamp,
        image,
        depth=None,
        intrinsics=None,
        imu_delta_pose=None,
        imu_confidence=1.0,
    ):
        """main thread - update map

        imu_delta_pose: lietorch SE3, scalar-batch (shape (7,)) — delta from frame t-1 to t. Layout: [tx, ty, tz, qx, qy, qz, qw]
        imu_confidence: float in [0, 1]. <0.1 disables the prior for this frame.
        """
        with torch.no_grad():
            # Stash IMU delta on the frontend so _init_next_state can read it.
            self.frontend.imu_delta_pose = imu_delta_pose
            self.frontend.imu_confidence = imu_confidence

            # DEBUG: count IMU stash calls.
            if not hasattr(self, "_imu_stats"):
                self._imu_stats = {"n_calls": 0, "n_usable": 0}
            self._imu_stats["n_calls"] += 1
            if imu_delta_pose is not None and imu_confidence > 0.1:
                self._imu_stats["n_usable"] += 1

            # check there is enough motion
            torch.cuda.synchronize()
            self.timing.start("motion_filter")
            self.filterx.track(tstamp, image, depth, intrinsics)
            torch.cuda.synchronize()
            self.timing.stop("motion_filter")

            # local bundle adjustment
            torch.cuda.synchronize()
            self.timing.start("frontend")
            self.frontend()
            torch.cuda.synchronize()
            self.timing.stop("frontend")

    def terminate(self, stream=None):
        """terminate the visualization process, return poses [t, q]"""

        del self.frontend

        torch.cuda.empty_cache()
        print("#" * 32)
        torch.cuda.synchronize()
        self.timing.start("backend_low")
        self.backend(7)
        torch.cuda.synchronize()
        self.timing.stop("backend_low")

        torch.cuda.empty_cache()
        print("#" * 32)
        torch.cuda.synchronize()
        self.timing.start("backend_high")
        self.backend(12)
        torch.cuda.synchronize()
        self.timing.stop("backend_high")

        torch.cuda.synchronize()
        self.timing.start("traj_filler")
        camera_trajectory = self.traj_filler(stream)
        torch.cuda.synchronize()
        self.timing.stop("traj_filler")

        return camera_trajectory.inv().data.cpu().numpy()
