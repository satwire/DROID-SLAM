import time

import torch
from factor_graph import FactorGraph
from lietorch import SE3


class DroidFrontend:
    def __init__(self, net, video, args, timing=None):
        self.video = video
        self.update_op = net.update
        self.graph = FactorGraph(
            video, net.update, max_factors=48, upsample=args.upsample
        )
        self.timing = timing

        # Local optimization window.
        self.t0 = 0
        self.t1 = 0

        # Frontend variables
        self.is_initialized = False
        self.count = 0

        self.max_age = 20
        self.iters1 = 3
        self.iters2 = 2

        self.keyframe_removal_index = 3

        self.warmup = args.warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius

        self.depth_window = 3

        self.motion_damping = 0.0
        if hasattr(args, "motion_damping"):
            self.motion_damping = args.motion_damping

        # Confidence-weighted blending of the IMU prior toward the motion
        # model's guess. False (default): full trust in the prior once its
        # confidence is above 0.1.
        self.imu_blend = bool(getattr(args, "imu_blend", False))
        # Frames whose start pose came from the prior (for checking a run).
        self.prior_applied = 0

    def _init_next_state(self):
        """Motion-model guess for the next slot, before its frame arrives.

        With motion_damping = 0 (the default, as in upstream DROID) this copies
        the previous pose. An IMU prior for the frame is applied later, by
        apply_prior, once the frame is in the video.
        """
        self.video.disps[self.t1] = torch.quantile(
            self.video.disps[self.t1 - 3 : self.t1 - 1], 0.5
        )

        poses = SE3(self.video.poses)
        if self.motion_damping >= 0 and self.t1 >= 2:
            vel = (poses[self.t1 - 1] * poses[self.t1 - 2].inv()).log()  # type: ignore
            guess = SE3.exp(self.motion_damping * vel) * poses[self.t1 - 1]
        else:
            guess = poses[self.t1 - 1]
        self.video.poses[self.t1] = guess.data  # type: ignore

    def apply_prior(self, delta, confidence):
        """Start the frame just appended (slot t1) from the IMU step ending at it.

        `delta` is the relative pose from the previous frame to this one, so the
        prior initializes the same step it describes. It replaces the motion
        model's guess that _init_next_state left in slot t1; with imu_blend it
        is blended toward that guess by the confidence instead. Droid.track
        calls this after the motion filter and before the update. Nothing
        happens if the filter dropped the frame, before initialization (the
        warmup frames start from identity, as in DROID), or when the
        confidence is 0.1 or lower.
        """
        if (
            delta is None
            or confidence <= 0.1
            or not self.is_initialized
            or self.video.counter.value != self.t1 + 1
        ):
            return
        t0 = time.perf_counter()
        poses = SE3(self.video.poses)
        # The stored poses are world-to-camera, so delta.inv() on the left moves
        # the camera by delta in its own frame (the opposite composition about
        # doubles the rotation error against ground truth).
        prior = delta.to(self.video.poses.device).inv() * poses[self.t1 - 1]
        if self.imu_blend:
            # Geodesic blend: confidence 1 gives the prior, 0 the motion model.
            guess = poses[self.t1]
            xi = (prior * guess.inv()).log()  # type: ignore
            prior = SE3.exp(confidence * xi) * guess
        self.video.poses[self.t1] = prior.data  # type: ignore
        self.prior_applied += 1
        if self.timing is not None:
            self.timing.record("frontend_apply_prior", time.perf_counter() - t0)

    def _update(self):
        """add edges, perform update"""

        self.count += 1
        self.t1 += 1

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(
            self.t1 - 5,
            max(self.t1 - self.frontend_window, 0),
            rad=self.frontend_radius,
            nms=self.frontend_nms,
            thresh=self.frontend_thresh,
            beta=self.beta,
            remove=True,
        )

        self.video.disps[self.t1 - 1] = torch.where(
            self.video.disps_sens[self.t1 - 1] > 0,
            self.video.disps_sens[self.t1 - 1],
            self.video.disps[self.t1 - 1],
        )

        for itr in range(self.iters1):
            self.graph.update(None, None, use_inactive=True)

        # set initial pose for next frame
        d = self.video.distance(
            [self.t1 - 4], [self.t1 - 2], beta=self.beta, bidirectional=True
        )

        if d.item() < 2 * self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1 - 3)

            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1

        else:
            for itr in range(self.iters2):
                self.graph.update(None, None, use_inactive=True)

        # set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1 - 1]
        self.video.disps[self.t1] = torch.quantile(
            self.video.disps[self.t1 - self.depth_window - 1 : self.t1 - 1], 0.7
        )

        # update visualization
        self.video.dirty[self.graph.ii.min() : self.t1] = True

    def _initialize(self):
        """initialize the SLAM system"""

        self.t0 = 0
        self.t1 = self.video.counter.value

        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        self.graph.add_proximity_factors(
            0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False
        )

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1 - 1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1 - 4 : self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1 - 1].clone()
        self.last_disp = self.video.disps[self.t1 - 1].clone()
        self.last_time = self.video.tstamp[self.t1 - 1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[: self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup - 4, store=True)

    def __call__(self):
        """main update"""

        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            t0 = time.perf_counter()
            self._initialize()
            if self.timing is not None:
                torch.cuda.synchronize()
                self.timing.record("frontend_initialize", time.perf_counter() - t0)
            t0 = time.perf_counter()
            self._init_next_state()
            if self.timing is not None:
                self.timing.record("frontend_init_next_state", time.perf_counter() - t0)

        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            t0 = time.perf_counter()
            self._update()
            if self.timing is not None:
                torch.cuda.synchronize()
                self.timing.record("frontend_update", time.perf_counter() - t0)
            t0 = time.perf_counter()
            self._init_next_state()
            if self.timing is not None:
                self.timing.record("frontend_init_next_state", time.perf_counter() - t0)
