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

        self.imu_delta_pose = None
        self.imu_confidence = 0.0
        # Confidence-weighted blending of the IMU prior toward the fallback
        # prediction. False (default) = legacy behavior: full trust in the
        # prior once confidence > 0.1.
        self.imu_blend = bool(getattr(args, "imu_blend", False))

    def _init_next_state(self):
        # DEBUG: count init_next_state invocations.
        self._init_calls = getattr(self, "_init_calls", 0) + 1

        self.video.disps[self.t1] = torch.quantile(
            self.video.disps[self.t1 - 3 : self.t1 - 1], 0.5
        )

        # Pose prediction. Priority: IMU prior > damped velocity > constant pose.
        use_imu = (
            self.imu_delta_pose is not None
            and self.imu_confidence > 0.1
            and self.t1 >= 1
        )

        poses = SE3(self.video.poses)
        # Fallback prediction (used on its own without the IMU prior, and as
        # the blend target with it): damped velocity when two poses exist,
        # else constant pose.
        if self.motion_damping >= 0 and self.t1 >= 2:
            vel = (poses[self.t1 - 1] * poses[self.t1 - 2].inv()).log()  # type: ignore
            fallback = SE3.exp(self.motion_damping * vel) * poses[self.t1 - 1]
        else:
            fallback = poses[self.t1 - 1]

        if use_imu:
            # DEBUG: count IMU branch hits.
            self._imu_hits = getattr(self, "_imu_hits", 0) + 1

            delta = self.imu_delta_pose.to(self.video.poses.device)  # type: ignore
            # IAAI stores the delta with inverse-frame convention relative to
            # DROID (the rotation matrix expresses pose[t-1] in frame t, not
            # frame t in frame t-1). Invert before forward-composing.
            imu_pred = delta.inv() * poses[self.t1 - 1]
            if self.imu_blend:
                # Confidence-weighted geodesic blend: conf=1 -> pure IMU
                # prediction, conf->0 -> fallback. xi is the tangent-space
                # difference between the two predictions.
                xi = (imu_pred * fallback.inv()).log()  # type: ignore
                next_pose = SE3.exp(self.imu_confidence * xi) * fallback
            else:
                next_pose = imu_pred
            self.video.poses[self.t1] = next_pose.data  # type: ignore
        else:
            self.video.poses[self.t1] = fallback.data  # type: ignore

        # Consume the stashed prior to prevent stale reuse on the next call.
        self.imu_delta_pose = None
        self.imu_confidence = 0.0

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
