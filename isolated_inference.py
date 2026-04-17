# isolated inference.py


"""
Minimal LeRobot VLA inference script.


Stripped down to just: load model → connect robot → observation → action → send.
No teleop, no dataset recording, no video encoding.


Data flow:
    LeRobotDataset(DATASET_PATH)        → ds_meta  (features + normalization stats)
    load_model(policy_path, ds_meta)    → policy, policy_cfg
    load_pipeline(policy_cfg, ds_meta)  → preprocessor, postprocessor
    run_inference(robot, policy, ...)   → sends actions to hardware
"""


import time
import logging

# used for performance analysis
# (trying to check bottlenecks during inference)
import matplotlib.pyplot as plt
TIMING_PLOT_NAME = "smolvla_timing_plot_torchautocast"
timing_history = {
    "camera_capture": [],
    "obs_processing": [],
    "predict_action": [],
}
plt.ion()  # interactive mode — allows non-blocking updates
fig, ax = plt.subplots()
ax.set_xlabel("iteration")
ax.set_ylabel("ms")
ax.set_title(f"{TIMING_PLOT_NAME}")

import torch
import numpy as np


from lerobot.configs.policies import PreTrainedConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    make_default_processors,
    RobotAction )
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import make_robot_from_config, so_follower  # noqa: F401
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging


from pathlib import Path




# ─────────────────────────────────────────────
# CONFIG — edit these for your setup
# ─────────────────────────────────────────────

USE_AUTOCAST = False

POLICY_PATH  = "grahamwichhh/v3_smolvla_so101-pick-up-lego"
DATASET_PATH = "grahamwichhh/eval_v2_so101_lego-to-mug_50ep"   # training dataset, ONLY needed for feature shapes + norm stats
# ROBOT_NAME = "ryan_gosling" # redundant
TASK         = "Grab the cube"              # natural language prompt passed to the VLA model
DEVICE       = "cuda"                       # "cuda", "mps", or "cpu"
FPS          = 30                           # control loop frequency
RUN_TIME_S   = 60                           # how long to run inference (seconds)

RENAME_MAP = {}

# camera ports senmt to OpenCVCameraConfig require Path objects:
CAMERA_VIDEO_1 = Path("/dev/video12")
CAMERA_VIDEO_2 = Path("/dev/video6")
CAMERA_VIDEO_13 = Path("/dev/video0")

ROBOT_CONFIG = so_follower.SO101FollowerConfig(
    port="/dev/ttyACM1",
    id="ryan_gosling",                                          # must match the id used during calibration
    calibration_dir=Path("~/.cache/huggingface/lerobot/calibration/robots/so_follower").expanduser(),
    cameras={
        "camera1": OpenCVCameraConfig(
            index_or_path=CAMERA_VIDEO_1,
            width=640,
            height=480,
            fps=30,
        ),
        "camera2": OpenCVCameraConfig(
            index_or_path=CAMERA_VIDEO_2,
            width=640,
            height=480,
            fps=30,
        ),
    },
)




# ─────────────────────────────────────────────
# 1. LOAD DATASET METADATA
# ─────────────────────────────────────────────
"""
    Fetch metadata from the training dataset without downloading any episode frames.
    LeRobotDataset(repo_id) pulls only the dataset card and stats JSON from HuggingFace.
    What we get back:
    ds_meta.features — shapes + dtypes the model expects (action, observations, etc.)
    ds_meta.stats    — per-feature mean/std used for normalization during training
    Both are required downstream:
    make_policy()             needs ds_meta to validate input/output feature shapes
    make_pre_post_processors() needs ds_meta.stats to build the normalization layers
"""

# test loading dataset here:
def load_dataset(dataset_path: str):

    logging.info(f"Loading dataset metadata from: {dataset_path}")
    # dataset = LeRobotDataset.create(repo_id=dataset_path, robot_type=ROBOT_NAME, fps=FPS, features=dataset_features)
    dataset=LeRobotDataset(dataset_path)
    logging.info(f"Dataset meta loaded:\n{dataset.meta}")
    return dataset


# ─────────────────────────────────────────────
# 2. LOAD MODEL
# ─────────────────────────────────────────────


def load_model(policy_path: str, device: str, ds_meta):
    """
    Load the pretrained VLA policy.


    PreTrainedConfig.from_pretrained() reads the model architecture config
    (e.g. Pi0, ACT, Diffusion) stored alongside the weights on HuggingFace.


    make_policy() instantiates the correct model class, loads weights, and uses
    ds_meta to validate that the model's expected input/output features match
    the dataset the policy was trained on.


    ds_meta comes from load_dataset_meta() — it must be the dataset the policy
    was trained on, not an arbitrary dataset.
    """
    logging.info(f"Loading policy from: {policy_path}")


    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = device


    # ds_meta is passed here instead of None — this is what resolves the
    # "Missing key(s) in state_dict" weight loading errors seen with ds_meta=None,
    # because make_policy uses ds_meta.features to correctly configure the model
    # head dimensions before loading weights.
    policy = make_policy(policy_cfg, ds_meta=ds_meta)
    policy.eval()   # disable dropout etc. — required for deterministic inference




    # ATTEMPTED OPTIMS

    # halve weights to fp16:
    # policy = policy.half()

    # policy = torch.compile(policy, mode="reduce-overhead")
    # mode options:
    #   "default"         — balanced
    #   "reduce-overhead" — best for repeated same-shape inputs (your case)
    #   "max-autotune"    — slowest to compile, fastest at runtime

    policy_cfg.use_amp = True




    logging.info("Policy loaded successfully.")
    return policy, policy_cfg




# ─────────────────────────────────────────────
# 3. LOAD PIPELINE (pre/postprocessors)
# ─────────────────────────────────────────────


def load_pipeline(policy_cfg, ds_meta, rename_map=None):
    """
    Build the normalization pipelines that wrap the model.


    Preprocessor:  raw obs dict → normalized tensors the model expects
    Postprocessor: raw model output tensors → denormalized joint targets


    ds_meta.stats contains the per-feature mean/std saved at training time.
    rename_stats() applies RENAME_MAP to those stat keys so they match
    whatever observation key names your robot produces at runtime.


    Mirrors lerobot_record.py lines 484-492.
    """
    rename_map = rename_map or {}


    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        dataset_stats=rename_stats(ds_meta.stats, rename_map),  # lerobot_record.py line 487
        preprocessor_overrides={
            "device_processor":              {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    return preprocessor, postprocessor




# ─────────────────────────────────────────────
# 4. INFERENCE LOOP
# ─────────────────────────────────────────────
def run_inference(robot, policy, policy_cfg, preprocessor, postprocessor, task, fps, run_time_s, dataset):

    _, robot_action_processor, robot_observation_processor = make_default_processors()

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    # Map from policy's expected keys → robot's raw output keys.
    # policy_cfg.input_features is the ground truth of what the model needs.
    # raw_obs uses short keys ('camera1', 'camera2'), policy expects
    # 'observation.images.camera1' etc. — we bridge that here.
    robot_to_policy_key_map = {
        "camera1": "observation.images.camera1",
        "camera2": "observation.images.camera2",
        # add camera3 here if/when you have a third camera connected
    }

    device = get_safe_torch_device(policy_cfg.device)
    logging.info(f"Starting inference loop | task='{task}' | fps={fps} | duration={run_time_s}s")

    start_t = time.perf_counter()
    timestamp = 0.0

    while timestamp < run_time_s:
        loop_start_t = time.perf_counter()

        # --- OBSERVE ---

        t0 = time.perf_counter() # start time

        raw_obs = robot.get_observation()
        t1 = time.perf_counter() # check how long it takes to get the robot state

        obs = robot_observation_processor(raw_obs)
        t2 = time.perf_counter() # check how long observation processing took


        # Build the observation batch directly from policy_cfg.input_features.
        # This bypasses build_dataset_frame entirely — we don't need dataset
        # storage format, we just need the tensors the model expects.
        observation_frame = {}



        # Joint state — robot outputs short keys like 'shoulder_pan.pos',
        # policy expects them aggregated under 'observation.state'
        state_keys = [
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
        ]

        # TODO: purpose? use for triton?
        # observation_frame["observation.state"] = state_tensor.to(device) 

        # state_tensor = torch.tensor(
        #     [obs[k] for k in state_keys], dtype=torch.float32
        # ).unsqueeze(0)  # shape: (1, 6) — batch dim required


        # Camera images — just remap the key, no conversion needed.
        # predict_action calls prepare_observation_for_inference internally,
        # which converts numpy arrays to tensors itself.
        for robot_key, policy_key in robot_to_policy_key_map.items():
            if robot_key in obs and policy_key in policy_cfg.input_features:
                observation_frame[policy_key] = obs[robot_key]  # raw numpy array, HWC uint8

        
        # Joint state — same, just pass the numpy array
        observation_frame["observation.state"] = np.array(
            [obs[k] for k in state_keys], dtype=np.float32
        )


        # --- PREDICT ---

        # call predict_action(), will return a base tensor:
        """
            example: 
                action_values are tensor([[  2.3541, -17.7406,  46.0053,  58.2234,  26.3015,  15.2118]])
        """

        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy_cfg.use_amp,
            task=task,
            robot_type=robot.robot_type,
        )
        t3 = time.perf_counter() # check how long it takes to prediction the action

        # must convert it to "action_processed_policy" via make_robot_action using dataset as well
        """
            should be like: 
                {'shoulder_pan.pos': 2.354058265686035, 
                'shoulder_lift.pos': -17.740571975708008, 
                'elbow_flex.pos': 46.00529479980469, 
                'wrist_flex.pos': 58.223419189453125, 
                'wrist_roll.pos': 26.301464080810547, 
                'gripper.pos': 15.21180248260498}

        """

        action_processed_policy: RobotAction = make_robot_action(action_values, dataset.features)




        # --- SEND ---
        robot_action_to_send = robot_action_processor((action_processed_policy, obs))
        robot.send_action(robot_action_to_send)



        # --- PACE TO FPS ---

        # loop_start_t ─────────────────────────────────────> now
        #       [observe → predict → send]  [sleep]
        #       |←────── dt_s ─────────────|←──────→|
        #       |←──────────── 1/fps (33.3ms) ──────→|

        dt_s = time.perf_counter() - loop_start_t
        sleep_time_s = 1.0 / fps - dt_s
        if sleep_time_s < 0:
            logging.warning(f"Loop running slow: {1/dt_s:.1f} Hz vs target {fps} Hz")
        precise_sleep(max(sleep_time_s, 0.0))


        if int(timestamp * FPS) % 10 == 0:
            logging.info(
                f"camera_capture={1000*(t1-t0):.1f}ms | "
                f"obs_processing={1000*(t2-t1):.1f}ms | "
                f"predict_action={1000*(t3-t2):.1f}ms"
            )


        # matplot:
        timing_history["camera_capture"].append(1000 * (t1 - t0))
        timing_history["obs_processing"].append(1000 * (t2 - t1))
        timing_history["predict_action"].append(1000 * (t3 - t2))

        if len(timing_history["camera_capture"]) % 10 == 0:
            ax.clear()
            iterations = range(len(timing_history["camera_capture"]))
            ax.plot(iterations, timing_history["camera_capture"], label="camera_capture")
            ax.plot(iterations, timing_history["obs_processing"], label="obs_processing")
            ax.plot(iterations, timing_history["predict_action"], label="predict_action")
            ax.legend()
            ax.set_xlabel("iteration")
            ax.set_ylabel("ms")
            ax.set_title("per-iteration timing")
            plt.pause(0.001)  # non-blocking draw — 1ms, won't affect FPS meaningfully


        timestamp = time.perf_counter() - start_t


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

def main():
    init_logging()


    # Step 1: fetch feature shapes + normalization stats from the training dataset
    # ds_meta = load_dataset_meta(DATASET_PATH)
    dataset=load_dataset(DATASET_PATH)


    # Step 2: load policy weights, validated against ds_meta feature shapes
    policy, policy_cfg = load_model(POLICY_PATH, DEVICE, dataset.meta)


    # Step 3: build normalization pipelines using ds_meta.stats
    preprocessor, postprocessor = load_pipeline(policy_cfg, dataset.meta, RENAME_MAP)


    # Step 4: connect robot and run
    # NOTE: robot calibration information is loaded (if present) here
    robot = make_robot_from_config(ROBOT_CONFIG)
    robot.connect()


    try:
        if (USE_AUTOCAST):
            print("\n\nrunning inference with torch.autocast\n\n")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                run_inference(
                    robot=robot,
                    policy=policy,
                    policy_cfg=policy_cfg,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    task=TASK,
                    fps=FPS,
                    run_time_s=RUN_TIME_S,
                    dataset=dataset
                )
        else:
            print("\n\nrunning standard inference\n\n")
            run_inference(
                robot=robot,
                policy=policy,
                policy_cfg=policy_cfg,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                task=TASK,
                fps=FPS,
                run_time_s=RUN_TIME_S,
                dataset=dataset
            )
    finally:

        # functionality to return so101 to resting position:
        # should be {'shoulder_pan.pos': 1.4575186633487363, 'shoulder_lift.pos': -99.75619666802113, 'elbow_flex.pos': 97.13261648745518, 'wrist_flex.pos': 72.54736842105262, 'wrist_roll.pos': -3.003663003663007, 'gripper.pos': 3.2860824742268044, 
        rest_position = {
            "shoulder_pan.pos":  1.4,    # replace with your actual rest values
            "shoulder_lift.pos": -99.0,
            "elbow_flex.pos":    97.0,
            "wrist_flex.pos":    72.0,
            "wrist_roll.pos":    -3.0,
            "gripper.pos":       3.2,
        }

        logging.info("Attempting to return to resting position.")
        rest_start_t = time.perf_counter()
        rest_duration_s = 5.0

        plt.ioff()
        plt.savefig(f"{TIMING_PLOT_NAME}.png")
        plt.close()


        # read curr pos:
        current_obs = robot.get_observation()
        current_pos = {k: current_obs[k] for k in rest_position}

        while time.perf_counter() - rest_start_t < rest_duration_s:
            # linearly interpolate from current position to rest over rest_duration_s
            # alpha goes 0.0 → 1.0 over the duration
            # (claude code, TODO: learn about interpolation)
            alpha = (time.perf_counter() - rest_start_t) / rest_duration_s
            alpha = min(alpha, 1.0)

            interpolated = {
                k: current_pos[k] + alpha * (rest_position[k] - current_pos[k])
                for k in rest_position
            }

            # print(interpolated)

            robot.send_action(interpolated)
            precise_sleep(1.0 / FPS)

        # Always disconnect cleanly even if an exception is raised mid-loop
        robot.disconnect()
        logging.info("Done.")




if __name__ == "__main__":
    main()
