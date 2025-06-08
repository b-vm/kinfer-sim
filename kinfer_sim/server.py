"""Server and simulation loop for KOS."""

import asyncio
import itertools
import logging
import time
import traceback
from pathlib import Path
from queue import Queue
from typing import Literal

import colorlogging
import numpy as np
import tarfile
import typed_argparse as tap
from kinfer.rust_bindings import PyModelRunner, metadata_from_json
from kmv.app.viewer import QtViewer
from kmv.utils.logging import VideoWriter, save_logs
from kscale import K
from kscale.web.gen.api import RobotURDFMetadataOutput
from kscale.web.utils import get_robots_dir, should_refresh_file

from kinfer_sim.keyboard_listener import KeyboardListener
from kinfer_sim.provider import ModelProvider
from kinfer_sim.simulator import MujocoSimulator
from kinfer_sim.reward_plotter import RewardPlotter

logger = logging.getLogger(__name__)


class ServerConfig(tap.TypedArgs):
    kinfer_path: str = tap.arg(positional=True, help="Path to the K-Infer model to load")

    # Mujoco settings
    mujoco_model_name: str = tap.arg(positional=True, help="Name of the Mujoco model to simulate")
    mujoco_scene: str = tap.arg(default="smooth", help="Mujoco scene to use")
    no_cache: bool = tap.arg(default=False, help="Don't use cached metadata")
    debug: bool = tap.arg(default=False, help="Enable debug logging")

    # Physics settings
    dt: float = tap.arg(default=0.0001, help="Simulation timestep")
    pd_update_frequency: float = tap.arg(default=1000.0, help="PD update frequency for the actuators (Hz)")
    no_gravity: bool = tap.arg(default=False, help="Enable gravity")
    start_height: float = tap.arg(default=1.1, help="Start height")
    quat_name: str = tap.arg(default="imu_site_quat", help="Name of the quaternion sensor")
    acc_name: str = tap.arg(default="imu_acc", help="Name of the accelerometer sensor")
    gyro_name: str = tap.arg(default="imu_gyro", help="Name of the gyroscope sensor")

    # Rendering settings
    no_render: bool = tap.arg(default=False, help="Enable rendering")
    render_frequency: float = tap.arg(default=1.0, help="Render frequency (Hz)")
    frame_width: int = tap.arg(default=640, help="Frame width")
    frame_height: int = tap.arg(default=480, help="Frame height")
    camera: str | None = tap.arg(default=None, help="Camera to use")
    save_path: str = tap.arg(default="logs", help="Path to save logs")
    save_video: bool = tap.arg(default=False, help="Save video")
    save_logs: bool = tap.arg(default=False, help="Save logs")
    plot_rewards: bool = tap.arg(default=False, help="Enable real-time reward plotting")

    # Keyboard settings
    use_keyboard: bool = tap.arg(default=False, help="Use keyboard to control the robot")
    command_type: Literal["joystick", "control_vector"] = tap.arg(default="joystick", help="Type of command to use")

    # Randomization settings
    command_delay_min: float | None = tap.arg(default=None, help="Minimum command delay")
    command_delay_max: float | None = tap.arg(default=None, help="Maximum command delay")
    drop_rate: float = tap.arg(default=0.0, help="Drop actions with this probability")
    joint_pos_delta_noise: float = tap.arg(default=0.0, help="Joint position delta noise (degrees)")
    joint_pos_noise: float = tap.arg(default=0.0, help="Joint position noise (degrees)")
    joint_vel_noise: float = tap.arg(default=0.0, help="Joint velocity noise (degrees/second)")
    joint_zero_noise: float = tap.arg(default=0.0, help="Joint zero noise (degrees)")
    accelerometer_noise: float = tap.arg(default=0.0, help="Accelerometer noise (m/s^2)")
    gyroscope_noise: float = tap.arg(default=0.0, help="Gyroscope noise (rad/s)")
    imu_quat_noise: float = tap.arg(default=0.0, help="IMU quaternion noise")


class SimulationServer:
    def __init__(
        self,
        model_path: str | Path,
        model_metadata: RobotURDFMetadataOutput,
        config: ServerConfig,
        key_queue: Queue | None,
        reset_queue: Queue | None,
        pause_queue: Queue | None,
    ) -> None:
        self.simulator = MujocoSimulator(
            model_path=model_path,
            model_metadata=model_metadata,
            dt=config.dt,
            gravity=not config.no_gravity,
            render_mode="offscreen" if config.no_render else "window",
            start_height=config.start_height,
            command_delay_min=config.command_delay_min,
            command_delay_max=config.command_delay_max,
            drop_rate=config.drop_rate,
            joint_pos_delta_noise=config.joint_pos_delta_noise,
            joint_pos_noise=config.joint_pos_noise,
            joint_vel_noise=config.joint_vel_noise,
            joint_zero_noise=config.joint_zero_noise,
            accelerometer_noise=config.accelerometer_noise,
            gyroscope_noise=config.gyroscope_noise,
            imu_quat_noise=config.imu_quat_noise,
            pd_update_frequency=config.pd_update_frequency,
            mujoco_scene=config.mujoco_scene,
            camera=config.camera,
            frame_width=config.frame_width,
            frame_height=config.frame_height,
        )
        self._kinfer_path = config.kinfer_path
        self._stop_event = asyncio.Event()
        self._step_lock = asyncio.Semaphore(1)
        self._video_render_decimation = int(1.0 / config.render_frequency)
        self._quat_name = config.quat_name
        self._acc_name = config.acc_name
        self._gyro_name = config.gyro_name
        self._save_path = Path(config.save_path).expanduser().resolve()
        self._save_video = config.save_video
        self._save_logs = config.save_logs
        self._key_queue = key_queue
        self._reset_queue = reset_queue
        self._pause_queue = pause_queue
        self._is_paused = False
        self._plot_rewards = config.plot_rewards

        # Initialize reward plotter if enabled
        self._reward_plotter = RewardPlotter(self.simulator._model) if self._plot_rewards else None

        self._video_writer: VideoWriter | None = None
        if self._save_video:
            self._save_path.mkdir(parents=True, exist_ok=True)

            fps = round(self.simulator._control_frequency)
            self._video_writer = VideoWriter(self._save_path / "video.mp4", fps=fps)

        # # Check command dimension matches model expectations
        # try:
        #     with tarfile.open(self._kinfer_path, "r:gz") as tar:
        #         metadata_file = tar.extractfile("metadata.json")
        #         if metadata_file is None:
        #             logger.warning("Could not validate command dimension: metadata.json not found in kinfer file")
        #             return

        #         metadata = metadata_from_json(metadata_file.read().decode("utf-8"))
        #         if metadata.num_commands is None:  # type: ignore[attr-defined]
        #             logger.warning("Could not validate command dimension: num_commands not specified in model metadata")
        #             return

        #         expected = metadata.num_commands  # type: ignore[attr-defined]
        #         actual = len(model_provider.command_array)
        #         if actual != expected:
        #             raise ValueError(
        #                 f"Command dimension mismatch: {type(model_provider).__name__} provides command"
        #                 f"with dim {actual} but model expects command with dim {expected}"
        #             )

        # except (tarfile.TarError, FileNotFoundError):
        #     logger.warning("Could not validate commandq dimension: unable to read kinfer file: %s", self._kinfer_path)

    async def _simulation_loop(self) -> None:
        """Run the simulation loop asynchronously."""
        start_time = time.perf_counter()
        last_fps_time = start_time
        num_steps = 0
        fps_update_interval = 1.0  # Update FPS every second
        ctrl_dt = 1.0 / self.simulator._control_frequency

        # Initialize the model runner on the simulator.
        model_provider = ModelProvider(
            self.simulator,
            quat_name=self._quat_name,
            acc_name=self._acc_name,
            gyro_name=self._gyro_name,
            key_queue=self._key_queue,
        )
        model_runner = PyModelRunner(str(self._kinfer_path), model_provider)

        loop = asyncio.get_running_loop()

        carry = model_runner.init()

        logs: list[dict[str, np.ndarray]] | None = None
        if self._save_logs:
            logs = []

        # Start the reward plotter if enabled
        if self._reward_plotter is not None:
            await self._reward_plotter.start()

        try:
            while not self._stop_event.is_set():
                loop_start_time = time.perf_counter()
                
                # Shut down if the viewer is closed.
                if isinstance(self.simulator._viewer, QtViewer):
                    if not self.simulator._viewer.is_open:
                        break

                # handle pause command
                if self._pause_queue is not None and not self._pause_queue.empty():
                    _ = self._pause_queue.get()
                    self._is_paused = not self._is_paused

                # Only simulate step if not paused
                if not self._is_paused:
                    model_provider.arrays.clear()

                    # handle sim reset command
                    if self._reset_queue is not None and not self._reset_queue.empty():
                        await self.simulator.reset()
                        carry = model_runner.init()
                        if self._reward_plotter is not None:
                            await self._reward_plotter.reset()
                        self._reset_queue.get()

                    # Runs the simulation for one step.
                    async with self._step_lock:
                        for _ in range(self.simulator._sim_decimation):
                            await self.simulator.step()

                    # Inference policy, offload blocking calls to the executor
                    output, carry = await loop.run_in_executor(None, model_runner.step, carry)
                    await loop.run_in_executor(None, model_runner.take_action, output)

                    # logging
                    self.simulator._viewer.push_plot_metrics(
                        scalars={f"{self.simulator._model.actuator(i).name}": x for i, x in enumerate(output)},
                        group="actions"
                    )
                    for sub_obs in model_provider.arrays:
                        self.simulator._viewer.push_plot_metrics(
                            scalars={f"{sub_obs}_{i}": x for i, x in enumerate(model_provider.arrays[sub_obs])},
                            group=sub_obs
                        )
                    if self._reward_plotter is not None:
                        await self._reward_plotter.add_data(self.simulator._data, model_provider.arrays, model_provider.heading)

                    if logs is not None:
                        logs.append(model_provider.arrays.copy())

                    if self._video_writer is not None and num_steps % self._video_render_decimation == 0:
                        self._video_writer.append(self.simulator.read_pixels())

                    num_steps += 1

                # Calculate and log FPS
                current_time = time.perf_counter()
                if current_time - last_fps_time >= fps_update_interval:
                    physics_fps = num_steps / (current_time - last_fps_time)
                    logger.info(
                        "Physics FPS: %.2f, Simulation time: %.3f, Wall time: %.3f",
                        physics_fps,
                        self.simulator.sim_time,
                        current_time,
                    )
                    num_steps = 0
                    last_fps_time = current_time

                # Sleep for the remaining time in this control step
                if self._is_paused:
                    await asyncio.sleep(0.01)
                else:
                    elapsed = time.perf_counter() - loop_start_time
                    sleep_duration = max(0, ctrl_dt - elapsed)
                    await asyncio.sleep(sleep_duration)

        except Exception as e:
            logger.error("Simulation loop failed: %s", e)
            logger.error("Traceback: %s", traceback.format_exc())

        finally:
            await self.stop()

            if self._video_writer is not None:
                self._video_writer.close()

            if isinstance(self.simulator._viewer, QtViewer):
                self.simulator._viewer.close()

            if logs is not None:
                save_logs(logs, self._save_path / "logs")

    async def start(self) -> None:
        """Start both the gRPC server and simulation loop asynchronously."""
        sim_task = asyncio.create_task(self._simulation_loop())

        try:
            await sim_task
        except asyncio.CancelledError:
            await self.stop()

    async def stop(self) -> None:
        """Stop the simulation and cleanup resources asynchronously."""
        logger.info("Shutting down simulation...")
        self._stop_event.set()
        if self._reward_plotter is not None:
            await self._reward_plotter.stop()
        await self.simulator.close()


async def get_model_metadata(api: K, model_name: str, cache: bool = True) -> RobotURDFMetadataOutput:
    model_path = get_robots_dir() / model_name / "metadata.json"
    if cache and model_path.exists() and not should_refresh_file(model_path):
        return RobotURDFMetadataOutput.model_validate_json(model_path.read_text())
    model_path.parent.mkdir(parents=True, exist_ok=True)
    robot_class = await api.get_robot_class(model_name)
    metadata = robot_class.metadata
    if metadata is None:
        raise ValueError(f"No metadata found for model {model_name}")
    model_path.write_text(metadata.model_dump_json())
    return metadata


async def serve(config: ServerConfig) -> None:
    async with K() as api:
        model_dir, model_metadata = await asyncio.gather(
            api.download_and_extract_urdf(config.mujoco_model_name, cache=(not config.no_cache)),
            get_model_metadata(api, config.mujoco_model_name),
        )

    model_path = next(
        (
            path
            for path in itertools.chain(
                model_dir.glob("*.mjcf"),
                model_dir.glob("*.xml"),
            )
        )
    )

    key_queue, reset_queue, pause_queue = None, None, None
    if config.use_keyboard:
        keyboard_listener = KeyboardListener()
        key_queue, reset_queue, pause_queue = keyboard_listener.get_queues()

    server = SimulationServer(
        model_path=model_path,
        model_metadata=model_metadata,
        config=config,
        key_queue=key_queue,
        reset_queue=reset_queue,
        pause_queue=pause_queue,
    )

    await server.start()


async def run_server(config: ServerConfig) -> None:
    await serve(config=config)


def runner(args: ServerConfig) -> None:
    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)
    asyncio.run(run_server(config=args))


def main() -> None:
    tap.Parser(ServerConfig).bind(runner).run()


if __name__ == "__main__":
    # python -m kinfer_sim.server
    main()
