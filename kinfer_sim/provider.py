"""Defines a K-Infer model provider for the Mujoco simulator."""

import logging
from typing import Sequence, cast
from queue import Queue

import numpy as np
from kinfer.rust_bindings import ModelProviderABC, PyModelMetadata

from kinfer_sim.simulator import MujocoSimulator

logger = logging.getLogger(__name__)


def rotate_vector_by_quat(vector: np.ndarray, quat: np.ndarray, inverse: bool = False, eps: float = 1e-6) -> np.ndarray:
    """Rotates a vector by a quaternion.

    Args:
        vector: The vector to rotate, shape (*, 3).
        quat: The quaternion to rotate by, shape (*, 4).
        inverse: If True, rotate the vector by the conjugate of the quaternion.
        eps: A small epsilon value to avoid division by zero.

    Returns:
        The rotated vector, shape (*, 3).
    """
    # Normalize quaternion
    quat = quat / (np.linalg.norm(quat, axis=-1, keepdims=True) + eps)
    w, x, y, z = np.split(quat, 4, axis=-1)

    if inverse:
        x, y, z = -x, -y, -z

    # Extract vector components
    vx, vy, vz = np.split(vector, 3, axis=-1)

    # Terms for x component
    xx = (
        w * w * vx
        + 2 * y * w * vz
        - 2 * z * w * vy
        + x * x * vx
        + 2 * y * x * vy
        + 2 * z * x * vz
        - z * z * vx
        - y * y * vx
    )

    # Terms for y component
    yy = (
        2 * x * y * vx
        + y * y * vy
        + 2 * z * y * vz
        + 2 * w * z * vx
        - z * z * vy
        + w * w * vy
        - 2 * w * x * vz
        - x * x * vy
    )

    # Terms for z component
    zz = (
        2 * x * z * vx
        + 2 * y * z * vy
        + z * z * vz
        - 2 * w * y * vx
        + w * w * vz
        + 2 * w * x * vy
        - y * y * vz
        - x * x * vz
    )

    return np.concatenate([xx, yy, zz], axis=-1)


class ModelProvider(ModelProviderABC):
    simulator: MujocoSimulator
    quat_name: str
    acc_name: str
    gyro_name: str
    arrays: dict[str, np.ndarray]
    key_queue: Queue | None

    def __new__(
        cls,
        simulator: MujocoSimulator,
        key_queue: Queue | None,
        quat_name: str = "imu_site_quat",
        acc_name: str = "imu_acc",
        gyro_name: str = "imu_gyro",
    ) -> "ModelProvider":
        self = cast(ModelProvider, super().__new__(cls))
        self.simulator = simulator
        self.quat_name = quat_name
        self.acc_name = acc_name
        self.gyro_name = gyro_name
        self.arrays = {}
        self.key_queue = key_queue
        self.command_array = np.zeros(6) # vx vy wz base_height roll pitch
        return self
    
    def process_key_queue(self):
        # Read keystrokes and update command array
        while not self.key_queue.empty():
            key = self.key_queue.get()
            key = key.strip("'")

            # reset commands
            if key == '0' or key == 'key.backspace':
                self.command_array *= 0
            
            # lin vel
            elif key == 'w':
                self.command_array[0] += 0.1
            elif key == 's':
                self.command_array[0] -= 0.1
            elif key == 'a':
                self.command_array[1] += 0.1
            elif key == 'd':
                self.command_array[1] -= 0.1

            # ang vel
            elif key == 'q':
                self.command_array[2] += 0.1
            elif key == 'e':
                self.command_array[2] -= 0.1

            # height
            elif key == '=':
                self.command_array[3] += 0.05
            elif key == '-':
                self.command_array[3] -= 0.05
            
            # base orient
            elif key == 'r':
                self.command_array[4] += 0.1
            elif key == 'f':
                self.command_array[4] -= 0.1
            elif key == 't':
                self.command_array[5] += 0.1
            elif key == 'g':
                self.command_array[5] -= 0.1

    def get_inputs(self, input_types: Sequence[str], metadata: PyModelMetadata) -> dict[str, np.ndarray]:
        """Get inputs for the model based on the requested input types.

        Args:
            input_types: List of input type names to retrieve
            metadata: Model metadata containing joint names and other info

        Returns:
            Dictionary mapping input type names to numpy arrays
        """
        inputs = {}

        for input_type in input_types:
            if input_type == "joint_angles":
                inputs[input_type] = self.get_joint_angles(metadata.joint_names)  # type: ignore[attr-defined]
            elif input_type == "joint_angular_velocities":
                inputs[input_type] = self.get_joint_angular_velocities(metadata.joint_names)  # type: ignore[attr-defined]
            elif input_type == "quaternion":
                inputs[input_type] = self.get_quaternion()
            elif input_type == "projected_gravity":
                inputs[input_type] = self.get_projected_gravity()
            elif input_type == "accelerometer":
                inputs[input_type] = self.get_accelerometer()
            elif input_type == "gyroscope":
                inputs[input_type] = self.get_gyroscope()
            elif input_type == "command":
                inputs[input_type] = self.get_command()
            elif input_type == "time":
                inputs[input_type] = self.get_time()
            else:
                raise ValueError(f"Unknown input type: {input_type}")

        return inputs

    def get_joint_angles(self, joint_names: Sequence[str]) -> np.ndarray:
        angles = [float(self.simulator._data.joint(joint_name).qpos) for joint_name in joint_names]
        angles_array = np.array(angles, dtype=np.float32)
        angles_array += np.random.normal(
            -self.simulator._joint_pos_noise, self.simulator._joint_pos_noise, angles_array.shape
        )
        self.arrays["joint_angles"] = angles_array
        return angles_array

    def get_joint_angular_velocities(self, joint_names: Sequence[str]) -> np.ndarray:
        velocities = [float(self.simulator._data.joint(joint_name).qvel) for joint_name in joint_names]
        velocities_array = np.array(velocities, dtype=np.float32)
        velocities_array += np.random.normal(
            -self.simulator._joint_vel_noise, self.simulator._joint_vel_noise, velocities_array.shape
        )
        self.arrays["joint_velocities"] = velocities_array
        return velocities_array

    def get_quaternion(self) -> np.ndarray:
        sensor = self.simulator._data.sensor(self.quat_name)
        quat_array = np.array(sensor.data, dtype=np.float32)

        quat_array += np.random.normal(
            -self.simulator._imu_quat_noise, self.simulator._imu_quat_noise, quat_array.shape
        )

        self.arrays["quaternion"] = quat_array
        return quat_array

    def get_projected_gravity(self) -> np.ndarray:
        gravity = self.simulator._model.opt.gravity
        quat_name = self.quat_name
        sensor = self.simulator._data.sensor(quat_name)
        proj_gravity = rotate_vector_by_quat(gravity, sensor.data, inverse=True)
        self.arrays["projected_gravity"] = proj_gravity
        return proj_gravity

    def get_accelerometer(self) -> np.ndarray:
        sensor = self.simulator._data.sensor(self.acc_name)
        acc_array = np.array(sensor.data, dtype=np.float32)
        acc_array += np.random.normal(
            -self.simulator._accelerometer_noise, self.simulator._accelerometer_noise, acc_array.shape
        )
        self.arrays["accelerometer"] = acc_array
        return acc_array

    def get_gyroscope(self) -> np.ndarray:
        sensor = self.simulator._data.sensor(self.gyro_name)
        gyro_array = np.array(sensor.data, dtype=np.float32)
        gyro_array += np.random.normal(
            -self.simulator._gyroscope_noise, self.simulator._gyroscope_noise, gyro_array.shape
        )
        self.arrays["gyroscope"] = gyro_array
        return gyro_array

    def get_time(self) -> np.ndarray:
        time = self.simulator._data.time
        time_array = np.array([time], dtype=np.float32)
        self.arrays["time"] = time_array
        return time_array

    def get_command(self) -> np.ndarray:
        # Process any queued keyboard commands
        if self.key_queue is not None: 
            self.process_key_queue()
            # TODO move cmd printing to kmv
            logging.info(f"Command: \033[31mVx={self.command_array[0]:.2f}\033[0m, "
                        f"\033[32mVy={self.command_array[1]:.2f}\033[0m, "
                        f"\033[33mωz={self.command_array[2]:.2f}\033[0m, "
                        f"\033[34mbaseheight={self.command_array[3]:.2f}\033[0m, "
                        f"\033[36mbaseroll={self.command_array[4]:.2f}\033[0m, "
                        f"\033[35mbasepitch={self.command_array[5]:.2f}\033[0m")

        command_obs = np.concatenate([
            self.command_array,
        ])

        # if self.command_array[0] > 0: 
        #     command_obs = np.array([0, 1, 0, 0]) # walk
        # elif self.command_array[2] > 0:
        #     command_obs = np.array([0, 0, 1, 0]) # turn left
        # elif self.command_array[2] < 0:
        #     command_obs = np.array([0, 0, 0, 1]) # turn right
        # else:
        #     command_obs = np.array([1, 0, 0, 0]) # stand

        self.arrays["command"] = command_obs
        return command_obs

    def take_action(self, action: np.ndarray, metadata: PyModelMetadata) -> None:
        joint_names = metadata.joint_names  # type: ignore[attr-defined]
        assert action.shape == (len(joint_names),)
        self.arrays["action"] = action
        self.simulator.command_actuators({name: {"position": action[i]} for i, name in enumerate(joint_names)})
