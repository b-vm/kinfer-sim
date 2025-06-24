"""Defines a K-Infer model provider for the Mujoco simulator."""

import logging
from typing import Sequence, cast
from queue import Queue

import numpy as np
from kinfer.rust_bindings import ModelProviderABC, PyModelMetadata

from kinfer_sim.simulator import MujocoSimulator

logger = logging.getLogger(__name__)


def ensure_quat_in_positive_hemisphere(quat_4: np.ndarray) -> np.ndarray:
    """Ensures a quaternion is in the positive hemisphere of the quaternion space."""
    return np.where(quat_4[..., 0] < 0, -quat_4, quat_4)


def euler_to_quat(euler_3: np.ndarray) -> np.ndarray:
    """Converts roll, pitch, yaw angles to a quaternion (w, x, y, z).

    Args:
        euler_3: The roll, pitch, yaw angles, shape (*, 3).

    Returns:
        The quaternion with shape (*, 4).
    """
    # Extract roll, pitch, yaw from input
    roll, pitch, yaw = np.split(euler_3, 3, axis=-1)

    # Calculate trigonometric functions for each angle
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    # Calculate quaternion components using the conversion formula
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    # Combine into quaternion [w, x, y, z]
    quat = np.concatenate([w, x, y, z], axis=-1)

    # Normalize the quaternion
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)

    return quat


def rotate_quat_by_quat(quat_to_rotate: np.ndarray, rotating_quat: np.ndarray, inverse: bool = False, eps: float = 1e-6) -> np.ndarray:
    """Rotates one quaternion by another quaternion through quaternion multiplication.
    
    This performs the operation: rotating_quat * quat_to_rotate * rotating_quat^(-1) if inverse=False
    or rotating_quat^(-1) * quat_to_rotate * rotating_quat if inverse=True
    
    Args:
        quat_to_rotate: The quaternion being rotated (w,x,y,z), shape (*, 4)
        rotating_quat: The quaternion performing the rotation (w,x,y,z), shape (*, 4)
        inverse: If True, rotate by the inverse of rotating_quat
        eps: Small epsilon value to avoid division by zero in normalization
        
    Returns:
        The rotated quaternion (w,x,y,z), shape (*, 4)
    """
    # Normalize both quaternions
    quat_to_rotate = quat_to_rotate / (np.linalg.norm(quat_to_rotate, axis=-1, keepdims=True) + eps)
    rotating_quat = rotating_quat / (np.linalg.norm(rotating_quat, axis=-1, keepdims=True) + eps)
    
    # If inverse requested, conjugate the rotating quaternion (negate x,y,z components)
    if inverse:
        rotating_quat = np.concatenate([rotating_quat[..., :1], -rotating_quat[..., 1:]], axis=-1)
    
    # Extract components of both quaternions
    w1, x1, y1, z1 = np.split(rotating_quat, 4, axis=-1)        # rotating quaternion 
    w2, x2, y2, z2 = np.split(quat_to_rotate, 4, axis=-1)      # quaternion being rotated
    
    # Quaternion multiplication formula
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    result = np.concatenate([w, x, y, z], axis=-1)
    
    # Normalize result
    return result / (np.linalg.norm(result, axis=-1, keepdims=True) + eps)


def quat_to_euler(quat_4: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalizes and converts a quaternion (w, x, y, z) to roll, pitch, yaw.

    Args:
        quat_4: The quaternion to convert, shape (*, 4).
        eps: A small epsilon value to avoid division by zero.

    Returns:
        The roll, pitch, yaw angles with shape (*, 3).
    """
    quat_4 = quat_4 / (np.linalg.norm(quat_4, axis=-1, keepdims=True) + eps)
    w, x, y, z = np.split(quat_4, 4, axis=-1)

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)

    # Handle edge cases where |sinp| >= 1
    pitch = np.where(
        np.abs(sinp) >= 1.0,
        np.sign(sinp) * np.pi / 2.0,  # Use 90 degrees if out of range
        np.arcsin(sinp),
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.concatenate([roll, pitch, yaw], axis=-1)


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
        self.heading = None
        return self
    
    def process_key_queue(self):
        # Read keystrokes and update command array
        while not self.key_queue.empty():
            key = self.key_queue.get()
            key = key.strip("'")

            # reset commands
            if key == '0':
                self.command_array *= 0
            elif key == 'key.backspace':
                self.command_array *= 0
                self.heading = None # lazy init this to prevent timing issues
            
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
            elif input_type == "initial_heading":
                inputs[input_type] = self.get_initial_heading()
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

        if self.heading == None: # lazy init heading after we are sure sim has been reset
            self.heading = quat_to_euler(quat_array)[2]

        quat_array += np.random.normal(
            -self.simulator._imu_quat_noise, self.simulator._imu_quat_noise, quat_array.shape
        )

        self.arrays["quaternion"] = quat_array
        return quat_array

    def get_initial_heading(self) -> np.ndarray:
        if self.heading == None: # lazy init heading after we are sure sim has been reset
            sensor = self.simulator._data.sensor(self.quat_name)
            quat_array = np.array(sensor.data, dtype=np.float32)
            self.heading = quat_to_euler(quat_array)[2]
        return np.array([self.heading], dtype=np.float32)

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

        if self.heading is not None:
            self.heading += self.command_array[2] * self.simulator._control_dt

        command_obs = np.concatenate([
            self.command_array[:3],
            np.zeros(1), # mask out carried heading
            self.command_array[3:],
        ])

        self.arrays["command"] = command_obs
        return command_obs

    def take_action(self, action: np.ndarray, metadata: PyModelMetadata) -> None:
        joint_names = metadata.joint_names  # type: ignore[attr-defined]
        assert action.shape == (len(joint_names),)
        self.arrays["action"] = action
        self.simulator.command_actuators({name: {"position": action[i]} for i, name in enumerate(joint_names)})
