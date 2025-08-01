"""Defines motion sequences for robot arm movements."""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

# Define the joint names and their default positions for KBOT
COMMANDS = [
    "xvel",
    "yvel",
    "angvel",
]
POSITIONS = [
    "base_height",
    "base_roll",
    "base_pitch",
    "dof_right_shoulder_pitch_03",
    "dof_right_shoulder_roll_03", 
    "dof_right_shoulder_yaw_02",
    "dof_right_elbow_02",
    "dof_right_wrist_00",
    "dof_left_shoulder_pitch_03",
    "dof_left_shoulder_roll_03",
    "dof_left_shoulder_yaw_02",
    "dof_left_elbow_02",
    "dof_left_wrist_00"
]

@dataclass
class Keyframe:
    """A keyframe in a motion sequence."""
    time: float
    positions: Dict[str, float]
    commands: Dict[str, float]

class Motion:
    """Represents a sequence of arm motions with keyframes and interpolation."""
    
    def __init__(self, keyframes: List[Keyframe], dt: float):
        """Initialize a motion sequence.
        
        Args:
            keyframes: List of keyframes defining the motion
            dt: Time step for interpolation
        """
        self.keyframes = sorted(keyframes, key=lambda k: k.time)
        self.total_duration = self.keyframes[-1].time
        self.dt = dt
        self.current_time = 0.0
        
    def get_next_motion_frame(self) -> Tuple[np.ndarray, np.ndarray] | None:
        """Get the next motion frame.
        
        Returns:
            Tuple of (commands, positions) as numpy arrays,
            or None if sequence is complete
        """
        if self.current_time > self.total_duration:
            return None
            
        # Find surrounding keyframes
        next_idx = 0
        while (next_idx < len(self.keyframes) and 
               self.keyframes[next_idx].time < self.current_time):
            next_idx += 1
            
        # Get interpolated positions
        if next_idx == 0:
            positions = self.keyframes[0].positions.copy()
            commands = self.keyframes[0].commands.copy()
        elif next_idx >= len(self.keyframes):
            positions = self.keyframes[-1].positions.copy()
            commands = self.keyframes[-1].commands.copy()
        else:
            # Interpolate between keyframes
            prev_frame = self.keyframes[next_idx - 1]
            next_frame = self.keyframes[next_idx]
            
            alpha = ((self.current_time - prev_frame.time) / 
                    (next_frame.time - prev_frame.time))
            
            positions = {}
            for joint in POSITIONS:
                prev_pos = prev_frame.positions.get(joint, 0.0)
                next_pos = next_frame.positions.get(joint, 0.0)
                positions[joint] = prev_pos + alpha * (next_pos - prev_pos)
            
            # Use the previous keyframe's commands
            commands = prev_frame.commands.copy()
        
        # Convert to ordered numpy arrays
        commands_array = np.array([commands.get(cmd, 0.0) for cmd in COMMANDS])
        positions_array = np.array([positions.get(joint, 0.0) for joint in POSITIONS])
        
        self.current_time += self.dt
        return commands_array, positions_array
    
    def reset(self):
        """Reset the motion sequence to start."""
        self.current_time = 0.0

def create_test_motion(joint_name: str, dt: float = 0.01) -> Motion:
    """Creates a test motion for a joint: 0° -> -90° -> 90° -> 0°
    
    Args:
        joint_name: Name of the joint to test
        dt: Time step between frames
    """
    keyframes = [
        Keyframe(
            time=0.0,
            positions={joint_name: math.radians(0.0)},
            commands={}
        ),
        Keyframe(
            time=1.0,
            positions={joint_name: math.radians(-90.0)},
            commands={}
        ),
        Keyframe(
            time=2.0,
            positions={joint_name: math.radians(90.0)},
            commands={}
        ),
        Keyframe(
            time=3.0,
            positions={joint_name: math.radians(0.0)},
            commands={}
        ),
    ]
    return Motion(keyframes, dt=dt)

def create_wave(dt: float = 0.01) -> Motion:
    """Creates a waving motion sequence."""
    keyframes = [
        Keyframe(
            time=0.0,
            positions={
                "dof_right_shoulder_roll_03": math.radians(-45.0),
                "dof_right_shoulder_yaw_02": 0.0,
                "dof_right_elbow_02": math.radians(90.0),
            },
            commands={}
        ),
        Keyframe(
            time=0.5,
            positions={
                "dof_right_shoulder_roll_03": math.radians(-45.0),
                "dof_right_shoulder_yaw_02": math.radians(45.0),
                "dof_right_elbow_02": math.radians(90.0),
            },
            commands={}
        ),
        Keyframe(
            time=1.0,
            positions={
                "dof_right_shoulder_roll_03": math.radians(-45.0),
                "dof_right_shoulder_yaw_02": math.radians(-45.0),
                "dof_right_elbow_02": math.radians(90.0),
            },
            commands={}
        ),
        Keyframe(
            time=1.5,
            positions={
                "dof_right_shoulder_roll_03": math.radians(-10.0),
                "dof_right_shoulder_yaw_02": 0.0,
                "dof_right_elbow_02": math.radians(90.0),
            },
            commands={}
        ),
    ]
    return Motion(keyframes, dt=dt)

def create_salute(dt: float = 0.01) -> Motion:
    """Creates a saluting motion sequence."""
    keyframes = [
        Keyframe(
            time=0.6,
            positions={
                "dof_right_shoulder_roll_03": math.radians(-90.0),
                "dof_right_elbow_02": math.radians(0.0),
            },
            commands={}
        ),
        Keyframe(
            time=1.1,
            positions={
                "dof_right_shoulder_roll_03": math.radians(-90.0),
                "dof_right_elbow_02": math.radians(90.0),
            },
            commands={}
        ),
        Keyframe(
            time=2.1,
            positions={
                "dof_right_shoulder_roll_03": math.radians(-90.0),
                "dof_right_elbow_02": math.radians(90.0),
            },
            commands={}
        ),
        Keyframe(
            time=2.6,
            positions={
                "dof_right_shoulder_roll_03": math.radians(-10.0),
                "dof_right_elbow_02": math.radians(0.0),
            },
            commands={}
        ),
    ]
    return Motion(keyframes, dt=dt)

def create_pickup(dt: float = 0.01) -> Motion:
    """Creates a pickup motion sequence."""
    keyframes = [
        Keyframe(
            time=0.0,
            positions={
                "dof_right_shoulder_pitch_03": 0.0,
                "dof_right_shoulder_roll_03": math.radians(10.0),
                "dof_right_elbow_02": 0.0,
                "dof_right_wrist_00": 0.0,
                "dof_left_shoulder_pitch_03": 0.0,
                "dof_left_shoulder_roll_03": math.radians(-10.0),
                "dof_left_elbow_02": 0.0,
                "dof_left_wrist_00": 0.0,
            },
            commands={}
        ),
        Keyframe(
            time=0.5,
            positions={
                "dof_right_shoulder_pitch_03": math.radians(-45.0),
                "dof_right_shoulder_roll_03": math.radians(20.0),
                "dof_right_elbow_02": math.radians(-10.0),
                "dof_right_wrist_00": 0.0,
                "dof_left_shoulder_pitch_03": math.radians(45.0),
                "dof_left_shoulder_roll_03": math.radians(-20.0),
                "dof_left_elbow_02": math.radians(10.0),
                "dof_left_wrist_00": 0.0,
                "base_pitch": math.radians(15.0),
            },
            commands={}
        ),
        Keyframe(
            time=1.0,
            positions={
                "dof_right_shoulder_pitch_03": math.radians(-90.0),
                "dof_right_shoulder_roll_03": math.radians(20.0),
                "dof_right_elbow_02": math.radians(-45.0),
                "dof_right_wrist_00": math.radians(20.0),
                "dof_left_shoulder_pitch_03": math.radians(90.0),
                "dof_left_shoulder_roll_03": math.radians(-20.0),
                "dof_left_elbow_02": math.radians(45.0),
                "dof_left_wrist_00": math.radians(-20.0),
                "base_height": -0.1,
                "base_pitch": math.radians(30.0),
            },
            commands={}
        ),
        Keyframe(
            time=1.3,
            positions={
                "dof_right_shoulder_pitch_03": math.radians(-90.0),
                "dof_right_shoulder_roll_03": math.radians(20.0),
                "dof_right_elbow_02": math.radians(-90.0),
                "dof_right_wrist_00": math.radians(30.0),
                "dof_left_shoulder_pitch_03": math.radians(90.0),
                "dof_left_shoulder_roll_03": math.radians(-20.0),
                "dof_left_elbow_02": math.radians(90.0),
                "dof_left_wrist_00": math.radians(-30.0),
                "base_height": -0.1,
                "base_pitch": math.radians(30.0),
            },
            commands={}
        ),
        Keyframe(
            time=1.8,
            positions={
                "dof_right_shoulder_pitch_03": math.radians(-45.0),
                "dof_right_shoulder_roll_03": math.radians(20.0),
                "dof_right_elbow_02": math.radians(-90.0),
                "dof_right_wrist_00": math.radians(30.0),
                "dof_left_shoulder_pitch_03": math.radians(45.0),
                "dof_left_shoulder_roll_03": math.radians(-20.0),
                "dof_left_elbow_02": math.radians(90.0),
                "dof_left_wrist_00": math.radians(-30.0),
                "base_pitch": math.radians(15.0),
            },
            commands={}
        ),
        Keyframe(
            time=2.3,
            positions={
                "dof_right_shoulder_pitch_03": 0.0,
                "dof_right_shoulder_roll_03": math.radians(10.0),
                "dof_right_elbow_02": 0.0,
                "dof_right_wrist_00": 0.0,
                "dof_left_shoulder_pitch_03": 0.0,
                "dof_left_shoulder_roll_03": math.radians(-10.0),
                "dof_left_elbow_02": 0.0,
                "dof_left_wrist_00": 0.0,
            },
            commands={}
        ),
    ]
    return Motion(keyframes, dt=dt)

def create_wild_walk(dt: float = 0.01) -> Motion:
    """Creates a wild walking motion with extreme arm movements."""
    keyframes = [
        Keyframe(
            time=0.0,
            positions={
                "dof_right_shoulder_pitch_03": 0.0,
                "dof_right_shoulder_roll_03": 0.0,
                "dof_right_shoulder_yaw_02": 0.0,
                "dof_right_elbow_02": 0.0,
                "dof_left_shoulder_pitch_03": 0.0,
                "dof_left_shoulder_roll_03": 0.0,
                "dof_left_shoulder_yaw_02": 0.0,
                "dof_left_elbow_02": 0.0,
            },
            commands={
                "xvel": 0.0,
                "yvel": 0.0,
                "angvel": 0.0,
            }
        ),
        # Start moving forward and spin while raising arms
        Keyframe(
            time=1.0,
            positions={
                "dof_right_shoulder_pitch_03": math.radians(-135.0),  # Extreme up and back
                "dof_right_shoulder_roll_03": math.radians(-90.0),    # Out to side
                "dof_right_shoulder_yaw_02": math.radians(90.0),      # Rotated forward
                "dof_right_elbow_02": math.radians(120.0),           # Sharp bend
                "dof_left_shoulder_pitch_03": math.radians(135.0),    # Mirror of right
                "dof_left_shoulder_roll_03": math.radians(90.0),
                "dof_left_shoulder_yaw_02": math.radians(-90.0),
                "dof_left_elbow_02": math.radians(-120.0),
            },
            commands={
                "xvel": 0.5,      # Move forward
                "yvel": 0.0,
                "angvel": 1.0,    # Spin clockwise
            }
        ),
        # Move sideways while windmilling arms
        Keyframe(
            time=2.0,
            positions={
                "dof_right_shoulder_pitch_03": math.radians(90.0),    # Forward
                "dof_right_shoulder_roll_03": math.radians(45.0),     # Diagonal up
                "dof_right_shoulder_yaw_02": math.radians(-90.0),     # Rotated back
                "dof_right_elbow_02": math.radians(-90.0),           # Sharp bend opposite
                "dof_left_shoulder_pitch_03": math.radians(-90.0),    # Mirror
                "dof_left_shoulder_roll_03": math.radians(-45.0),
                "dof_left_shoulder_yaw_02": math.radians(90.0),
                "dof_left_elbow_02": math.radians(90.0),
            },
            commands={
                "xvel": 0.0,
                "yvel": 0.5,      # Move right
                "angvel": -1.0,   # Spin counter-clockwise
            }
        ),
        # Diagonal movement with arms in maximum angles
        Keyframe(
            time=3.0,
            positions={
                "dof_right_shoulder_pitch_03": math.radians(-180.0),  # Maximum back
                "dof_right_shoulder_roll_03": math.radians(-120.0),   # Maximum out
                "dof_right_shoulder_yaw_02": math.radians(180.0),     # Maximum rotation
                "dof_right_elbow_02": math.radians(145.0),           # Maximum bend
                "dof_left_shoulder_pitch_03": math.radians(180.0),    # Mirror maxima
                "dof_left_shoulder_roll_03": math.radians(120.0),
                "dof_left_shoulder_yaw_02": math.radians(-180.0),
                "dof_left_elbow_02": math.radians(-145.0),
            },
            commands={
                "xvel": 0.5,      # Move diagonal
                "yvel": -0.5,     # Move diagonal
                "angvel": 2.0,    # Fast spin
            }
        ),
        # Return to neutral
        Keyframe(
            time=4.0,
            positions={
                "dof_right_shoulder_pitch_03": 0.0,
                "dof_right_shoulder_roll_03": 0.0,
                "dof_right_shoulder_yaw_02": 0.0,
                "dof_right_elbow_02": 0.0,
                "dof_left_shoulder_pitch_03": 0.0,
                "dof_left_shoulder_roll_03": 0.0,
                "dof_left_shoulder_yaw_02": 0.0,
                "dof_left_elbow_02": 0.0,
            },
            commands={
                "xvel": 0.0,
                "yvel": 0.0,
                "angvel": 0.0,
            }
        ),
    ]
    return Motion(keyframes, dt=dt)

def create_zombie_walk(dt: float = 0.01) -> Motion:
    """Creates a classic zombie shambling motion with stiff arms."""
    keyframes = [
        # Start pose - classic zombie arms out
        Keyframe(
            time=0.0,
            positions={
                "dof_right_shoulder_pitch_03": math.radians(30.0),    # Arms forward
                "dof_right_shoulder_roll_03": math.radians(-30.0),    # Slightly out
                "dof_right_shoulder_yaw_02": math.radians(-20.0),     # Rotated down
                "dof_right_elbow_02": math.radians(-10.0),           # Slightly bent
                "dof_right_wrist_00": math.radians(-30.0),           # Limp wrist
                "dof_left_shoulder_pitch_03": math.radians(30.0),
                "dof_left_shoulder_roll_03": math.radians(30.0),
                "dof_left_shoulder_yaw_02": math.radians(20.0),
                "dof_left_elbow_02": math.radians(10.0),
                "dof_left_wrist_00": math.radians(30.0),
                "base_pitch": math.radians(15.0),                     # Leaning forward
            },
            commands={
                "xvel": 0.0,
                "yvel": 0.0,
                "angvel": 0.0,
            }
        ),
        # Shamble right with tilt
        Keyframe(
            time=1.0,
            positions={
                "dof_right_shoulder_pitch_03": math.radians(45.0),
                "dof_right_shoulder_roll_03": math.radians(-40.0),
                "dof_right_shoulder_yaw_02": math.radians(-30.0),
                "dof_right_elbow_02": math.radians(-15.0),
                "dof_right_wrist_00": math.radians(-40.0),
                "dof_left_shoulder_pitch_03": math.radians(20.0),
                "dof_left_shoulder_roll_03": math.radians(20.0),
                "dof_left_shoulder_yaw_02": math.radians(10.0),
                "dof_left_elbow_02": math.radians(5.0),
                "dof_left_wrist_00": math.radians(20.0),
                "base_roll": math.radians(10.0),                      # Tilt right
                "base_pitch": math.radians(20.0),
            },
            commands={
                "xvel": 0.2,                                         # Slow forward
                "yvel": 0.1,                                         # Drift right
                "angvel": -0.2,                                      # Slight turn
            }
        ),
        # Shamble left with tilt
        Keyframe(
            time=2.0,
            positions={
                "dof_right_shoulder_pitch_03": math.radians(20.0),
                "dof_right_shoulder_roll_03": math.radians(-20.0),
                "dof_right_shoulder_yaw_02": math.radians(-10.0),
                "dof_right_elbow_02": math.radians(-5.0),
                "dof_right_wrist_00": math.radians(-20.0),
                "dof_left_shoulder_pitch_03": math.radians(45.0),
                "dof_left_shoulder_roll_03": math.radians(40.0),
                "dof_left_shoulder_yaw_02": math.radians(30.0),
                "dof_left_elbow_02": math.radians(15.0),
                "dof_left_wrist_00": math.radians(40.0),
                "base_roll": math.radians(-10.0),                     # Tilt left
                "base_pitch": math.radians(20.0),
            },
            commands={
                "xvel": 0.2,
                "yvel": -0.1,                                        # Drift left
                "angvel": 0.2,
            }
        ),
        # Lunge forward
        Keyframe(
            time=2.5,
            positions={
                "dof_right_shoulder_pitch_03": math.radians(60.0),    # Arms reach forward
                "dof_right_shoulder_roll_03": math.radians(-45.0),
                "dof_right_shoulder_yaw_02": math.radians(-20.0),
                "dof_right_elbow_02": math.radians(-5.0),
                "dof_right_wrist_00": math.radians(-45.0),
                "dof_left_shoulder_pitch_03": math.radians(60.0),
                "dof_left_shoulder_roll_03": math.radians(45.0),
                "dof_left_shoulder_yaw_02": math.radians(20.0),
                "dof_left_elbow_02": math.radians(5.0),
                "dof_left_wrist_00": math.radians(45.0),
                "base_pitch": math.radians(30.0),                     # Strong forward lean
            },
            commands={
                "xvel": 0.4,                                         # Faster lunge
                "yvel": 0.0,
                "angvel": 0.0,
            }
        ),
        # Return to start pose with slight variation
        Keyframe(
            time=3.0,
            positions={
                "dof_right_shoulder_pitch_03": math.radians(35.0),
                "dof_right_shoulder_roll_03": math.radians(-35.0),
                "dof_right_shoulder_yaw_02": math.radians(-25.0),
                "dof_right_elbow_02": math.radians(-15.0),
                "dof_right_wrist_00": math.radians(-35.0),
                "dof_left_shoulder_pitch_03": math.radians(35.0),
                "dof_left_shoulder_roll_03": math.radians(35.0),
                "dof_left_shoulder_yaw_02": math.radians(25.0),
                "dof_left_elbow_02": math.radians(15.0),
                "dof_left_wrist_00": math.radians(35.0),
                "base_roll": math.radians(0.0),
                "base_pitch": math.radians(15.0),
            },
            commands={
                "xvel": 0.1,                                         # Slow again
                "yvel": 0.0,
                "angvel": 0.0,
            }
        ),
    ]
    return Motion(keyframes, dt=dt)

# Dictionary mapping motion names to their creation functions
MOTIONS = {
    'wave': create_wave,
    'salute': create_salute,
    'pickup': create_pickup,
    'wild_walk': create_wild_walk,
    'zombie_walk': create_zombie_walk,
    # Test motions - automatically generate test functions for each joint
    **{
        f'test_{"".join(word[0].lower() for word in joint_name.split("_")[1:-1])}': 
            lambda dt=0.01, joint=joint_name: create_test_motion(joint, dt)
        for joint_name in POSITIONS[3:]
    }
}