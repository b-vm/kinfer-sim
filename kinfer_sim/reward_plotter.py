import asyncio
import importlib.util
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
from jaxtyping import Array
import pyqtgraph as pg
import numpy as np
import mujoco

import xax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Trajectory:
    """ Simplified trajectory class mimicking the Trajectory class in ksim.types"""

    qpos: Array
    qvel: Array
    xpos: Array
    xquat: Array
    ctrl: Array
    obs: dict[str, Array]
    command: dict[str, Array]
    # event_state: xax.FrozenDict[str, Array]
    action: Array
    done: Array
    # success: Array
    # timestep: Array
    # termination_components: xax.FrozenDict[str, Array]
    # aux_outputs: xax.FrozenDict[str, PyTree]


class RewardPlotter:
    def __init__(self, mujoco_model: mujoco.MjModel):
        path_to_train_file = "/home/bart/kscale/kbot-joystick/train.py"
        # path_to_train_file = "/home/bart/kscale/kbot-walking/train.py"

        # replace error line
        with open(path_to_train_file, 'r') as f:
            content = f.read()
        modified_content = content.replace(
            'FeetAirtimeReward(scale=0.8, ctrl_dt=self.config.ctrl_dt, touchdown_penalty=0.4),',
            'FeetAirtimeReward(scale=0.8, ctrl_dt=0.02, touchdown_penalty=0.4),'
        ).replace(
            'SingleFootContactReward(scale=0.5, ctrl_dt=self.config.ctrl_dt, grace_period=0.1),',
            'SingleFootContactReward(scale=0.5, ctrl_dt=0.02, grace_period=0.1),'
        )
        temp_path = '/tmp/modified_train.py'
        with open(temp_path, 'w') as f:
            f.write(modified_content)

        spec = importlib.util.spec_from_file_location("train", temp_path)
        train = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train)
        
        # Get the actual rewards being used in train.py
        self.rewards = train.HumanoidWalkingTask.get_rewards(self=None, physics_model=mujoco_model)
        self.rewards = {reward.__class__.__name__: reward for reward in self.rewards}
        print("\n=== Found Reward Classes ===")
        for i, (reward_name, reward) in enumerate(self.rewards.items(), 1):
            print(f"\n{i}. {reward_name}")
            print(f"   {reward.__doc__ or 'No description available'}")
        print("\n" + "=" * 30 + "\n")

        # Initialize PyQtPlot window and widgets
        self.app = pg.mkQApp()
        self.win = pg.GraphicsLayoutWidget()
        
        # Create dictionaries to store plots, curves and data
        self.traj_data, self.plots, self.curves, self.plot_data = {}, {}, {}, {}

        self.setup_plots()
        self.win.show()
        
        # Create a queue for communication between sim and plot threads
        self.plot_queue = asyncio.Queue()
        
        # Data processing and rendering flags
        self.data_needs_update = False

        # Start the tasks
        self.data_task = None
        self.render_task = None

        self.executor = ThreadPoolExecutor(max_workers=1)

    def setup_plots(self):
        def make_plot(name: str):
            self.plots[name] = self.win.addPlot(title=name.capitalize())
            self.win.nextRow()
            self.plots[name].setXLink(self.plots['Total Reward'])
            self.plots[name].showGrid(x=True, y=True, alpha=0.3)
            self.curves[name] = self.plots[name].plot(pen='y')
            self.plot_data[name] = []

        # Setup reward plots
        make_plot('Total Reward')
        for name in self.rewards.keys():
            make_plot(name)

        # Setup command + obsplots
        additional_metrics = ['feet_force_touch_observation', 'linvel', 'angvel', 'base_height', 'xyorientation']
        for metric in additional_metrics:
            make_plot(metric)
            self.plots[metric].addLegend()

            if metric == 'linvel':
                self.curves[metric] = {
                    'x_cmd': self.plots[metric].plot(pen=pg.mkPen('r', width=2, style=pg.QtCore.Qt.DashLine), name='X Command'),
                    'x_real': self.plots[metric].plot(pen=pg.mkPen('r', width=2), name='X Actual'),
                    'y_cmd': self.plots[metric].plot(pen=pg.mkPen('g', width=2, style=pg.QtCore.Qt.DashLine), name='Y Command'),
                    'y_real': self.plots[metric].plot(pen=pg.mkPen('g', width=2), name='Y Actual')
                }
            elif metric == 'angvel':
                self.curves[metric] = {
                    'wz_cmd': self.plots[metric].plot(pen=pg.mkPen('y', width=2, style=pg.QtCore.Qt.DashLine), name='ωz Command'),
                    'wz_real': self.plots[metric].plot(pen=pg.mkPen('y', width=2), name='ωz Actual'),
                }
            elif metric == 'base_height':
                self.curves[metric] = {
                    'base_height_cmd': self.plots[metric].plot(pen=pg.mkPen('b', width=2, style=pg.QtCore.Qt.DashLine), name='Height Command'),
                    'base_height_real': self.plots[metric].plot(pen=pg.mkPen('b', width=2), name='Height Actual')
                }
            elif metric == 'xyorientation':
                self.curves[metric] = {
                    'pitch_cmd': self.plots[metric].plot(pen=pg.mkPen('m', width=2, style=pg.QtCore.Qt.DashLine), name='Pitch Command'),
                    # 'pitch_real': self.plots[metric].plot(pen=pg.mkPen('m', width=2), name='Pitch Actual'),
                    'roll_cmd': self.plots[metric].plot(pen=pg.mkPen('c', width=2, style=pg.QtCore.Qt.DashLine), name='Roll Command'),
                    # 'roll_real': self.plots[metric].plot(pen=pg.mkPen('c', width=2), name='Roll Actual')
                }
            elif metric == 'feet_force_touch_observation':
                self.curves[metric] = {
                    'left_foot_force': self.plots[metric].plot(pen=pg.mkPen('r', width=2, style=pg.QtCore.Qt.DashLine), name='Left Foot Force'),
                    'right_foot_force': self.plots[metric].plot(pen=pg.mkPen('g', width=2, style=pg.QtCore.Qt.DashLine), name='Right Foot Force')
                }


    async def start(self):
        """Start both the data processing and rendering tasks"""
        self.running = True
        self.data_task = asyncio.create_task(self._data_loop())
        self.render_task = asyncio.create_task(self._render_loop())

    async def stop(self):
        """Stop all tasks gracefully"""
        self.running = False
        if self.data_task:
            await self.data_task
        if self.render_task:
            await self.render_task
        self.executor.shutdown(wait=True)

    async def reset(self):
        """Reset all plots by clearing data while preserving structure"""
        # Recursively clear nested dictionaries while preserving structure
        def clear_data_structure(data):
            if isinstance(data, dict):
                return {k: clear_data_structure(v) for k, v in data.items()}
            elif isinstance(data, list):
                return []
            else:
                return data
            
        # clear queue and then wait for it to be processed by the data loop
        self.plot_queue = asyncio.Queue()
        await asyncio.sleep(0.5)

        self.plot_data = clear_data_structure(self.plot_data)
        self.traj_data = clear_data_structure(self.traj_data)
        self.data_needs_update = True
        
    async def _data_loop(self):
        """Process incoming data in background"""
        while self.running:
            try:
                await self.collect_and_organize_data()
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Error in data loop: {e}")
                import traceback
                print(f"Full traceback:\n{traceback.format_exc()}")
                await asyncio.sleep(1)

    def _process_data_sync(self):
        """Process data from the queue, ran through executor in separate thread to avoid blocking the main thread"""
        new_data = False
        while not self.plot_queue.empty():
            new_data = True
            # Get data from queue synchronously
            mjdata, obs_arrays = self.plot_queue.get_nowait()

            # mjdata
            for key in ['qpos', 'qvel', 'xpos', 'xquat', 'ctrl', 'action']:
                self.traj_data.setdefault(key, []).append(mjdata[key])

            # commands
            if not 'command' in self.traj_data:
                self.traj_data['command'] = {
                    'unified_command': []
                }
            unified_command = obs_arrays['command']
            self.traj_data['command']['unified_command'].append(unified_command)

            # some obs
            if not 'obs' in self.traj_data:
                self.traj_data['obs'] = {
                    'sensor_observation_base_site_linvel': [],
                    'sensor_observation_base_site_angvel': [],
                    'sensor_observation_left_foot_touch': [],
                    'sensor_observation_right_foot_touch': []
                }
            self.traj_data['obs']['sensor_observation_base_site_linvel'].append(mjdata['base_site_linvel'])
            self.traj_data['obs']['sensor_observation_base_site_angvel'].append(mjdata['base_site_angvel'])
            self.traj_data['obs']['sensor_observation_left_foot_touch'].append(mjdata['left_foot_touch'])
            self.traj_data['obs']['sensor_observation_right_foot_touch'].append(mjdata['right_foot_touch'])

        if not new_data:
            return False

        traj = Trajectory(
            qpos=jnp.stack(self.traj_data['qpos']),
            qvel=jnp.stack(self.traj_data['qvel']),
            xpos=jnp.stack(self.traj_data['xpos']),
            xquat=jnp.stack(self.traj_data['xquat']),
            command={k: jnp.stack(v) for k, v in self.traj_data['command'].items()},
            obs={k: jnp.stack(v) for k, v in self.traj_data['obs'].items()},
            ctrl=jnp.stack(self.traj_data['ctrl']),
            done=jnp.zeros((len(self.traj_data['qpos']),), dtype=jnp.bool_),
            action=jnp.stack(self.traj_data['action'])
        )

        for reward_name, reward in self.rewards.items():
            try:
                name = reward_name
                if hasattr(reward, 'get_reward_stateful'):
                    reward_values, _ = reward.get_reward_stateful(traj, reward.initial_carry(None))
                else:
                    reward_values = reward.get_reward(traj)
                self.plot_data[name] = [float(x) for x in reward_values.flatten()]
            except Exception as e:
                import traceback
                print(f"Full traceback:\n{traceback.format_exc()}")
                print(f"Error computing reward for {reward_name}: {e}")

        # Calculate total reward
        total_reward = np.zeros(len(self.traj_data['qpos']))
        for reward_name, reward in self.rewards.items():
            if reward_name in self.plot_data:
                total_reward += np.array(self.plot_data[reward_name]) * reward.scale
        self.plot_data['Total Reward'] = total_reward.tolist()
        
        # Adjust y-axis range for total reward plot
        min_reward = np.min(total_reward)
        max_reward = np.max(total_reward)
        y_min = min(0, min_reward)  # Start at 0 or lower if there are negative values
        y_max = max(0, max_reward)  # End at 0 or higher if there are positive values
        self.plots['Total Reward'].setYRange(y_min, y_max, padding=0.1)  # Add 10% padding

        base_eulers = xax.quat_to_euler(traj.xquat[:, 1, :])
        base_eulers = base_eulers.at[:, :2].set(0.0)
        heading_quats = xax.euler_to_quat(base_eulers)
        local_frame_linvel = xax.rotate_vector_by_quat(traj.obs['sensor_observation_base_site_linvel'], heading_quats, inverse=True)

        self.plot_data['feet_force_touch_observation'] = {
            'left_foot_force': [float(x[0]) for x in self.traj_data['obs']['sensor_observation_left_foot_touch']],
            'right_foot_force': [float(x[0]) for x in self.traj_data['obs']['sensor_observation_right_foot_touch']]
        }
        self.plot_data['linvel'] = {
            'x_cmd': [float(x[0]) for x in self.traj_data['command']['unified_command']],
            'x_real': [float(x[0]) for x in local_frame_linvel],
            'y_cmd': [float(x[1]) for x in self.traj_data['command']['unified_command']],
            'y_real': [float(x[1]) for x in local_frame_linvel]
        }
        self.plot_data['angvel'] = {
            'wz_cmd': [float(x[2]) for x in self.traj_data['command']['unified_command']],
            'wz_real': [float(x[2]) for x in self.traj_data['obs']['sensor_observation_base_site_angvel']], # TODO BUG not correct
        }
        standard_height = self.rewards['BaseHeightReward'].standard_height
        self.plot_data['base_height'] = {
            'base_height_cmd': [float(x[4]+standard_height) for x in self.traj_data['command']['unified_command']],
            'base_height_real': [float(x[1, 2]) for x in self.traj_data['xpos']]
        }
        self.plot_data['xyorientation'] = {
            'roll_cmd': [float(x[5]) for x in self.traj_data['command']['unified_command']],
            # 'pitch_real': [float(x[0]) for x in self.traj_data['xquat'][:, 0]],
            'pitch_cmd': [float(x[6]) for x in self.traj_data['command']['unified_command']],
            # 'roll_real': [float(x[1]) for x in self.traj_data['xquat'][:, 1]]
        }

        self.data_needs_update = True
        return True

    async def collect_and_organize_data(self):
        """Run the entire data processing in an executor"""
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._process_data_sync
        )

    async def _render_loop(self):
        """Render all plots at a fixed rate"""
        while self.running:
            try:
                if self.data_needs_update:
                    for name, curves in self.curves.items():
                        if isinstance(curves, dict):
                            # Multiple curves per plot
                            for curve_name, curve in curves.items():
                                values = self.plot_data[name][curve_name]
                                x = list(range(len(values)))
                                curve.setData(x, values)
                        else:
                            # Single curve
                            values = self.plot_data[name]
                            x = list(range(len(values)))
                            curves.setData(x, values)
                        self.plots[name].enableAutoRange()
                    self.data_needs_update = False
                
                self.app.processEvents()
                await asyncio.sleep(1/60)
                
            except Exception as e:
                print(f"Error in render loop: {e}")
                import traceback
                print(f"Full traceback:\n{traceback.format_exc()}")
                await asyncio.sleep(1)

        
    async def add_data(self, mjdata, obs_arrays, action):
        """Copy simulation data to be plotted asynchronously"""
        mjdata_copy = {
            'qpos': np.array(mjdata.qpos, copy=True),
            'qvel': np.array(mjdata.qvel, copy=True),
            'xpos': np.array(mjdata.xpos, copy=True),
            'xquat': np.array(mjdata.xquat, copy=True),
            'time': float(mjdata.time),
            'base_site_linvel': np.array(mjdata.sensor('base_site_linvel').data, copy=True),
            'base_site_angvel': np.array(mjdata.sensor('base_site_angvel').data, copy=True),
            'left_foot_touch': np.array(mjdata.sensor('left_foot_touch').data, copy=True),
            'right_foot_touch': np.array(mjdata.sensor('right_foot_touch').data, copy=True),
            'ctrl': np.array(mjdata.ctrl, copy=True),
            'action': np.array(action, copy=True)
        }
        obs_arrays_copy = {k: np.array(v, copy=True) for k, v in obs_arrays.items()}
        await self.plot_queue.put((mjdata_copy, obs_arrays_copy))
