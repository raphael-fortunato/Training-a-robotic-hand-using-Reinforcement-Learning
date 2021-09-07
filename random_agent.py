import gym
import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder
env = gym.make('HandManipulateBlock-v0')
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)  # Create a window to init GLFW.
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()

def record():
    try:
        if not os.path.exists("videos"):
            os.mkdir('videos')
        recorder = VideoRecorder(env, path=f'videos/random_agent.mp4')
        for _ in range():
            print(_)
            env.reset()
            done =False
            for __ in range(env._max_episode_steps):
                recorder.capture_frame()
                action = env.action_space.sample()
                env.step(action)
        recorder.close()
    except Exception as e:
        print(e)

record()