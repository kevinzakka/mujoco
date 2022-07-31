from typing import Tuple
import mujoco
import time
from PIL import Image
import numpy as np

_XML_PATH = "/Users/kevin/dev/mujoco/model/humanoid/humanoid.xml"
_FLIPPED_XML_PATH = "/Users/kevin/dev/mujoco/model/humanoid/humanoid_flipped.xml"
_NUM_IMAGES = 100
_SMALL_RESOLUTION = (64, 64)
_MEDIUM_RESOLUTION = (128, 128)
_LARGE_RESOLUTION = (640, 480)


def old(res: Tuple[int, int]) -> float:
    w, h = res

    gl = mujoco.GLContext(w, h)
    gl.make_current()

    model = mujoco.MjModel.from_xml_path(_XML_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    scene = mujoco.MjvScene(model, maxgeom=10000)
    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        mujoco.MjvPerturb(),
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL.value,
        scene,
    )

    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, context)

    viewport = mujoco.MjrRect(0, 0, w, h)
    upside_down_image = np.empty((h, w, 3), dtype=np.uint8)

    times = []
    for _ in range(_NUM_IMAGES):
        mujoco.mj_step(model, data)
        mujoco.mjv_updateScene(
            model,
            data,
            mujoco.MjvOption(),
            mujoco.MjvPerturb(),
            mujoco.MjvCamera(),
            mujoco.mjtCatBit.mjCAT_ALL.value,
            scene,
        )
        tic = time.time()
        mujoco.mjr_render(viewport, scene, context)
        mujoco.mjr_readPixels(upside_down_image, None, viewport, context)
        image = np.flipud(upside_down_image)
        times.append(time.time() - tic)
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"(old) Time taken to render {_NUM_IMAGES} images: {mean_time}s (+/- {std_time})")
    # Image.fromarray(image).show()
    return mean_time


def new(res: Tuple[int, int]) -> float:
    w, h = res

    gl = mujoco.GLContext(w, h)
    gl.make_current()

    model = mujoco.MjModel.from_xml_path(_FLIPPED_XML_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    scene = mujoco.MjvScene(model, maxgeom=10000)
    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), mujoco.MjvPerturb(),
        mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL.value, scene)

    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, context)

    viewport = mujoco.MjrRect(0, 0, w, h)
    image = np.empty((h, w, 3), dtype=np.uint8)

    times = []
    for _ in range(_NUM_IMAGES):
        mujoco.mj_step(model, data)
        mujoco.mjv_updateScene(
            model,
            data,
            mujoco.MjvOption(),
            mujoco.MjvPerturb(),
            mujoco.MjvCamera(),
            mujoco.mjtCatBit.mjCAT_ALL.value,
            scene,
        )
        tic = time.time()
        mujoco.mjr_render(viewport, scene, context)
        mujoco.mjr_readPixels(image, None, viewport, context)
        times.append(time.time() - tic)
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"(new) Time taken to render {_NUM_IMAGES} images: {mean_time}s (+/- {std_time})")
    Image.fromarray(image).show()
    return mean_time



if __name__ == "__main__":
    t_old = old(_MEDIUM_RESOLUTION)
    t_new = new(_MEDIUM_RESOLUTION)
    speedup = t_old / t_new
    pct_speedup = (t_old - t_new) / t_new * 100
    print(f"Speedup: {speedup}x")
    print(f"Percent speedup: {pct_speedup:.2f}%")
