import json
from typing import Sequence
import mujoco
import numpy as np
from dataclasses import dataclass
from google.protobuf import json_format


_HTML = """
<html>
  <head>
    <title>MuJoCo visualizer</title>
    <style>
      body {
        margin: 0;
        padding: 0;
      }
      #mujoco-viewer {
        margin: 0;
        padding: 0;
        height: <!-- viewer height goes here -->;
      }
    </style>
  </head>
  <body>
    <script type="application/javascript">
      var system = <!-- system json goes here -->;
    </script>
    <div id="mujoco-viewer"></div>
    <script type="module">
      import {Viewer} from 'https://cdn.jsdelivr.net/gh/google/brax@v0.0.13/js/viewer.js';
      const domElement = document.getElementById('mujoco-viewer');
      var viewer = new Viewer(domElement, system);
    </script>
  </body>
</html>
"""


@dataclass(frozen=True)
class CartesianPose:
  xpos: np.ndarray
  xquat: np.ndarray

  @staticmethod
  def from_mjdata(data: mujoco.MjData) -> "CartesianPose":
    return CartesianPose(data.xpos.copy(), data.xquat.copy())


def render(
    config,
    frames: Sequence[CartesianPose],
    height: int = 480,
) -> str:
    """Returns an HTML page that visualizes the system and its configuration."""
    d = {
      "config": json_format.MessageToDict(config, True),
      "pos": [frame.xpos.tolist() for frame in frames],
      "rot": [frame.xquat.tolist() for frame in frames],
    }
    system = json.dumps(d)
    html = _HTML.replace('<!-- system json goes here -->', system)
    html = html.replace('<!-- viewer height goes here -->', f'{height}px')
    return html