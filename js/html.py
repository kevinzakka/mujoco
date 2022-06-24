import json
import mujoco


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
      import {Viewer} from './viewer.js';
      const domElement = document.getElementById('mujoco-viewer');
      var viewer = new Viewer(domElement, system);
    </script>
  </body>
</html>
"""


def render(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    height: int = 480,
) -> str:
    """Returns an HTML page that visualizes the system and its configuration."""

    # Extract all positions and rotations.
    # Convert to json.
    # Replace in _HTML string.

    html = _HTML.replace('<!-- viewer height goes here -->', f'{height}px')
    return html


if __name__ == "__main__":
    # Load the model.
    model = mujoco.MjModel.from_xml_path("./ant.xml")
    data = mujoco.MjData(model)

    # Apply random controls and step the simulation.

    # Render the trajectory.


