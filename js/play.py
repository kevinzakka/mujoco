import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

import dcargs
import numpy as np
from dm_control import mjcf
from dm_control.mjcf import constants
from dm_robotics.transformations import transformations as tr

MjcfElement = Any

_IDENTITY_QUAT = np.array([1, 0, 0, 0], dtype=np.float32)


# =============================================================== #
# Helper functions
# =============================================================== #


def _get_rotation(elem: MjcfElement) -> np.ndarray:
    """Safely gets the rotation of an element in case it is None."""
    if elem.tag == constants.WORLDBODY:
        return _IDENTITY_QUAT
    if elem.euler is not None:
        return tr.euler_to_quat(elem.euler)
    if elem.quat is not None:
        return np.array(elem.quat, dtype=np.float32, copy=True)
    if elem.axisangle is not None:
        return tr.axisangle_to_quat(elem.axisangle)
    return _IDENTITY_QUAT


def _get_position(elem: MjcfElement) -> np.ndarray:
    """Safely gets the position of an element in case it is None."""
    if elem.pos is None:
        return np.zeros(3, dtype=np.float32)
    else:
        return np.array(elem.pos, dtype=np.float32, copy=True)


def _local_position(pos: np.ndarray, body: Optional[MjcfElement]) -> np.ndarray:
    """Converts a position from global to local coordinates."""
    if body and not body.tag == constants.WORLDBODY:
        return pos - _get_position(body)
    else:
        return pos


# =============================================================== #
# Parser functions
# =============================================================== #


def _parse_sphere(sphere: MjcfElement) -> Dict[str, Any]:
    if sphere.size is None:
        raise ValueError("Sphere size is None")
    radius = sphere.size[0]
    position = _get_position(sphere)
    return {"radius": radius, "position": position}


def _parse_capsule(capsule: MjcfElement, add_radius: bool = True) -> Dict[str, Any]:
    size = capsule.size
    radius = size[0]
    if capsule.fromto is not None:
        start = capsule.fromto[0:3]
        end = capsule.fromto[3:6]
        direction = end - start
        length = np.linalg.norm(direction)
        quat = tr.quat_between_vectors(direction, [0, 0, 1])
        position = _local_position((start + end) / 2.0, capsule.parent)
    else:
        length = size[1] * 2
        position = _get_position(capsule)
        quat = _IDENTITY_QUAT
    geom_rotation = _get_rotation(capsule)
    quat = tr.quat_mul(quat, geom_rotation)
    if add_radius:
        length += 2 * radius
    return {"radius": radius, "length": length, "position": position, "quat": quat}


def _parse_box(box: MjcfElement) -> Dict[str, Any]:
    raise NotImplementedError


def _parse_cylinder(cylinder: MjcfElement) -> Dict[str, Any]:
    raise NotImplementedError


def _parse_mesh(mesh: MjcfElement) -> Dict[str, Any]:
    raise NotImplementedError


def _parse_plane(plane: MjcfElement) -> Dict[str, Any]:
    raise NotImplementedError


def parse_geom(geom: MjcfElement) -> Dict[str, Any]:
    if geom.type == "box":
        return _parse_box(geom)
    elif geom.type == "sphere":
        return _parse_sphere(geom)
    elif geom.type == "cylinder":
        return _parse_cylinder(geom)
    elif geom.type == "capsule":
        return _parse_capsule(geom)
    elif geom.type == "mesh":
        return _parse_mesh(geom)
    elif geom.type == "plane":
        return _parse_plane(geom)
    else:
        raise ValueError(f"Unknown geom type: {geom.type}")


def add_body(body: MjcfElement, parent_body: Optional[MjcfElement]) -> None:
    body_dict = {}

    if not parent_body:
        body_dict["name"] = constants.WORLDBODY
    else:
        body_dict["name"] = body.name if body.name else "no_name"
    print(f"Body: {body_dict['name']}")

    body_quat = _get_rotation(body)
    body_dict["quat"] = body_quat
    print(f"\tBody rotation: {body_quat}")

    geoms = body.geom if hasattr(body, "geom") else []
    print(f"\tGeoms: {len(geoms)}")

    body_dict["geoms"] = {}
    for geom in geoms:
        body_dict["geoms"][geom.name] = parse_geom(geom)

    for child_body in body.body:
        add_body(child_body, body)


@dataclass(frozen=True)
class Args:
    xml_path: str = "ant.xml"


def main(args: Args) -> None:
    xml_path = pathlib.Path(args.xml_path)
    print(f"Loading XML file: {xml_path}")

    # Things we care about:
    # Purely visual and collision geometry
    # Body frames (position and orientation) over time
    # Joints and actuators? No.

    mjcf_model = mjcf.from_path(str(xml_path))

    add_body(mjcf_model.worldbody, None)


if __name__ == "__main__":
    main(dcargs.cli(Args))
