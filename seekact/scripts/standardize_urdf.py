"""Derive a standard-URDF copy of the RB-Y1 vendor model for generic tools (pinocchio, yourdfpy, trimesh).

    python scripts/standardize_urdf.py <vendor model.urdf> <out.urdf>

The vendor file (rby1-sdk models/rby1a/urdf) uses two things standard URDF parsers reject:
`<capsule radius= length=>` collision geometry and wheel `<limit>` without `effort`. The
copy replaces each capsule (axis z of its origin) with the identical union of a cylinder and
two end spheres, and gives the wheel limits effort="0". Visual meshes, inertias, joints and
limits are unchanged. The vendor files stay as shipped; this only writes a new file.
"""
import sys
import xml.etree.ElementTree as ET

import numpy as np


def rpy_matrix(rpy):
    r, p, y = rpy
    cx, sx, cy, sy, cz, sz = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
                     [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
                     [-sy, sx * cy, cx * cy]])


def standardize(source, target):
    tree = ET.parse(source)
    capsules = 0
    for link in tree.getroot().iter('link'):
        for collision in list(link.findall('collision')):
            capsule = collision.find('geometry/capsule')
            if capsule is None:
                continue
            origin = collision.find('origin')
            xyz = np.array([float(v) for v in (origin.get('xyz', '0 0 0') if origin is not None else '0 0 0').split()])
            rpy = [float(v) for v in (origin.get('rpy', '0 0 0') if origin is not None else '0 0 0').split()]
            radius, length = float(capsule.get('radius')), float(capsule.get('length'))
            axis = rpy_matrix(rpy)[:, 2]
            index = list(link).index(collision)
            link.remove(collision)
            parts = [('cylinder', xyz), ('sphere', xyz + axis * length / 2), ('sphere', xyz - axis * length / 2)]
            for k, (kind, centre) in enumerate(parts):
                new = ET.Element('collision', {'name': f'capsule_{capsules}_{kind}{k}'})
                ET.SubElement(new, 'origin', {'xyz': ' '.join(f'{v:.6g}' for v in centre),
                                              'rpy': ' '.join(f'{v:.6g}' for v in rpy)})
                geometry = ET.SubElement(new, 'geometry')
                attrs = {'radius': f'{radius:.6g}'}
                if kind == 'cylinder':
                    attrs['length'] = f'{length:.6g}'
                ET.SubElement(geometry, kind, attrs)
                link.insert(index + k, new)
            capsules += 1
    efforts = 0
    for joint in tree.getroot().iter('joint'):
        limit = joint.find('limit')
        if limit is not None and limit.get('effort') is None:
            limit.set('effort', '0')
            efforts += 1
    tree.getroot().insert(0, ET.Comment(
        f' Derived from {source.rsplit("/", 1)[-1]} by scripts/standardize_urdf.py: {capsules} capsules '
        f'-> cylinder + 2 spheres each, effort="0" added to {efforts} limits; otherwise unchanged. '))
    tree.write(target, encoding='utf-8', xml_declaration=True)
    return capsules, efforts


if __name__ == '__main__':
    print(standardize(sys.argv[1], sys.argv[2]))
