"""Export all visible mesh objects of a .blend file as one PLY in Blender world coordinates.

Run with Blender (not with the project's Python):
  blender -b path/to/lego.blend --python scripts/blender_export_mesh.py -- data/reference/lego.ply

Modifiers are applied and object transforms are baked, so the output is in the same world frame as
the NeRF-Synthetic transforms_*.json cameras. Check alignment afterwards by rendering the reference
points into a training view (frecon/benchmark/reference_geometry.check_alignment).
"""
import sys

import bpy  # noqa: provided by Blender

out = sys.argv[sys.argv.index("--") + 1]
deps = bpy.context.evaluated_depsgraph_get()
verts, faces = [], []
for obj in bpy.context.scene.objects:
    if obj.type != "MESH" or obj.hide_render or obj.hide_get():
        continue
    ev = obj.evaluated_get(deps)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    base = len(verts)
    mw = obj.matrix_world
    verts.extend([tuple(mw @ v.co) for v in me.vertices])
    faces.extend([tuple(base + i for i in t.vertices) for t in me.loop_triangles])
    ev.to_mesh_clear()

with open(out, "w") as f:
    f.write(f"ply\nformat ascii 1.0\nelement vertex {len(verts)}\nproperty float x\nproperty float y\n"
            f"property float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    for v in verts:
        f.write(f"{v[0]} {v[1]} {v[2]}\n")
    for t in faces:
        f.write(f"3 {t[0]} {t[1]} {t[2]}\n")
print(f"exported {len(verts)} vertices, {len(faces)} triangles -> {out}")
