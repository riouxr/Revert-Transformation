bl_info = {
    "name": "Match Transformations From Master (Affine)",
    "author": "ChatGPT",
    "version": (1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Tool",
    "description": "Restore transforms using a master mesh — supports non-uniform scale / shear (affine).",
    "category": "Object",
}

import bpy
import math
from mathutils import Vector, Matrix, Quaternion

EPS = 1e-9
KEEP_MASTER_TRANSFORM = True  # set False to reset master to identity after operation

# -----------------------
# Helper: Horn quaternion (used as fallback)
# -----------------------
def compute_rotation_quaternion_from_cov(H):
    # same robust Horn-power-iteration as before
    Sxx = H[0][0]; Sxy = H[0][1]; Sxz = H[0][2]
    Syx = H[1][0]; Syy = H[1][1]; Syz = H[1][2]
    Szx = H[2][0]; Szy = H[2][1]; Szz = H[2][2]

    N = Matrix((
        ( Sxx + Syy + Szz, Syz - Szy,       Szx - Sxz,       Sxy - Syx      ),
        ( Syz - Szy,       Sxx - Syy - Szz, Sxy + Syx,       Szx + Sxz      ),
        ( Szx - Sxz,       Sxy + Syx,      -Sxx + Syy - Szz, Syz + Szy      ),
        ( Sxy - Syx,       Szx + Sxz,       Syz + Szy,      -Sxx - Syy + Szz)
    ))

    q = Vector((1.0, 0.0, 0.0, 0.0))
    for _ in range(200):
        qn = N @ q
        if not math.isfinite(qn.x + qn.y + qn.z + qn.w):
            return None
        ln = qn.length
        if ln < EPS:
            return None
        q = qn / ln
    if not math.isfinite(q.x + q.y + q.z + q.w):
        return None
    return Quaternion((q[0], q[1], q[2], q[3]))


def compute_affine_from_point_sets(src_points, dst_points, fallback_to_uniform=True):
    """
    Compute 3x3 affine linear matrix A and translation T that best map:
      A * src + T ≈ dst
    src_points / dst_points are lists of Vector with same length.
    Returns (A (3x3 Matrix), T Vector)
    If the normal matrix is singular, optionally fallback to uniform-scale method;
    otherwise raises ValueError.
    """
    if len(src_points) != len(dst_points) or len(src_points) == 0:
        raise ValueError("Point lists must be same non-zero length")

    src = [Vector(p) for p in src_points]
    dst = [Vector(p) for p in dst_points]

    # centroids
    c_src = sum(src, Vector()) / len(src)
    c_dst = sum(dst, Vector()) / len(dst)

    # demean
    A_pts = [p - c_src for p in src]   # a_i
    B_pts = [p - c_dst for p in dst]   # b_i

    # Check variance
    sumA2 = sum(v.length_squared for v in A_pts)
    sumB2 = sum(v.length_squared for v in B_pts)
    if sumA2 < EPS or sumB2 < EPS:
        raise ValueError("Degenerate point sets (zero variance)")

    # Build S = Σ a a^T  (3x3)
    S = Matrix(((0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0)))
    # Build D = Σ b a^T  (3x3)
    D = Matrix(((0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0)))

    for a, b in zip(A_pts, B_pts):
        # S += a * a^T
        S[0][0] += a.x * a.x; S[0][1] += a.x * a.y; S[0][2] += a.x * a.z
        S[1][0] += a.y * a.x; S[1][1] += a.y * a.y; S[1][2] += a.y * a.z
        S[2][0] += a.z * a.x; S[2][1] += a.z * a.y; S[2][2] += a.z * a.z

        # D += b * a^T
        D[0][0] += b.x * a.x; D[0][1] += b.x * a.y; D[0][2] += b.x * a.z
        D[1][0] += b.y * a.x; D[1][1] += b.y * a.y; D[1][2] += b.y * a.z
        D[2][0] += b.z * a.x; D[2][1] += b.z * a.y; D[2][2] += b.z * a.z

    # Try to invert S
    try:
        S_inv = S.inverted()
    except Exception:
        S_inv = None

    if S_inv is not None:
        # A = D * S_inv
        A_mat = D @ S_inv
        T = c_dst - (A_mat @ c_src)

        # Sanity check finite
        if not all(math.isfinite(v) for row in A_mat for v in row) or not math.isfinite(T.x + T.y + T.z):
            raise ValueError("Non-finite affine result")
        return A_mat, T

    # If S is singular and we allow fallback, attempt robust uniform-scale Horn method
    if fallback_to_uniform:
        # Build cross-covariance H = D (which is Σ b a^T)
        H = D
        q = compute_rotation_quaternion_from_cov(H)
        if q is None:
            raise ValueError("Singular normal matrix and uniform fallback failed")
        R = q.to_matrix()
        # RMS uniform scale
        denom = sum((R @ a).length_squared for a in A_pts)
        numer = sum(b.length_squared for b in B_pts)
        if denom < EPS:
            raise ValueError("Degenerate denom in uniform fallback")
        scale = (numer / denom) ** 0.5
        T = c_dst - (R @ (c_src * scale))
        A_mat = Matrix(((R[0][0]*scale, R[0][1]*scale, R[0][2]*scale),
                        (R[1][0]*scale, R[1][1]*scale, R[1][2]*scale),
                        (R[2][0]*scale, R[2][1]*scale, R[2][2]*scale)))
        return A_mat, T

    raise ValueError("Normal matrix singular and fallback disabled")


# -----------------------
# Operator
# -----------------------
class OBJECT_OT_match_transformations(bpy.types.Operator):
    bl_idname = "object.match_transformations"
    bl_label = "Revert Transformations"
    bl_description = "Use active master to restore transforms (supports non-uniform scale & shear)."

    def execute(self, context):
        objs = list(context.selected_objects)
        if len(objs) < 2:
            self.report({'ERROR'}, "Select at least master + one object.")
            return {'CANCELLED'}

        master = context.active_object
        if master is None or master not in objs:
            self.report({'ERROR'}, "Master (reference) must be the active object (last selected).")
            return {'CANCELLED'}

        if master.type != 'MESH':
            self.report({'ERROR'}, "Master must be a mesh.")
            return {'CANCELLED'}

        # master points in world space
        master_world_pts = [master.matrix_world @ v.co for v in master.data.vertices]
        n_master = len(master_world_pts)
        if n_master == 0:
            self.report({'ERROR'}, "Master mesh has no vertices.")
            return {'CANCELLED'}

        processed = 0
        skipped = []

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        for obj in objs:
            if obj == master:
                continue
            if obj.type != 'MESH':
                skipped.append((obj.name, "not a mesh"))
                continue

            if len(obj.data.vertices) != n_master:
                skipped.append((obj.name, f"vertex count mismatch ({len(obj.data.vertices)} != {n_master})"))
                continue

            obj_world_pts = [(obj.matrix_world @ v.co) for v in obj.data.vertices]

            try:
                A_mat, T = compute_affine_from_point_sets(master_world_pts, obj_world_pts, fallback_to_uniform=True)
            except ValueError as e:
                skipped.append((obj.name, str(e)))
                continue

            # Build 4x4 matrix from A_mat and T
            M = Matrix((
                (A_mat[0][0], A_mat[0][1], A_mat[0][2], T.x),
                (A_mat[1][0], A_mat[1][1], A_mat[1][2], T.y),
                (A_mat[2][0], A_mat[2][1], A_mat[2][2], T.z),
                (0.0,         0.0,         0.0,         1.0),
            ))

            # Replace mesh with copy of master
            obj.data = master.data.copy()

            # Assign matrix_world (Blender will show location/rotation/scale via decomposition)
            try:
                obj.matrix_world = M
            except Exception as e:
                skipped.append((obj.name, f"failed to set matrix: {e}"))
                continue

            processed += 1

        if not KEEP_MASTER_TRANSFORM:
            master.matrix_world = Matrix.Identity(4)

        msg = f"Processed: {processed}"
        if skipped:
            msg += f"; Skipped: {len(skipped)} (console details)"
            print("Match Transformations (affine) skipped objects:")
            for name, reason in skipped:
                print(f"  {name}: {reason}")

        self.report({'INFO'}, msg)
        return {'FINISHED'}


# -----------------------
# Panel
# -----------------------
class OBJECT_PT_match_transformations_panel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_match_transformations"
    bl_label = "Revert Transformations"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout
        layout.label(text="Select objects then master")
        layout.operator("object.match_transformations", text="Revert Transformations")

# -----------------------
# Register
# -----------------------
classes = (
    OBJECT_OT_match_transformations,
    OBJECT_PT_match_transformations_panel,
)

def register():
    for c in classes:
        bpy.utils.register_class(c)

def unregister():
    for c in classes:
        bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()
