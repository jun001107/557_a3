# Name: Junghoon Cho
# ID: 260854962

import numpy as np
from pyglm import glm


class HalfEdge:

    def __init__(self, head: 'Vertex', face: 'Face', twin: 'HalfEdge'):
        self.head = head  # the vertex at the "head" of this half-edge
        self.face = face  # left face that this half-edge borders
        self.twin = twin  # the twin half-edge (None if boundary)
        self.next = None  # the next half-edge in the face (to be set later)
        self.edge_collapse_data = None  # data for edge collapse operation, to be set later
        if head.he is None:
            head.he = self  # set the vertex's half-edge if not already set
        if face.he is None:
            face.he = self  # set the face's half-edge if not already set

    def tail(self):
        """ Get the tail of this half-edge."""
        he = self
        while he.next is not self:
            he = he.next  # previous half edge
        return he.head

    def __str__(self):
        return f"~~ HE with Head {self.head.index}, Face {self.face.index} ~~"


class Face:

    def __init__(self, index: int, he: HalfEdge = None):
        """ A face in the half-edge data structure. """
        self.index = index
        self.he = he  # one of the half-edges bordering this face
        self.normal = None  # normal of this face, for visualization and otherwise only for inital quadric computation
        self.center = None  # center of this face, for visualization
        self.M = None  # model matrix for text rendering
        self.text_scale = None  # scale for text rendering

    def get_normal(self):
        """ Return this face's normal. Will compute when called for the first time on a face."""
        if self.normal is not None:
            return self.normal
        v0 = self.he.head.pos
        v1 = self.he.next.head.pos
        v2 = self.he.next.next.head.pos
        n = glm.normalize(glm.cross(glm.vec3(*(v1 - v0)), glm.vec3(*(v2 - v0))))
        self.normal = n
        return n

    def get_center(self):
        """ Return this face's centroid. Will compute when called for the first time on a face."""
        if self.center is not None:
            return self.center
        v0 = self.he.head.pos
        v1 = self.he.next.head.pos
        v2 = self.he.next.next.head.pos
        c = (v0 + v1 + v2) / 3.0
        self.center = c
        return c

    def draw_debug(self, P: glm.mat4, V: glm.mat4, faces: np.ndarray, vert_objs: list, text_renderer):
        """Render the index of this face, for debug purposes."""
        if self.M is None:
            # Cache the necessary quantities, with redo/undo of collapses causing cache recompute by setting M to None
            # We're using the np faces array to get the vertex indices for this face
            #   because the half-edge structure may have changed
            v0 = vert_objs[faces[self.index, 0]].pos
            v1 = vert_objs[faces[self.index, 1]].pos
            v2 = vert_objs[faces[self.index, 2]].pos
            ave_edge_length = (glm.length(v0 - v1) + glm.length(v1 - v2) + glm.length(v2 - v0)) / 3.0
            center = (v0 + v1 + v2) / 3.0
            n = glm.cross(v1 - v0, v2 - v0)
            if glm.length(n) < 1e-6:
                n = glm.vec3(0, 0, 1)
            else:
                n = glm.normalize(n)
            t = glm.normalize(v1 - v0)
            b = glm.normalize(glm.cross(n, t))
            self.M = glm.mat4(
                glm.vec4(t, 0.0),  # X axis
                glm.vec4(b, 0.0),  # Y axis
                glm.vec4(n, 0.0),  # Z axis
                glm.vec4(center + n * 0.01, 1.0)  # Translation
            )
            self.text_scale = ave_edge_length * 0.1
        text_renderer.render_text(str(self.index), P, V * self.M, color=glm.vec4(1, 1, 1, 1),
                                  char_width=self.text_scale, centered=True, view_aligned=False)

    def __str__(self):
        return f"~~ Face with Index {self.index}, Referencing HE {self.he} ~~"


class Vertex:

    def __init__(self, index: int, pos: np.ndarray, he: HalfEdge):
        """ A vertex in the half-edge data structure
        Args:
            index: index of this vertex in the vertex list
            pos: 3D position of this vertex (np for convenience, as this is coming from trimesh)
            he: one of the half-edges ending at this vertex
        """
        self.index = index
        self.pos = glm.vec3(*pos)  # 3D position of this vertex
        self.Q = glm.mat4(1)  # Quadric
        self.he = he  # one of the half-edges ending at this vertex
        self.normal = None  # average normal of faces around this vertex, for visualization
        self.removed_at_level = None  # level of detail at which this vertex was removed
        self.cost = 0  # cost of this vertex living where it is

        self.text_pos = None  # Data for debug text
        self.text_scale = None

    def compute_Q(self):
        """ Compute the quadric for this vertex from the surrounding faces.
        It gets stored in the parameter self.Q"""

        self.Q = glm.mat4(0)

        # TODO: Objective 5: Compute the quadric matrix Q for this vertex
        start = self.he
        if start is None:
            return

        h = start
        visited = set()
        while True:
            if h is None or h in visited:
                break
            visited.add(h)
            face = h.face
            if face is not None and face.he is not None:
                normal = face.get_normal()
                if glm.length(normal) > 1e-12:
                    # Plane equation: n.x * X + n.y * Y + n.z * Z + d = 0
                    d = -glm.dot(normal, self.pos)
                    plane = glm.vec4(normal.x, normal.y, normal.z, d)
                    self.Q += glm.outerProduct(plane, plane)
            next_he = h.next
            if next_he is None:
                break
            h = next_he.twin
            if h is None or h == start:
                break


    def get_normal(self) -> glm.vec3:
        """ Compute the average normal of faces adjacent to this vertex.
        The value is cached after first computation.
        This is currently only used for visualization, but could also be
        used for smooth shading of the mesh."""
        if self.normal is not None:
            return self.normal
        n = glm.vec3(0, 0, 0)
        h = self.he
        while True:
            # Accumulate normal value
            n += h.face.get_normal()
            h = h.next.twin
            if h == self.he:
                break
        if glm.length(n) > 1e-6:
            n = glm.normalize(n)
        self.normal = n
        return n

    def compute_debug_viz_data(self):
        """ Compute data for debug visualization (text position and scale)
        Note that this should be called when the vertex is first created so
        that it has access to a valid half-edge structure around it."""
        # Use the average edge length around this vertex to scale the text
        edge_length = 0.0
        num_edges = 0
        h = self.he
        while True:
            tail = h.tail().pos
            edge_length += glm.length(self.pos - tail)
            num_edges += 1
            h = h.next.twin
            if h == self.he:
                break
        avg_edge_length = edge_length / num_edges if num_edges > 0 else 0.0
        self.text_scale = avg_edge_length * 0.1
        self.text_pos = self.pos + self.get_normal() * avg_edge_length * 0.1

    def draw_debug(self, P: glm.mat4, V: glm.mat4, text_renderer):
        """Render the index of this vertex, for debug purposes."""
        text_renderer.render_text(str(self.index), P, V, pos=self.text_pos, char_width=self.text_scale,
                                  color=glm.vec4(0, 0.75, 0, 1), centered=True, view_aligned=True)

    def __str__(self):
        return f"~~ Vertex with Index {self.index}, Referencing HE {self.he} ~~"


class EdgeCollapseData:
    """ Data structure to hold the data for an edge collapse operation, comparable by cost (i.e., for priority queue)"""

    def __init__(self, he: HalfEdge):
        """ Compute the edge collapse data for the given half-edge.
        Store the cost, optimal position, and quadric matrix for the edge collapse. """

        self.he = he
        # store link to Edge collapse data in both half edges
        self.he.edge_collapse_data = self
        self.he.twin.edge_collapse_data = self

        # TODO: Objective 5: Compute cost, optimal position, and quadric matrix for edge collapse
        v_tail = he.tail()
        v_head = he.head
        if v_tail is None or v_head is None:
            self.cost = float('inf')
            self.Q = glm.mat4(0)
            self.v_opt = glm.vec3(0)
            return

        self.Q = v_tail.Q + v_head.Q

        # Extract A (3x3) and b (3x1) from the quadric matrix
        A = np.array([[float(self.Q[c][r]) for c in range(3)] for r in range(3)], dtype=float)
        b = np.array([float(self.Q[3][r]) for r in range(3)], dtype=float)

        candidates = []
        detA = np.linalg.det(A)
        if abs(detA) > 1e-10:
            try:
                v_opt_np = -np.linalg.solve(A, b)
                if np.all(np.isfinite(v_opt_np)):
                    v_opt = glm.vec3(float(v_opt_np[0]), float(v_opt_np[1]), float(v_opt_np[2]))
                    candidates.append(v_opt)
            except np.linalg.LinAlgError:
                pass

        tail_pos = v_tail.pos
        head_pos = v_head.pos
        mid_pos = (tail_pos + head_pos) * 0.5
        candidates.extend([tail_pos, head_pos, mid_pos])

        def quadric_cost(pos: glm.vec3) -> float:
            vec = glm.vec4(pos.x, pos.y, pos.z, 1.0)
            return float(glm.dot(vec, self.Q * vec))

        best_cost = float('inf')
        best_pos = glm.vec3(0)
        for candidate in candidates:
            cost = quadric_cost(candidate)
            if np.isfinite(cost) and cost < best_cost:
                best_cost = cost
                best_pos = candidate

        self.v_opt = best_pos
        self.cost = best_cost

    def __lt__(self, other):
        if self.cost == other.cost:
            return id(self) < id(other)  # ensure a consistent ordering
        return self.cost < other.cost

    def __eq__(self, other):
        return id(self) == id(other)  # equal cost is not enough, must be the same edge


class CollapseRecord:
    """ data structure to hold the data for an edge collapse operation, for LOD tracking.
        Use a list of Faces, rather than indices, as face indices will change as we collapse."""

    def __init__(self, affected_faces: list[Face], old_indices: np.ndarray, new_indices: np.ndarray):
        self.affected_faces = affected_faces  # faces that were removed during this collapse
        self.old_indices = old_indices.copy()  # to be safe, make our own copy
        self.new_indices = new_indices.copy()

    def redo(self, faces: np.ndarray):
        """ Apply this collapse record to the given faces array."""
        for i, f in enumerate(self.affected_faces):
            f.M = None  # invalidate cached model matrix for text rendering
            faces[f.index, :] = self.new_indices[i]

    def undo(self, faces: np.ndarray):
        """ Undo this collapse record on the given faces array. """
        for i, f in enumerate(self.affected_faces):
            f.M = None  # invalidate cached model matrix for text rendering
            faces[f.index, :] = self.old_indices[i]


def build_heds(F: np.ndarray, vert_objs: list[Vertex]) -> (list[HalfEdge], list[Face]):
    """ Build a half-edge data structure from the given vertices and faces.
    Args:
        F: (num_faces, 3) array of vertex indices for each triangular face
        vert_objs: a list of vertices to set as head of the half edges
    Returns:
        List of *all* HalfEdge objects
        List of *all* Face objects
    """

    he_list = []
    face_objs = []

    # TODO: Objective 1: Build the half-edge data structure
    edge_map: dict[tuple[int, int], HalfEdge] = {}

    for face_idx, (v0_idx, v1_idx, v2_idx) in enumerate(F):
        face = Face(face_idx)
        face_objs.append(face)

        # create the three half-edges for this face (tail -> head ordering follows F winding)
        he0 = HalfEdge(vert_objs[v1_idx], face, None)  # v0 -> v1
        he1 = HalfEdge(vert_objs[v2_idx], face, None)  # v1 -> v2
        he2 = HalfEdge(vert_objs[v0_idx], face, None)  # v2 -> v0

        # link the cycle around the face
        he0.next = he1
        he1.next = he2
        he2.next = he0

        he_list.extend((he0, he1, he2))

        for tail_idx, head_idx, he in (
            (v0_idx, v1_idx, he0),
            (v1_idx, v2_idx, he1),
            (v2_idx, v0_idx, he2),
        ):
            twin_key = (head_idx, tail_idx)
            if twin_key in edge_map:
                twin_he = edge_map[twin_key]
                he.twin = twin_he
                twin_he.twin = he
            edge_map[(tail_idx, head_idx)] = he

    return he_list, face_objs