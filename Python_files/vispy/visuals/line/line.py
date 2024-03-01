# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
"""
Line visual implementing Agg- and GL-based drawing modes.
"""

from __future__ import division

import numpy as np

from ... import gloo, glsl
from ...color import Color, ColorArray, get_colormap
from ...ext.six import string_types
from ..shaders import Function
from ..visual import Visual, CompoundVisual
from ...util.profiler import Profiler

from .dash_atlas import DashAtlas


vec2to4 = Function("""
    vec4 vec2to4(vec2 inp) {
        return vec4(inp, 0, 1);
    }
""")

vec3to4 = Function("""
    vec4 vec3to4(vec3 inp) {
        return vec4(inp, 1);
    }
""")


"""
TODO:

* Agg support is very minimal; needs attention.
* Optimization--avoid creating new buffers, avoid triggering program
  recompile.
"""


joins = {'miter': 0, 'round': 1, 'bevel': 2}

caps = {'': 0, 'none': 0, '.': 0,
        'round': 1, ')': 1, '(': 1, 'o': 1,
        'triangle in': 2, '<': 2,
        'triangle out': 3, '>': 3,
        'square': 4, '=': 4, 'butt': 4,
        '|': 5}


class LineVisual(CompoundVisual):
    """Line visual

    Parameters
    ----------
    pos : array
        Array of shape (..., 2) or (..., 3) specifying vertex coordinates.
    color : Color, tuple, or array
        The color to use when drawing the line. If an array is given, it
        must be of shape (..., 4) and provide one rgba color per vertex.
        Can also be a colormap name, or appropriate `Function`.
    width:
        The width of the line in px. Line widths > 1px are only
        guaranteed to work when using 'agg' method.
    connect : str or array
        Determines which vertices are connected by lines.

            * "strip" causes the line to be drawn with each vertex
              connected to the next.
            * "segments" causes each pair of vertices to draw an
              independent line segment
            * numpy arrays specify the exact set of segment pairs to
              connect.

    method : str
        Mode to use for drawing.

            * "agg" uses anti-grain geometry to draw nicely antialiased lines
              with proper joins and endcaps.
            * "gl" uses OpenGL's built-in line rendering. This is much faster,
              but produces much lower-quality results and is not guaranteed to
              obey the requested line width or join/endcap styles.

    antialias : bool
        Enables or disables antialiasing.
        For method='gl', this specifies whether to use GL's line smoothing,
        which may be unavailable or inconsistent on some platforms.
    """
    def __init__(self, pos=None, color=(0.5, 0.5, 0.5, 1), width=1,
                 connect='strip', method='gl', antialias=False):
        self._line_visual = None

        self._changed = {'pos': False, 'color': False, 'width': False,
                         'connect': False}

        self._pos = None
        self._color = None
        self._width = None
        self._connect = None
        self._bounds = None
        self._antialias = None
        self._method = 'none'

        CompoundVisual.__init__(self, [])

        # don't call subclass set_data; these often have different
        # signatures.
        LineVisual.set_data(self, pos=pos, color=color, width=width,
                            connect=connect)
        self.antialias = antialias
        self.method = method

    @property
    def antialias(self):
        return self._antialias

    @antialias.setter
    def antialias(self, aa):
        self._antialias = bool(aa)
        self.update()

    @property
    def method(self):
        """The current drawing method"""
        return self._method

    @method.setter
    def method(self, method):
        if method not in ('agg', 'gl'):
            raise ValueError('method argument must be "agg" or "gl".')
        if method == self._method:
            return

        self._method = method
        if self._line_visual is not None:
            self.remove_subvisual(self._line_visual)

        if method == 'gl':
            self._line_visual = _GLLineVisual(self)
        elif method == 'agg':
            self._line_visual = _AggLineVisual(self)
        self.add_subvisual(self._line_visual)

        for k in self._changed:
            self._changed[k] = True

    def set_data(self, pos=None, color=None, width=None, connect=None):
        """Set the data used to draw this visual.

        Parameters
        ----------
        pos : array
            Array of shape (..., 2) or (..., 3) specifying vertex coordinates.
        color : Color, tuple, or array
            The color to use when drawing the line. If an array is given, it
            must be of shape (..., 4) and provide one rgba color per vertex.
        width:
            The width of the line in px. Line widths < 1 px will be rounded up
            to 1 px when using the 'gl' method.
        connect : str or array
            Determines which vertices are connected by lines.

                * "strip" causes the line to be drawn with each vertex
                  connected to the next.
                * "segments" causes each pair of vertices to draw an
                  independent line segment
                * int numpy arrays specify the exact set of segment pairs to
                  connect.
                * bool numpy arrays specify which _adjacent_ pairs to connect.

        """
        if pos is not None:
            self._bounds = None
            self._pos = pos
            self._changed['pos'] = True

        if color is not None:
            self._color = color
            self._changed['color'] = True

        if width is not None:
            self._width = width
            self._changed['width'] = True

        if connect is not None:
            self._connect = connect
            self._changed['connect'] = True

        self.update()

    @property
    def color(self):
        return self._color

    @property
    def width(self):
        return self._width

    @property
    def connect(self):
        return self._connect

    @property
    def pos(self):
        return self._pos

    def _interpret_connect(self):
        if isinstance(self._connect, np.ndarray):
            # Convert a boolean connection array to a vertex index array
            if self._connect.ndim == 1 and self._connect.dtype == bool:
                index = np.empty((len(self._connect), 2), dtype=np.uint32)
                index[:] = np.arange(len(self._connect))[:, np.newaxis]
                index[:, 1] += 1
                return index[self._connect]
            elif self._connect.ndim == 2 and self._connect.shape[1] == 2:
                return self._connect.astype(np.uint32)
            else:
                raise TypeError("Got invalid connect array of shape %r and "
                                "dtype %r" % (self._connect.shape,
                                              self._connect.dtype))
        else:
            return self._connect

    def _interpret_color(self, color_in=None):
        color_in = self._color if color_in is None else color_in
        if isinstance(color_in, string_types):
            try:
                colormap = get_colormap(color_in)
                color = Function(colormap.glsl_map)
            except KeyError:
                color = Color(color_in).rgba
        elif isinstance(color_in, Function):
            color = Function(color_in)
        else:
            color = ColorArray(color_in).rgba
            if len(color) == 1:
                color = color[0]
        return color

    def _compute_bounds(self, axis, view):
        """Get the bounds

        Parameters
        ----------
        mode : str
            Describes the type of boundary requested. Can be "visual", "data",
            or "mouse".
        axis : 0, 1, 2
            The axis along which to measure the bounding values, in
            x-y-z order.
        """
        # Can and should we calculate bounds?
        if (self._bounds is None) and self._pos is not None:
            pos = self._pos
            self._bounds = [(pos[:, d].min(), pos[:, d].max())
                            for d in range(pos.shape[1])]
        # Return what we can
        if self._bounds is None:
            return
        else:
            if axis < len(self._bounds):
                return self._bounds[axis]
            else:
                return (0, 0)

    def _prepare_draw(self, view):
        if self._width == 0:
            return False
        CompoundVisual._prepare_draw(self, view)


class _GLLineVisual(Visual):
    VERTEX_SHADER = """
        varying vec4 v_color;

        void main(void) {
            gl_Position = $transform($to_vec4($position));
            v_color = $color;
        }
    """

    FRAGMENT_SHADER = """
        varying vec4 v_color;
        void main() {
            gl_FragColor = v_color;
        }
    """

    def __init__(self, parent):
        self._parent = parent
        self._pos_vbo = gloo.VertexBuffer()
        self._color_vbo = gloo.VertexBuffer()
        self._connect_ibo = gloo.IndexBuffer()
        self._connect = None
        
        Visual.__init__(self, vcode=self.VERTEX_SHADER,
                        fcode=self.FRAGMENT_SHADER)
        
        self.set_gl_state('translucent')

    def _prepare_transforms(self, view):
        xform = view.transforms.get_transform()
        view.view_program.vert['transform'] = xform

    def _prepare_draw(self, view):
        prof = Profiler()

        if self._parent._changed['pos']:
            if self._parent._pos is None:
                return False
            # todo: does this result in unnecessary copies?
            pos = np.ascontiguousarray(self._parent._pos.astype(np.float32))
            self._pos_vbo.set_data(pos)
            self.shared_program.vert['position'] = self._pos_vbo
            if pos.shape[-1] == 2:
                self.shared_program.vert['to_vec4'] = vec2to4
            elif pos.shape[-1] == 3:
                self.shared_program.vert['to_vec4'] = vec3to4
            else:
                raise TypeError("Got bad position array shape: %r"
                                % (pos.shape,))
            self._parent._changed['pos'] = False

        if self._parent._changed['color']:
            color = self._parent._interpret_color()
            # If color is not visible, just quit now
            if isinstance(color, Color) and color.is_blank:
                return False
            if isinstance(color, Function):
                # TODO: Change to the parametric coordinate once that is done
                self.shared_program.vert['color'] = color(
                    '(gl_Position.x + 1.0) / 2.0')
            else:
                if color.ndim == 1:
                    self.shared_program.vert['color'] = color
                else:
                    self._color_vbo.set_data(color)
                    self.shared_program.vert['color'] = self._color_vbo
            self._parent._changed['color'] = False

        # Do we want to use OpenGL, and can we?
        GL = None
        from ...app._default_app import default_app
        if default_app is not None and \
                default_app.backend_name != 'ipynb_webgl':
            try:
                import OpenGL.GL as GL
            except Exception:  # can be other than ImportError sometimes
                pass

        # Turn on line smooth and/or line width
        if GL:
            if self._parent._antialias:
                GL.glEnable(GL.GL_LINE_SMOOTH)
            else:
                GL.glDisable(GL.GL_LINE_SMOOTH)
            px_scale = self.transforms.pixel_scale
            width = px_scale * self._parent._width
            GL.glLineWidth(max(width, 1.))

        if self._parent._changed['connect']:
            self._connect = self._parent._interpret_connect()
            if isinstance(self._connect, np.ndarray):
                self._connect_ibo.set_data(self._connect)
            self._parent._changed['connect'] = False
        if self._connect is None:
            return False

        prof('prepare')

        # Draw
        if isinstance(self._connect, string_types) and \
                self._connect == 'strip':
            self._draw_mode = 'line_strip'
            self._index_buffer = None
        elif isinstance(self._connect, string_types) and \
                self._connect == 'segments':
            self._draw_mode = 'lines'
            self._index_buffer = None
        elif isinstance(self._connect, np.ndarray):
            self._draw_mode = 'lines'
            self._index_buffer = self._connect_ibo
        else:
            raise ValueError("Invalid line connect mode: %r" % self._connect)

        prof('draw')


class _AggLineVisual(Visual):
    _agg_vtype = np.dtype([('a_position', np.float32, 2),
                           ('a_tangents', np.float32, 4),
                           ('a_segment',  np.float32, 2),
                           ('a_angles',   np.float32, 2),
                           ('a_texcoord', np.float32, 2),
                           ('alength', np.float32),
                           ('color', np.float32, 4)])

    VERTEX_SHADER = glsl.get('lines/agg.vert')
    FRAGMENT_SHADER = glsl.get('lines/agg.frag')

    def __init__(self, parent):
        self._parent = parent
        self._vbo = gloo.VertexBuffer()

        self._pos = None
        self._color = None

        self._da = DashAtlas()
        dash_index, dash_period = self._da['solid']
        self._U = dict(dash_index=dash_index, dash_period=dash_period,
                       linejoin=joins['round'],
                       linecaps=(caps['round'], caps['round']),
                       dash_caps=(caps['round'], caps['round']),
                       antialias=1.0)
        self._dash_atlas = gloo.Texture2D(self._da._data)

        Visual.__init__(self, vcode=self.VERTEX_SHADER,
                        fcode=self.FRAGMENT_SHADER)
        self._index_buffer = gloo.IndexBuffer()
        self.set_gl_state('translucent', depth_test=False)
        self._draw_mode = 'triangles'

    def _prepare_transforms(self, view):
        data_doc = view.get_transform('visual', 'document')
        doc_px = view.get_transform('document', 'framebuffer')
        px_ndc = view.get_transform('framebuffer', 'render')

        vert = view.view_program.vert
        vert['transform'] = data_doc
        vert['doc_px_transform'] = doc_px
        vert['px_ndc_transform'] = px_ndc

    def _prepare_draw(self, view):
        bake = False
        if self._parent._changed['pos']:
            if self._parent._pos is None:
                return False
            # todo: does this result in unnecessary copies?
            self._pos = np.ascontiguousarray(
                self._parent._pos.astype(np.float32))
            bake = True

        if self._parent._changed['color']:
            self._color = self._parent._interpret_color()
            bake = True

        if self._parent._changed['connect']:
            if self._parent._connect not in [None, 'strip']:
                raise NotImplementedError("Only 'strip' connection mode "
                                          "allowed for agg-method lines.")

        if bake:
            V, idxs = self._agg_bake(self._pos, self._color)
            self._vbo.set_data(V)
            self._index_buffer.set_data(idxs)

        #self.shared_program.prepare()
        self.shared_program.bind(self._vbo)
        uniforms = dict(closed=False, miter_limit=4.0, dash_phase=0.0,
                        linewidth=self._parent._width)
        for n, v in uniforms.items():
            self.shared_program[n] = v
        for n, v in self._U.items():
            self.shared_program[n] = v
        self.shared_program['u_dash_atlas'] = self._dash_atlas

    @classmethod
    def _agg_bake(cls, vertices, color, closed=False):
        """
        Bake a list of 2D vertices for rendering them as thick line. Each line
        segment must have its own vertices because of antialias (this means no
        vertex sharing between two adjacent line segments).
        """

        n = len(vertices)
        P = np.array(vertices).reshape(n, 2).astype(float)
        idx = np.arange(n)  # used to eventually tile the color array

        dx, dy = P[0] - P[-1]
        d = np.sqrt(dx*dx+dy*dy)

        # If closed, make sure first vertex = last vertex (+/- epsilon=1e-10)
        if closed and d > 1e-10:
            P = np.append(P, P[0]).reshape(n+1, 2)
            idx = np.append(idx, idx[-1])
            n += 1

        V = np.zeros(len(P), dtype=cls._agg_vtype)
        V['a_position'] = P

        # Tangents & norms
        T = P[1:] - P[:-1]

        N = np.sqrt(T[:, 0]**2 + T[:, 1]**2)
        # T /= N.reshape(len(T),1)
        V['a_tangents'][+1:, :2] = T
        V['a_tangents'][0, :2] = T[-1] if closed else T[0]
        V['a_tangents'][:-1, 2:] = T
        V['a_tangents'][-1, 2:] = T[0] if closed else T[-1]

        # Angles
        T1 = V['a_tangents'][:, :2]
        T2 = V['a_tangents'][:, 2:]
        A = np.arctan2(T1[:, 0]*T2[:, 1]-T1[:, 1]*T2[:, 0],
                       T1[:, 0]*T2[:, 0]+T1[:, 1]*T2[:, 1])
        V['a_angles'][:-1, 0] = A[:-1]
        V['a_angles'][:-1, 1] = A[+1:]

        # Segment
        L = np.cumsum(N)
        V['a_segment'][+1:, 0] = L
        V['a_segment'][:-1, 1] = L
        # V['a_lengths'][:,2] = L[-1]

        # Step 1: A -- B -- C  =>  A -- B, B' -- C
        V = np.repeat(V, 2, axis=0)[1:-1]
        V['a_segment'][1:] = V['a_segment'][:-1]
        V['a_angles'][1:] = V['a_angles'][:-1]
        V['a_texcoord'][0::2] = -1
        V['a_texcoord'][1::2] = +1
        idx = np.repeat(idx, 2)[1:-1]

        # Step 2: A -- B, B' -- C  -> A0/A1 -- B0/B1, B'0/B'1 -- C0/C1
        V = np.repeat(V, 2, axis=0)
        V['a_texcoord'][0::2, 1] = -1
        V['a_texcoord'][1::2, 1] = +1
        idx = np.repeat(idx, 2)

        idxs = np.resize(np.array([0, 1, 2, 1, 2, 3], dtype=np.uint32),
                         (n-1)*(2*3))
        idxs += np.repeat(4*np.arange(n-1, dtype=np.uint32), 6)

        # Length
        V['alength'] = L[-1] * np.ones(len(V))

        # Color
        if color.ndim == 1:
            color = np.tile(color, (len(V), 1))
        elif color.ndim == 2 and len(color) == n:
            color = color[idx]
        else:
            raise ValueError('Color length %s does not match number of '
                             'vertices %s' % (len(color), n))
        V['color'] = color

        return V, idxs
