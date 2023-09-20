# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.5
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info
if version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_glextlib', [dirname(__file__)])
        except ImportError:
            import _glextlib
            return _glextlib
        if fp is not None:
            try:
                _mod = imp.load_module('_glextlib', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _glextlib = swig_import_helper()
    del swig_import_helper
else:
    import _glextlib
del version_info
try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.


def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr_nondynamic(self, class_type, name, static=1):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    if (not static):
        return object.__getattr__(self, name)
    else:
        raise AttributeError(name)

def _swig_getattr(self, class_type, name):
    return _swig_getattr_nondynamic(self, class_type, name, 0)


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object:
        pass
    _newclass = 0



_glextlib.GL_COMPILE_STATUS_swigconstant(_glextlib)
GL_COMPILE_STATUS = _glextlib.GL_COMPILE_STATUS

_glextlib.GL_FRAGMENT_SHADER_swigconstant(_glextlib)
GL_FRAGMENT_SHADER = _glextlib.GL_FRAGMENT_SHADER

_glextlib.GL_LINK_STATUS_swigconstant(_glextlib)
GL_LINK_STATUS = _glextlib.GL_LINK_STATUS

_glextlib.GL_VALIDATE_STATUS_swigconstant(_glextlib)
GL_VALIDATE_STATUS = _glextlib.GL_VALIDATE_STATUS

_glextlib.GL_VERTEX_SHADER_swigconstant(_glextlib)
GL_VERTEX_SHADER = _glextlib.GL_VERTEX_SHADER

def glCheckFramebufferStatusEXT(arg1):
    return _glextlib.glCheckFramebufferStatusEXT(arg1)
glCheckFramebufferStatusEXT = _glextlib.glCheckFramebufferStatusEXT

def glActiveTexture(arg1):
    return _glextlib.glActiveTexture(arg1)
glActiveTexture = _glextlib.glActiveTexture

def glCreateShader(arg1):
    return _glextlib.glCreateShader(arg1)
glCreateShader = _glextlib.glCreateShader

def glAttachShader(arg1, arg2):
    return _glextlib.glAttachShader(arg1, arg2)
glAttachShader = _glextlib.glAttachShader

def glCompileShader(arg1):
    return _glextlib.glCompileShader(arg1)
glCompileShader = _glextlib.glCompileShader

def glCreateProgram():
    return _glextlib.glCreateProgram()
glCreateProgram = _glextlib.glCreateProgram

def glGetProgramiv(arg1, arg2, aInt):
    return _glextlib.glGetProgramiv(arg1, arg2, aInt)
glGetProgramiv = _glextlib.glGetProgramiv

def glGetProgramInfoLog(arg1, arg2, arg3, arg4):
    return _glextlib.glGetProgramInfoLog(arg1, arg2, arg3, arg4)
glGetProgramInfoLog = _glextlib.glGetProgramInfoLog

def glGetShaderiv(arg1, arg2, aInt):
    return _glextlib.glGetShaderiv(arg1, arg2, aInt)
glGetShaderiv = _glextlib.glGetShaderiv

def glGetShaderInfoLog(arg1, arg2, arg3, arg4):
    return _glextlib.glGetShaderInfoLog(arg1, arg2, arg3, arg4)
glGetShaderInfoLog = _glextlib.glGetShaderInfoLog

def glGetUniformLocation(arg1, arg2):
    return _glextlib.glGetUniformLocation(arg1, arg2)
glGetUniformLocation = _glextlib.glGetUniformLocation

def glLinkProgram(arg1):
    return _glextlib.glLinkProgram(arg1)
glLinkProgram = _glextlib.glLinkProgram

def glShaderSource(arg1, arg2, arg3, arg4):
    return _glextlib.glShaderSource(arg1, arg2, arg3, arg4)
glShaderSource = _glextlib.glShaderSource

def glUniform1i(arg1, arg2):
    return _glextlib.glUniform1i(arg1, arg2)
glUniform1i = _glextlib.glUniform1i

def glUniform1f(arg1, arg2):
    return _glextlib.glUniform1f(arg1, arg2)
glUniform1f = _glextlib.glUniform1f

def glUniform4f(arg1, arg2, arg3, arg4, arg5):
    return _glextlib.glUniform4f(arg1, arg2, arg3, arg4, arg5)
glUniform4f = _glextlib.glUniform4f

def glUseProgram(arg1):
    return _glextlib.glUseProgram(arg1)
glUseProgram = _glextlib.glUseProgram

def glValidateProgram(arg1):
    return _glextlib.glValidateProgram(arg1)
glValidateProgram = _glextlib.glValidateProgram

_glextlib.GL_COLOR_ATTACHMENT0_EXT_swigconstant(_glextlib)
GL_COLOR_ATTACHMENT0_EXT = _glextlib.GL_COLOR_ATTACHMENT0_EXT

_glextlib.GL_DEPTH_ATTACHMENT_EXT_swigconstant(_glextlib)
GL_DEPTH_ATTACHMENT_EXT = _glextlib.GL_DEPTH_ATTACHMENT_EXT

_glextlib.GL_DEPTH_STENCIL_EXT_swigconstant(_glextlib)
GL_DEPTH_STENCIL_EXT = _glextlib.GL_DEPTH_STENCIL_EXT

_glextlib.GL_FRAMEBUFFER_EXT_swigconstant(_glextlib)
GL_FRAMEBUFFER_EXT = _glextlib.GL_FRAMEBUFFER_EXT

_glextlib.GL_STENCIL_ATTACHMENT_EXT_swigconstant(_glextlib)
GL_STENCIL_ATTACHMENT_EXT = _glextlib.GL_STENCIL_ATTACHMENT_EXT

_glextlib.GL_FRAMEBUFFER_COMPLETE_EXT_swigconstant(_glextlib)
GL_FRAMEBUFFER_COMPLETE_EXT = _glextlib.GL_FRAMEBUFFER_COMPLETE_EXT

def glBindFramebufferEXT(arg1, arg2):
    return _glextlib.glBindFramebufferEXT(arg1, arg2)
glBindFramebufferEXT = _glextlib.glBindFramebufferEXT

def glFramebufferTexture2DEXT(arg1, arg2, arg3, arg4, arg5):
    return _glextlib.glFramebufferTexture2DEXT(arg1, arg2, arg3, arg4, arg5)
glFramebufferTexture2DEXT = _glextlib.glFramebufferTexture2DEXT

def glGenFramebuffersEXT(arg1, arg2):
    return _glextlib.glGenFramebuffersEXT(arg1, arg2)
glGenFramebuffersEXT = _glextlib.glGenFramebuffersEXT

_glextlib.GL_ARRAY_BUFFER_ARB_swigconstant(_glextlib)
GL_ARRAY_BUFFER_ARB = _glextlib.GL_ARRAY_BUFFER_ARB

_glextlib.GL_ELEMENT_ARRAY_BUFFER_swigconstant(_glextlib)
GL_ELEMENT_ARRAY_BUFFER = _glextlib.GL_ELEMENT_ARRAY_BUFFER

_glextlib.GL_STATIC_DRAW_ARB_swigconstant(_glextlib)
GL_STATIC_DRAW_ARB = _glextlib.GL_STATIC_DRAW_ARB

_glextlib.GL_DYNAMIC_DRAW_ARB_swigconstant(_glextlib)
GL_DYNAMIC_DRAW_ARB = _glextlib.GL_DYNAMIC_DRAW_ARB

_glextlib.GL_STREAM_DRAW_ARB_swigconstant(_glextlib)
GL_STREAM_DRAW_ARB = _glextlib.GL_STREAM_DRAW_ARB

_glextlib.GL_DEPTH_TEXTURE_MODE_swigconstant(_glextlib)
GL_DEPTH_TEXTURE_MODE = _glextlib.GL_DEPTH_TEXTURE_MODE

_glextlib.GL_TEXTURE_COMPARE_MODE_swigconstant(_glextlib)
GL_TEXTURE_COMPARE_MODE = _glextlib.GL_TEXTURE_COMPARE_MODE

_glextlib.GL_DEPTH_COMPONENT32_swigconstant(_glextlib)
GL_DEPTH_COMPONENT32 = _glextlib.GL_DEPTH_COMPONENT32

_glextlib.GL_CLAMP_TO_BORDER_swigconstant(_glextlib)
GL_CLAMP_TO_BORDER = _glextlib.GL_CLAMP_TO_BORDER

_glextlib.GL_TEXTURE0_swigconstant(_glextlib)
GL_TEXTURE0 = _glextlib.GL_TEXTURE0

_glextlib.GL_TEXTURE1_swigconstant(_glextlib)
GL_TEXTURE1 = _glextlib.GL_TEXTURE1

_glextlib.GL_TEXTURE2_swigconstant(_glextlib)
GL_TEXTURE2 = _glextlib.GL_TEXTURE2

_glextlib.GL_TEXTURE3_swigconstant(_glextlib)
GL_TEXTURE3 = _glextlib.GL_TEXTURE3

_glextlib.GL_TEXTURE4_swigconstant(_glextlib)
GL_TEXTURE4 = _glextlib.GL_TEXTURE4

_glextlib.GL_TEXTURE5_swigconstant(_glextlib)
GL_TEXTURE5 = _glextlib.GL_TEXTURE5

_glextlib.GL_TEXTURE6_swigconstant(_glextlib)
GL_TEXTURE6 = _glextlib.GL_TEXTURE6

_glextlib.GL_TEXTURE7_swigconstant(_glextlib)
GL_TEXTURE7 = _glextlib.GL_TEXTURE7

_glextlib.GL_TEXTURE8_swigconstant(_glextlib)
GL_TEXTURE8 = _glextlib.GL_TEXTURE8

_glextlib.GL_TEXTURE9_swigconstant(_glextlib)
GL_TEXTURE9 = _glextlib.GL_TEXTURE9

_glextlib.GL_TEXTURE10_swigconstant(_glextlib)
GL_TEXTURE10 = _glextlib.GL_TEXTURE10

def glGenBuffersARB(arg1):
    return _glextlib.glGenBuffersARB(arg1)
glGenBuffersARB = _glextlib.glGenBuffersARB

def glBindBufferARB(arg1, arg2):
    return _glextlib.glBindBufferARB(arg1, arg2)
glBindBufferARB = _glextlib.glBindBufferARB

def glBufferDataARB(arg1, arg2, arg3, arg4):
    return _glextlib.glBufferDataARB(arg1, arg2, arg3, arg4)
glBufferDataARB = _glextlib.glBufferDataARB

def glDeleteBuffersARB(arg1, arg2):
    return _glextlib.glDeleteBuffersARB(arg1, arg2)
glDeleteBuffersARB = _glextlib.glDeleteBuffersARB
# This file is compatible with both classic and new-style classes.


