#include "pfr_native_env.h"

/* Expose heatmap + reset_to_map as custom Python methods */
#define PY_ARRAY_UNIQUE_SYMBOL PFR_NATIVE_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static PyObject* get_heatmap(PyObject* self, PyObject* args) {
    (void)self; (void)args;
    pfr_heatmap_ensure_alloc();
    if (!g_pfr_heatmap) Py_RETURN_NONE;

    npy_intp dims[2] = { PFR_HEATMAP_H, PFR_HEATMAP_W };
    PyObject* arr = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, g_pfr_heatmap);
    if (!arr) Py_RETURN_NONE;
    PyArray_CLEARFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA);
    return arr;
}

/* reset_to_map(env_handle, map_id, x, y) — teleport agent to any map/position */
static PyObject* py_reset_to_map(PyObject* self, PyObject* args) {
    (void)self;
    PyObject* handle_obj;
    int map_id, x, y;
    if (!PyArg_ParseTuple(args, "Oiii", &handle_obj, &map_id, &x, &y))
        return NULL;
    Env* env = (Env*)PyLong_AsVoidPtr(handle_obj);
    if (!env) {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle");
        return NULL;
    }
    int ok = pfr_native_reset_to_map(&env->core,
        (PfrNativeMapId)map_id, (int16_t)x, (int16_t)y, PFR_NATIVE_DIR_SOUTH);
    pfr_extract_obs(env);
    return PyLong_FromLong(ok);
}

/* get_map_count() — return number of maps */
static PyObject* py_get_map_count(PyObject* self, PyObject* args) {
    (void)self; (void)args;
    return PyLong_FromSize_t(gPfrNativeMapCount);
}

/* get_map_info(map_id) — return (name, width, height, warp_count) */
static PyObject* py_get_map_info(PyObject* self, PyObject* args) {
    (void)self;
    int map_id;
    if (!PyArg_ParseTuple(args, "i", &map_id))
        return NULL;
    if (map_id < 0 || (size_t)map_id >= gPfrNativeMapCount) {
        PyErr_SetString(PyExc_IndexError, "map_id out of range");
        return NULL;
    }
    const PfrNativeMap* m = &gPfrNativeMaps[map_id];
    return Py_BuildValue("(siii)", m->name, (int)m->width, (int)m->height, (int)m->warp_count);
}

#define MY_METHODS \
    {"get_heatmap", get_heatmap, METH_NOARGS, "Get global heatmap (H,W) float32"}, \
    {"reset_to_map", py_reset_to_map, METH_VARARGS, "reset_to_map(handle, map_id, x, y)"}, \
    {"get_map_count", py_get_map_count, METH_NOARGS, "Number of maps"}, \
    {"get_map_info", py_get_map_info, METH_VARARGS, "get_map_info(map_id) -> (name, w, h, warps)"}

#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs)
{
    pfr_engine_init(&env->core);
    pfr_heatmap_ensure_alloc();
    c_reset(env);
    return 0;
}

static int my_log(PyObject *dict, Log *log)
{
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "unique_tiles", log->unique_tiles);
    assign_to_dict(dict, "unique_maps", log->unique_maps);
    assign_to_dict(dict, "warps_taken", log->warps_taken);
    return 0;
}
