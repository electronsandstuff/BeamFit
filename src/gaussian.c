/*
BeamFit - Robust laser and charged particle beam image analysis
Copyright (C) 2020 Christopher M. Pierce (contact@chris-pierce.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "Python.h"
#include "math.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"

static PyMethodDef GaussMethods[] = {
        {NULL, NULL, 0, NULL}
};

static void double_supergaussian_internal(char **args, const long *dimensions, const long* steps, void* data){
    npy_intp i,j;
    npy_intp n = dimensions[0];
    double *args_copy[11];

    // Copy over the args to hold onto their pointers
    for(j=0; j<11; j++){
      args_copy[j] = (double *)(args[j]);
    }

    double TMP_177, TMP_171, TMP_179;
    for (i=0; i<n; i++) {
        TMP_177 = 1/(-1*(*args_copy[5])*(*args_copy[5]) + (*args_copy[4])\
          *(*args_copy[6]));
        TMP_171 = *args_copy[1] - *args_copy[3];
        TMP_179 = *args_copy[0] - *args_copy[2];
        *args_copy[10]  = *args_copy[8];
        *args_copy[10] /= exp(pow(fabs(TMP_171*(*args_copy[4]*TMP_171*TMP_177\
          -*args_copy[5]*TMP_177*TMP_179) + TMP_179*(-(*args_copy[5]*TMP_171\
            *TMP_177)+*args_copy[6]*TMP_177*TMP_179)), *args_copy[7])/\
            pow(2.0, *args_copy[7]));
        *args_copy[10] += *args_copy[9];

        // Increment the pointers
        for(j=0; j<11; j++){
          args_copy[j] = (double *) ((char *)args_copy[j] + steps[j]);
        }
    }
}

static void double_supergaussian_grad_internal(char **args, const long *dimensions, const long* steps, void* data){
  npy_intp i,j;
  npy_intp n = dimensions[0];
  double *args_copy[20];

  // Copy over the args to hold onto their pointers
  for(j=0; j<20; j++){
    args_copy[j] = (double *)(args[j]);
  }

  double TMP_180, TMP_172, TMP_178, TMP_185, TMP_189, TMP_191, TMP_170, TMP_171;
  double TMP_192, TMP_193, TMP_195, TMP_199, TMP_205, TMP_206, TMP_200, TMP_202;
  double TMP_204, TMP_214, TMP_216, TMP_212, TMP_175;
  for (i=0; i<n; i++) {
      TMP_180 = *args_copy[2] - *args_copy[0];
      TMP_172 = *args_copy[5]**args_copy[5];
      TMP_178 = *args_copy[4]**args_copy[3]**args_copy[3];
      TMP_185 = *args_copy[4]**args_copy[1]**args_copy[1];
      TMP_189 = *args_copy[2]**args_copy[5] -(*args_copy[5]**args_copy[0]) + *args_copy[4]**args_copy[1];
      TMP_191 = TMP_178 + *args_copy[6]*TMP_180*TMP_180 + 2.0**args_copy[5]**args_copy[1]*TMP_180 + TMP_185 -2.0**args_copy[3]*TMP_189;
      TMP_170 = pow(2.0, 1.0-*args_copy[7]);
      TMP_171 = pow(2.0, -*args_copy[7]);
      TMP_175 = (*args_copy[4]**args_copy[6]) - TMP_172;
      TMP_192 = TMP_191/TMP_175;
      TMP_193 = pow(fabs(TMP_192),*args_copy[7]);
      TMP_195 = exp(-(TMP_171*TMP_193));
      TMP_199 = pow(fabs(*args_copy[4]**args_copy[6]-TMP_172),-*args_copy[7]);
      TMP_205 = -1.0 + *args_copy[7];
      TMP_206 = pow(fabs(TMP_191) + 1.0e-100, TMP_205);
      TMP_200 = *args_copy[3]**args_copy[5];
      TMP_202 = *args_copy[6]**args_copy[0];
      TMP_204 = TMP_200 -(*args_copy[2]**args_copy[6]) + TMP_202 -(*args_copy[5]**args_copy[1]);
      TMP_214 = 1/TMP_175/TMP_175;
      TMP_216 = pow(fabs(TMP_192) + 1.0e-100,TMP_205);
      TMP_212 = *args_copy[3]**args_copy[4] -(*args_copy[2]**args_copy[5]) + *args_copy[5]**args_copy[0] -(*args_copy[4]**args_copy[1]);

      *args_copy[10] = *args_copy[8]**args_copy[7]*TMP_170*TMP_195*TMP_199*TMP_204*TMP_206;
      *args_copy[11] = -(*args_copy[8]**args_copy[7]*TMP_170*TMP_195*TMP_199*TMP_206*TMP_212);
      *args_copy[12] = *args_copy[8]* *args_copy[7]*TMP_171*TMP_195*TMP_204*TMP_204*TMP_214*TMP_216;
      *args_copy[13] = *args_copy[8]**args_copy[7]*TMP_170*(-(*args_copy[3]**args_copy[4]) + TMP_189)*TMP_195*TMP_204*TMP_214*TMP_216;
      *args_copy[14] = *args_copy[8]**args_copy[7]*TMP_171*TMP_195*TMP_212*TMP_212*TMP_214*TMP_216;
      if(TMP_191 < 1e-50){
        *args_copy[15] = 0.0;
      }
      else{
        *args_copy[15] = *args_copy[8]*TMP_171*TMP_193*TMP_195*(log(fabs(2.0**args_copy[4]**args_copy[6] - 2.*TMP_172)) - log(fabs(TMP_191)));
      }
      *args_copy[16] = TMP_195;
      *args_copy[17] = 1;

      // Increment the pointers
      for(j=0; j<20; j++){
        args_copy[j] = (double *) ((char *)args_copy[j] + steps[j]);
      }
  }
}

PyUFuncGenericFunction funcs_gaussian[1] = {&double_supergaussian_internal};
static char types_gaussian[11] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
                                  NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
                                  NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
                                  NPY_DOUBLE, NPY_DOUBLE};

PyUFuncGenericFunction funcs_grad[1] = {&double_supergaussian_grad_internal};
static char types_grad[20] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
                              NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
                              NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
                              NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
                              NPY_DOUBLE, NPY_DOUBLE};

static void *data[1] = {NULL};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "gaussufunc",
    NULL,
    -1,
    GaussMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_gaussufunc(void){
    PyObject *m, *supergaussian, *grad, *d;

    // Setup our module
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    // Initialize the numpy API
    import_array();
    import_umath();

    d = PyModule_GetDict(m);

    // Add supergaussian
    supergaussian = PyUFunc_FromFuncAndData(funcs_gaussian, data, types_gaussian, 1, 10, 1, PyUFunc_None,
    "supergaussian_internal", "", 0);
    PyDict_SetItemString(d, "supergaussian_internal", supergaussian);
    Py_DECREF(supergaussian);

    // Add the gradient
    grad = PyUFunc_FromFuncAndData(funcs_grad, data, types_grad, 1, 10, 8, PyUFunc_None,
    "supergaussian_grad_internal", "", 0);
    PyDict_SetItemString(d, "supergaussian_grad_internal", grad);
    Py_DECREF(grad);

    return m;
}