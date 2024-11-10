import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from sympy import sin, cos, symbols, lambdify, sympify
import sympy2jax

# Failed with batching inputs, therefore not used


def convert_sindy_model_to_sympyjax_model(model, quantize=False, quantize_round_to=3):
    feature_library_names = model.feature_library.get_feature_names().copy()
    coefs = model.coefficients()
    feature_names = model.feature_names
    return convert_sindy_model_to_sympyjax_model_core(feature_library_names, feature_names, coefs, quantize=quantize, quantize_round_to=quantize_round_to)


def convert_sindy_model_to_sympyjax_model_core(feature_library_names, feature_names, coefs, quantize=False, quantize_round_to=3):
    feature_library_names = [fn.replace(' ', '*') for fn in feature_library_names]
    fln_l = []
    for fln in feature_library_names:
        for i in range(len(feature_names)):
            fln = fln.replace(f'x{i}', feature_names[i])
        fln_l.append(fln)
    feature_library_names = fln_l
    str_exp = ''
    mods = []
    str_exps = []
    for j in range(coefs.shape[0]):
        for i, coef in enumerate(coefs[0]):
            if np.abs(coef) > 1e-3:
                if quantize:
                    coef = np.round(coef, quantize_round_to)
                str_exp += f'+{coef}*' + feature_library_names[i]
        if not np.all(coefs == 0):
            expr = sympify(str_exp)
        else:
            expr = sympify('0.0')
        mod = sympy2jax.SymbolicModule(expr)
        mods.append(mod)
        str_exps.append(str_exp)
    # def f(**kwargs):
    #     return np.array([mod(**kwargs) for mod in mods])
    return mods, str_exps


def convert_sindy_model_to_sympy_model(model, quantize=False):
    feature_library_names = model.feature_library.get_feature_names().copy()
    coefs = model.coefficients()
    feature_names = model.feature_names
    feature_library_names = [fn.replace(' ', '*') for fn in feature_library_names]
    fln_l = []
    for fln in feature_library_names:
        for i in range(len(feature_names)):
            fln = fln.replace(f'x{i}', feature_names[i])
        fln_l.append(fln)
    feature_library_names = fln_l
    str_exp = ''
    exprs = []
    for j in range(coefs.shape[0]):
        for i, coef in enumerate(coefs[0]):
            if np.abs(coef) > 1e-3:
                if quantize:
                    coef = np.round(coef, 2)
                str_exp += f'{coef}*' + feature_library_names[i]
        expr = sympify(str_exp)
        exprs.append(expr)
    # def f(**kwargs):
    #     return np.array([expr(**kwargs) for expr in exprs])
    return exprs, str_exp