import cgp

class Const255(cgp.OperatorNode):
    _arity = 0
    _def_output = "255"


class Const0(cgp.OperatorNode):
    _arity = 0
    _def_output = "0"


class Identity(cgp.OperatorNode):
    _arity = 1
    _def_output = "x_0"


class Inversion(cgp.OperatorNode):
    _arity = 1
    _def_output = "255-x_0"


class Max(cgp.OperatorNode):
    _arity = 2
    _def_output = "max(x_0,x_1)"
    _def_numpy_output = "np.max([x_0,x_1])"
    _def_torch_output = "torch.max(x_0,x_1)"
    _def_sympy_output = "max(x_0,x_1)"


class Min(cgp.OperatorNode):
    _arity = 2
    _def_output = "min(x_0,x_1)"
    _def_numpy_output = "np.min([x_0,x_1])"
    _def_torch_output = "torch.min(x_0,x_1)"
    _def_sympy_output = "min(x_0,x_1)"


class Div2(cgp.OperatorNode):
    _arity = 1
    _def_output = "x_0 // 2"
    _def_numpy_output = "x_0 // 2"
    _def_torch_output = "x_0 // 2"
    _def_sympy_output = "x_0 // 2"


class Div4(cgp.OperatorNode):
    _arity = 1
    _def_output = "x_0 // 4"
    _def_numpy_output = "x_0 // 4"
    _def_torch_output = "x_0 // 4"
    _def_sympy_output = "x_0 // 4"


class Add(cgp.OperatorNode):
    _arity = 2
    _def_output = "x_0 + x_1"
    _def_numpy_output = "x_0 + x_1"
    _def_torch_output = "x_0 + x_1"
    _def_sympy_output = "x_0 + x_1"


class ConditionalAssignment(cgp.OperatorNode):
    _arity = 2
    _def_output = "x_1 if x_0 > 127 else x_0"
    _def_numpy_output = "x_1[0] if x_0[0] > 127 else x_0[0]"
    _def_torch_output = "x_1 if x_0 > 127 else x_0"
    _def_sympy_output = " Piecewise((x_1, x_0 > 127), (x_0, True))"


class Sub(cgp.OperatorNode):
    _arity = 2
    _def_output = "x_0 - x_1"
    _def_numpy_output = "x_0 - x_1"
    _def_torch_output = "x_0 - x_1"
    _def_sympy_output = "x_0 - x_1"


class AddS(cgp.OperatorNode):
    _arity = 2
    _def_output = "min(x_0 + x_1, 255)"
    _def_numpy_output = "np.min([x_0[0] + x_1[0], 255])"
    _def_torch_output = "torch.min(x_0 + x_1, 255)"
    _def_sympy_output = "min(x_0 + x_1, 255)"


class SubS(cgp.OperatorNode):
    _arity = 2
    _def_output = "max(x_0 - x_1, 0)"
    _def_numpy_output = "np.max([x_0[0] - x_1[0], 0])"
    _def_torch_output = "torch.max(x_0 - x_1, 0)"
    _def_sympy_output = "max(x_0 - x_1, 0)"


class Avg(cgp.OperatorNode):
    _arity = 2
    _def_output = "(x_0 + x_1) // 2"
    _def_numpy_output = "(x_0 + x_1) // 2"
    _def_torch_output = "(x_0 + x_1) // 2"
    _def_sympy_output = "(x_0 + x_1) // 2"


class AbsDiff(cgp.OperatorNode):
    _arity = 2
    _def_output = "abs(x_0-x_1)"
    _def_numpy_output = "abs(x_0-x_1)"
    _def_torch_output = "abs(x_0-x_1)"
    _def_sympy_output = "abs(x_0-x_1)"


class Mul2(cgp.OperatorNode):
    _arity = 1
    _def_output = "x_0 * 2"
    _def_numpy_output = "x_0 * 2"
    _def_torch_output = "x_0 * 2"
    _def_sympy_output = "x_0 * 2"


class Mul4(cgp.OperatorNode):
    _arity = 1
    _def_output = "x_0 * 4"
    _def_numpy_output = "x_0 * 4"
    _def_torch_output = "x_0 * 4"
    _def_sympy_output = "x_0 * 4"
