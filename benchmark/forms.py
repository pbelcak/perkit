# general and math imports
import numpy as np
from numpy import random
from typing import Callable
from math import gcd # Python versions 3.5 and above
from functools import reduce # Python version 3.x
from typeguard import typechecked
import math

class FunctionForm:
    @typechecked
    def __init__(self, is_periodic: bool = False, is_discontinuous: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.is_periodic = is_periodic
        self.is_discontinuous = is_discontinuous
    def isConstant(self):
        return False
    def getPeriodicRoots(self):
        return []
    
    def __call__(self, other):
        return ComposedFunctionForm(fnOuter=self, fnInner=other)
    def __add__(self, other):
        return AddedFunctionForm(fnLeft=self, fnRight=other)
    def __mul__(self, other):
        return MultipliedFunctionForm(fnLeft=self, fnRight=other)
    
class FunctionVariant:
    @typechecked
    def __init__(self, form: FunctionForm, coeffs: dict):
        self.form = form
        self.coeffs = coeffs

    def __str__(self):
        if isinstance(self.form, EFF):
            s = str(self.form)
            for key, value in self.coeffs.items():
                s = s.replace(key, "{value: .3f}".format(value=value))
            return s
        
        if isinstance(self.form, ComposedFunctionForm):
            return str(self.coeffs['fnOuter']) + "(" + str(self.coeffs['fnInner']) + ")"

        if isinstance(self.form, AddedFunctionForm):
            return str(self.coeffs['fnLeft']) + " + " + str(self.coeffs['fnRight'])

        if isinstance(self.form, MultipliedFunctionForm):
            return str(self.coeffs['fnLeft']) + " * " + str(self.coeffs['fnRight'])

        raise Exception("Unknow type of self.form")

    
    def getPeriodMultipliers(self, particularCoeffs: dict = None) -> list:
        coeffs = self.coeffs if particularCoeffs is None else particularCoeffs
        
        periodMultipliers: list = []
        if 'period_multiplier' in coeffs:
            periodMultipliers.append(coeffs['period_multiplier'])
            
        # print(coeffs)
        
        if 'fnInner' in coeffs:
            periodMultipliers.extend(coeffs['fnInner'].getPeriodMultipliers())
        if 'fnOuter' in coeffs:
            periodMultipliers.extend(coeffs['fnOuter'].getPeriodMultipliers())
        if 'fnLeft' in coeffs:
            periodMultipliers.extend(coeffs['fnLeft'].getPeriodMultipliers())
        if 'fnRight' in coeffs:
            periodMultipliers.extend(coeffs['fnRight'].getPeriodMultipliers())
        
        return periodMultipliers
    
    def enforceCoeffs(self, coeffsToEnforce: dict, particularCoeffs: dict = None):
        coeffs = self.coeffs if particularCoeffs is None else particularCoeffs
        
        # print(coeffsToEnforce)
        for key, value in coeffsToEnforce.items():
            if key in coeffs:
                coeffs[key] = value
            
        if 'fnInner' in coeffs:
            coeffs['fnInner'].enforceCoeffs(coeffsToEnforce)
        if 'fnOuter' in coeffs:
            coeffs['fnOuter'].enforceCoeffs(coeffsToEnforce)
        if 'fnLeft' in coeffs:
            coeffs['fnLeft'].enforceCoeffs(coeffsToEnforce)
        if 'fnRight' in coeffs:
            coeffs['fnRight'].enforceCoeffs(coeffsToEnforce)
        
        
    def execute(self, x: float, *args, **kwargs):
        # print(str(self), self.coeffs)
        return self.form.execute(x, *args, **self.coeffs, **kwargs)

class EFF(FunctionForm): # EFF is short for encapsulated function form
    def __init__(self, fn: Callable, fnStringForm: str, builder: Callable=None, isConstantFn: Callable=None, classes: set = None, **kwargs):
        super().__init__(**kwargs)
        self.fn = fn
        self.builder = builder
        self.isConstantFn = isConstantFn
        self.fnStringForm = fnStringForm
        self.classes = classes if classes is not None else set()
    def isConstant(self):
        return self.isConstantFn()
    def getPeriodicRoots(self):
        return [ self ] if self.is_periodic else [ ]
    
    def build(self, **kwargs):
        return FunctionVariant(form=self, coeffs=self.builder(**kwargs))
    def execute(self, x: float, *args, **kwargs):
        return self.fn(x, *args, **kwargs)
    
    def __str__(self):
        return self.fnStringForm

class PeriodicizedFunctionForm(FunctionForm):
    def __init__(self, baseFn, classes: set = None, **kwargs):
        super().__init__(is_periodic=True, **kwargs)
        if isinstance(baseFn, str):
            self.baseFn = atomic_forms[baseFn]
        else:
            self.baseFn = baseFn
        self.classes = classes if classes is not None else set()
    def isConstant(self):
        return self.baseFn.isConstant()
    def getPeriodicRoots(self):
        return [ self ]
    
    def build(self, bounds: dict, **kwargs):
        return FunctionVariant(form=self, coeffs={
            'period_multiplier': random.randint(low=bounds['period_multiplier'][0], high=bounds['period_multiplier'][1]),
            'base_period': None,
            'phase_shift': random.uniform(bounds['phase_shift'][0], bounds['phase_shift'][1]),
            'centre_of_symmetry': random.uniform(bounds['centre_of_symmetry'][0], bounds['centre_of_symmetry'][1]),
            'centre_of_laterality': random.uniform(bounds['centre_of_laterality'][0], bounds['centre_of_laterality'][1]),
            **self.builder(bounds=bounds, **kwargs)
        })
    def execute(self, x: float, period_multiplier: float, base_period: float, phase_shift: float, centre_of_symmetry: float, centre_of_laterality: float, *args, **kwargs):
        periodRemainder = np.remainder( x/(period_multiplier * base_period) + phase_shift, 1.0)

        sign = np.where(periodRemainder < centre_of_laterality, False, True)

        x_lat = np.where(
            periodRemainder < centre_of_laterality,
            periodRemainder / centre_of_laterality,
            1 - (periodRemainder - centre_of_laterality)/( 1 - centre_of_laterality)
        )

        x_sym = np.where(
            x_lat < centre_of_symmetry,
            x_lat / centre_of_symmetry,
            1 - (x_lat - centre_of_symmetry)/( 1 - centre_of_symmetry)
        )

        return self.baseFn(x=sign * x_sym, *args, **kwargs)

    def __str__(self):
        return "wave_symmetric_bilateral(" + str(self.baseFn) + ", COS=centre_of_symmetry, COL=centre_of_laterality, period=period_multiplier*base_period, PS=phase_shift)"
        
class ComposedFunctionForm(FunctionForm):
    def __init__(self, fnOuter: FunctionForm, fnInner: FunctionForm, **kwargs):
        super().__init__(
            is_periodic=fnInner.is_periodic,
            is_discontinuous=fnOuter.is_discontinuous or fnInner.is_discontinuous, # potentially -- depends on the particular composition
            **kwargs
        )
        self.fnOuter = fnOuter
        self.fnInner = fnInner
    def isConstant(self):
        return self.fnOuter.isConstant() or self.fnInner.isConstant()
    def getPeriodicRoots(self):
        return [*self.fnOuter.getPeriodicRoots(), *self.fnInner.getPeriodicRoots()]
    
    def build(self, **kwargs):
        fnOuterParticular = self.fnOuter.build(**kwargs)
        fnInnerParticular = self.fnInner.build(**kwargs)
        return FunctionVariant(form=self, coeffs={'fnOuter': fnOuterParticular, 'fnInner': fnInnerParticular})
    def execute(self, x: float, fnOuter: FunctionVariant, fnInner: FunctionVariant, *args, **kwargs):
        return fnOuter.execute(fnInner.execute(x, *args, **kwargs), *args, **kwargs)
    
    def __str__(self):
        return "(" + str(self.fnOuter) + ").(" + str(self.fnInner) + ")"

class AddedFunctionForm(FunctionForm):
    def __init__(self, fnLeft: FunctionForm, fnRight: FunctionForm, **kwargs):
        super().__init__(
            is_periodic=fnLeft.is_periodic and fnRight.is_periodic,
            is_discontinuous=fnLeft.is_discontinuous or fnRight.is_discontinuous,
            **kwargs
        )
        self.fnLeft = fnLeft
        self.fnRight = fnRight
    def isConstant(self):
        return self.fnLeft.isConstant() and self.fnRight.isConstant()
    def getPeriodicRoots(self):
        return [*self.fnLeft.getPeriodicRoots(), *self.fnRight.getPeriodicRoots()]
    
    def build(self, **kwargs):
        fnLeft = self.fnLeft.build(**kwargs)
        fnRight = self.fnRight.build(**kwargs)
        return FunctionVariant(form=self, coeffs={'fnLeft': fnLeft, 'fnRight': fnRight})
    def execute(self, x: float, fnLeft: FunctionVariant, fnRight: FunctionVariant, *args, **kwargs):
        return fnLeft.execute(x, *args, **kwargs) + fnRight.execute(x, *args, **kwargs)
    
    def __str__(self):
        return str(self.fnLeft) + " + " + str(self.fnRight)
    
class MultipliedFunctionForm(FunctionForm):
    def __init__(self, fnLeft: FunctionForm, fnRight: FunctionForm, **kwargs):
        super().__init__(
            is_periodic=fnLeft.is_periodic and fnRight.is_periodic,
            is_discontinuous=fnLeft.is_discontinuous or fnRight.is_discontinuous,
            **kwargs
        )
        self.fnLeft = fnLeft
        self.fnRight = fnRight
    def isConstant(self):
        return self.fnLeft.isConstant() and self.fnRight.isConstant()
    def getPeriodicRoots(self):
        return [*self.fnLeft.getPeriodicRoots(), *self.fnRight.getPeriodicRoots()]
    
    def build(self, **kwargs):
        fnLeft = self.fnLeft.build(**kwargs)
        fnRight = self.fnRight.build(**kwargs)
        return FunctionVariant(form=self, coeffs={'fnLeft': fnLeft, 'fnRight': fnRight})
    def execute(self, x: float, fnLeft: FunctionVariant, fnRight: FunctionVariant, *args, **kwargs):
        return fnLeft.execute(x, *args, **kwargs) + fnRight.execute(x, *args, **kwargs)
    
    def __str__(self):
        return "(" + str(self.fnLeft) + ") * (" + str(self.fnRight) + ")"

def lcm(denominators):
    return reduce(lambda a,b: a*b // gcd(a,b), denominators, 1)

atomic_forms = {
    "constant_zero": EFF(
        fn=(lambda x, c0: np.full(x.shape, c0)),
        fnStringForm="c0",
        builder=lambda bounds: {
            'c0': 0.0
        },
        isConstantFn=lambda x: True,
        is_periodic=False,
        classes={ "constant" }
    ),
    "constant_one": EFF(
        fn=(lambda x, c0: np.full(x.shape, c0)),
        fnStringForm="c0",
        builder=lambda bounds: {
            'c0': 1.0
        },
        isConstantFn=lambda x: True,
        is_periodic=False,
        classes={ "constant" }
    ),
    "constant_arbitrary": EFF(
        fn=(lambda x, c0: np.full(x.shape, c0)),
        fnStringForm="c0",
        builder=lambda bounds: {
            'c0': random.uniform(bounds['c0'][0], bounds['c0'][1])
        },
        isConstantFn=lambda x: True,
        is_periodic=False,
        classes={ "constant" }
    ),
    "linear": EFF(
        fn=(lambda x, c0, c1: c1*x + c0),
        fnStringForm="c1*x+c0",
        builder=lambda bounds: {
            'c0': random.uniform(bounds['c0'][0], bounds['c0'][1]),
            'c1': random.uniform(bounds['c1'][0], bounds['c1'][1])
        },
        isConstantFn=lambda self: self.coeffs['c0'] == 0.0,
        classes={ "polynomial" }
    ),
    "quadratic": EFF(
        fn=(lambda x, c0, c1, c2: c2*x**2 + c1*x + c0),
        fnStringForm="c2*x^2+c1*x+c0",
        builder=lambda bounds: {
            'c0': random.uniform(bounds['c0'][0], bounds['c0'][1]),
            'c1': random.uniform(bounds['c1'][0], bounds['c1'][1]),
            'c2': random.uniform(bounds['c2'][0], bounds['c2'][1])
        },
        isConstantFn=lambda self: self.coeffs['c0'] == 0.0 and self.coeffs['c1'] == 0.0 and self.coeffs['c2'] == 0.0,
        classes={ "polynomial" }
    ),
    "cubic": EFF(
        fn=(lambda x, c0, c1, c2, c3: c3*x**3 + c2*x**2 + c1*x + c0),
        fnStringForm="c3*x^3+c2*x^2+c1*x+c0",
        builder=lambda bounds: {
            'c0': random.uniform(bounds['c0'][0], bounds['c0'][1]),
            'c1': random.uniform(bounds['c1'][0], bounds['c1'][1]),
            'c2': random.uniform(bounds['c2'][0], bounds['c2'][1]),
            'c3': random.uniform(bounds['c3'][0], bounds['c3'][1]),
        },
        isConstantFn=lambda self: self.coeffs['c0'] == 0.0 and self.coeffs['c1'] == 0.0 and self.coeffs['c2'] == 0.0 and self.coeffs['c3'] == 0.0,
        classes={ "polynomial" }
    ),
    "quadric": EFF(
        fn=(lambda x, c0, c1, c2, c3, c4: c4*x**4 + c3*x**3 + c2*x**2 + c1*x + c0),
        fnStringForm="c4*x^4+c3*x^3+c2*x^2+c1*x+c0",
        builder=lambda bounds: {
            'c0': random.uniform(bounds['c0'][0], bounds['c0'][1]),
            'c1': random.uniform(bounds['c1'][0], bounds['c1'][1]),
            'c2': random.uniform(bounds['c2'][0], bounds['c2'][1]),
            'c3': random.uniform(bounds['c3'][0], bounds['c3'][1]),
            'c4': random.uniform(bounds['c4'][0], bounds['c4'][1]),
        },
        isConstantFn=lambda self: self.coeffs['c0'] == 0.0 and self.coeffs['c1'] == 0.0 and self.coeffs['c2'] == 0.0 and self.coeffs['c3'] == 0.0 and self.coeffs['c4'] == 0.0,
        classes={ "polynomial" }
    ),
    
    "sin": EFF(
        fn=(lambda x, period_multiplier, base_period, phase_shift: np.sin( 2*math.pi*1.0/(period_multiplier * base_period)*(x + phase_shift) )),
        fnStringForm="sin(2*pi*(period_multiplier * base_period)*(x + phase_shift))",
        builder=lambda bounds: {
            'period_multiplier': random.randint(low=bounds['period_multiplier'][0], high=bounds['period_multiplier'][1]),
            'base_period': None,
            'phase_shift': random.uniform(bounds['phase_shift'][0], bounds['phase_shift'][1])
        },
        isConstantFn=lambda self: self.coeffs['period'] == 0.0,
        is_periodic=True,
        classes={ "wave", "wave_symmetric", "wave_bilateral", "wave_trigonometric" }
    ),
    "tan": EFF(
        fn=(lambda x, period_multiplier, base_period, phase_shift: np.tan( 2*math.pi*1.0/(period_multiplier * base_period)*(x + phase_shift) )),
        fnStringForm="tan(2*pi*(period_multiplier * base_period)*(x + phase_shift))",
        builder=lambda bounds: {
            'period_multiplier': random.randint(low=bounds['period_multiplier'][0], high=bounds['period_multiplier'][1]),
            'base_period': None,
            'phase_shift': random.uniform(bounds['phase_shift'][0], bounds['phase_shift'][1])
        },
        isConstantFn=lambda self: self.coeffs['period'] == 0.0,
        is_periodic=True,
        is_discontinuous=True,
        classes={ "wave", "wave_symmetric", "wave_bilateral", "wave_trigonometric" }
    ),

    "saw": EFF( # SAWTOOTH
        fn=(lambda x, period_multiplier, base_period, phase_shift: np.remainder( x/(period_multiplier * base_period) + phase_shift, 1.0)),
        fnStringForm="saw(x/(period_multiplier * base_period) + phase_shift, 1.0)",
        builder=lambda bounds: {
            'period_multiplier': random.randint(low=bounds['period_multiplier'][0], high=bounds['period_multiplier'][1]),
            'base_period': None,
            'phase_shift': random.uniform(bounds['phase_shift'][0], bounds['phase_shift'][1])
        },
        isConstantFn=lambda self: False,
        is_periodic=True,
        is_discontinuous=True,
        classes={ "wave", "wave_asymmetric", "wave_polynomial", "wave_polynomial_asymmetric" }
    ),

    "wave_constant": EFF( # 50% DUTY CYCLE CLOCK SIGNAL
        fn=lambda x, period_multiplier, base_period, phase_shift: np.where(
            np.remainder( x/(period_multiplier * base_period) + phase_shift, 1.0) < 0.5,
            1.0,
            0.0
        ),
        fnStringForm="constant_wave(x/(period_multiplier * base_period) + phase_shift, 1.0)",
        builder=lambda bounds: {
            'period_multiplier': random.randint(low=bounds['period_multiplier'][0], high=bounds['period_multiplier'][1]),
            'base_period': None,
            'phase_shift': random.uniform(bounds['phase_shift'][0], bounds['phase_shift'][1])
        },
        isConstantFn=lambda self: False,
        is_periodic=True,
        is_discontinuous=True,
        classes={ "wave", "wave_symmetric", "wave_constant", "wave_polynomial", "wave_polynomial_symmetric" }
    ),
}
advanced_atomic_forms = {
    "wave_linear": EFF(
        fn=(lambda x, period_multiplier, base_period, phase_shift, c1, c0: c0 + c1*np.remainder( x/(period_multiplier * base_period) + phase_shift, 1.0)),
        fnStringForm="c0 + c1*saw(x/(period_multiplier * base_period) + phase_shift, 1.0)",
        builder=lambda bounds: {
            'period_multiplier': random.randint(low=bounds['period_multiplier'][0], high=bounds['period_multiplier'][1]),
            'base_period': None,
            'phase_shift': random.uniform(bounds['phase_shift'][0], bounds['phase_shift'][1]),
            'c1': random.uniform(bounds['c1'][0], bounds['c1'][1]),
            'c0': random.uniform(bounds['c0'][0], bounds['c0'][1]),
        },
        isConstantFn=lambda self: False,
        is_periodic=True,
        is_discontinuous=True,
        classes={ "wave", "wave_asymmetric", "wave_polynomial", "wave_polynomial_asymmetric" }
    ),

    "wave_linear_symmetric_bilateral": PeriodicizedFunctionForm(
        baseFn="linear",
        classes={ "wave", "wave_symmetric", "wave_bilateral", "wave_polynomial", "wave_polynomial_symmetric", "wave_polynomial_bilateral", "wave_polynomial_symmetric_bilateral" }
    ),
    "wave_quadratic_symmetric_bilateral": PeriodicizedFunctionForm(
        baseFn="quadratic",
        classes={ "wave", "wave_symmetric", "wave_bilateral", "wave_polynomial", "wave_polynomial_symmetric", "wave_polynomial_bilateral", "wave_polynomial_symmetric_bilateral" }
    ),
    "wave_cubic_symmetric_bilateral": PeriodicizedFunctionForm(
        baseFn="cubic",
        classes={ "wave", "wave_symmetric", "wave_bilateral", "wave_polynomial", "wave_polynomial_symmetric", "wave_polynomial_bilateral", "wave_polynomial_symmetric_bilateral" }
    ),
    "wave_quadric_symmetric_bilateral": PeriodicizedFunctionForm(
        baseFn="quadric",
        classes={ "wave", "wave_symmetric", "wave_bilateral", "wave_polynomial", "wave_polynomial_symmetric", "wave_polynomial_bilateral", "wave_polynomial_symmetric_bilateral" }
    )
}