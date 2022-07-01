# general and math imports
import math
import itertools
import numpy as np
from numpy import random
from typing import Tuple, List
from copy import deepcopy
from typeguard import typechecked

from .forms import *

class Benchmark:
    @classmethod
    # classes of atomic (elementary) forms to be an argument to this one
    def getAtomicForms(cls, classes:set = set({"polynomial", "wave_symmetric", "wave_trigonometric"})):
        atomic_forms_to_return = []

        for atomic_form in atomic_forms.values():
            if any(c in atomic_form.classes for c in classes):
                atomic_forms_to_return.append(atomic_form)

        return atomic_forms_to_return

    @classmethod
    def transform(cls, sourceForms: list, transformationForms: list, transformations: set = {"offset", "amplitude", "phase"}):
        listsToMerge = []

        # offset addition
        if "offset" in transformations or "o" in transformations:
            for transformationForm in transformationForms:
                listsToMerge.append([transformationForm + sourceForm for sourceForm in sourceForms])

        # amplitude multiplication
        if "amplitude" in transformations or "a" in transformations:
            for transformationForm in transformationForms:
                listsToMerge.append([transformationForm * sourceForm for sourceForm in sourceForms])

        # amplitude multiplication
        if "phase" in transformations or "p" in transformations:
            for transformationForm in transformationForms:
                listsToMerge.append([transformationForm(sourceForm) for sourceForm in sourceForms])
        
        ret = list(itertools.chain.from_iterable(listsToMerge))
        return ret
        
    @classmethod
    def generateForms(cls, sourceForms: list, order: int, transformations: set = {"offset", "amplitude", "phase"}, include_lower_orders: bool = True):
        core = None
        if order > 0:
            core = cls.generateForms(sourceForms=sourceForms, order=order-1, transformations=transformations)
        else:
            return deepcopy(sourceForms)
        
        listsToMerge = [core] if include_lower_orders else []
        
        listsToMerge.append(cls.transform(sourceForms=core, transformationForms=sourceForms, transformations=transformations))
            
        new_core = list(itertools.chain.from_iterable(listsToMerge))
        return new_core
    
    @classmethod
    @typechecked
    def generateVariants(cls, forms: List[FunctionForm], variants_per_form: int, master_period_range: Tuple[float, float] = (0.5, 1.0)):
        variants: List[FunctionVariant] = []
        for form in forms:
            for variantId in range(0, variants_per_form):
                particularFn = form.build(bounds={
                    'c0': (0, 1),
                    'c1': (0, 1),
                    'c2': (0, 1),
                    'c3': (0, 1),
                    'c4': (0, 1),
                    'period_multiplier': (1, 10),
                    'phase_shift': (0, 1)
                })
                
                periodMultipliers = particularFn.getPeriodMultipliers()
                if periodMultipliers:
                    periodMultiplierLCM = lcm(periodMultipliers)
                    master_period = random.uniform(master_period_range[0], master_period_range[1])
                    base_period = master_period / periodMultiplierLCM
                    particularFn.enforceCoeffs({'base_period': base_period})
                    particularFn.base_period = base_period
                    particularFn.master_period = master_period
                    particularFn.periodMultiplierLCM = periodMultiplierLCM
                else:
                    particularFn.base_period = 1.0
                    particularFn.master_period = 1.0
                    particularFn.periodMultiplierLCM = 1
                
                variants.append(particularFn)
                
        return variants
