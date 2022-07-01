# local utils
from .forms import *
from .benchmark import *

# python imports
import argparse
import dill as pickle
import os

def main():
	args = parse_args()
	# print(args)

	# generate function forms
	elementary_forms = Benchmark.getAtomicForms(
		classes={"wave_trigonometric", "wave_polynomial_asymmetric"}
	)
	periodicForms = Benchmark.generateForms(
		sourceForms=elementary_forms,
		order=1,
		transformations={"offset", "amplitude"}
	)
	scaledForms = Benchmark.transform(
		sourceForms=periodicForms,
		transformationForms=[
			atomic_forms['constant_arbitrary'],
		],
		transformations = {"amplitude"}
	)
	finalizedForms = Benchmark.transform(
		sourceForms=scaledForms,
		transformationForms=[
			atomic_forms['linear'],
		],
		transformations = {"offset"}
	)
	# INSTEAD DO:
	# finalizedForms = scaledForms
	
	# generate various particular function variants for each form
	variants = Benchmark.generateVariants(
		forms=finalizedForms,
		variants_per_form=args.variants_per_form,
		master_period_range = (0.5, 1.0)
	)

	for variant, i in zip(variants, range(0, len(variants))):
		variant.id = i

	pickle_out = open(os.path.join(args.output_path, "variants.pickle"), "wb")
	pickle.dump(variants, pickle_out)
	pickle_out.close()

def parse_args():
	parser = argparse.ArgumentParser(
		description = "A tool for benchmarking of the ability of various neural network architectures " \
			"to perform periodic extrapolation of the learned signal beyond the training domain"
	)

	# parser argument setup
	parser.add_argument(
		"--variants-per-form",
		action="store",
		type=int,
		required=False,
		default=1,
		help="How many variants per function form to generate"
	)
	parser.add_argument(
		"--output-path",
		action="store",
		type=str,
		required=False,
		default=".",
		help="The path of the directory to be used for output"
	)

	# parse the args
	args = parser.parse_args()

	return args

if __name__ == "__main__":
	main()