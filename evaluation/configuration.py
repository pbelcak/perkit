import argparse

from .models import model_map

def parse_cmd_arguments():
	parser = argparse.ArgumentParser(
		description = "A tool for benchmarking of the ability of various neural network architectures " \
			"to perform periodic extrapolation of the learned signal beyond the training domain"
	)

	# the only (and the key) positional argument
	parser.add_argument(
		"models",
		action="append",
		choices=model_map.keys(),
		nargs='+',
		help="Which models to test"
	)

	# runtime bench setup config
	parser.add_argument(
		"--optimizers",
		action="store",
		choices=optimizers_set,
		nargs='+',
		default=["sgd", "rmsprop", "adam"],
		help="Which optimizers to use"
	)
	parser.add_argument(
		"--trials-count",
		action="store",
		type=int,
		required=False,
		default=5,
		help="How many times to repeat the training and evaluation process"
	)

	# model config
	parser.add_argument(
		"--width-min",
		action="store",
		type=int,
		required=False,
		default=2,
		help="The minimum hidden layer width of tested networks"
	)
	parser.add_argument(
		"--width-max",
		action="store",
		type=int,
		required=False,
		default=2,
		help="The maximum hideen layer width of tested networks"
	)
	parser.add_argument(
		"--width-step",
		action="store",
		type=int,
		required=False,
		default=1,
		help="The increment to be used to generate width between width_min and width_max+1"
	)
	parser.add_argument(
		"--sequence-length",
		action="store",
		type=int,
		required=False,
		default=7,
		help="The length of the recurrent sequences"
	)
	
	# training config
	parser.add_argument(
		"--epochs-max",
		action="store",
		type=int,
		required=False,
		default=10,
		help="How many epochs at most to run"
	)
	parser.add_argument(
		"--radius-training",
		action="store",
		type=int,
		required=False,
		default=5,
		help="How many master periods away from origin to use for training"
	)
	parser.add_argument(
		"--sampling-method-training",
		action="store",
		choices=[ "random", "linspace" ],
		default="random",
		help="Which method to use for sampling of training data"
	)
	parser.add_argument(
		"--sampling-method-evaluation",
		action="store",
		choices=[ "random", "linspace" ],
		default="linspace",
		help="Which method to use for sampling of evaluation data"
	)
	parser.add_argument(
		"--sampling-sides",
		action="store",
		choices=[ "positive", "negative", "both" ],
		default="both",
		help="Which method to use of sampling"
	)
	parser.add_argument(
		"--samples-per-period",
		action="store",
		type=int,
		required=False,
		default=100,
		help="Sampling rate (#/period) for training and evaluation"
	)
	parser.add_argument(
		"--bins-per-period",
		action="store",
		type=int,
		required=False,
		default=10,
		help="If binned random observations are to be used in a recurrent model, then this parameter specifies how many bins to locate per period"
	)
	parser.add_argument(
		"--noise-stddev",
		action="store",
		type=float,
		required=False,
		default=0.0,
		help="The standard deviation of the normal noise. Is applied after normalisation. 0.0 means no noise (default), we recommend that 3stddev<2"
	)
	
	# eval config
	parser.add_argument(
		"--radius-evaluation",
		action="store",
		type=int,
		required=False,
		default=10,
		help="How many master periods away from origin to use for evaluation"
	)
	parser.add_argument(
		"--eval-in-training-radius",
		action="store_true",
		required=False,
		default=False,
		help="Allow evaluation data to be also drawn from periods in the training radius"
	)
	parser.add_argument(
		"--epsilon-metric-fraction",
		action="store",
		type=float,
		required=False,
		default=0.05,
		help="The fraction of the master period to be used as epsilon"
	)
	parser.add_argument(
		"--epsilon-metric-samples",
		action="store",
		type=int,
		required=False,
		default=10,
		help="The number of samples to take of the epsilon range for optimisation of base metrics"
	)

	# genetic config
	parser.add_argument(
		"--max-generations",
		action="store",
		type=int,
		required=False,
		default=10,
		help="The number of generations to use in the training of genetic models"
	)
	parser.add_argument(
		"--initial-population-size",
		action="store",
		type=int,
		required=False,
		default=8,
		help="The number of pops to begin with in the training of genetic models"
	)
	parser.add_argument(
		"--root-population-size",
		action="store",
		type=int,
		required=False,
		default=3,
		help="The number of pops ancestrally related to initial pops to preserve"
	)
	parser.add_argument(
		"--children-population-size",
		action="store",
		type=int,
		required=False,
		default=6,
		help="The number of children to generate in a single generation"
	)
	
	# IO config
	parser.add_argument(
		"--variants-path",
		action="store",
		type=str,
		required=False,
		default="variants.pickle",
		help="Path to the pickle file (including the name and extension) containing the variants"
	)
	parser.add_argument(
		"--job-id",
		action="store",
		type=int,
		required=False,
		default=0,
		help="The job ID to be used as a prefix for output files"
	)
	parser.add_argument(
		"--output-path",
		action="store",
		type=str,
		required=False,
		default=".",
		help="The path of the directory to be used for output"
	)

	##################
	# parse the args #
	##################
	args = parser.parse_args()

	return args

optimizers_set = { "sgd", "rmsprop", "adam", "adadelta", "adamax", "nadam" }
