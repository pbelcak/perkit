# PerKit -- A toolkit for the study of periodicity in neural networks

PerKit is a toolkit consisting of two parts -- a bechmarking dataset generator, and an evaluation setup featuring a plethora of models and metrics, readily implemented.

## Using the Benchmarking Dataset Generator
This is the `benchmark` python module encapsulated as a directory within this repository.

```bash
> python -m perkit.benchmark --help
usage: __main__.py [-h] [--variants-per-form VARIANTS_PER_FORM] [--output-path OUTPUT_PATH]

A tool for benchmarking of the ability of various neural network architectures to perform periodic extrapolation of
the learned signal beyond the training domain

optional arguments:
  -h, --help            show this help message and exit
  --variants-per-form VARIANTS_PER_FORM
                        How many variants per function form to generate
  --output-path OUTPUT_PATH
                        The path of the directory to be used for output
```

## Using the Evaluator
The evaluator is available as a module in the `evaluation` directory.

```bash
>python -m perkit.evaluation --help
usage: __main__.py [-h]
                   [--optimizers {adamax,rmsprop,adam,nadam,sgd,adadelta} [{adamax,rmsprop,adam,nadam,sgd,adadelta} ...]]
                   [--trials-count TRIALS_COUNT] [--width-min WIDTH_MIN] [--width-max WIDTH_MAX]
                   [--width-step WIDTH_STEP] [--sequence-length SEQUENCE_LENGTH] [--epochs-max EPOCHS_MAX]
                   [--radius-training RADIUS_TRAINING] [--sampling-method-training {random,linspace}]
                   [--sampling-method-evaluation {random,linspace}] [--sampling-sides {positive,negative,both}]
                   [--samples-per-period SAMPLES_PER_PERIOD] [--bins-per-period BINS_PER_PERIOD]
                   [--noise-stddev NOISE_STDDEV] [--radius-evaluation RADIUS_EVALUATION] [--eval-in-training-radius]
                   [--epsilon-metric-fraction EPSILON_METRIC_FRACTION]
                   [--epsilon-metric-samples EPSILON_METRIC_SAMPLES] [--max-generations MAX_GENERATIONS]
                   [--initial-population-size INITIAL_POPULATION_SIZE] [--root-population-size ROOT_POPULATION_SIZE]
                   [--children-population-size CHILDREN_POPULATION_SIZE] [--variants-path VARIANTS_PATH]
                   [--job-id JOB_ID] [--output-path OUTPUT_PATH]
                   {ff_sin,ff_cos,ff_sincos,ff_xsin,ff_xcos,ff_snake,ff_tsnake,decoder_srnn,decoder_gru,decoder_lstm,bayes,genetic_pareto,genetic_nfittest}
                   [{ff_sin,ff_cos,ff_sincos,ff_xsin,ff_xcos,ff_snake,ff_tsnake,decoder_srnn,decoder_gru,decoder_lstm,bayes,genetic_pareto,genetic_nfittest} ...]

A tool for benchmarking of the ability of various neural network architectures to perform periodic extrapolation of
the learned signal beyond the training domain

positional arguments:
  {ff_sin,ff_cos,ff_sincos,ff_xsin,ff_xcos,ff_snake,ff_tsnake,decoder_srnn,decoder_gru,decoder_lstm,bayes,genetic_pareto,genetic_nfittest}
                        Which models to test

optional arguments:
  -h, --help            show this help message and exit
  --optimizers {adamax,rmsprop,adam,nadam,sgd,adadelta} [{adamax,rmsprop,adam,nadam,sgd,adadelta} ...]
                        Which optimizers to use
  --trials-count TRIALS_COUNT
                        How many times to repeat the training and evaluation process
  --width-min WIDTH_MIN
                        The minimum hidden layer width of tested networks
  --width-max WIDTH_MAX
                        The maximum hideen layer width of tested networks
  --width-step WIDTH_STEP
                        The increment to be used to generate width between width_min and width_max+1
  --sequence-length SEQUENCE_LENGTH
                        The length of the recurrent sequences
  --epochs-max EPOCHS_MAX
                        How many epochs at most to run
  --radius-training RADIUS_TRAINING
                        How many master periods away from origin to use for training
  --sampling-method-training {random,linspace}
                        Which method to use for sampling of training data
  --sampling-method-evaluation {random,linspace}
                        Which method to use for sampling of evaluation data
  --sampling-sides {positive,negative,both}
                        Which method to use of sampling
  --samples-per-period SAMPLES_PER_PERIOD
                        Sampling rate (#/period) for training and evaluation
  --bins-per-period BINS_PER_PERIOD
                        If binned random observations are to be used in a recurrent model, then this parameter
                        specifies how many bins to locate per period
  --noise-stddev NOISE_STDDEV
                        The standard deviation of the normal noise. Is applied after normalisation. 0.0 means no noise
                        (default), we recommend that 3stddev<2
  --radius-evaluation RADIUS_EVALUATION
                        How many master periods away from origin to use for evaluation
  --eval-in-training-radius
                        Allow evaluation data to be also drawn from periods in the training radius
  --epsilon-metric-fraction EPSILON_METRIC_FRACTION
                        The fraction of the master period to be used as epsilon
  --epsilon-metric-samples EPSILON_METRIC_SAMPLES
                        The number of samples to take of the epsilon range for optimisation of base metrics
  --max-generations MAX_GENERATIONS
                        The number of generations to use in the training of genetic models
  --initial-population-size INITIAL_POPULATION_SIZE
                        The number of pops to begin with in the training of genetic models
  --root-population-size ROOT_POPULATION_SIZE
                        The number of pops ancestrally related to initial pops to preserve
  --children-population-size CHILDREN_POPULATION_SIZE
                        The number of children to generate in a single generation
  --variants-path VARIANTS_PATH
                        Path to the pickle file (including the name and extension) containing the variants
  --job-id JOB_ID       The job ID to be used as a prefix for output files
  --output-path OUTPUT_PATH
                        The path of the directory to be used for output
```