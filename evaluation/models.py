
# model imports
from .feedforward.sin_model import SinModel
from .feedforward.cos_model import CosModel
from .feedforward.sincos_model import SinCosModel
from .feedforward.xsin_model import XSinModel
from .feedforward.xcos_model import XCosModel
from .feedforward.snake_model import SnakeModel, TSnakeModel

from .decoder.SRNN_model import SRNNModel
from .decoder.GRU_model import GRUModel
from .decoder.LSTM_model import LSTMModel
from .genetic.bayes import BayesModel
from .genetic.pareto import ParetoModel
from .genetic.nfittest import nFittestModel

model_map = {
	# name: (python type, is sequence based?)
	"ff_sin":			(SinModel,		False),
	"ff_cos":			(CosModel,		False),
	"ff_sincos":		(SinCosModel,	False),
	"ff_xsin":			(XSinModel,		False),
	"ff_xcos":			(XCosModel,		False),
	"ff_snake":			(SnakeModel,	False),
	"ff_tsnake":		(TSnakeModel,	False),

	"decoder_srnn":		(SRNNModel,		True),
	"decoder_gru":		(GRUModel,		True),
	"decoder_lstm":		(LSTMModel,		True),

	"bayes":			(BayesModel,	False),
	"genetic_pareto":	(ParetoModel,	False),
	"genetic_nfittest":	(nFittestModel,	False),
}
