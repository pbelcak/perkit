from .metrics import distance_adjusted_mse, distance_adjusted_mae, least_metrics_by_periodic_shift, least_metrics_by_periodic_acceleration, least_metrics_by_periodic_speedup

class EvaluationModel:
	def __init__(self, width, args):
		self.width = width
		self.isRecurrent = False
	
	def train(self, x, y, pivots, optimizer, epochs_max):
		pass

	def evaluate(self, x, y, pivots, train_y, y_normalizer: float, variant, training_radius: int, epsilon: float, samples_per_epsilon: int):
		variant_evaluation_x, variant_evaluation_y, variant_evaluation_predictions, eval_mse, eval_mae = self.prepare_evaluation_data(x, y, train_y, y_normalizer)

		training_radius = training_radius * variant.master_period

		da_mse = distance_adjusted_mse(
			inputs=variant_evaluation_x,
			y_true=variant_evaluation_y,
			y_pred=variant_evaluation_predictions,
			training_radius=training_radius,
			alpha = 1.0
		)
		da_mae = distance_adjusted_mae(
			inputs=variant_evaluation_x,
			y_true=variant_evaluation_y,
			y_pred=variant_evaluation_predictions,
			training_radius=training_radius,
			alpha = 1.0
		)

		least_mse_by_periodic_shift, least_mae_by_periodic_shift = least_metrics_by_periodic_shift(
			epsilon=epsilon,
			variant=variant,
			inputs=variant_evaluation_x,
			predictions=variant_evaluation_predictions,
			training_radius=training_radius,
			metrics=[distance_adjusted_mse, distance_adjusted_mae],
			y_normalizer=y_normalizer,
			sample_count=samples_per_epsilon
		)

		least_mse_by_periodic_speedup, least_mae_by_periodic_speedup = least_metrics_by_periodic_speedup(
			epsilon=epsilon,
			variant=variant,
			inputs=variant_evaluation_x,
			predictions=variant_evaluation_predictions,
			training_radius=training_radius,
			metrics=[distance_adjusted_mse, distance_adjusted_mae],
			y_normalizer=y_normalizer,
			sample_count=samples_per_epsilon
		)

		least_mse_by_periodic_acceleration, least_mae_by_periodic_acceleration = least_metrics_by_periodic_acceleration(
			epsilon=epsilon,
			variant=variant,
			inputs=variant_evaluation_x,
			predictions=variant_evaluation_predictions,
			training_radius=training_radius,
			metrics=[distance_adjusted_mse, distance_adjusted_mae],
			y_normalizer=y_normalizer,
			sample_count=samples_per_epsilon
		)

		evaluation_row_basis = {
			'Evaluation MSE': float(eval_mse),
			'Evaluation MAE': float(eval_mae),
			'Distance-Adjusted MSE': float(da_mse),
			'Distance-Adjusted MAE': float(da_mae),
			
			'Least DA-MSE by Periodic Shift': float(least_mse_by_periodic_shift),
			'Least DA-MAE by Periodic Shift': float(least_mae_by_periodic_shift),
			'Least DA-MSE by Periodic Speedup': float(least_mse_by_periodic_speedup),
			'Least DA-MAE by Periodic Speedup': float(least_mae_by_periodic_speedup),
			'Least DA-MSE by Periodic Acceleration': float(least_mse_by_periodic_acceleration),
			'Least DA-MAE by Periodic Acceleration': float(least_mae_by_periodic_acceleration)
		}

		return evaluation_row_basis
