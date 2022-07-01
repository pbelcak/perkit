import numpy as np
import math

def make_binned_series(x, y, bin_range: float, fn):
    time_range_low = x.min()
    time_range_high = x.max()
    time_range = time_range_high - time_range_low

    ret_x = []
    ret_y = []
    ret_indices = []
    
    bins: int = math.ceil(time_range / bin_range)
    for i in range(0, bins):
        ts_span_min = time_range_low + i * bin_range
        ts_span_max = ts_span_min + bin_range

        bin_x = None
        bin_y = None
        bin_indices = None

        indices_of_points_in_span = np.logical_and(ts_span_min <= x, x < ts_span_max).nonzero()[0]
        if not np.size(indices_of_points_in_span):
            bin_x = np.array([ (ts_span_min+ts_span_max)/2 ])
            bin_y = np.array([ fn( (ts_span_min+ts_span_max)/2 ) ])
            bin_indices = np.array([ None ])
        else:
            bin_x = x[indices_of_points_in_span]
            bin_y = y[indices_of_points_in_span]
            bin_indices = indices_of_points_in_span

        ret_x.append(bin_x)
        ret_y.append(bin_y)
        ret_indices.append(bin_indices)
        
    return ret_x, ret_y, ret_indices

def make_timeseries_from_binned_series(x_binned, y_binned, x_true_source, sequence_length: int, timestep_size: int = 1, stride: int = 1, iterations: int = 1):
    input_series = []
    series_outputs = []
    xses_for_targets = []
    for iteration in range(0, iterations):
        for offset in range(0, len(x_binned)-timestep_size*sequence_length, stride):
            series = []
            for ts in range(0, sequence_length):
                x_bin_index = offset + timestep_size*ts
                x_bin = x_binned[x_bin_index]
                x_choice = np.random.randint(low=0, high=x_bin.size)
                series.append(x_bin[x_choice])

            y_bin_index = offset + timestep_size*sequence_length
            y_bin = y_binned[y_bin_index]
            yx_bin = x_true_source[y_bin_index]
            y_choice = np.random.randint(low=0, high=y_bin.size)
            series_outputs.append(y_bin[y_choice])
            xses_for_targets.append(yx_bin[y_choice])

            input_series.append(np.array(series)[:, np.newaxis])
    
    return np.array(input_series), np.array(series_outputs), np.array(xses_for_targets)

def continuous_prediction(model, past, window_length, future_length: int):
    past = np.array(past)
    future = np.empty(shape=(past.size+future_length,))
    future[0:past.size] = past
    for i in range(past.size, past.size+future_length):
        relevant_recent_past = np.array(future[i-window_length:i])[:,np.newaxis]
        future[i] = model.predict(np.array([ relevant_recent_past ]))
    
    return future