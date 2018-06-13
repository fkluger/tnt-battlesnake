import tensorflow as tf
import numpy as np
from tensorboard import summary
from tensorboard.plugins.custom_scalar import layout_pb2


class MetricType:
    Value = 0
    Histogram = 1
    Image = 2


class Metric:

    def __init__(self, name, metric_type, value, global_step):
        self.name = name
        self.metric_type = metric_type
        self.value = value
        self.global_step = global_step


class TensorboardLogger:

    def __init__(self, output_directory, actor_idx=None):
        if actor_idx is not None:
            self.output_directory = f'{output_directory}/actor-{actor_idx}'
        else:
            self.output_directory = f'{output_directory}/learner'

        self.writer = tf.summary.FileWriter(self.output_directory)
        if actor_idx == 1:
            self._init_custom_scalar_layout()

    def log(self, metric):
        if metric.metric_type == MetricType.Value:
            self._log_value(metric)
        elif metric.metric_type == MetricType.Histogram:
            self._log_histogram(metric)
        self.writer.flush()

    def _init_custom_scalar_layout(self):
        layout = layout_pb2.Layout(
            category=[
                layout_pb2.Category(
                    title='mean rewards',
                    chart=[
                        layout_pb2.Chart(
                            title='mean rewards per actor',
                            multiline=layout_pb2.MultilineChartContent(tag=[r'actor-\d+/mean rewards']))
                    ]),
                layout_pb2.Category(
                    title='mean episode lengths',
                    chart=[
                        layout_pb2.Chart(
                            title='mean episode length per actor',
                            multiline=layout_pb2.MultilineChartContent(tag=[r'actor-\d+/mean episode lengths']))
                    ]),
                layout_pb2.Category(
                    title='mean fruits eaten',
                    chart=[
                        layout_pb2.Chart(
                            title='mean fruits eaten per actor',
                            multiline=layout_pb2.MultilineChartContent(tag=[r'actor-\d+/mean fruits eaten']))
                    ])
            ])
        self.writer.add_summary(summary.custom_scalar_pb(layout))

    def _log_value(self, metric):
        self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(
            tag=metric.name, simple_value=metric.value)]), global_step=metric.global_step)

    def _log_histogram(self, metric):
        self.writer.add_summary(self._create_histogram(metric.name, metric.value), global_step=metric.global_step)

    def _create_histogram(self, tag, values, bins=1000):

        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        return tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
