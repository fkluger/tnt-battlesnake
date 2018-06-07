import tensorflow as tf
import numpy as np
from keras.callbacks import TensorBoard

def log_histogram(tag, values, bins=1000):

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

def write_summaries(summary_writer, metrics, steps=None):
    for metric in metrics:
        if metric['type'] == 'value':
            summary_writer.add_summary(
                tf.Summary(value=[
                    tf.Summary.Value(
                        tag=metric['name'], simple_value=metric['value'])
                ]),
                global_step=steps)
        elif metric['type'] == 'histogram':
            if metric['value']:
                summary_writer.add_summary(
                    log_histogram(tag=metric['name'], values=metric['value']),
                    global_step=steps)
    summary_writer.flush()


class CustomTensorboard(TensorBoard):

    callbacks = []
    global_step = 0
    summaries = []

    def __init__(self, report_interval, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_interval = report_interval

    def register_metrics_callback(self, cb):
        self.callbacks.append(cb)

    def on_epoch_end(self, epoch, logs=None):

        if self.global_step % self.report_interval == 0:
            metrics = []
            for cb in self.callbacks:
                metrics.extend(cb())
            
            result = self.sess.run([self.merged])
            summary_str = result[0]
            self.writer.add_summary(summary_str, self.global_step)

            write_summaries(self.writer, metrics, self.global_step)
    
    def on_train_end(self, _):
        return

