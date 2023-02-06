import tensorflow as tf

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, data=value, step=step)


    def histo_summary(self, tag, values, step, bins=100):
        """Log a histogram of the tensor of values."""
        with self.writer.as_default():
            tf.summary.histogram('{}'.format(tag), values, buckets=bins, step=step)



