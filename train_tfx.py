from typing import List, Text
import absl
import tensorflow as tf
import tensorflow_transform as tft
from tfx import v1 as tfx
from tfx_bsl.public import tfxio

NUM_EPOCHS = 4

DENSE_FLOAT_FEATURE_KEYS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12',
                                'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'Amount', 'Time']
LABEL_KEY = 'Class'

_DENSE_FLOAT_FEATURE_KEYS = DENSE_FLOAT_FEATURE_KEYS
_LABEL_KEY = LABEL_KEY


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """
    Returns a function that parses a serialized tf.Example and applies TFT.
    """
    model.tft_layer = tf_transform_output.transform_features_layer()
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(_LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)
    return serve_tf_examples_fn


def _input_fn(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
    """
    Generates features and label for tuning/training.
    Args:
      file_pattern: List of paths or patterns of input tfrecord files.
      data_accessor: DataAccessor for converting input to RecordBatch.
      tf_transform_output: A TFTransformOutput.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch

    Returns:
      A dataset that contains (features, indices) tuple where features are a
      dictionary of Tensors, and indices are  single Tensors of label indices.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=_LABEL_KEY),
        tf_transform_output.transformed_metadata.schema)

def _create_model():
    """
    Build a simple keras wide and deep model.
    Args:
    Returns:
      Keras model
    """

    input_layers = {
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.float32)
        for colname in _DENSE_FLOAT_FEATURE_KEYS
    }

    output = tf.keras.layers.Dense(30, activation='relu')(input_layers)
    output = tf.keras.layers.Dense(40, activation='relu')(output)
    output = tf.keras.layers.Dense(50, activation='relu')(output)
    output = tf.keras.layers.Dense(1)(output)

    model = tf.keras.Model(input_layers, output)
    model.compile(
        loss      = tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer = tf.keras.optimizers.Adam(lr=0.001),
        metrics   = [tf.keras.metrics.BinaryAccuracy()])

    model.summary(print_fn=absl.logging.info)
    return model

def run_fn(fn_args: tfx.components.FnArgs):
    """
    Train the model based on given args.
    Args:
      fn_args: Holds args used to train the model as name/value pairs.

    """

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor,
                              tf_transform_output, 32)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor,
                             tf_transform_output, 32)

    model = _create_model()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch')

    model.fit(
        train_dataset,
        epochs=NUM_EPOCHS,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback])

    signatures = {
        'serving_default':
            _get_serve_tf_examples_fn(model,
                                      tf_transform_output).get_concrete_function(
                tf.TensorSpec(
                    shape=[None],
                    dtype=tf.string,
                    name='examples')),
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)