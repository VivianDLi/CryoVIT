Running from the command line
=====================================

This section goes over how to use the CryoViT CLI
to train and predict on a custom dataset.

.. admonition:: Note

   This guide assumes you have already installed CryoViT
   by following the instructions in :ref:`Installing CryoViT <installing cryovit>`.
   If you have not done so, please do that first.

   You also need a valid training dataset. See :ref:`Setting Up a Dataset <setting up a dataset>`
   for instructions on how to set one up.

   A GPU is also highly recommended.

.. highlight:: console

=========================
Training a Model
=========================

You can train a new model using the ``train`` command:

.. figure:: ../_static/tutorial/cryovit_cli_train_output.png
    :align: center
    :width: 95%

    Output of ``cryovit train --help`` command.

So, to train a model, you would run a command like the following: ::

    $ cryovit train train.txt labels.txt mito --val-data val.txt --val-labels val_labels.txt --labels mito membrane --name mito_model

This command would train a model to segment mitochondria,
given label files that have both mitochondria and membranes labeled.

    - The training data is specified in ``train.txt``,
      and the corresponding labels are in ``labels.txt``.
    - The validation data is specified in ``val.txt``,
      and the corresponding labels are in ``val_labels.txt``.
    - After training, the model will be saved as ``mito_model.model``
      in the current directory.

If your training dataset is organized in folders, then you would replace
``*.txt`` with the folder names.

.. important::

    If using ``--model cryovit`` (the default), then your training and
    validation data must either contain DINOv2 features, or you must
    compute the features first using the ``cryovit features`` command.
    See :ref:`Quick Start Guide <quick start guide>` for an example of this.

Monitoring Training
-------------------------

If you specify the ``--log-training`` argument (or ``-l`` for short), then
you can monitor the training progress using TensorBoard.

When training starts, a message will be printed of the form: ::

    $ INFO  Setup TensorBoard logger. View logs with `tensorboard --logdir
            <path-to-log-dir>

Running this command in a separate terminal will then give you a local
URL, which you can use to monitor the training progress in your browser.

Evaluating the Model
-------------------------

After training, you can evaluate the model on a test dataset using the
``evaluate`` command:

.. figure:: ../_static/tutorial/cryovit_cli_evaluate_output.png
    :align: center
    :width: 95%

    Output of ``cryovit evaluate --help`` command.

This command takes similar arguments to the ``train``
command, except that you specify the test data and labels instead of
the train data and labels, and you also need to specify a pre-trained
model file. For example: ::

    $ cryovit evaluate test.txt test_labels.txt mito_model.model

This results in a ``.csv`` file being created in a ``results`` folder in
the current directory, containing the **DICE** scores for each tomogram.

If you specify the ``--visualize`` argument (or ``-v`` for short),
then the predicted segmentations will be saved as ``.hdf`` files
in a ``predictions`` folder in the result directory (i.e., the
current directory).

=========================
Using a Pre-trained Model
=========================

