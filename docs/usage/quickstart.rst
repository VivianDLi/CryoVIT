Quick Start Guide
=========================

This guide goes over a quick example of using CryoViT
to segment mitochondria in a group of Cryo-ET tomograms
using a pre-trained model.

.. admonition:: Note

   This guide assumes you have already installed CryoViT
   by following the instructions in :ref:`Installing CryoViT <installing cryovit>`.
   If you have not done so, please do that first.

   You do not need a working installation of napari
   to follow this guide, but a GPU is recommended for faster inference.

.. highlight:: console

First, download the example data from `here <example_data>`_ and its
contents: ::

    $ tar -xzf example_data.tar.gz

.. TODO: add method/location to download example_data as .tar.gz
.. _example_data: https://github.com/VivianDLi/CryoViT-Example-Data

This will extract a directory ``example_data`` containing a
folder of tomograms ``data/`` and a pre-trained model
file ``pretrained_model.model``.

.. TODO: replace code blocks with actual results

============================
Viewing Tomogram Data
============================

CryoViT supports most common file formats for tomogram data,
including ``.mrc``, ``.tiff``, and ``.hdf`` formats, expecting
the tomogram data to be stored as a 3D array with shape
``(D, H, W)``.

You can preview the tomogram data with :py:func:`cryovit.utils.load_data`,
which returns the data as a `numpy array`_:

.. tip::

    For ``.hdf`` files, which can contain multiple keyed datasets,
    you can specify which dataset to load by passing in the ``key``
    argument to :py:func:`cryovit.utils.load_data`.

    Otherwise, the dataset with the most *unique* values will be loaded by default, and :py:func:`cryovit.utils.load_data` will return the key found.

.. _numpy array: https://numpy.org/doc/stable/reference/arrays.ndarray.html

.. code-block:: python

    >>> from cryovit.utils import load_data
    >>> data, key = load_data("example_data/data/tomogram_01.hdf")
    >>> data
    array([[[...]]], dtype=float32)
    >>> key
    'data'
    >>> print(type(data))
    <class 'numpy.ndarray'>
    >>> print(data.shape)
    (100, 512, 512)
    >>> print(data.dtype.name)
    'float32'

============================
Viewing Model Information
============================

CryoViT uses a custom file extension ``.model`` to save pre-trained
model weights and metadata about the model. You can view the model
data with :py:func:`cryovit.utils.load_model`, which returns a tuple
containing the model (a `pytorch model`_), and its metadata.

.. _pytorch model: https://pytorch.org/docs/stable/generated/torch.nn.Module.html

.. tip::

    If you only want to view the metadata without loading the model,
    you can pass in the argument ``load_model=False`` to
    :py:func:`cryovit.utils.load_model`.

.. code-block:: python

    >>> from cryovit.utils import load_model
    >>> model, model_type, name, label = load_model("example_data/pretrained_model.model")
    >>> print(model)
    CryoViT(
      (patch_embed): PatchEmbed(
        (proj): Conv3d(1, 96, kernel_size=(4, 4, 4), stride=(4, 4, 4))
        (norm): LayerNorm((96,), eps=1e-06, elementwise_affine=True)
      )
      ...
      (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (head): Linear(in_features=768, out_features=2, bias=True)
    )
    >>> print(model_type.name,  model_type.value)
    ModelType.CRYOVIT 'cryovit'
    >>> print(name)
    'pretrained_example'
    >>> print(label)
    'mito'

We see that the ``model_type`` is ``ModelType.CRYOVIT``,
indicating that this is a CryoViT segmentation model, and the
``label`` is ``mito``, indicating that this model segments mitochondria.

============================
Running Inference Script
============================

The main utilities of CryoViT can be run through command-line scripts.
You can see all available scripts by running: ::

    $ cryovit --help
    # or
    $ cryovit

.. TODO: insert screenshot of `cryovit --help` output

and the arguments for a specific script by running: ::

    $ cryovit <script_name> --help
    # or
    $ cryovit <script_name>

We see the available scripts are ``features``, ``train``, ``evaluate``,
and ``inference``. For this quick start guide, we will be using the
``inference`` script to segment the tomograms using the pre-trained model.

.. important::

    Since the model is a CryoViT model, we need to run the ``features``
    script first to extract the high-level ViT features from the tomograms.

.. TODO: insert screenshot of cryovit features --help output

To run the ``features`` script, we need to specify the input tomogram folder and the output directory to save the extracted features: ::

    $ cryovit features example_data/data example_data/features

.. tip::

    This step requires a GPU, and is possibly very memory-intensive. If you run into out-of-memory issues, try reducing the ``--batch-size`` or ``--window-size`` arguments. Reducing the batch size is preferable, as reducing the window size will affect the quality of the extracted features.

Then, we can run the ``inference`` script on the extracted features,
storing the results in a ``predictions`` folder: ::

    $ cryovit inference example_data/features --model example_data/pretrained_model.model --result-folder example_data/predictions

============================
Viewing Segmentation Results
============================

The segmentation results will be saved as ``.hdf`` files in the
``example_data/predictions`` folder, each containing a ``data`` dataset
with the original data, and a ``<label>_preds`` dataset with the predicted
segmentation masks.

While you can still load the predicted segmentations using
:py:func:`cryovit.utils.load_data` or :py:func:`cryovit.utils.load_labels`,
it is recommended to use a visualization tool like `ChimeraX`_
to view the results in 3D, as shown below:

.. _ChimeraX: https://www.rbvi.ucsf.edu/chimerax/

.. TODO: insert screenshot of ChimeraX visualization
