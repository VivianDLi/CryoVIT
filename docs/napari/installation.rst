Setting up Napari
=================

To train new models with CryoViT,
you will need to provide a labeled training dataset.

For this, we recommend using `napari`_, an open-source,
Python-based medical image viewer with a rich plugin ecosystem.

.. tip::
    We recommend installing napari in its own separate environment,
    as described in the `napari installation instructions`_ and below.

    Then, you can install the CryoViT napari plugin using **pip**
    in the same environment, as an alternative to the **Plugin Manager**.

.. _napari: https://napari.org/stable/index.html
.. _napari installation instructions: https://napari.org/stable/tutorials/fundamentals/installation.html#napari-installation

CryoViT provides a `napari plugin`_ to help create and manage
training datasets, which can be installed directly from PyPI,
or through the **Plugin Manager** within napari.

.. _napari plugin: https://github.com/VivianDLi/CryoVIT-Napari

.. note::
    If you do not plan to train your own models,
    and only want to use the provided pre-trained models,
    you do not need to install napari or the CryoViT napari plugin.

========================
Installing Napari
========================

.. highlight:: console

First, create a new conda environment for napari: ::

    $ conda create -n cryovit-napari
    $ conda activate cryovit-napari

Then, install napari using ``mamba`` (recommended) or ``conda``: ::

    $ mamba install -c conda-forge napari
    # or
    $ conda install -c conda-forge napari

Launch napari by activating the environment and
running the ``napari`` command: ::

    $ conda activate cryovit-napari
    $ napari

Finally, check out the `napari documentation`_ for help
on using the application.

.. _napari documentation: https://napari.org/stable/usage.html

========================================
Installing the CryoViT Napari Plugin
========================================

Using the Plugin Manager
----------------------------------------

To install the CryoViT napari plugin using the `Plugin Manager`_:

.. _Plugin Manager: https://napari.org/napari-plugin-manager/

1. Launch napari
2. Open the Plugin Manager from the "Plugins" menu
   under "Install/Uninstall Plugins".
3. Search for ``cryovit-napari`` and click "Install"

Using pip
----------------------------------------

You can install the CryoViT napari plugin using **pip**
with the napari environment activated: ::

    $ conda activate cryovit-napari
    $ pip install cryovit-napari
