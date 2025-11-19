Installing CryoViT
========================

CryoViT is a Python application, and is installable
via :ref:`pip <pypi package>`.
For most users, this is the recommended installation method.

If you plan on using CryoViT for complicated experiments,
or want to extend its functionality,
you may want to install it :ref:`from source <installation from source>`.

.. highlight:: console

After installation, you can check that CryoViT is available by running ::

    $ cryovit --help

This should display the basic usage information for the
CryoViT command-line interface (CLI).

.. tip::

    For local development or even usage, it is recommended to install CryoViT
    into a separate non-global environment (e.g., using `venv`_ or `conda`_ environments).
    This prevents dependency conflicts with other Python packages you may have installed.

    The authors recommend using `miniforge`_, a lightweight version of conda, to manage environments and dependencies.

.. _venv: https://docs.python.org/3/library/venv.html
.. _conda: https://conda.io/projects/conda/en/latest/user-guide/getting-started.html
.. _miniforge: https://github.com/conda-forge/miniforge

========================
PyPI package
========================

CryoViT is available on the `Python Package Index`_ (PyPI).
The preferred tool for installing packages from PyPI is **pip**,
which is included by default with all modern versions of Python.

.. _Python Package Index: https://pypi.org/project/cryovit/

To install the latest stable release of CryoViT, run ::

    $ pip install -U cryovit

Installing from PyPI will naturally give access to all features covered
in :ref:`the user guide <user guide>`,
including the CLI and napari plugin.
However, some advanced features, such as training on clusters and large-scale
experiments, require building from source, as described below.

========================
Installation from Source
========================

You can install CryoViT directly from a clone of the `Git repository`_.
This requires having `git`_ installed on your system, as well as a virtual environment manager and Python distribution such as `miniforge`_.    

.. _Git repository: https://github.com/sanketx/CryoVIT
.. _git: https://git-scm.com/downloads
.. _miniforge: https://github.com/conda-forge/miniforge

First, clone the repository: ::

    $ git clone https://github.com/sanketx/CryoVIT.git
    $ cd CryoVIT

Next, create a conda environment based on the provided environment file: ::

    $ conda create --name cryovit_env

This will create a conda environment named `cryovit_env`. You can change this name to whatever you like.

The final step is then to activate the environment
and install CryoViT in editable mode: ::

    $ conda activate cryovit_env
    $ pip install -e .

This will install all the packages necessary and track any changes you make to the source code, allowing you to use CryoViT for general use.

You can download the additional packages for development and testing by running instead: ::

    $ pip install -e .[dev]
