.. ATTENTION::
    This library is in experimental stages! Please pin deployments to a specific release, and consider every release as breaking.

===============
Aerosense Tools
===============

.. epigraph::
   *"Aerosense Tools" ~ wrappers and functions for querying the Aerosense database and working with the resulting data.*


Data Origins
============

The data flow from the aerosense sensor modules looks like this:

.. code-block::

   Node (edge processor on-blade)
     >  Receiver (bluetooth equipment in-nacelle)
       >  Gateway (data manager and uploader on-nacelle)
         >  Ingress (server to receive data on-cloud)
           >  GretaDB (BigQuery database)
                or
           >  GretaStore (object store for acoustic data)


The ``data-gateway`` library is responsible for processing all that data and ingress.

This library, ``aerosense-tools``, is used by a python client, such as:
   - a dashboard server
   - a jupyter notebook
   - a service within the Digital Twin
   - a researchers own local script
to access and manipulate data from GretaDB or the GretaStore


.. toctree::
   :maxdepth: 2

   self
   installation
   api
