===============
Aerosense Tools
===============

``aerosense-tools`` is a library of wrappers and functions for querying the Aerosense data warehouse and working with the resulting data.

The data flow from the aerosense sensor modules looks like this:

.. code-block::

   Node (edge processor on-blade)
     >  Receiver (bluetooth equipment on-tower)
       >  Gateway (data manager and uploader on-tower)
         >  Ingress (server to receive data on-cloud)
           >  GretaDB (GCP BigQuery database for installation and most sensor data)
                or
           >  GretaStore (GCP Object Store for acoustic data)


The ``data-gateway`` library is responsible for the data collection and ingress. 
This library, ``aerosense-tools``, is used to access and manipulate 
data from GretaDB or the GretaStore by any python client:

.. code-block::

   GretaDB or GretaStore
      > dashboard server
      > jupyter notebooks
      > service within the Digital Twin
      > local scripts run by researchers


Contents
========

.. toctree::
   :maxdepth: 2

   installation
   authentication
   api
