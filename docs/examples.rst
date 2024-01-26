Examples
========

An example is given in  ``main.py``. The example uses a dataset imported from the ``scikit-learn`` library and a generator from ``synthcity``. Clearly, you can replace these with your dataset and generative models respectively.

You first instantiate an object using the class ``MmbrshpRsk``. The class will partition the input real dataset (using the default parameters) and return a training dataset. Internally, the indices of the training observations are retained for further calculations. You use the training data with any generative model to generate your synthetic data. Finally, you pass the synthetic data to the previously defined ``MmbrshpRsk`` object to calculate  the F1 risk scores. Some parameters can be adjusted for risk calculations, e.g. the hamming distance threshold  ``h``. Selected parameters can be passed as arguments to the class..  For further information, please refer to the comments in the script ``src/synthprivacy/mmbrshp_rsk.py``.