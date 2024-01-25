# Synthprivacy

The ```synthprivacy``` package currently calculates membership disclosure risk that is associated with synthetic data. It is developed by [Electronic Health Information Lab ](https://www.ehealthinformation.ca/) as an implementation of the paper:

```
El Emam K, Mosquera L, Fang X. Validating a membership disclosure metric for synthetic health data. 
JAMIA Open. 2022 Oct 11;5(4):ooac083. doi: 10.1093/jamiaopen/ooac083. PMID: 36238080; PMCID: PMC9553223.
```

An example is given in  ```main.py```. You first instantiate an object using the class ```MmbrshpRsk```. The class will partition the input real dataset (using the default parameters) and return a training dataset. Internally, the indices of the training observations are retained for further calculations. You use the training data with any generative model to generate your synthetic data. Finally, you pass the synthetic data to the previously defined```MmbrshpRsk``` object to calculate  the F1 risk scores. Some parameters can be adjusted for risk calculations, e.g. the hamming distance threshold  ```h```. If you like to change these parameters, please make sure to change them in the class itself.  For further information, please refer to the comments in the script ```src/synthprivacy/mmbrshp_rsk.py```. 
