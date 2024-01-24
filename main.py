import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_diabetes

import synthcity.logger as log
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
log.add(sink=sys.stderr, level="INFO")

from mmbrshp_rsk import MmbrshpRsk

if __name__ == "__main__":
    
    # load sample real data 
    real_data, y = load_diabetes(return_X_y=True, as_frame=True)
    real_data["target"] = y

    # Construct membership disclosure object and get the training dataset 
    mmbr=MmbrshpRsk(real_data, population_size=10*len(real_data))

    # Synthesize
    loader = GenericDataLoader(mmbr.train_data)
    syn_model = Plugins().get('marginal_distributions')
    syn_model.fit(loader)
    syn_data=syn_model.generate(count=len(real_data)).dataframe()
    
    # Calculate membership disclosure risk
    rel_f1,naive_f1=mmbr.calc_risk(syn_data)
 
    
    