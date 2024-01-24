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
    real_df, y = load_diabetes(return_X_y=True, as_frame=True)
    real_df["target"] = y
    loader = GenericDataLoader(real_df,target_column="target", sensitive_columns=["sex"])
    
    # synthesize
    syn_model = Plugins().get('marginal_distributions')
    syn_model.fit(loader)
    syn_df=syn_model.generate(count=len(real_df)).dataframe()
    
    # calculate membership risk
    rsk=MmbrshpRsk(real_df, syn_df)
    rel_f1, naive_f1=rsk.calc_risk(population_size=10*len(real_df), h=9)
    print(rel_f1, naive_f1)