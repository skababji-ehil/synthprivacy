import pandas as pd
import numpy as np
import hashlib
import time
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
import copy


class MmbrshpRsk():
    
    def __init__(self, real_df: pd.DataFrame , syn_df: pd.DataFrame):
        """Calculates the membership privacy risk associated with synthetic data.
        
        Args:
            real_df: Real Dataset without Unique Identifier column. 
            syn_df: Synthetic Dataset with same shape and variable names of the real dataset 
        
        """
        assert real_df.shape == syn_df.shape, "Real and Synthetic dataset should have the same shape"
        assert (real_df.columns == syn_df.columns).all(), "Real and Synthetic dataset should have the same variable names"
        assert (real_df.dtypes == syn_df.dtypes).all(), "Real and Synthetic dataset should have the same variable types"
        
        self.real_df=real_df
        self.syn_df=syn_df
    
    def partition(self,  population_size: int):
        '''Partitions the REAL dataset into TRAINING and ATTACK datasets.
        
        Args:
            population_size: The population size of from which the real dataset was sampled.

        Returns:
            Training and attack data frames. Both dataframes includes an additional column 'ID' for record index track against the real data.
        '''
        real_data=copy.deepcopy(self.real_df)
        real_data['ID'] = range(len(real_data))    # add an id variable to real data
        t = len(real_data) / population_size
        idx_train, idx_attack=train_test_split(np.arange(len(real_data)),test_size=0.2-(t*0.2), train_size=0.8+(t*0.2)) 
        train_data_w_id = real_data.iloc[idx_train,:]
        attack_data_w_id = real_data.iloc[idx_attack,:]
        attack_data_w_id = pd.concat([attack_data_w_id, train_data_w_id.sample(int(np.ceil(t*len(real_data)*0.2)))], axis=0)
        print({"No of Attack Records":len(attack_data_w_id), "No of of Attack Records in Training":int(np.ceil(t*len(real_data)*0.2))} )
        attack_data_w_id.reset_index(inplace=True, drop=True)
        train_data_w_id.reset_index(inplace=True, drop=True)
        return train_data_w_id, attack_data_w_id
    
    def calc_risk(self, population_size:int ,h: int, quasiID=None, max_n_cores=None, mmbr_bins=20):
        ''' Calculates the membership disclosure risk.
        
        Args:
            population_size: The population size of from which the real dataset was sampled.
            h: An integer representing the hamming distance threshold to be used when calculating teh membership disclosure risk. A smaller number indicates a more conservative model when calculating the membership disclosure risk. A good value can the number of variables-2.


        Returns:
            Training and attack data frames. Both dataframes includes an additional column 'ID' for record index track against the real data.
        '''
        
        
        assert isinstance(self.real_data, pd.DataFrame)
        assert population_size >= len(self.real_data)
        assert h > 0
        
        print('Partitioning real data into training and attack datasets...')
        train_data_w_id, attack_data_w_id=self.partition(population_size)
        
        if quasiID is None:
            quasiID = list(self.real_data.columns)
        
        # detect types of variables
        dataTypes=type_detector(self.real_data.loc[:,quasiID]) #Note: Ensure that the output data types dataframe does NOT include the ID column 
        
        #discretize data 
        syn_data_disrete=copy.deepcopy(self.syn_data)
        attack_data_w_id_discrete=copy.deepcopy(attack_data_w_id)
        
        for k in dataTypes.loc[(dataTypes['Type'] == "Discrete") | (dataTypes['Type'] == "Continuous"), "Name"]:
            discretizer = KBinsDiscretizer(n_bins=mmbr_bins, strategy='uniform', encode='ordinal')
            syn_data_disrete[k] = discretizer.fit_transform(syn_data_disrete[k].values.reshape(-1,1))
            attack_data_w_id_discrete[k] = discretizer.transform(attack_data_w_id_discrete[k].values.reshape(-1,1)) #ID column will be excluded from discretization 

        sim_match = hamming_min_match(attack_data_w_id_discrete, syn_data_disrete, quasiID, max_n_cores=max_n_cores)
        
        #print("Computing the F1 score")
        pp=np.nansum(sim_match["DIST"] <= h)
        tp=np.nansum(np.in1d(attack_data_w_id_discrete['ID'][sim_match['DIST']<=h], train_data_w_id['ID'])) #True positive means a match (i.e. the attacker finds a records which means a patient is identified as a member in the training dataset.)
        p=np.nansum(np.in1d(attack_data_w_id_discrete['ID'], train_data_w_id['ID']))
        precision = 0 if pp == 0 else tp / pp
        recall = tp / p
        f1_baseMh = 0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
        fyard_baseMh = p / attack_data_w_id_discrete.shape[0] #naive_f1
        fnorm_baseMh = (f1_baseMh - fyard_baseMh) / (1 - fyard_baseMh) #rel_f1
        
        return pp, fnorm_baseMh, fyard_baseMh
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

def type_detector(dataTable: pd.DataFrame, IDCol = None) -> pd.DataFrame:
    '''Detects the type of each variable in the input table.
    
    Args:
        dataTable: Input dataframe that includes mixed types of variables as columns. 
        
    Returns:
        A dataframe listing the types of each input variable.
    '''
    
    for x in dataTable.columns:
        if dataTable[x].dtype == 'object':
            dataTable[x] = pd.factorize(dataTable[x])[0]
            dataTable[x]=dataTable[x].astype('category') #added line by SMK for python code
    
    Fits = pd.DataFrame({'Column': range(1, len(dataTable.columns) + 1), 'Name': dataTable.columns, 'Type': 'Filler', 'NAPercent': np.nan})
    iterRange = range(len(dataTable.columns))
    
    if IDCol is not None:
        Fits.loc[Fits['Name'] == IDCol, 'Type'] = 'IDColumn'
        Fits.loc[Fits['Name'] == IDCol, 'NAPercent'] = sum(dataTable[IDCol].isna()) / len(dataTable)
        iterRange = Fits.loc[Fits['Name'] != IDCol].index
    
    #SDGS defined dtypes include: IDColumn, Constant, Binary (0 or 1), Factors (i.e. categorical with cardinality <=10), Continuous, BinaryFactor (e.g. M orF, 1 or 2..etc) and Discrete (any integer with cardinality > 10), 
    for ii in iterRange:
        Fits.loc[ii, 'NAPercent'] = sum(dataTable.iloc[:, ii].isna()) / len(dataTable)
        if (len(dataTable.iloc[:, ii].dropna().unique()) == len(dataTable)) & (dataTable.iloc[:, ii].dtype != 'float64'): #last condition added my SMK
            Fits.loc[ii, 'Type'] = 'IDColumn'
        elif len(dataTable.iloc[:, ii].dropna().unique()) == 1:
            Fits.loc[ii, 'Type'] = 'Constant'
        elif (dataTable.iloc[:, ii].dtype=='int64') & sum((dataTable.iloc[:, ii] == 0) | (dataTable.iloc[:, ii] == 1) & (~dataTable.iloc[:, ii].isna())) == len(dataTable.iloc[:, ii].dropna()): #added first condition by SMK
            Fits.loc[ii, 'Type'] = 'Binary'
        elif dataTable.iloc[:, ii].dtype == 'float64':
            if len(dataTable.iloc[:, ii].dropna().unique()) <= 10:
                Fits.loc[ii, 'Type'] = 'Factors'
            else:
                Fits.loc[ii, 'Type'] = 'Continuous'
        elif dataTable.iloc[:, ii].dtype.name == 'category':
            if len(dataTable.iloc[:, ii].dropna().unique()) == 2:
                Fits.loc[ii, 'Type'] = 'BinaryFactor'
            else:
                Fits.loc[ii, 'Type'] = 'Factors'
        elif dataTable.iloc[:, ii].dtype == 'int64':
            if sum((dataTable.iloc[:, ii] == 0) | (dataTable.iloc[:, ii] == 1) & (~dataTable.iloc[:, ii].isna())) == len(dataTable.iloc[:, ii].dropna()):
                Fits.loc[ii, 'Type'] = 'Binary'
            elif len(dataTable.iloc[:, ii].dropna().unique()) == 2:
                Fits.loc[ii, 'Type'] = 'BinaryFactor'
            elif len(dataTable.iloc[:, ii].dropna().unique()) <= 10:
                Fits.loc[ii, 'Type'] = 'Factors'
            else:
                Fits.loc[ii, 'Type'] = 'Discrete'
    
    return Fits

##############################

def which_min(vec):
    minima = np.where(vec == np.min(vec))[0] #the minma  is a vector in case there are more than one minima
    if len(minima) > 1:
        minima = np.random.choice(minima, 1) 
    return minima[0]

def calc_hamm(i, a_data: pd.DataFrame, b_data: pd.DataFrame) -> list:
        #print(f'PID: {os.getpid()}')
        ham_match_i=np.sum(a_data.values[i,:] != b_data.values, axis=1) #ham_match_i is a vector with a length of b_data
        min_i=which_min(ham_match_i)
        return  [min_i, ham_match_i[min_i]]
    
def hamming_min_match(attack_data: pd.DataFrame, syn_data: pd.DataFrame, QIDSet: list, max_n_cores: int) -> pd.DataFrame:
    ''' Calculates the hamming distance matching table. process can be: hi for maximum number of cores, lo for 5 cores or  or None for no multiprocessing package
    '''
    start=time.time()
    
    a_data = attack_data[QIDSet]
    b_data = syn_data[QIDSet]
    
    if max_n_cores==None:
        res=pd.DataFrame(columns=['b_ID','DIST'])
        for i in range(a_data.shape[0]):
            ham_match_i=np.sum(a_data.values[i,:] != b_data.values, axis=1) #ham_match_i is a vector with a length of b_data. Every variable in a_data row is comapred with corresponding variable in b_data. A mistach is recorded is TRUE (1), then the number of mistaches are summed accorss that row. So, the higher teh number the more teh mismatches
            min_i=which_min(ham_match_i) #acorss all rows of b_data, the record that has the minimum mismatches is recorded. 
            res.loc[len(res)]=[min_i, ham_match_i[min_i]]
        return res
    elif max_n_cores == np.inf:
        n_cores = cpu_count()
    elif max_n_cores <np.inf:
        n_cores = min(cpu_count(), max_n_cores)
    
    calc_hamm_i=partial(calc_hamm,a_data=a_data, b_data=b_data )
    with Pool(n_cores) as p:
        res_lst=p.map(calc_hamm_i,range(a_data.shape[0]))
    res=pd.DataFrame(res_lst)
    res.columns=['b_ID','DIST']
    return res

##############################



##############################
