import os
import sys
from pathlib import Path

# project_path = Path(__file__).absolute().parents[2]
project_path = Path().cwd()

os.sys.path.append(project_path.as_posix())

from src.MyModule.utils import *

print(f" this is project path : {project_path} ")

#%% settings

age = 50

#%% path settings

config = load_config()

input_path = get_path("data/processed/2_produce_data/synthetic_restore")

synthetic_path = input_path.joinpath(f"Synthetic_data_epsilon10000_{age}.csv")

# synthetic_data_path_list = [input_path.joinpath(
#     f"Synthetic_data_epsilon{eps}_{age}.csv") for eps in config['epsilon']]

ori_path = get_path(f"data/processed/preprocess_1/original_{age}.pkl")

output_path = get_path("data/processed/3_evaluate_data/")

if not output_path.exists():
    output_path.mkdir(parents=True)

#%%

syn = pd.read_csv(synthetic_path, encoding = 'cp949')
ori = pd.read_pickle(ori_path)

#%%


#%%

class Tester:

    def __init__(self, test_criteria, information) :
        self.test_criteria = test_criteria
        self.information = information

    def do_test(self) :
        NotImplementedError

    def convert_dates(self, data, column) :
        try : 
            data[column] = data[column].astype('datetime64[ns]')
        except :
            try : 
                data[column] = data[column].replace('0', np.nan)
                data[column] = data[column].astype('datetime64[ns]')
            except :
                print("check the values of the columns")
                raise ValueError 
        return data

#%%
class InclusionExclusion(Tester):

    def __init__(self):
        information = "under 30 days of follow up days are excluded"
        super().__init__(test_criteria = "inclusion/exclusion", information = information)

    def do_test(self, data, is_original = True): 
        '''
        returns 30 days, 60 days under follow up patient ratio
        '''
        self.test_stage_value_null(data)
        
        if not is_original :
            return (data.OVR_SURV < 30).sum() / data.shape[0],  (data.OVR_SURV < 60).sum() / data.shape[0]
        
        data = self.convert_dates(data, ["CENTER_LAST_VST_YMD", "BSPT_FRST_DIAG_YMD"])
        days = (data.CENTER_LAST_VST_YMD - data.BSPT_FRST_DIAG_YMD).dt.days 
        return (days < 30).sum() / days.shape[0], (days < 60).sum() / days.shape[0]

    def test_stage_value_null(self, data) :
        if data['BSPT_STAG_VL'].isna().sum() >= 0 :
            print("there is null value in stage values")


class BasicInfo(Tester):

    def __init__(self):
        information = "basic patient information"
        super().__init__(test_criteria = "basic information testing", information = information)

    def do_test(self, data): 
        '''
        gender distribution, and age and stage value information
        '''
        male, female = self.test_sex(data)
        median, min_val, max_val = self.test_median_age(data)
        diagnosis_info = self.test_diagnosis_per_patients(data)
        stage_info = self.test_basic_stage_value_and_ratio(data)

        gender_info = (male, female)
        age_info = (median, min_val, max_val)

        return gender_info, age_info, diagnosis_info, stage_info

    def test_sex(self, data):
        male_counts = data["BSPT_SEX_CD"].value_counts()["M"]
        female_counts = data["BSPT_SEX_CD"].value_counts()["F"]
        total = male_counts + female_counts 

        return male_counts / total, female_counts / total

    def test_median_age(self, data) : 
        median_value = data["BSPT_IDGN_AGE"].median()
        min_value, max_value = data["BSPT_IDGN_AGE"].min(), data["BSPT_IDGN_AGE"].max()
        return median, min_value, max_value
        
    def test_basic_stage_value_and_ratio(self, data) :
        df = data["BSPT_STAG_VL"].value_counts().sort_index().reset_index(name="counts")
        df['ratio'] = df['counts'].apply(lambda x : x / df.shape[0])
        return df

    def test_diagnosis_per_patients(self, data) :
        df = data["BSPT_FRST_DIAG_NM"].value_counts().sort_index().reset_index(name = 'counts')
        df['ratio'] = df['counts'].apply(lambda x : x / df.shape[0])
        return df

#%%

class Relapse(Tester):

    def __init__(self):
        information = "for each stage, calculate the relapse information"
        super().__init__(test_criteria = "relapse", information = information)

    def do_test(self, data, is_original=True): 
        '''
        1) stage per relapse count, 
        2) relapse - first diagnosis
        3) stage per relapse within 3 years counts and ratio
        '''
        data = data.copy()

        if not is_original :
            data = data.rename(columns = {"RLPS_YMD" : "RLPS_DIAG_YMD"})

        stage_info = self.test_stage_per_relapse(data)
        first_diag_to_recur = self.first_diagnosis_to_recur(data)

        return  stage_info, first_diag_to_recur
    
    def test_stage_per_relapse(self, data):
        df = data[['BSPT_STAG_VL', 'RLPS']].copy()
        df = df.groupby("BSPT_STAG_VL", as_index=False)["RLPS"].sum().rename(columns = {"RLPS":"counts"})
        df['ratio'] = df["RLPS"].apply(lambda x : x / df.shape[0])
        return df
    
    def first_diagnosis_to_recur(self, data):
        data = self.convert_dates(data, ["RLPS_DIAG_YMD","BSPT_FRST_DIAG_YMD"])
        diff_days = data["RLPS_DIAG_YMD"] - data["BSPT_FRST_DIAG_YMD"]
        diff_days = diff_days.dt.days

        return min(diff_days), max(diff_days)

    def test_stage_per_3_year_recur(self, data) :
        data = self.convert_dates(data, ["RLPS_DIAG_YMD","BSPT_FRST_DIAG_YMD"])
        df = data[data.RLPS == 1].copy()

        df["RLPS_DIFF"] = (df["RLPS_DIAG_YMD"] - df["BSPT_FRST_DIAG_YMD"]).dt.days
        df = df[['BSPT_STAG_VL', 'RLPS_DIFF']].copy()

        df = df[df.RLPS_DIFF <= (365 * 3)]["BSPT_STAG_VL"]

        df = df.value_counts().sort_index().reset_index(name = 'counts')
        df['ratio'] = df['counts'].apply(lambda x : x / data.shape[0])
        return df

#%%
