import os
import pandas as pd
import numpy as np
import glob
import datetime
import socket
from sklearn import preprocessing
from model import Config
from fns_categorical_encoding import preprocess_data
from constants import sample_seed, shuffle_seed, clf_seed

CE_HOME = os.environ.get('CE_HOME')


def get_data_folder():
    hostname = socket.gethostname()
    if hostname in ['drago', 'drago2', 'drago3']:
        data_folder = '/storage/store/work/pcerda/data'
    elif hostname in ['paradox', 'paradigm']:
        data_folder = '/storage/local/pcerda/data'
    else:
        data_folder = os.path.join(CE_HOME, 'data')
    return data_folder


def create_folder(path, folder):
    if not os.path.exists(os.path.join(path, folder)):
        os.makedirs(os.path.join(path, folder))


def print_unique_values(df):
    for col in df.columns:
        print(col, df[col].unique().shape)
        print(df[col].unique())
        print('\n')


class Data:
    def __init__(self, name):
        self.name = name
        self.configs = None
        self.xcols, self.ycol = None, None

        ''' Given the dataset name, return the respective dataframe as well as
        the the action for each column.'''

        if name in ['adult', 'adult2', 'adult3']:
            '''Source: https://archive.ics.uci.edu/ml/datasets/adult'''
            data_path = os.path.join(get_data_folder(), 'adult_dataset')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'adult.data')

        if name == 'beer_reviews':
            '''Source: BigML'''
            data_path = os.path.join(get_data_folder(), 'bigml/beer_reviews/')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'beer_reviews.csv')

        if name == 'midwest_survey':
            '''FiveThirtyEight Midwest Survey
            Original source: https://github.com/fivethirtyeight/data/tree/
                             master/region-survey
            Source: BigML'''
            data_path = os.path.join(get_data_folder(),
                                     'bigml/FiveThirtyEight_Midwest_Survey')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'FiveThirtyEight_Midwest_Survey.csv')

        if name == 'indultos_espana':
            '''Source: '''
            data_path = os.path.join(get_data_folder(),
                                     'bigml/Indultos_en_Espana_1996-2013')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'Indultos_en_Espana_1996-2013.csv')

        if name == 'docs_payments':
            '''Source: '''
            data_path = os.path.join(get_data_folder(), 'docs_payments')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'output', 'DfD.h5')

        if name == 'medical_charge':
            '''Source: BigML'''
            data_path = os.path.join(get_data_folder(),
                                     'bigml/MedicalProviderChargeInpatient')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'MedicalProviderChargeInpatient.csv')

        if name == 'road_safety':
            '''Source: https://data.gov.uk/dataset/road-accidents-safety-data
            '''
            data_path = os.path.join(get_data_folder(), 'road_safety')
            create_folder(data_path, 'output/results')
            data_file = [os.path.join(data_path, 'raw', '2015_Make_Model.csv'),
                         os.path.join(data_path, 'raw', 'Accidents_2015.csv'),
                         os.path.join(data_path, 'raw', 'Casualties_2015.csv'),
                         os.path.join(data_path, 'raw', 'Vehicles_2015.csv')]

        if name == 'consumer_complaints':
            '''Source: https://catalog.data.gov/dataset/
                       consumer-complaint-database
               Documentation: https://cfpb.github.io/api/ccdb//fields.html'''
            data_path = os.path.join(get_data_folder(), 'consumer_complaints')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'Consumer_Complaints.csv')

        if name == 'traffic_violations':
            '''Source: https://catalog.data.gov/dataset/
                       traffic-violations-56dda
               Source2: https://data.montgomerycountymd.gov/Public-Safety/
                        Traffic-Violations/4mse-ku6q'''
            data_path = os.path.join(get_data_folder(), 'traffic_violations')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'Traffic_Violations.csv')

        if name == 'crime_data':
            '''Source: https://catalog.data.gov/dataset/
                       crime-data-from-2010-to-present
               Source2: https://data.lacity.org/A-Safe-City/
                        Crime-Data-from-2010-to-Present/y8tr-7khq'''
            data_path = os.path.join(get_data_folder(), 'crime_data')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'Crime_Data_from_2010_to_Present.csv')

        if name == 'employee_salaries':
            '''Source: https://catalog.data.gov/dataset/
                       employee-salaries-2016'''
            data_path = os.path.join(get_data_folder(), 'employee_salaries')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'Employee_Salaries_-_2016.csv')

        # add here the path to a new dataset ##################################
        if name == 'new_dataset':
            '''Source: '''
            data_path = os.path.join(get_data_folder(), 'new_dataset')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'data_file.csv')
        #######################################################################

        self.file = data_file
        self.path = data_path

    def preprocess(self, n_rows=-1, str_preprocess=True, clf_type='regression'):
        if n_rows == -1:
            df = self.df.sample(frac=1, random_state=sample_seed
                                ).reset_index(drop=True)
        else:
            df = self.df.sample(frac=1, random_state=sample_seed
                                ).reset_index(drop=True)[:n_rows]
        if str_preprocess:
            df = preprocess_data(df,
                                 [key for key in self.col_action
                                  if self.col_action[key] == 'se'])
        xcols = [key for key in self.col_action
                 if self.col_action[key] is not 'y']
        ycol = [key for key in self.col_action
                if self.col_action[key] is 'y'][0]
        self.df = df.dropna(axis=0, subset=[c for c in xcols
                                            if self.col_action[c]
                                            is not 'del'] + [ycol])
        self.xcols, self.ycol = xcols, ycol

        return

    def make_configs(self, **kw):
        if self.df is None:
            raise ValueError('need data to make column config')
        self.configs = [Config(name=name, kind=self.col_action.get(name), **kw) for name in self.df.columns
                        if name in self.col_action.keys()]
        self.configs = [c for c in self.configs if not (c.kind in ('del', 'y'))]
        print(self.configs)

    def get_df(self):
        if self.name == 'adult':
            header = ['age', 'workclass', 'fnlwgt', 'education',
                      'education-num', 'marital-status', 'occupation',
                      'relationship', 'race', 'sex', 'capital-gain',
                      'capital-loss', 'hours-per-week', 'native-country',
                      'income']
            df = pd.read_csv(self.file, names=header)
            df = df[df['occupation'] != ' ?']
            df = df.reset_index()
            df['income'] = (df['income'] == ' >50K')
            col_action = {'age': 'num',
                          'workclass': 'ohe',
                          'fnlwgt': 'del',
                          'education': 'ohe',
                          'education-num': 'num',
                          'marital-status': 'ohe',
                          'occupation': 'se',
                          'relationship': 'ohe',
                          'race': 'ohe',
                          'sex': 'ohe',
                          'capital-gain': 'num',
                          'capital-loss': 'num',
                          'hours-per-week': 'num',
                          'native-country': 'ohe',
                          'income': 'y'}
            self.clf_type = 'binary_clf'

        if self.name == 'beer_reviews':
            df = pd.read_csv(self.file)
            df.shape
            df = df.dropna(axis=0, how='any')

            # print_unique_values(df)
            col_action = {'brewery_id': 'del',
                          'brewery_name': 'del',
                          'review_time': 'del',
                          'review_overall': 'del',
                          'review_aroma': 'num',
                          'review_appearance': 'num',
                          'review_profilename': 'del',
                          'beer_style': 'y',
                          'review_palate': 'num',
                          'review_taste': 'num',
                          'beer_name': 'se',
                          'beer_abv': 'del',
                          'beer_beerid': 'del'}
            self.clf_type = 'multiclass_clf'

        if self.name == 'midwest_survey':
            df = pd.read_csv(self.file)
            # print_unique_values(df)
            col_action = {'RespondentID': 'del',
                          'In your own words, what would you call the part ' +
                          'of the country you live in now?': 'se',
                          'Personally identification as a Midwesterner?':
                              'ohe',
                          'Illinois in MW?': 'ohe-1',
                          'Indiana in MW?': 'ohe-1',
                          'Iowa in MW?': 'ohe-1',
                          'Kansas in MW?': 'ohe-1',
                          'Michigan in MW?': 'ohe-1',
                          'Minnesota in MW?': 'ohe-1',
                          'Missouri in MW?': 'ohe-1',
                          'Nebraska in MW?': 'ohe-1',
                          'North Dakota in MW?': 'ohe-1',
                          'Ohio in MW?': 'ohe-1',
                          'South Dakota in MW?': 'ohe-1',
                          'Wisconsin in MW?': 'ohe-1',
                          'Arkansas in MW?': 'ohe-1',
                          'Colorado in MW?': 'ohe-1',
                          'Kentucky in MW?': 'ohe-1',
                          'Oklahoma in MW?': 'ohe-1',
                          'Pennsylvania in MW?': 'ohe-1',
                          'West Virginia in MW?': 'ohe-1',
                          'Montana in MW?': 'ohe-1',
                          'Wyoming in MW?': 'ohe-1',
                          'ZIP Code': 'del',
                          'Gender': 'ohe',
                          'Age': 'ohe',
                          'Household Income': 'ohe',
                          'Education': 'ohe',
                          'Location (Census Region)': 'y'}

            le = preprocessing.LabelEncoder()
            ycol = [col for col in col_action if col_action[col] == 'y']
            df[ycol] = le.fit_transform(df[ycol[0]].astype(str))
            self.clf_type = 'multiclass_clf'

        if self.name == 'indultos_espana':
            df = pd.read_csv(self.file)
            col_action = {'Fecha BOE': 'del',
                          'Ministerio': 'ohe-1',
                          'Ministro': 'ohe',
                          'Partido en el Gobierno': 'ohe-1',
                          'Género': 'ohe-1',
                          'Tribunal': 'ohe',
                          'Región': 'ohe',
                          'Fecha Condena': 'del',
                          'Rol en el delito': 'se',
                          'Delito': 'se',
                          'Año Inicio Delito': 'num',
                          'Año Fin Delito': 'num',
                          'Tipo de Indulto': 'y',
                          'Fecha Indulto': 'del',
                          'Categoría Cod.Penal': 'se',
                          'Subcategoría Cod.Penal': 'se',
                          'Fecha BOE.año': 'num',
                          'Fecha BOE.mes': 'num',
                          'Fecha BOE.día del mes': 'num',
                          'Fecha BOE.día de la semana': 'num',
                          'Fecha Condena.año': 'num',
                          'Fecha Condena.mes': 'num',
                          'Fecha Condena.día del mes': 'num',
                          'Fecha Condena.día de la semana': 'num',
                          'Fecha Indulto.año': 'num',
                          'Fecha Indulto.mes': 'num',
                          'Fecha Indulto.día del mes': 'num',
                          'Fecha Indulto.día de la semana': 'num'}
            df['Tipo de Indulto'] = (df['Tipo de Indulto']
                                     == 'indultar')
            self.clf_type = 'binary_clf'

        if self.name == 'docs_payments':
            # Variable names in Dollars for Docs dataset ######################
            pi_specialty = ['Physician_Specialty']
            drug_nm = ['Name_of_Associated_Covered_Drug_or_Biological1']
            #    'Name_of_Associated_Covered_Drug_or_Biological2',
            #    'Name_of_Associated_Covered_Drug_or_Biological3',
            #    'Name_of_Associated_Covered_Drug_or_Biological4',
            #    'Name_of_Associated_Covered_Drug_or_Biological5']
            dev_nm = ['Name_of_Associated_Covered_Device_or_Medical_Supply1']
            #  'Name_of_Associated_Covered_Device_or_Medical_Supply2',
            #  'Name_of_Associated_Covered_Device_or_Medical_Supply3',
            #  'Name_of_Associated_Covered_Device_or_Medical_Supply4',
            #  'Name_of_Associated_Covered_Device_or_Medical_Supply5']
            corp = ['Applicable_Manufacturer_or_Applicable_GPO_Making_' +
                    'Payment_Name']
            amount = ['Total_Amount_of_Payment_USDollars']
            dispute = ['Dispute_Status_for_Publication']
            ###################################################################

            if os.path.exists(self.file):
                df = pd.read_hdf(self.file)
                # print('Loading DataFrame from:\n\t%s' % self.file)
            else:
                hdf_files = glob.glob(os.path.join(self.path, 'hdf', '*.h5'))
                hdf_files_ = []
                for file_ in hdf_files:
                    if 'RSRCH_PGYR2013' in file_:
                        hdf_files_.append(file_)
                    if 'GNRL_PGYR2013' in file_:
                        hdf_files_.append(file_)

                dfd_cols = pi_specialty + drug_nm + dev_nm + corp + amount + dispute
                df_dfd = pd.DataFrame(columns=dfd_cols)
                for hdf_file in hdf_files_:
                    if 'RSRCH' in hdf_file:
                        with pd.HDFStore(hdf_file) as hdf:
                            for key in hdf.keys():
                                df = pd.read_hdf(hdf_file, key)
                                df = df[dfd_cols]
                                df['status'] = 'allowed'
                                df = df.drop_duplicates(keep='first')
                                df_dfd = pd.concat([df_dfd, df],
                                                   ignore_index=True)
                                print('size: %d, %d' % tuple(df_dfd.shape))
                unique_vals = {}
                for col in df_dfd.columns:
                    unique_vals[col] = set(list(df_dfd[col].unique()))

                for hdf_file in hdf_files_:
                    if 'GNRL' in hdf_file:
                        with pd.HDFStore(hdf_file) as hdf:
                            for key in hdf.keys():
                                df = pd.read_hdf(hdf_file, key)
                                df = df[dfd_cols]
                                df['status'] = 'disallowed'
                                df = df.drop_duplicates(keep='first')
                                # remove all value thats are not in RSRCH
                                # for col in pi_specialty+drug_nm+dev_nm+corp:
                                #     print(col)
                                #     s1 = set(list(df[col].unique()))
                                #     s2 = unique_vals[col]
                                #     df = df.set_index(col).drop(labels=s1-s2)
                                #            .reset_index()
                                df_dfd = pd.concat([df_dfd, df],
                                                   ignore_index=True)
                                print('size: %d, %d' % tuple(df_dfd.shape))
                df_dfd = df_dfd.drop_duplicates(keep='first')
                df_dfd.to_hdf(self.file, 't1')
                df = df_dfd
            df['status'] = (df['status'] == 'allowed')
            # print_unique_values(df)
            col_action = {pi_specialty[0]: 'del',
                          drug_nm[0]: 'del',
                          dev_nm[0]: 'del',
                          corp[0]: 'se',
                          amount[0]: 'num',
                          dispute[0]: 'ohe-1',
                          'status': 'y'}
            self.clf_type = 'binary_clf'

        if self.name == 'medical_charge':
            df = pd.read_csv(self.file)
            # print_unique_values(df)
            col_action = {'State': 'ohe',
                          'Total population': 'del',
                          'Median age': 'del',
                          '% BachelorsDeg or higher': 'del',
                          'Unemployment rate': 'del',
                          'Per capita income': 'del',
                          'Total households': 'del',
                          'Average household size': 'del',
                          '% Owner occupied housing': 'del',
                          '% Renter occupied housing': 'del',
                          '% Vacant housing': 'del',
                          'Median home value': 'del',
                          'Population growth 2010 to 2015 annual': 'del',
                          'House hold growth 2010 to 2015 annual': 'del',
                          'Per capita income growth 2010 to 2015 annual':
                              'del',
                          '2012 state winner': 'del',
                          'Medical procedure': 'se',
                          'Total Discharges': 'del',
                          'Average Covered Charges': 'num',
                          'Average Total Payments': 'y'}
            self.clf_type = 'regression'  # opts: 'regression',
            # 'binary_clf', 'multiclass_clf'

        if self.name == 'road_safety':
            files = self.file
            for filename in files:
                if filename.split('/')[-1] == '2015_Make_Model.csv':
                    df_mod = pd.read_csv(filename)
                    df_mod['Vehicle_Reference'] = (df_mod['Vehicle_Reference']
                                                   .map(str))
                    df_mod['Vehicle_Index'] = (df_mod['Accident_Index'] +
                                               df_mod['Vehicle_Reference'])
                    df_mod = df_mod.set_index('Vehicle_Index')
                    df_mod = df_mod.dropna(axis=0, how='any', subset=['make'])
            for filename in files:
                if filename.split('/')[-1] == 'Accidents_2015.csv':
                    df_acc = pd.read_csv(filename).set_index('Accident_Index')
            for filename in files:
                if filename.split('/')[-1] == 'Vehicles_2015.csv':
                    df_veh = pd.read_csv(filename)
                    df_veh['Vehicle_Reference'] = (df_veh['Vehicle_Reference']
                                                   .map(str))
                    df_veh['Vehicle_Index'] = (df_veh['Accident_Index'] +
                                               df_veh['Vehicle_Reference'])
                    df_veh = df_veh.set_index('Vehicle_Index')
            for filename in files:
                if filename.split('/')[-1] == 'Casualties_2015.csv':
                    df_cas = pd.read_csv(filename)
                    df_cas['Vehicle_Reference'] = (df_cas['Vehicle_Reference']
                                                   .map(str))
                    df_cas['Vehicle_Index'] = (df_cas['Accident_Index'] +
                                               df_cas['Vehicle_Reference'])
                    df_cas = df_cas.set_index('Vehicle_Index')

            df = df_cas.join(df_mod, how='left', lsuffix='_cas',
                             rsuffix='_model')
            df = df.dropna(axis=0, how='any', subset=['make'])
            df = df[df['Sex_of_Driver'] != 3]
            df = df[df['Sex_of_Driver'] != -1]
            df['Sex_of_Driver'] = df['Sex_of_Driver'] - 1
            # print_unique_values(df)
            # col_action = {'Casualty_Severity': 'y',
            #               'Casualty_Class': 'num',
            #               'make': 'ohe',
            #               'model': 'se'}
            col_action = {'Sex_of_Driver': 'y',
                          'model': 'se',
                          'make': 'ohe'}
            df = df.dropna(axis=0, how='any', subset=list(col_action.keys()))
            self.clf_type = 'binary_clf'  # opts: 'regression',
            # 'binary_clf', 'multiclass_clf'
            self.file = self.file[0]

        if self.name == 'consumer_complaints':
            df = pd.read_csv(self.file)
            # print_unique_values(df)
            col_action = {'Date received': 'del',
                          'Product': 'ohe',
                          'Sub-product': 'ohe',
                          'Issue': 'ohe',
                          'Sub-issue': 'ohe',
                          'Consumer complaint narrative': 'se',  # too long
                          'Company public response': 'ohe',
                          'Company': 'se',
                          'State': 'del',
                          'ZIP code': 'del',
                          'Tags': 'del',
                          'Consumer consent provided?': 'del',
                          'Submitted via': 'ohe',
                          'Date sent to company': 'del',
                          'Company response to consumer': 'ohe',
                          'Timely response?': 'ohe-1',
                          'Consumer disputed?': 'y',
                          'Complaint ID': 'del'
                          }
            for col in col_action:
                if col_action[col] in ['ohe', 'se']:
                    df = df.fillna(value={col: 'nan'})
            df = df.dropna(axis=0, how='any', subset=['Consumer disputed?'])
            df.loc[:, 'Consumer disputed?'] = (df['Consumer disputed?'] ==
                                               'Yes')
            self.clf_type = 'binary_clf'  # opts: 'regression',
            # 'binary_clf', 'multiclass_clf'

        if self.name == 'traffic_violations':
            df = pd.read_csv(self.file)
            # print_unique_values(df)
            col_action = {'Date Of Stop': 'del',
                          'Time Of Stop': 'del',
                          'Agency': 'del',
                          'SubAgency': 'del',  # 'ohe'
                          'Description': 'se',
                          'Location': 'del',
                          'Latitude': 'del',
                          'Longitude': 'del',
                          'Accident': 'del',
                          'Belts': 'ohe-1',
                          'Personal Injury': 'del',
                          'Property Damage': 'ohe-1',
                          'Fatal': 'ohe-1',
                          'Commercial License': 'ohe-1',
                          'HAZMAT': 'ohe',
                          'Commercial Vehicle': 'ohe-1',
                          'Alcohol': 'ohe-1',
                          'Work Zone': 'ohe-1',
                          'State': 'del',  #
                          'VehicleType': 'del',  # 'ohe'
                          'Year': 'num',
                          'Make': 'del',
                          'Model': 'del',
                          'Color': 'del',
                          'Violation Type': 'y',
                          'Charge': 'del',  # 'y'
                          'Article': 'del',  # 'y'
                          'Contributed To Accident': 'del',  # 'y'
                          'Race': 'ohe',
                          'Gender': 'ohe',
                          'Driver City': 'del',
                          'Driver State': 'del',
                          'DL State': 'del',
                          'Arrest Type': 'ohe',
                          'Geolocation': 'del'
                          }
            for col in col_action:
                if col_action in ['ohe', 'se']:
                    df = df.fillna(value={col: 'nan'})
            self.clf_type = 'multiclass_clf'  # opts: 'regression',
            # 'binary_clf', 'multiclass_clf'

        if self.name == 'crime_data':
            df = pd.read_csv(self.file)
            # print_unique_values(df)
            col_action = {'DR Number': 'del',
                          'Date Reported': 'del',
                          'Date Occurred': 'del',
                          'Time Occurred': 'del',
                          'Area ID': 'del',
                          'Area Name': 'del',
                          'Reporting District': 'del',
                          'Crime Code': 'del',
                          'Crime Code Description': 'y',
                          'MO Codes': 'del',  # 'se'
                          'Victim Age': 'num',
                          'Victim Sex': 'ohe',
                          'Victim Descent': 'ohe',
                          'Premise Code': 'del',
                          'Premise Description': 'ohe',
                          'Weapon Used Code': 'del',
                          'Weapon Description': 'ohe',
                          'Status Code': 'del',
                          'Status Description': 'del',
                          'Crime Code 1': 'del',
                          'Crime Code 2': 'del',
                          'Crime Code 3': 'del',
                          'Crime Code 4': 'del',
                          'Address': 'del',
                          'Cross Street': 'se',  # 'se'
                          'Location ': 'del'
                          }
            for col in col_action:
                if col_action in ['ohe', 'se']:
                    df = df.fillna(value={col: 'nan'})
            self.clf_type = 'multiclass_clf'  # opts: 'regression',
            # 'binary_clf', 'multiclass_clf'

        if self.name == 'employee_salaries':
            df = pd.read_csv(self.file)
            col_action = {'Full Name': 'del',
                          'Gender': 'ohe',
                          'Current Annual Salary': 'y',
                          '2016 Gross Pay Received': 'del',
                          '2016 Overtime Pay': 'del',
                          'Department': 'del',
                          'Department Name': 'ohe',
                          'Division': 'ohe',  # 'se'
                          'Assignment Category': 'ohe-1',
                          'Employee Position Title': 'se',
                          'Underfilled Job Title': 'del',
                          'Date First Hired': 'num'
                          }
            df['Current Annual Salary'] = [float(s[1:]) for s
                                           in df['Current Annual Salary']]
            df['Date First Hired'] = [datetime.datetime.strptime(
                d, '%m/%d/%Y').year for d
                                      in df['Date First Hired']]
            for col in col_action:
                if col_action in ['ohe', 'se']:
                    df = df.fillna(value={col: 'nan'})
            self.clf_type = 'regression'  # opts: 'regression',
            # 'binary_clf', 'multiclass_clf'

        # add here info about the dataset #####################################
        if self.name == 'new_dataset':
            df = pd.read_csv(self.file)
            col_action = {}
            for col in col_action:
                if col_action in ['ohe', 'se']:
                    df = df.fillna(value={col: 'nan'})
            self.clf_type = 'multiclass_clf'  # opts: 'regression',
            # 'binary_clf', 'multiclass_clf'
        #######################################################################

        self.df = df
        self.col_action = {k: col_action[k] for k in col_action
                           if col_action[k] != 'del'}  # why not but not coherent with the rest --> self.preprocess
        return self
