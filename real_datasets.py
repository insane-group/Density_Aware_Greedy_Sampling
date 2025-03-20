import os
import numpy  as np
import pandas as pd

def data_preparation(sourceFile=None, research_data="zifs_diffusivity") -> list:

    Y = []
    X = []

    if research_data != "zifs_diffusivity" and sourceFile is None:
        print("Error. Source file path for real data not provided.")
        exit()

    if research_data == "zifs_diffusivity":
        if sourceFile is not None:
            data_from_file = zif_diffus_data(sourceFile)
        else:
            data_from_file = zif_diffus_data()

        Y = ["logD"]
        X = ['diameter','mass','ascentricF', 'kdiameter','ionicRad',
            'MetalNum','MetalMass','σ_1', 'e_1',
            'linker_length1', 'linker_length2', 'linker_length3',
            'linker_mass1', 'linker_mass2', 'linker_mass3',
            'func1_length', 'func2_length', 'func3_length', 
            'func1_mass', 'func2_mass', 'func3_mass']
    
    elif research_data == "co2":
        data_from_file = pd.read_csv(sourceFile)
        data_from_file = data_from_file.rename(columns={'CO2_working_capacity(mol/kg)':'working_capacity', 'mof_name':'type'})

        # One Hot Encode Data
        features = ["Nodular_BB1", "Nodular_BB2", "Connecting_BB1", "Connecting_BB2"]
        data_from_file = pd.get_dummies(data_from_file, columns=features,dtype=int)

        Y = ["working_capacity"]
        X = [feature_label for base_label in features for feature_label in list(data_from_file.columns) if base_label in feature_label]

    elif research_data == "co2/n2":
        data_from_file = pd.read_csv(sourceFile)
        data_from_file = data_from_file.rename(columns={'CO2/N2_selectivity':'selectivity', 'mof_name':'type'})

        # One Hot Encode Data
        features = ["Nodular_BB1", "Nodular_BB2", "Connecting_BB1", "Connecting_BB2"]
        data_from_file = pd.get_dummies(data_from_file, columns=features,dtype=int)

        Y = ["selectivity"]
        X = [feature_label for base_label in features for feature_label in list(data_from_file.columns) if base_label in feature_label]    

    elif research_data == "o2":
        data_from_file = pd.read_csv(sourceFile)

        
        data_from_file["SelfdiffusionofO2cm2s"] = np.log(data_from_file["SelfdiffusionofO2cm2s"])
        data_from_file = data_from_file.rename(columns={'SelfdiffusionofO2cm2s':'logSelfD','MOFRefcodes': 'type'})
        Y = ["logSelfD"]
        
        X = ['LCD',	'PLD',	'LFPD',	'Volume',	'ASA_m2_g',	
             'ASA_m2_cm3',	'NASA_m2_g', 'NASA_m2_cm3',	
             'AV_VF',	'AV_cm3_g',	'NAV_cm3_g', ' H', 'C',	'N', 'metal type', 
             ' total degree of unsaturation', 'metalic percentage',	' oxygetn-to-metal ratio',	
             'electronegtive-to-total ratio', ' weighted electronegativity per atom', 
             ' nitrogen to oxygen ']  #, 'mass',	'ascentricF',	'diameter',	'kdiameter'

    elif research_data == "n2":
        data_from_file = pd.read_csv(sourceFile)

        
        data_from_file["SelfdiffusionofN2cm2s"] = np.log(data_from_file["SelfdiffusionofN2cm2s"])
        data_from_file = data_from_file.rename(columns={'SelfdiffusionofN2cm2s':'logSelfD','MOFRefcodes': 'type'})
        Y = ["logSelfD"]
        
        X = ['LCD',	'PLD',	'LFPD',	'Volume',	'ASA_m2_g',	
             'ASA_m2_cm3',	'NASA_m2_g', 'NASA_m2_cm3',	
             'AV_VF',	'AV_cm3_g',	'NAV_cm3_g', ' H', 'C',	'N', 'metal type', 
             ' total degree of unsaturation', 'metalic percentage',	' oxygetn-to-metal ratio',	
             'electronegtive-to-total ratio', ' weighted electronegativity per atom', 
             ' nitrogen to oxygen ']  #, 'mass',	'ascentricF',	'diameter',	'kdiameter'         

    elif research_data == "ch4":
        data_from_file = pd.read_csv(sourceFile)

        
        data_from_file["D_CH4 (cm2/s)"] = np.log(data_from_file["D_CH4 (cm2/s)"])
        data_from_file = data_from_file.rename(columns={'D_CH4 (cm2/s)':'logSelfD','MOF name': 'type'})
        Y = ["logSelfD"]
        
        X = ['LCD (Å)','LCD/PLD','PLD (Å)','density (g/cm3)','Pore Volume (cm3/g)','Porosity','Surface Area (m2/g)',
        '%C','%H','%O','%N','%Halojen','%Metalloids','%Ametal','%Metal','O-to-M','MP','TDU','DU']  #, 'mass',	'ascentricF',	'diameter',	'kdiameter'

    elif research_data == "h2":
        data_from_file = pd.read_csv(sourceFile)

        
        data_from_file["D_H2 (cm2/s)"] = np.log(data_from_file["D_H2 (cm2/s)"])
        data_from_file = data_from_file.rename(columns={'D_H2 (cm2/s)':'logSelfD','MOF': 'type'})
        Y = ["logSelfD"]
        
        X = ['LCD (Å)','LCD/PLD','PLD (Å)','density (g/cm3)','Pore Volume (cm3/g)','Porosity','Surface Area (m2/g)',
        '%C','%H','%O','%N','%Halojen','%Metalloids','%Ametal','%Metal','O-to-M','MP','TDU','DU']

    elif research_data == "he":
        data_from_file = pd.read_csv(sourceFile)

        
        data_from_file["D_He (cm2/s)"] = np.log(data_from_file["D_He (cm2/s)"])
        data_from_file = data_from_file.rename(columns={'D_He (cm2/s)':'logSelfD','MOF': 'type'})
        Y = ["logSelfD"]
        
        X = ['LCD (Å)','LCD/PLD','PLD (Å)','density (g/cm3)','Pore Volume (cm3/g)','Porosity','Surface Area (m2/g)','%C','%H','%O','%N','%Halojen','%Metalloids','%Ametal','%Metal','TDU','DU','MP','O-to-M']    

    elif research_data == "methane":
        data_from_file = pd.read_csv(sourceFile)
        data_from_file = data_from_file.rename(columns={' absolute methane uptake high P [v STP/v]':'methane_uptake', ' name':'type'})        

        Y = ['methane_uptake']
        X = ['dimensions', ' supercell volume [A^3]', ' density [kg/m^3]',
             ' surface area [m^2/g]', ' num carbon', ' num hydrogen',
             ' num nitrogen', ' num oxygen', ' num sulfur', ' num silicon',
             ' vertices', ' edges', ' genus', ' largest included sphere diameter [A]',
             ' largest free sphere diameter [A]', ' largest included sphere along free sphere path diameter [A]']

    else:
        print("Error the provided dataset name does not exist.")
        exit()

    return data_from_file, X, Y

def zif_diffus_data(source_file = './TrainData.xlsx') -> pd.DataFrame:

    # Read file
    df = pd.DataFrame()
    if source_file.split('.')[-1] == 'csv':
        df = pd.read_csv(source_file)
    else:
        df=pd.read_excel(source_file)

    if 'diffusivity' in df.columns:
        df['logD'] = np.log10(df['diffusivity'])

    # Keep appropriate columns
    approriate_columns = [ 'type', 'gas', 'MetalNum', 'aperture', 'size - van der Waals (Å)','mass', 'ascentricF', 'logD', 'size - kinetic diameter (Å)', 'ionicRad', 
        'Μ-N_lff', 'Μ-N_kFF', 'MetalCharge', 'MetalMass',
        'σ_1', 'e_1', 'σ_2', 'e_2', 'σ_3', 'e_3', 'linker_length1', 'linker_length2',
        'linker_length3', 'linker_mass1', 'linker_mass2', 'linker_mass3',
        'func1_length', 'func2_length', 'func3_length', 'func1_mass',  
        'func2_mass', 'func3_mass', 'func1_charge', 'func2_charge',
        'func3_charge',]

    cleaned_original_df = pd.DataFrame()
    for col in approriate_columns:
        if col in df.columns:
            if col == 'size - van der Waals (Å)':
                cleaned_original_df['diameter'] =df[col]
            elif col == 'size - kinetic diameter (Å)':
                cleaned_original_df['kdiameter'] =df[col]
            else:
                cleaned_original_df[col] =df[col]
        elif col == 'size - van der Waals (Å)':
            cleaned_original_df['diameter'] =df['diameter']
        elif col == 'size - kinetic diameter (Å)':
            cleaned_original_df['kdiameter'] =df['diameter']
        else:
            continue

    # Clear NA entries
    cleaned_original_df = cleaned_original_df.dropna()
    # Remove outlier molecule (?)    
    cleaned_original_df=cleaned_original_df.reset_index(drop=True)
   
    return cleaned_original_df

def plot_data_exists(data_path) -> bool:

    """ Check wheather plot data already exist and return the respective truth value.
        data_path1:     The path to look for the set of data."""

    if not os.path.exists(data_path):
        return False

    return True
