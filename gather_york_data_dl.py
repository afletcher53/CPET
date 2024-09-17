import os
import pandas as pd
from Classes.ProjectStrings import ProjectStrings
from bin_dl_data import main as binned_normalised
from tqdm import tqdm

def get_files(folder, extension):
    """Get all files in a folder with a given extension."""
    import os

    files = []
    for file in os.listdir(folder):
        if file.endswith(extension):
            files.append(os.path.join(folder, file))
    if not files:
        raise FileNotFoundError(
            f"No files with extension {extension} found in folder {folder}"
        )

    return files


def extract_bxb_data_from_file(file, ps):
    """Extract BxB data from a given file."""

    with open(file, "r") as f:
        content = f.read()

    bxb_start = content.find("$;4000;BxBSection;;")
    bxb_end = content.find("$;4999;BxBSectionEnd;;")
    if bxb_start != -1 and bxb_end != -1:
        bxb_data = content[bxb_start:bxb_end].split("\n")
        bxb_data = [line.split(";") for line in bxb_data if line]
        if len(bxb_data) > 1:
            bxb_data[1].pop(2)
    else:
        bxb_data = []
    if len(bxb_data) > 3:
        # drop first row
        bxb_data.pop(0)
        bxb_df = pd.DataFrame(bxb_data, columns=bxb_data[0])
        # drop coloums 0 and 1
        bxb_df.drop(bxb_df.columns[[0, 1]], axis=1, inplace=True)
        # Set row 0 as header
        bxb_df.columns = bxb_df.iloc[0]
        # drop row 0
        bxb_df = bxb_df[1:]

        bxb_df.columns = bxb_df.columns.map(str)
        # map coloumns to the correct names using ps.basic_feature_maps
        reverse_map = {str(v): k for k, v in ps.basic_feature_maps.items()}
        bxb_df.rename(columns=reverse_map, inplace=True)
        # drop row 0
        bxb_df = bxb_df[1:]

        columns_to_keep = [
            "Breath",
            "ExerTime_sec sec",
            "VO2_Lmin L/min",
            "VO2_kg mL/kg/min",
            "VCO2_Lmin L/min",
            "RER",
            "HRR_BPM BPM",
            "VO2HR mL/beat",
            "VE_BTPS L/min",
            "RR br/min",
            "VEVO2",
            "VEVCO2",
            "Speed_RPM RPM",
            "Vt_BTPS_L L",
            "Work_Watts Watts",
            "HRR %",
            "VO2Pred %",
            "HRPred %",
            "PETCO2 mmHg",
            "PETO2 mmHg",
            "PECO2 mmHg",
            "FETO2_Fr Fraction",
            "FETCO2_Fr Fraction",
            "Ti sec",
            "Te sec",
            "VtTi mL/sec",
            "VtET mL/sec",
            "Vd_est mL",
            "VdVt_est",
            "VO2WorkSlope mL/min/watt"
            # "VO2_BSA mL/m^2",
            # "VCO2_BSA mL/m^2",
        ]


        bxb_df = bxb_df[columns_to_keep]
        # cast all to int that can be cast
        bxb_df = bxb_df.apply(pd.to_numeric, errors="coerce")
        # reset index
        bxb_df.reset_index(drop=True, inplace=True)
        bxb_df["Phase"] = 0
        bxb_df.loc[
            (bxb_df["Speed_RPM RPM"] > 0) & (bxb_df["Work_Watts Watts"] == 0), "Phase"
        ] = 1
        bxb_df.loc[
            (bxb_df["Speed_RPM RPM"] > 0) & (bxb_df["Work_Watts Watts"] > 0), "Phase"
        ] = 2
        peak_load_index = bxb_df["Work_Watts Watts"].idxmax()
        bxb_df.loc[
            (bxb_df["Phase"] == 2) & (bxb_df.index > peak_load_index), "Phase"
        ] = 3

        return bxb_df


def extract_value(file, identifier):
    with open(file, "r") as file:
        content = file.read()
    part = content.split(identifier)[1].split("$;")[0]
    return part.split(";")[-1].strip()


def extract_single_variable_data_from_file(file, ps, bxb_data):
    """Extract single variable data from a given file."""

    OUES = float(extract_value(file, "$;3900;OUES Slope;"))
    VE_VO2_SLOPE = float(extract_value(file, "$;3901;VE/VCO2 Slope;"))
    VO2_WORK_SLOPE = float(extract_value(file, "$;3902;VO2/Work Slope;"))
    CHRONOTROPIC_INDEX = float(extract_value(file, "$;3903;Chronotropic Index;"))
    EXPROTOCOL = extract_value(file, "$;6098;EXProtocol;;")
    HEIGHT = extract_value(file, "$;6020;Height;;")
    WEIGHT = extract_value(file, "$;6021;Weight;;")
    SEX = extract_value(file, "$;6009;Sex;")
    BSA = extract_value(file, "$;6022;BSA;;")
    RAMPPROTOCOL = extract_value(file, "$;6098;EXProtocol;;")


    # exercise time is the phase from the start of test (phase 1) to the end of test (last phase 2)

    def get_exercise_time(bxb_data):
        try:
            #end sec is the LAST phase 2
            end_sec = bxb_data.loc[bxb_data["Phase"] == 2, "ExerTime_sec sec"].values[-1]
            start_sec = bxb_data.loc[bxb_data["Phase"] == 2, "ExerTime_sec sec"].values[
                0
            ]
            duration = end_sec - start_sec
            return duration
        except:
            return -1

    EXERCISE_TIME = get_exercise_time(bxb_data)

    return (
        OUES,
        VE_VO2_SLOPE,
        VO2_WORK_SLOPE,
        CHRONOTROPIC_INDEX,
        EXPROTOCOL,
        HEIGHT,
        WEIGHT,
        SEX,
        BSA,
        EXERCISE_TIME,
        RAMPPROTOCOL,
    )


def extract_bxb_data(ps):
    """Extract BxB data from all .sum files in the anonymised folder."""
    
    files = list(get_files(ps.anonymised, ".sum"))
    outcomes = pd.read_excel(os.path.join(ps.york, "outcomes.xlsx"))
    no_bxb_data = []
    no_research_id_in_outcomes = []

    CPET_data = pd.read_excel(os.path.join(ps.data_path, "CPETdb.xlsx"))

    with tqdm(files, desc="Extracting BxB data", unit="file") as pbar:
        for file in pbar:
            reseach_id = file.split("/")[-1].split("_")[0].split(".")[0]
            outcome_rd = outcomes.loc[outcomes["Research number"] == int(reseach_id)]
            cpet_data_rd = CPET_data.loc[CPET_data["Research number"] == int(reseach_id)]
            pbar.set_description(f"Processing {file.split('/')[-1]}")    
            bxb_data = extract_bxb_data_from_file(file, ps)
            single_variable_data = extract_single_variable_data_from_file(
                file, ps, bxb_data
            )

            if not outcome_rd.empty and not cpet_data_rd.empty:
                single_variable_data = list(single_variable_data)
                single_variable_data.append(outcome_rd["Ethnicity"].values[0])
                single_variable_data.append(outcome_rd["IMD_SCORE"].values[0])
                single_variable_data.append(outcome_rd['Date of CPET test'].values[0])
                single_variable_data.append(outcome_rd['Date of operation'].values[0])
                single_variable_data.append(outcome_rd['Operation specialty'].values[0])
                single_variable_data.append(outcome_rd['Operation subcategory'].values[0])
                single_variable_data.append(outcome_rd['Haemoglobin g/L'].values[0])
                single_variable_data.append(outcome_rd[r"White cell count x10\^9/L"].values[0])
                single_variable_data.append(outcome_rd[r"Platelets x10\^9/L"].values[0])
                single_variable_data.append(outcome_rd['Sodium mmol/L'].values[0])
                single_variable_data.append(outcome_rd['Potassium mmol/L'].values[0])
                single_variable_data.append(outcome_rd['Urea mmol/L'].values[0])
                single_variable_data.append(outcome_rd['Creatinine umol/L'].values[0])
                single_variable_data.append(outcome_rd['Total Protein g/L'].values[0])
                single_variable_data.append(outcome_rd['Albumin g/L'].values[0])
                single_variable_data.append(outcome_rd['Total bilirubin umol/L'].values[0])
                single_variable_data.append(outcome_rd['Alkaline phosphatase IU/L'].values[0])
                single_variable_data.append(outcome_rd['ALT U/L'].values[0])
                single_variable_data.append(outcome_rd['Calcium mmol/L'].values[0])
                single_variable_data.append(outcome_rd['Adjusted calcium mmol/L'].values[0])
                single_variable_data.append(outcome_rd['eGFR mls/min/1.73 m2'].values[0])
                single_variable_data.append(outcome_rd['CC_BOOKED_AT_LISTING'].values[0])
                # sex can be blank so default to 0
                if len(cpet_data_rd['Sex']) > 0:
                    single_variable_data.append(cpet_data_rd['Sex'].values[0])
                else:
                    single_variable_data.append(0)

                if len(cpet_data_rd['Age at test']) > 0:
                    single_variable_data.append(cpet_data_rd['Age at test'].values[0])
                else: 
                    single_variable_data.append(0) 
                    
                # single_variable_data.append(cpet_data_rd['Age at test'].values[0]) 
                single_variable_data.append(cpet_data_rd['Myocardial infarction'].values[0]) 
                single_variable_data.append(cpet_data_rd['Ischaemic Heart Disease'].values[0]) 
                single_variable_data.append(cpet_data_rd['Angina'].values[0]) 
                single_variable_data.append(cpet_data_rd['CABG / PCI'].values[0])
                single_variable_data.append(cpet_data_rd['Heart Failure'].values[0])
                single_variable_data.append(cpet_data_rd['CerebroVasc Dis.'].values[0])
                single_variable_data.append(cpet_data_rd['Diabetes'].values[0])
                single_variable_data.append(cpet_data_rd["No. of Lee's CRI factors"].values[0])
                single_variable_data.append(cpet_data_rd['Dysrhythmia'].values[0])
                single_variable_data.append(cpet_data_rd['Hypertension'].values[0])
                single_variable_data.append(cpet_data_rd['COPD'].values[0])
                single_variable_data.append(cpet_data_rd['Asthma'].values[0])
                single_variable_data.append(cpet_data_rd['Other lung disease'].values[0])
                single_variable_data.append(cpet_data_rd['Beta blocker'].values[0])
                single_variable_data.append(cpet_data_rd['ACE inhibitors'].values[0])
                single_variable_data.append(cpet_data_rd['Calcium channel blocker'].values[0])
                single_variable_data.append(cpet_data_rd['Diuretic'].values[0])
                single_variable_data.append(cpet_data_rd['Other antihypertensive'].values[0])
                single_variable_data.append(cpet_data_rd['Aspirin'].values[0])
                single_variable_data.append(cpet_data_rd['Clopidogrel'].values[0])
                single_variable_data.append(cpet_data_rd['Other anticoagulant'].values[0])
                single_variable_data.append(cpet_data_rd['Statin'].values[0])
                single_variable_data.append(cpet_data_rd['Digoxin'].values[0])
                single_variable_data.append(cpet_data_rd['Insulin'].values[0])
                single_variable_data.append(cpet_data_rd['Oral hypoglycaemic'].values[0])
                single_variable_data.append(cpet_data_rd['Inhalers'].values[0])
                single_variable_data.append(cpet_data_rd['Systemic steroid'].values[0])
                single_variable_data.append(cpet_data_rd['Anaerobic Threshold  ml/kg/min)'].values[0])
                single_variable_data.append(cpet_data_rd['Peak VO2'].values[0])
                single_variable_data.append(cpet_data_rd['VE/VCO2 at AT'].values[0])
                single_variable_data.append(cpet_data_rd['VO2/HR at AT'].values[0])
                single_variable_data.append(cpet_data_rd['Oxygen pulse response'].values[0])
                single_variable_data.append(cpet_data_rd['VO2/Workrate response'].values[0])


                # if any of the values are NaN, set them to -1
                single_variable_data = [-1 if pd.isna(x) else x for x in single_variable_data]

                # convert any that can be into floats
               
                for i in range(len(single_variable_data)):
                    try:
                        # skip the date columns
                        if i in [12, 13]:
                            continue
                        single_variable_data[i] = float(single_variable_data[i])
                    except:
                        pass

                # convert any Nos into 0s and Yes into 1s
                single_variable_data = [0 if x == "No" else x for x in single_variable_data]
                single_variable_data = [1 if x == "Yes" else x for x in single_variable_data]
                # convert any females into 0 and males into 1
                single_variable_data = [0 if x == "Female" else x for x in single_variable_data]
                single_variable_data = [1 if x == "Male" else x for x in single_variable_data]
                # convert any Normals into 0 and Abnormals into 1
                single_variable_data = [0 if x == "Normal" else x for x in single_variable_data]
                single_variable_data = [1 if x == "Abnormal" else x for x in single_variable_data]
                
            else:
                no_research_id_in_outcomes.append(file)
                continue
            if bxb_data is None:
                no_bxb_data.append(file)
                continue
            bxb_data.to_csv(
                ps.york_dl + f"/{file.split('/')[-1].replace('.sum', '_bxb.csv')}",
                index=False,
            )
            single_variable_data = pd.DataFrame(single_variable_data).T
            single_variable_data.columns = [
                "OUES",
                "VE_VO2_SLOPE",
                "VO2_WORK_SLOPE",
                "CHRONOTROPIC_INDEX",
                "EXPROTOCOL",
                "HEIGHT",
                "WEIGHT",
                "SEX",
                "BSA",
                "EXERCISE_TIME",
                "RAMPPROTOCOL",



        
 
     
                #CPD data
                "ETHNICITY",
                "IMD_SCORE",
                'DATE_OF_CPET_TEST',
                'DATE_OF_OPERATION',
                'OPERATION_SPECIALTY',
                'OPERATION_SUBCATEGORY',
                'HAEMOGLOBIN',
                'WBC',
                'PLATELETS',
                'SODIUM',
                'POTASSIUM',
                'UREA',
                'CREATININE',
                'TOTAL PROTEIN',
                'ALBUMIN',
                'TOTAL BILIRUBIN',
                'ALP',
                'ALT',
                'CALCIUM',
                'ADJUSTED_CALCIUM',
                'EGFR',
                'CC_BOOKED_AT_LISTING',
                # CPET data
                'CPET_SEX',
                'AGE',
                'MI',
                'IHD',
                'ANGINA',
                'CABG_PCI',
                'HF',
                'CVD',
                'DIABETES',
                'CRI',
                'DYSRHYTHMIA',
                'HYPERTENSION',
                'COPD',
                'ASTHMA',
                'OTHER_LUNG_DISEASE',
                'BETA_BLOCKER',
                'ACE_INHIBITORS',
                'CALCIUM_BLOCKER',
                'DIURETIC',
                'OTHER_ANTI_HYPERTENSIVE',
                'ASPIRIN',
                'CLOPIDOGREL',
                'OTHER_ANTICOAGULANT',
                'STATIN',
                'DIGOXIN',
                'INSULIN',
                'ORAL_HYPOGLYCAEMIC',
                'INHALERS',
                'SYSTEMIC_STEROID',
                'AT',
                'PEAK_VO2',
                'VE_VCO2_AT',
                'VO2_HR_AT',
                'OXYGEN_PULSE',
                'OXYGVO2_WORKRATE',



            ]
            single_variable_data.to_csv(
                ps.york_dl
                + f"/{file.split('/')[-1].replace('.sum', '_single_variable_data.csv')}",
                index=False,
            )

    # save files with no bxb data
    with open(ps.york_dl + "/empty_bxb_data.txt", "w") as f:
        f.write("\n".join(no_bxb_data))

    # save files with no research id in outcomes
    with open(ps.york_dl + "/no_research_id_in_outcomes.txt", "w") as f:
        f.write("\n".join(no_research_id_in_outcomes))

def main():
    ps = ProjectStrings()
    extract_bxb_data(ps)
    binned_normalised()


if __name__ == "__main__":
    main()
