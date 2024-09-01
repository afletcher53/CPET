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
            "VO2WorkSlope mL/min/watt",
            "VO2_BSA mL/m^2",
            "VCO2_BSA mL/m^2",
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

        # calculate the phase of test similar to check_integrities
        return bxb_df


def extract_value(file, identifier):
    with open(file, "r") as file:
        content = file.read()
    part = content.split(identifier)[1].split("$;")[0]
    return part.split(";")[-1].strip()


def extract_single_variable_data_from_file(file, ps, bxb_data):
    """Extract single variable data from a given file."""
    with open(file, "r") as f:
        content = f.read()

    OUES = float(extract_value(file, "$;3900;OUES Slope;"))
    VE_VO2_SLOPE = float(extract_value(file, "$;3901;VE/VCO2 Slope;"))
    VO2_WORK_SLOPE = float(extract_value(file, "$;3902;VO2/Work Slope;"))
    CHRONOTROPIC_INDEX = float(extract_value(file, "$;3903;Chronotropic Index;"))
    EXPROTOCOL = extract_value(file, "$;6098;EXProtocol;;")
    HEIGHT = extract_value(file, "$;6020;Height;;")
    WEIGHT = extract_value(file, "$;6021;Weight;;")
    SEX = extract_value(file, "$;6009;Sex;")
    BSA = extract_value(file, "$;6022;BSA;;")

    # exercise time is the phase from the start of test (phase 1) to the end of test (phase 3)

    def get_exercise_time(bxb_data):
        try:
            end_sec = bxb_data.loc[bxb_data["Phase"] == 3, "ExerTime_sec sec"].values[0]
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
    )


def extract_bxb_data(ps):
    """Extract BxB data from all .sum files in the anonymised folder."""
    
    files = list(get_files(ps.anonymised, ".sum"))
    
    with tqdm(files, desc="Extracting BxB data", unit="file") as pbar:
        for file in pbar:
            pbar.set_description(f"Processing {file.split('/')[-1]}")    
            bxb_data = extract_bxb_data_from_file(file, ps)
            single_variable_data = extract_single_variable_data_from_file(
                file, ps, bxb_data
            )
            if bxb_data is None:
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
            ]
            single_variable_data.to_csv(
                ps.york_dl
                + f"/{file.split('/')[-1].replace('.sum', '_single_variable_data.csv')}",
                index=False,
            )
def main():
    ps = ProjectStrings()
    # extract_bxb_data(ps)
    binned_normalised()


if __name__ == "__main__":
    main()
