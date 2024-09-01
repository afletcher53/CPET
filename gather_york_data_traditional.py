import os

from Classes.ProjectStrings import ProjectStrings
import warnings
from tqdm import tqdm
import pandas as pd

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


def extract_gxt_data(file):
    """Extract the GXT data from a sum file."""

    with open(file, "r") as file:
        content = file.read()

    gxt_start = content.find("$;1000;GXTestDataSection;")
    gxt_end = content.find("$;3999;GXTestDataSectionEnd;;")

    if gxt_start == -1 or gxt_end == -1:
        print("Warning: GXT section markers not found in the content.")
        return {}
    content = content[gxt_start:gxt_end]
    lines = content.strip().split("\n")
    header = ["Combined"] + lines[0].split(";")[3:]

    import pandas as pd

    data = []
    for line in lines[1:]:
        parts = line.split(";")
        combined = ";".join(parts[:3]) + ";"
        values = [combined] + parts[3:]
        data.append(values)
    df = pd.DataFrame(data, columns=header)

    for column in df.columns[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df.set_index("Combined", inplace=True)

    df["identifier"] = df.index.str.split(";").str[-2]

    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    df.set_index("identifier", inplace=True)

    keep = [
        "VO2_mLmin mL/min",
        "VCO2_mLmin mL/min",
        "VO2_kg mL/kg/min",
        "VEVO2",
        "VEVCO2",
        "HR BPM",
        "HR_Filtered BPM",
        "HRR %",
        "VO2HR mL/beat",
        "Work_Watts Watts",
        "VO2WorkSlope mL/min/watt",
        "Ti sec",
        "Te sec",
        "VtET mL/sec",
        "VtTi mL/sec",
        "Vd_est mL",
        "VdVt_est",
        "HRPred %",
        "VO2Pred %",
    ]

    existing_rows = [row for row in keep if row in df.index]

    col_keep = [
        "PredMax",
        "Rest",
        "AT",
        "VO2Max",
        "WorkMax",
        "MaxValue",
        "%VO2Max/Pred",
        "%WorkMax/Pred",
        "%AT/VO2Max",
        "%AT/Pred",
        "%AT/WorkMax",
    ]
    exisiting_cols = [col for col in col_keep if col in df.columns]
    df = df.loc[existing_rows, exisiting_cols]
    return df


def extract_value(file, identifier):
    with open(file, "r") as file:
        content = file.read()
    part = content.split(identifier)[1].split("$;")[0]
    return part.split(";")[-1].strip()
def extract_summaries(ps):
    warnings.filterwarnings("ignore")
    sum_files = list(get_files(ps.anonymised, "sum"))
    
    with tqdm(sum_files, desc="Extracting GXT data", unit="file") as pbar:
        for file in pbar:
            # Update the description with the current file name
            pbar.set_description(f"Processing {os.path.basename(file)}")
            
            df = extract_gxt_data(file)
            OUES = float(extract_value(file, "$;3900;OUES Slope;"))
            VE_VO2_SLOPE = float(extract_value(file, "$;3901;VE/VCO2 Slope;"))
            VO2_WORK_SLOPE = float(extract_value(file, "$;3902;VO2/Work Slope;"))
            CHRONOTROPIC_INDEX = float(extract_value(file, "$;3903;Chronotropic Index;"))
            EXPROTOCOL = extract_value(file, "$;6098;EXProtocol;;")
            HEIGHT = extract_value(file, "$;6020;Height;;")
            WEIGHT = extract_value(file, "$;6021;Weight;;")
            SEX = extract_value(file, "$;6009;Sex;")
            BSA = extract_value(file, "$;6022;BSA;;")
            
            df.loc["OUES"] = [OUES] + [None] * (df.shape[1] - 1)
            df.loc["VE/VCO2 Slope"] = [VE_VO2_SLOPE] + [None] * (df.shape[1] - 1)
            df.loc["VO2/Work Slope"] = [VO2_WORK_SLOPE] + [None] * (df.shape[1] - 1)
            df.loc["Chronotropic Index"] = [CHRONOTROPIC_INDEX] + [None] * (df.shape[1] - 1)
            df.loc["EXProtocol"] = [EXPROTOCOL] + [None] * (df.shape[1] - 1)
            df.loc["Height"] = [HEIGHT] + [None] * (df.shape[1] - 1)
            df.loc["Weight"] = [WEIGHT] + [None] * (df.shape[1] - 1)
            df.loc["Sex"] = [SEX] + [None] * (df.shape[1] - 1)
            df.loc["BSA"] = [BSA] + [None] * (df.shape[1] - 1)
            
            df = df.loc[~df.index.duplicated(keep="first")]
            
            output_file = os.path.basename(file).split(".")[0]
            output_path = os.path.join(ps.york_traditional, f"{output_file}.csv")
            df.to_csv(output_path)


def generate_flat_output(ps):
    files = get_files(ps.york_traditional, "csv")
    files = [f for f in files if "flat_output_final" not in f]


    all_rows = []

    for idx, file in enumerate(tqdm(files, desc="Processing files")):
        df = pd.read_csv(file)

        index = df.iloc[:, 0].values
        columns = df.columns

        headers = [
            f"{i} @ {c}"
            for i in index
            for c in columns
            if c
            not in [
                "identifier",
                "OUES",
                "VE/VCO2 Slope",
                "VO2/Work Slope",
                "Chronotropic Index",
                "EXProtocol",
                "Height",
                "Weight",
                "Sex",
                "BSA",
            ]
        ]

        headers += [
            "OUES",
            "VE/VCO2 Slope",
            "VO2/Work Slope",
            "Chronotropic Index",
            "EXProtocol",
            "Height",
            "Weight",
            "Sex",
            "BSA",
        ]

        row_data = {}

        for i in index:
            if i in [
                "OUES",
                "VE/VCO2 Slope",
                "VO2/Work Slope",
                "Chronotropic Index",
                "Height",
                "Weight",
                "Sex",
                "BSA",
                "EXProtocol",
            ]:
                continue
            for c in columns:
                if c not in [
                    "identifier",
                ]:
                    row_data[f"{i} @ {c}"] = df.loc[
                        df.iloc[:, 0].values == i, c
                    ].values[0]

                    try:
                        row_data[f"{i} @ {c}"] = float(row_data[f"{i} @ {c}"])
                    except:
                        pass

        for var in [
            "OUES",
            "VE/VCO2 Slope",
            "VO2/Work Slope",
            "Chronotropic Index",
            "EXProtocol",
            "Height",
            "Weight",
            "Sex",
            "BSA",
        ]:
            
            row = df.loc[df["identifier"] == var]
            
            value = row.iloc[0, 1]

            
            try:
                value = float(value)
            except:
                pass

            row_data[var] = value
            
            row_data["research id"] = float(file.split("/")[-1].split(".")[0])
        
        all_rows.append(row_data)
        
        if idx > 0 and idx % 50 == 0:
            headers = headers + ["research id"]
            
            df_flat = pd.DataFrame(all_rows, columns=headers)

            
            columns_to_drop = [
                'HRR % @ PredMax',
                'Te sec @ %AT/Pred',
                'VtTi mL/sec @ %AT/Pred',
                "VdVt_est @ PredMax",
                "VdVt_est @ %VO2Max/Pred",
                "VdVt_est @ %WorkMax/Pred",
                "VdVt_est @ %AT/Pred",
                'VtET mL/sec @ PredMax',
                "HRPred % @ PredMax",
                "HRPred % @ %VO2Max/Pred",
                "HRPred % @ %WorkMax/Pred",
                "HRPred % @ %AT/Pred",
                "VO2Pred % @ PredMax",
                "VO2Pred % @ %VO2Max/Pred",
                "VO2Pred % @ %WorkMax/Pred",
                "VO2Pred % @ %AT/Pred",
                "Vd_est mL @ %AT/Pred",
                "Vd_est mL @ %WorkMax/Pred",
                "Vd_est mL @ %VO2Max/Pred",
                "Vd_est mL @ PredMax",
                "VtTi mL/sec @ %WorkMax/Pred",
                "VtTi mL/sec @ %VO2Max/Pred",
                "VtTi mL/sec @ PredMax",
                "VtET mL/sec @ %AT/Pred",
                "VtET mL/sec @ %WorkMax/Pred",
                "VtET mL/sec @ %VO2Max/Pred",
                "Te sec @ %WorkMax/Pred",
                "Te sec @ %VO2Max/Pred",
                "Te sec @ PredMax",
                "Ti sec @ %AT/Pred",
                "Ti sec @ %WorkMax/Pred",
                "Ti sec @ %VO2Max/Pred",
                "Ti sec @ PredMax",
                'HRR % @ %VO2Max/Pred',
                "VO2WorkSlope mL/min/watt @ %AT/Pred",
                "VO2WorkSlope mL/min/watt @ %WorkMax/Pred",
                "VO2WorkSlope mL/min/watt @ %VO2Max/Pred",
                "2HRR % @ %VO2Max/Pred",
                "Work_Watts Watts @ Rest",
                "VO2WorkSlope mL/min/watt @ PredMax",
                "VO2WorkSlope mL/min/watt @ Rest",
                "HRR % @ %WorkMax/Pred",
                "HRR % @ %AT/Pred",
                "OUES @",
                "VE/VCO2 Slope @",
                "VO2/Work Slope @",
                "Chronotropic Index @",
                "EXProtocol @",
                "Height @",
                "Weight @",
                "BSA @",
                "Sex @",
            ]
            
            df_flat = df_flat.loc[
                :, ~df_flat.columns.str.startswith(tuple(columns_to_drop))
            ]

            
            if os.path.exists(ps.york_flat):
                existing = pd.read_csv(ps.york_flat)
                existing = pd.concat([existing, df_flat])
                
                existing = existing.sort_values("research id")
                existing.to_csv(
                    ps.york_flat, index=False
                )
            else:
                df_flat.to_csv(
                    ps.york_flat, index=False
                )
                
            all_rows = []

    df_flat = pd.DataFrame(all_rows, columns=headers)

    columns_to_drop = [
                'HRR % @ PredMax',
                'Te sec @ %AT/Pred',
                'VtTi mL/sec @ %AT/Pred',
                "VdVt_est @ PredMax",
                "VdVt_est @ %VO2Max/Pred",
                "VdVt_est @ %WorkMax/Pred",
                "VdVt_est @ %AT/Pred",
                'VtET mL/sec @ PredMax',
                "HRPred % @ PredMax",
                "HRPred % @ %VO2Max/Pred",
                "HRPred % @ %WorkMax/Pred",
                "HRPred % @ %AT/Pred",
                "VO2Pred % @ PredMax",
                "VO2Pred % @ %VO2Max/Pred",
                "VO2Pred % @ %WorkMax/Pred",
                "VO2Pred % @ %AT/Pred",
                "Vd_est mL @ %AT/Pred",
                "Vd_est mL @ %WorkMax/Pred",
                "Vd_est mL @ %VO2Max/Pred",
                "Vd_est mL @ PredMax",
                "VtTi mL/sec @ %WorkMax/Pred",
                "VtTi mL/sec @ %VO2Max/Pred",
                "VtTi mL/sec @ PredMax",
                "VtET mL/sec @ %AT/Pred",
                "VtET mL/sec @ %WorkMax/Pred",
                "VtET mL/sec @ %VO2Max/Pred",
                "Te sec @ %WorkMax/Pred",
                "Te sec @ %VO2Max/Pred",
                "Te sec @ PredMax",
                "Ti sec @ %AT/Pred",
                "Ti sec @ %WorkMax/Pred",
                "Ti sec @ %VO2Max/Pred",
                "Ti sec @ PredMax",
                'HRR % @ %VO2Max/Pred',
                "VO2WorkSlope mL/min/watt @ %AT/Pred",
                "VO2WorkSlope mL/min/watt @ %WorkMax/Pred",
                "VO2WorkSlope mL/min/watt @ %VO2Max/Pred",
                "2HRR % @ %VO2Max/Pred",
                "Work_Watts Watts @ Rest",
                "VO2WorkSlope mL/min/watt @ PredMax",
                "VO2WorkSlope mL/min/watt @ Rest",
                "HRR % @ %WorkMax/Pred",
                "HRR % @ %AT/Pred",
                "OUES @",
                "VE/VCO2 Slope @",
                "VO2/Work Slope @",
                "Chronotropic Index @",
                "EXProtocol @",
                "Height @",
                "Weight @",
                "BSA @",
                "Sex @",
            ]
            
    df_flat = df_flat.loc[
                :, ~df_flat.columns.str.startswith(tuple(columns_to_drop))
            ]
    existing = pd.read_csv(ps.york_flat)
    existing = pd.concat([existing, df_flat])
    existing.to_csv(ps.york_flat, index=False)

    return ps.york_flat


def main():

    ps = ProjectStrings()
    # extract_summaries(ps)
    generate_flat_output(ps)


if __name__ == "__main__":
    main()
