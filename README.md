# CPET File Linking Project

This project links CPET (Cardiopulmonary Exercise Testing) files with database entries. Follow the instructions below to set up and run the project successfully.



## Directory Structure
```
project_root/
│
├── Classes/
│   ├── integrity_checks.py
│   ├── mine_cpet_data.py
│   └── ProjectStrings.py
├── data/
│   ├── cpet raw/
│   │   └── (all .sum files)
│   │
│   ├── CPETdb.xlsx
│   ├── linked files.csv
|   └── anonymised/
│       ├── (all anonymised .sum files)
│       └── lined data with db.csv (anonymised linked patient data)
│
├── init_project.py
└── link_files.py
```
## Running the File Linking Process

### First Run
Execute the following command:
`python main.py`

The programme will halt after creating the directory structure, then place sum files into cpet raw directory and CPETdn.xlsx in the data directory.

### Start Matching
Rerun the program to match, anonymise and extract features, which are output to ./data/anonymised/linked data with db.csv

`python main.py`

### Debugging and Logs

To view debugging information and logs, use:
`python main.py --log-level INFO`

## Output

The script generates a file named `linked_data.csv` in the `data/` directory. This file contains information about the links created and the method used for each link.

### Linking Algorithm

For each patient in CPETDB we take {OPERATION DATE, PATIENT ID, NHS NUMBER, HOSPITAL NUMBER, MACHINE NUMBER, DATE OF BIRTH, TEST DATE}

- Load up all SUM files to create a CPET match pool.
- Remove any SUM files from the pool with no valid BxB Section
    -  As determined by no line of $;4000;BxBSection;; within the SUM file.
- Remove any SUM file that VISITDATE is not within 6 months of the OPERATION DATE
- Init a MATCH LIST = []
- For each remaining SUM file in match pool, look for matches:
    - for SUM file in CPET match pool: if PatientID == CPETdb PATIENT ID -> add to MATCH LIST if PatientID == CPETdb NHS NUMBER -> add to MATCH LIST if PatientID == CPETdb MACHINE NUMBER -> add to MATCH LIST if PatientID == CPETdb HOSPITAL NUMBER -> add to MATCH LIST if No matches on SUM file so far: if BIRTHDAY == CPETdb DATE OF BITH && VisitDateTime == CPETdb TEST DATE --> add to MATCH LIST
- Sort MATCH LIST
    - SORT by CPETdb PATIENT ID > CPETdb NHS NUMBER > CPETdb HOSPITAL NUMBER > CPETdb MACHINE NUMBER > DOB&TEST (i.e. our measure of link strength)
- IF multiple SUMs on same day, REMOVE ALL BUT FIRST from MATCH LIST.
- Take first Match list result

### Adding new features to be extracted.

sum_features.txt contains all the names of features that are to be extracted to the final data
If you input a feature that is NOT in ProjectString's featuremap, it will raise an error. 
Current GXT/BxB Feature pulling is not supported.
All potential Features are in options.txt

