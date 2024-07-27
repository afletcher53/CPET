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

