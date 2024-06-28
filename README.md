# CPET File Linking Project

This project links CPET (Cardiopulmonary Exercise Testing) files with database entries. Follow the instructions below to set up and run the project successfully.

## Project Setup

1. Initialize the project structure:

`python init_project.py`

2. Add required files:
- Place all `.sum` files in `data/cpet raw/`
- Add your CPET database file as `CPETdb.xlsx` in the `/data/` directory

## Directory Structure
```
project_root/
│
├── data/
│   ├── cpet raw/
│   │   └── (all .sum files)
│   │
│   ├── CPETdb.xlsx
│   └── linked files.csv
│
├── init_project.py
└── link_files.py
```
## Running the File Linking Process

Execute the following command:
`python link_files.py`

### Debugging and Logs

To view debugging information and logs, use:
`python link_files.py --log-level INFO`

## Output

The script generates a file named `linked_data.csv` in the `data/` directory. This file contains information about the links created and the method used for each link.

## Troubleshooting

If the project fails to run, ensure that:
- All required files are in their correct locations
- The directory structure has been properly initialized
- You have the necessary dependencies installed.
