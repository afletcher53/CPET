

For each patient in CPETDB we take {OPERATION DATE, PATIENT ID, NHS NUMBER, HOSPITAL NUMBER, MACHINE NUMBER, DATE OF BIRTH, TEST DATE}

1) Load up all SUM files to create a CPET match pool. 
2) Remove any SUM files from the pool with no valid BxB Section
    - As determined by no line of $;4000;BxBSection;; within the SUM file. 
3) Remove any SUM file that VISITDATE is not within 6 months of the OPERATION DATE
3) Init a MATCH LIST = []
4) For each remaining SUM file in match pool, look for matches:
    - for SUM file in CPET match pool:
        if PatientID == CPETdb PATIENT ID -> add to MATCH LIST
        if PatientID == CPETdb NHS NUMBER -> add to MATCH LIST
        if PatientID == CPETdb MACHINE NUMBER -> add to MATCH LIST
        if PatientID == CPETdb HOSPITAL NUMBER -> add to MATCH LIST
        if No matches on SUM file so far:
            if BIRTHDAY == CPETdb DATE OF BITH && VisitDateTime == CPETdb TEST DATE --> add to MATCH LIST
5) Sort MATCH LIST
    - SORT by CPETdb PATIENT ID > CPETdb NHS NUMBER > CPETdb HOSPITAL NUMBER > CPETdb MACHINE NUMBER > DOB&TEST (i.e. our measure of link strength)
6) IF multiple SUMs on same day, REMOVE ALL BUT FIRST from MATCH LIST. 
6) Take first Match list result 

