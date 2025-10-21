# Implement-speech-dementia
```
├── raw_data/
|   ├── sub-1/
|   |   ├── sub-1
|   |   |   ├── sub-1_sub.wav
│   │   │   └── sub-1_tech.wav
|   |   ├── sub-2
|   ├── sub-2/
│   └── ...
├── data/
│   ├── metadata.csv
│   ├── audio_files/
│   │   ├── participant_001/
│   │   │   ├── participant_001_Q4.wav
│   │   │   ├── participant_001_Q6.wav
│   │   │   ├── participant_001_Q10.wav
│   │   │   └── participant_001_Q12.wav
│   │   └── participant_002/
│   │       └── ...
├── feats/
│   ├── CognoSpeak_eGeMAPSv02.csv
│   └── CognoSpeak_ComParE_2016.csv
├── processed_data/
|   ├── segments/
|   |   ├── sub-1
|   |   |   ├── sub-1_sub_000.wav -> change name into participant_001_Qx_sub.wav (becasue the flow is the same in all record and align with protocol)
|   |   |   ├── sub-1_sub_001.wav ->
|   |   |   ├── sub-1_sub_002.wav ->
│   │   │   └── sub-1_tech_000.wav -> change name into participant_001_Qx_tech.wav (becasue the flow is the same in all record and align with protocol)
|   |   ├── sub-2
│   │   └──...
|   └── slate_positions/
|   |   ├── sub-1_sub.json
│   │   └── sub-1_tech.json
└── results/
    └── CognoSpeak_results_2024-08-23_14-30-15.csv
```

# Flow to run
Raw data -> preprocessing (clapperboard-detection.ipynb) in processed_data with two dir segments and slate_positions -> Rename file into participant_001_Qx_sub.wav ..... 

# Note
Q1: Maximum Phonation Time

Q2: Read sentences

Q3: Immediately recall the story

Q4: Picture description

Q5: Picture story narrative

Q6: Phonetic fluency task

Q7: Senmantic fluency task

Q8: Procedural discourse task

Q9: Simple calculation task

Q10: Calling object task

Q11: Delayed story recall task