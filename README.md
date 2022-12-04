# Sequentional Music Generation using GANs
*autor: Mateusz Szczepanowski*

# Generate music
You can generate a sample via command line by typing: `python src/generate.py` <br>
You can also use falgs:<br>
**--seq_len [VALUE]** (default: 3)<br>
**--resolution [VALUE]** (default: 6)<br>
**--threshold [VALUE]** (default 0.5)<br>

# Quality evaluation
You can read evaluation report in **./reports** directory. There is also **./samples** directory where you can see the progression that the model makes along with the learning process. In **./reports** directory there is also a CSV file with raw metrics data - **metrics_eval.csv**.

# Training data
The training data was too big to add it to git repository. Formats that are acceptable are: **.midi** and **.npz**. In ./src/data_preparation.py there is prepare_data function with these arguments:
* file_path - absolute path to a data file
* length - length of the array, from which the music will be generated
* music_info_threshold - minimum musical information contained in the array (for example 0.04 means 4%)
* file_extension - .midi or .npz
* pianoroll_idx - for multitrack you have to determine this value. For example MAESTRO dataset has index 0 and LPD dataset has index 1 for piano data
* do_filtration - Whether to reject data samples that have too little musical information

# Additional remarks
Remember to install fluidsunth to be able to convert **midi** to **wav** (and add it to PATH for Windows).