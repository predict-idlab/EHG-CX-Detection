

# Install the requirements

Run `pip -r requirements.txt`

# Setting up the directories

## Download the required data

* Download the [Term-Preterm EHG DataSet with Tocogram](https://physionet.org/content/tpehgt/1.0.0/) and put it in a directory called `tpehgts/`

* Download the [Icelandic 16-electrode Electrohysterogram (EHG) Database](https://archive.physionet.org/pn6/ehgdb/), store it in `pn6/`

## Create an empty `output/` directory

# Running it

Set your parameters in `config.json` and run `python3 pipeline.py`. The output will be saved in the `output/` directory.