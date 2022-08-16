import os
import sys

import json
import h5py
import pyedflib
import tqdm

print("\n Converting EDF file and annotations into H5 file")

data_dir = sys.argv[1]
h5_dir = sys.argv[2]
if not os.path.isdir(h5_dir):
    os.makedirs(h5_dir)

records = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))] # record directory
annotations = {'sleep_stage_annotation.json': ['wake', 'light', 'deep', 'rem']} # annotations contained in record dir.


for record in tqdm.tqdm(records):
    edf_filename = data_dir + record + ".edf"
    h5_filename = '{}/{}.h5'.format(h5_dir, record)

    with h5py.File(h5_filename, 'w') as h5:

        # Extract each annotation file from the record directory:
        for anno_filename, anno_labels in annotations.items():
            anno_filename = data_dir + record + anno_filename

            for anno_label in anno_labels:
                events = [
                    (x["start"], x["end"] - x["start"]) for x in json.load(open(anno_filename))
                ]

                starts, durations = list(zip(*events))
                h5.create_group(anno_label)
                h5.create_dataset("{}/start".format(anno_label), data=starts)
                h5.create_dataset("{}/duration".format(anno_label), data=durations)

        # Extract signals from edf
        with pyedflib.EdfReader(edf_filename) as f:
            labels = f.getSignalLabels()
            frequencies = f.getSampleFrequencies().astype(int).tolist()

            for i, (label, frequency) in enumerate(zip(labels, frequencies)):

                path = "{}".format(label.lower())
                data = f.readSignal(i)
                h5.create_dataset(path, data=data)
                h5[path].attrs["fs"] = frequency
