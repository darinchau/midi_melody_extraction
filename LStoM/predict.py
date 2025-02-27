import os
import tqdm
import torch
from train import LStoM, scale_data
from data_tools import *
from process_data import *

def predict_from_file(fpath, model_path, output_dir, preprocess=True):
    model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False).to("cpu").eval()

    os.makedirs(output_dir, exist_ok=True)
    input_notes = read_MIDI_file(fpath)

    if preprocess:
        notes = align_and_quantise_notes(input_notes)

    key_sig_event = KeySignature(name="GM")
    # TODO: Here to plug ways of detecting the MIDI time signature. Now it assumes 4/4.
    time_sig_event = TimeSignature(name="4/4")
    # compute the features
    x = compute_features(notes, key_sig_event, time_sig_event, include_melody_note_tag=False)

    x = (x - x.mean()) / x.std()

    x = torch.tensor(x.astype("float32"))
    x = torch.unsqueeze(x, dim=1)
    x = x.permute([2, 1, 0])  # [l, batch_size, no. of features]
    x = x.to("cpu")
    print(x.shape)

    with torch.no_grad():
        y = model(x)

    y = np.squeeze(y.numpy())

    mel_notes_loc = [bool(loc) for loc in np.round(y)]
    melody_notes = np.array(input_notes)[mel_notes_loc]
    print(melody_notes)
    output = os.path.join(output_dir, os.path.basename(fpath).split(".")[0] + "_melody" + ".mid")
    write_MIDI_melody_file(melody_notes, output)
    return melody_notes

if __name__ == "__main__":
    import sys
    predict_from_file(
        sys.argv[1],
        "model_results/test_model.pt",
        "./output",
        preprocess=True
    )
