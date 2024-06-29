import requests
import json
import base64
import pandas as pd
import argparse
import glob, os

parser = argparse.ArgumentParser()

parser.add_argument("--audio_dir",  default="sample_data/audio")
parser.add_argument("--save_dir",   default="sample_data/json")
args = parser.parse_args()

for path in glob.glob(os.path.join(args.audio_dir, "*.wav")):
    fname = os.path.basename(path)
    
    with open(path, 'rb') as fh:
        content = fh.read()
    res = requests.post(
        'https://demo.api.gowajee.ai/speech-to-text/transcribe',
        data=json.dumps({
                "audioData": base64.encodebytes(content).decode('utf-8')
            }),
        headers={
        'Content-type': 'application/json',
        'x-api-key':'<api-key>'
        }
    )
    os.makedirs(args.save_dir, exist_ok=True)
    result = res.json()

    df = pd.DataFrame(result['results']).rename(columns={
        'transcript': 'text',
        'start_time': 'start',
        'end_time': 'end',
    })

    new_dict = df.to_dict(orient='records')

    new_result = {
        'output': new_dict,
    }

    save_fpath = os.path.join(args.save_dir, fname.replace(".wav", '') + "-off0.json")
    with open(save_fpath, 'w') as f:
        json.dump(new_result, f, ensure_ascii=False, indent=True)
