import pandas as pd
import json
import glob

from dotenv import load_dotenv
import os

load_dotenv()

SAMBA_API_KEY = os.getenv("SAMBA_API_KEY")
import argparse

from rapper import RetrievalAugmentation, BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel, RetrievalAugmentationConfig, chat
parser = argparse.ArgumentParser()

parser.add_argument("--json_path",  default="sample_data/json_data")
parser.add_argument("--save_dir",   default="md_render")
parser.add_argument("--save_fname", default="feedback.md")
parser.add_argument("--save_text", default="script.txt")
args = parser.parse_args()

# You can define your own Summarization model by extending the base Summarization Class. 
class TyphoonSummarizationModel(BaseSummarizationModel):
    def __init__(self, model_name="llama3-70b-typhoon"):
        # Initialize the tokenizer and the pipeline for the typhoon model
        self.model_name = model_name

    def summarize(self, context, max_tokens=256):
        # Format the prompt for summarization
        
        # Generate the summary using the typhoon.
        system_prompt = "You are a helpful assistant named MITM. You always answer in Thai."
        user_prompt = f"Write a summary of the following, including as many key details as possible: {context}:"
        summary = chat.get_response(
            system_prompt, 
            user_prompt,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            model=self.model_name,
            max_tokens_to_generate=max_tokens,
            api_key=SAMBA_API_KEY,
            )
        
        # Extracting and returning the generated summary
        return summary

class TyphoonQATranscriptionModel(BaseQAModel):
    def __init__(self, model_name="llama3-70b-typhoon"):
        # Initialize the tokenizer and the pipeline for the typhoon model
        self.model_name = model_name

    def answer_question(self, context, question):
        # Apply the chat template for the context and question
        system_prompt = "You are assistant, efficient in answer question in meeting transcription. You always answer in Thai."
        user_prompt = f"{question}\n```transcription\n{context}\n```"

        # Generate the answer using typhoon
        answer = chat.get_response(
            system_prompt, 
            user_prompt,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            model=self.model_name,
            api_key=SAMBA_API_KEY,
            )
        return answer

from sentence_transformers import SentenceTransformer
class BGEm3EmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)

# Init Rag
RAC = RetrievalAugmentationConfig(
    summarization_model=TyphoonSummarizationModel(), 
    qa_model=TyphoonQATranscriptionModel(), 
    embedding_model=BGEm3EmbeddingModel(),
    tb_max_tokens=512,
    tb_summarization_length=256,
    )

RA_MITM = RetrievalAugmentation(config=RAC,)

def load_df(path):
    with open(path, 'r') as f:
        d = json.load(f)

    df = pd.DataFrame(d['output'])
    df['speaker'] = os.path.basename(path).split('-')[0]
    return df

def get_script_with_timestamp_sort(df):
    script = ""
    for i, r in df.sort_values(["start"]).iterrows():
        script += f"[{r.speaker},{r.start:.1f}-{r.end:.1f}]: {r.text}" 
        script += "\n"
    return script

def get_script_without_timestamp_sort(df):
    script = ""
    for i, r in df.sort_values(["start"]).iterrows():
        script += f"[{r.speaker}]: {r.text}" 
        script += "\n"
    return script

# load json and turn into script
json_paths = glob.glob(os.path.join(args.json_path, "*.json"))
all_df = [load_df(p) for p in json_paths]
all_df =pd.concat(all_df).reset_index(drop=True)

for p in json_paths:
    fname = os.path.basename(p)
    fname = fname.replace(".json", '')
    speaker, offset = fname.split("-")
    offset = int(offset.replace("off",''))
    
    all_df.loc[all_df['speaker'] == speaker,'start'] += offset
    all_df.loc[all_df['speaker'] == speaker, 'end'] += offset

script = get_script_without_timestamp_sort(all_df)
with open(os.path.join(args.save_dir, args.save_text), 'w') as f:
    f.write(script)

RA_MITM.add_documents(script)

# Create Prompt
all_users = all_df['speaker'].unique().tolist()

querys = [
    'มี user คนไหนพูดในประชุมนี้บ้าง เอามาจากใน [name] เท่านั้น ตอบเป็น bullet',
    'Agenda 3 ข้อหลักของประชุมนี้คืออะไร บอกเป็น ordered lists',
    'สรุปประชุมนีให้หน่อย บอกเป็น bullet สั้นๆ',
    'ในประชุมนี้ไม่ควรพูดเรื่องอะไรบ้าง ทำไมถึงเป็นอย่างนั้น อธิบายทีละสเต็ป บอกเป็นข้อๆ แบบ ordered lists',
    'ถ้าจะให้การประชุมนีดีขึ้นต้องทำยังไงบ้าง ในเชิงของวิธีการพูด ทำไมถึงเป็นอย่างนั้น อธิบายทีละสเต็ป',
]

qtitles = [
    'มีใครเข้าในประชุมนี้บ้าง',
    'Agenda หลักของประชุมนี้',
    'สรุปประชุม',
    'ในประชุมนี้ไม่ควรพูดเรื่องอะไรบ้าง',
    'ถ้าจะให้การประชุมนีดีขึ้นต้องทำยังไงบ้าง',
]

# add user feedback
for user in all_users:
    q = f'ถ้าจะให้การประชุมนีดีขึ้นต้องทำยังไงบ้าง สำหรับ "[{user}]" ในเชิงของวิธีการพูด ทำไมถึงเป็นอย่างนั้น อธิบายทีละสเต็ป'
    querys.append(q)
    qtitles.append(f'ถ้าจะให้การประชุมนีดีขึ้นต้องทำยังไงบ้าง สำหรับ "[{user}]"')
    q = f'ในประชุมนี้ไม่ควรพูดเรื่องอะไรบ้าง สำหรับ "[{user}]" ทำไมถึงเป็นอย่างนั้น อธิบายทีละสเต็ป บอกเป็นข้อๆ แบบ ordered lists'
    querys.append(q)
    qtitles.append(f'ในประชุมนี้ไม่ควรพูดเรื่องอะไรบ้าง สำหรับ "[{user}]"')


all_response = []
for i, q in enumerate(querys):
    answer = RA_MITM.answer_question(q, top_k=10, collapse_tree=True,)
    context = RA_MITM.retrieve(q)
    all_response.append({'question': qtitles[i], 'context': context, 'answer': answer})

# title 
title_sum = RA_MITM.answer_question("สรุปข้อมูลออกมาเป็น หนึ่งประโยคสั้นๆ ออกมาเป็นหัวข้อของรายงานการประชุม", top_k=10, collapse_tree=True,)

# create md render
md_render = ""
md_render += f"# {title_sum}\n"
dfr = pd.DataFrame(all_response)
for i, r in dfr.iterrows():
    q = r.question
    c = r.context
    a = r.answer

    md_render += f"## {q}  \n"
    md_render += f"{a}  \n"

# write files
os.makedirs(args.save_dir, exist_ok=True)
with open(os.path.join(args.save_dir, args.save_fname), 'w') as f:
    f.write(md_render)

# to do อาจจะปรับ parameter 
# context len ของการ ตัด chunk ให้ยาวขึ้น
