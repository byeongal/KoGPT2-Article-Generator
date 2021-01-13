# git clone https://github.com/SKT-AI/KoGPT2.git
# cd KoGPT2
# pip install -r requirements.txt
# pip install .

from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel
import torch
import threading
import time
from queue import Queue, Empty
from flask import Flask, render_template, request, redirect, url_for, send_file


app = Flask(__name__)
requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1


if torch.cuda.device_count():
  PU = 'cuda'
  print('cuda')
else:
  PU = 'cpu'
  print('cpu')


def dataset (file_path):
  data = []
  tokenizer = SentencepieceTokenizer(get_tokenizer())
  f = open(file_path,'r',encoding='utf-8')

  while True:
    file = f.readline()

    if not file:
      break
    line = tokenizer(file[:-1])
    indexing_word = [vocab[vocab.bos_token]]+ vocab[line] + [vocab[vocab.eos_token]]
    data.append(indexing_word)

  f.close()

  return data

model, vocab = get_pytorch_kogpt2_model()

load_path = 'KoGPT2_checkpoint/KoGPT2_checkpoint.tar'
checkpoint = torch.load(load_path, map_location=torch.device(PU))

model.to(torch.device(PU)) #모델 연산 유닛 설정
torch.load(load_path, map_location=torch.device(PU))

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

del model

save_path = 'KoGPT2_checkpoint/'

kogpt2_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 0.000025,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50000
}

checkpoint = torch.load(save_path+'KoGPT2_checkpoint.tar', map_location=PU)

kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))

kogpt2model.load_state_dict(checkpoint['model_state_dict'])

kogpt2model.eval()

kogpt2model.to(torch.device(PU))

model = kogpt2model

Tokenizer = SentencepieceTokenizer(get_tokenizer(), num_best=0, alpha=0)

def make(start_msg):
  global Tokenizer
  sentence = start_msg
  toked = Tokenizer(sentence)
  temp = []
  cnt = 0
  while True:
    input_ids = torch.tensor([vocab[vocab.bos_token],] + vocab[toked]).unsqueeze(0)
    pred = model(input_ids)[0]

    gen = vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist())
    # print(gen)
    # print(gen[-1])
    gen = gen[-1]
    cnt += 1

    if cnt == 50:
      break

    if '</s>' == gen:
      break
    sentence += gen.replace('▁', ' ')
    toked = Tokenizer(sentence)

  print(sentence)
  return sentence

@app.route('/queue-clear')
def queue_debug():
    try:
        requests_queue.queue.clear()
        return 'Clear', 200
    except Exception:
        return jsonify({'message': 'Queue clear error'}), 400


# request handling
def handle_requests_by_batch():
    try:
        while True:
            requests_batch = []
            while not (len(requests_batch) >= BATCH_SIZE):
                try:
                    requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
                except Empty:
                    continue

            batch_outputs = ['done']

            for request in requests_batch:
              request["result"] = make(request['input'][0])
            for request, output in zip(requests_batch, batch_outputs):
                request["output"] = output

    except Exception as e:
        while not requests_queue.empty():
            requests_queue.get()
        print(e)

threading.Thread(target=handle_requests_by_batch).start()

    
@app.route('/healthz')
def health_page():
    return 'ok'

@app.route('/api', methods=['GET', 'POST'])
def api():
  if requests_queue.qsize() > BATCH_SIZE: 
    return Response("Too many requests", status=429)
  try:
    args = []
    if request.method =='GET':
      start_msg=request.args.get('start_msg')
      args.append(start_msg)
  except Exception:
    print("Wrong")
    return Response("fail", status=400)
  req = {
    'input': args
  }
  requests_queue.put(req)

  while 'result' not in req:
    time.sleep(CHECK_INTERVAL)
  return req['result']

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == "__main__":
  app.run(host='0.0.0.0', port=80)




