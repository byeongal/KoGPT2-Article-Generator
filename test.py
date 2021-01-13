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

if torch.cuda.device_count():
  PU = 'cuda'
else:
  PU = 'cpu'

  #토큰화와 인덱싱을해서 리턴하는 클래스
class Data_Set(Dataset):

  def __init__ (self, file_path,vocab,tokenizer):
    self.data = []
    self.vocab = vocab
    self.tokenizer = tokenizer

    f = open(file_path,'r',encoding='utf-8')

    file = f.read()
    file = file.split('\n')

    dataset = []
    now = ''

    for i, line in enumerate(file):
      if i % 30 == 0 and i != 0:
        dataset.append(now)
        now = ''

      now = now + '\n' + line

    for line in dataset:
      tokenized_line = tokenizer(line[:-1])

      indexing_word = [vocab[vocab.bos_token], ]+ vocab[tokenized_line] + [vocab[vocab.eos_token]]
      self.data.append(indexing_word)

    f.close()

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    return self.data[index]



model, vocab = get_pytorch_kogpt2_model()

model.to(torch.device(PU)) #모델 연산 유닛 설정
model.train() #모델 학습모드로 변경

save_path = 'KoGPT2_checkpoint/'


kogpt2_config = {
		"initializer_range": 0.02,
		"layer_norm_epsilon": 0.000001,
		"n_ctx": 1024,
		"n_embd": 768,
		"n_head": 12,
		"n_layer": 12,
		"n_positions": 1024,
		"vocab_size": 50000,
		"activation_function": "gelu"
}

# checkpoint = torch.load(save_path+'KoGPT2_checkpoint.tar', map_location=PU)

kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))

# kogpt2model.load_state_dict(checkpoint['model_state_dict'])

kogpt2model.train()

kogpt2model.to(torch.device(PU))

model = kogpt2model

file_path = 'poem_data/dataset_ver3.txt'
tokenizer = SentencepieceTokenizer(get_tokenizer(), num_best=0, alpha=0)

data = Data_Set(file_path, vocab, tokenizer)

dataset = DataLoader(data, batch_size=8, shuffle=True, pin_memory=True)

learning_rate = 0.00005
epochs = 300
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(0, epochs+1):
  cnt = 0

  for data in dataset:
    optimizer.zero_grad()

    data = torch.stack(data)
    data = data.transpose(1,0)
    data = data.to(PU)

    output = model(data,labels=data)
    loss, logits = output[:2]
    loss.backward()
    optimizer.step()

    if cnt % 20 == 0:
      print("[+] epoch : {}, cnt : {}, loss : {} [+]".format(epoch, cnt+1, str(loss)[7:12]))

    if epoch % 20 == 0 and cnt == 1:
      torch.save({
          'epoch': epoch,
          'cnt': cnt,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': loss,
          }, save_path+'KoGPT2_checkpoint_'+str(epoch)+'.tar')
      
    cnt += 1