#Llama2-7B notebook

# downloading every needed library

!pip install langchain
!pip install transformers
!pip install datasets
!pip install peft
!pip install accelerate
!pip install bitsandbytes
!pip install trl
!pip install safetensors
!pip install gdown
!pip install -U langchain-community
!pip install huggingface-hub -q
!pip install -q -U faiss-cpu tiktoken sentence-transformers
!pip install rouge_score

# downloading csv dataset

!gdown https://drive.google.com/file/d/1Y9-2jj8QMcBbvgxHmgDzkmL24Ixc81F3/view --fuzzy

# importing dataset

from langchain.document_loaders.csv_loader import CSVLoader

review_loader = CSVLoader(file_path="top_5000_reviews.csv")

reviews = review_loader.load()

#defining dataset splitting function

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(chunk_size,overlap):
  text_splitter = RecursiveCharacterTextSplitter(
  	chunk_size = chunk_size, # the character length of the chunk (initially 1000)
  	chunk_overlap = overlap, # the character length of the overlap between chunks (initially 100)
  	length_function = len, # the length function - in this case, character length (aka the python len() fn.)
  )
  reviews_documents = text_splitter.transform_documents(reviews)

  return reviews_documents

# defining function to create vector store

from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore

def create_vector_store(emb_model,store,reviews_documents,chunk_size,overlap):

  embedder = CacheBackedEmbeddings.from_bytes_store(emb_model, store, namespace=f"{emb_model.model_name}_{chunk_size}_{overlap}")
  vector_store = FAISS.from_documents(reviews_documents, embedder)

  return vector_store

# defining function to create embedding model

def create_emb_model(emb_model_name):
  return HuggingFaceEmbeddings(model_name=emb_model_name)

# downloading the llama2 7b model from hugging face

import torch
import transformers
from huggingface_hub import login

# log in to Hugging Face with generated token
login(token="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")

model_id = "meta-llama/Llama-2-7b-chat-hf"
# change with another to use a different mode
#meta-llama/Llama-2-13b-chat-hf
#Qwen/Qwen2-7B-Instruct
#meta-llama/Meta-Llama-3-8B-Instruct

bnb_config = transformers.BitsAndBytesConfig(
	load_in_4bit=True,
	bnb_4bit_quant_type='nf4',
	bnb_4bit_use_double_quant=True,
	bnb_4bit_compute_dtype=torch.bfloat16
)

model_config = transformers.AutoConfig.from_pretrained(
	model_id
)

model = transformers.AutoModelForCausalLM.from_pretrained(
	model_id,
	trust_remote_code=True,
	config=model_config,
	quantization_config=bnb_config,
	device_map='auto'
)

model.eval()

# defining llama7b tokenizer

tokenizer = transformers.AutoTokenizer.from_pretrained(
	model_id
)

# starting an empy model dictionary

model_dict = {}

import os

def save_response(filename, response):
	if not os.path.exists(filename):
    	with open(filename, 'w') as file:
        	file.write(response + "\n")
        	file.write("\n" + "="*50 + "\n\n")
	else:
    	with open(filename, 'a') as file:
        	file.write(response + "\n")
        	file.write("\n" + "="*50 + "\n\n")

# defining ask function

from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
import re

def ask(query,emb_model, chunk_size, overlap, temperature, model_dict,max_new_tokens=500,min_new_tokens=0):

  key = f"{emb_model.model_name},{chunk_size},{overlap}"

  if key not in model_dict:
	print("Building new model")
	reviews_documents = split_text(chunk_size=chunk_size,overlap=overlap)
	vector_store = create_vector_store(emb_model=emb_model,store=LocalFileStore("./cache/"),reviews_documents=reviews_documents,
                                   	chunk_size=chunk_size, overlap=overlap)
	model_dict[key]={"reviews_documents":reviews_documents,"vector_store":vector_store}
  else:
	print("Re-using previously built model")

  generate_text = transformers.pipeline(
	model=model,
	tokenizer=tokenizer,
	task="text-generation",
	return_full_text=True,
	temperature=temperature,
	max_new_tokens=max_new_tokens,
	min_new_tokens=min_new_tokens
  )


  llm = HuggingFacePipeline(pipeline=generate_text)

  retriever = model_dict[key]["vector_store"].as_retriever(search_kwargs={"k": 10})

  handler = StdOutCallbackHandler()

  qa_with_sources_chain = RetrievalQA.from_chain_type(
  	llm=llm,
  	retriever=retriever,
  	callbacks=[handler],
  	return_source_documents=True
  )

  response = qa_with_sources_chain.invoke({"query":query})

  formatted_response=""

  helpful_answer_start = response['result'].find('Helpful Answer:')
  unsatisfactory_answer_start = response['result'].find('Unsatisfactory Answer:')  #sometimes the answers also contain these after the Helpful answer
  unhelpful_answer_start = response['result'].find('Unhelpful Answer:')
  if unhelpful_answer_start!=-1 and unsatisfactory_answer_start!=-1:
	response_end=min(unhelpful_answer_start,unsatisfactory_answer_start)
  elif unhelpful_answer_start!=-1 and unsatisfactory_answer_start==-1:
	response_end=unhelpful_answer_start
  elif unhelpful_answer_start==-1 and unsatisfactory_answer_start!=-1:
	response_end=unsatisfactory_answer_start
  else:
	response_end=len(response['result'])


  helpful_answer = response['result'][helpful_answer_start+15:response_end].strip()


  # Extracting source documents details
  source_documents = response['source_documents']

  # Printing the results
  formatted_response += query
  formatted_response += "\n\n"
  formatted_response += key
  formatted_response += "\n\n**Helpful Answer:**\n"
  formatted_response += helpful_answer
  formatted_response += "\n\n**Source Documents:**"
  for i,doc in enumerate(source_documents):
	formatted_response += f"\n\nDocument n˚{i+1}\n"
	formatted_response += "Source: "+doc.metadata['source']+"\n"
	formatted_response += "Line: "+str(doc.metadata['row'])+"\n"
	formatted_response += doc.page_content
  return formatted_response


print(ask(query="what do you think about extreme sports?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100))

lm_name="llama2-7B"
temperature=0.2

print(ask(query="My friend nico has down syndrome, suggest a game that i could play with him?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100))

#running questions on all-minilm-l6 embeddings with T=0.2
response0 = ask(query="How is Catan according to the reviews?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"all-MiniLM-L6-v2-{lm_name}-{temperature}", response0)

response1 = ask(query="How is Brass: Birmingham according to the reviews",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"all-MiniLM-L6-v2-{lm_name}-{temperature}", response1)

response2 = ask(query="Is Brass: Birmingham a good family game?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"all-MiniLM-L6-v2-{lm_name}-{temperature}", response2)

response3 = ask(query="Break down the gameplay of Catan.",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"all-MiniLM-L6-v2-{lm_name}-{temperature}", response3)

response4 = ask(query="Which game would you suggest to bring to some friends' home who are not accustomed to board games?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"all-MiniLM-L6-v2-{lm_name}-{temperature}", response4)

response5 = ask(query="My friends find Brass: Birmingham boring. Suggest a board game that they might like, break down the features of this game, and compare it with Brass: Birmingham",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"all-MiniLM-L6-v2-{lm_name}-{temperature}", response5)

response6 = ask(query="Which game would you suggest to bring to some friends' home who are not accustomed to board games?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"all-MiniLM-L6-v2-{lm_name}-{temperature}", response6)

#running questions on multi-qa-minilm-l6 embedding with T=0.2
response0 = ask(query="How is Catan according to the reviews?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"multi-qa-MiniLM-L6-cos-v1-{lm_name}-{temperature}", response0)

response1 = ask(query="How is Brass: Birmingham according to the reviews",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"multi-qa-MiniLM-L6-cos-v1-{lm_name}-{temperature}", response1)

response2 = ask(query="Is Brass: Birmingham a good family game?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"multi-qa-MiniLM-L6-cos-v1-{lm_name}-{temperature}", response2)

response3 = ask(query="Break down the gameplay of Catan.",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"multi-qa-MiniLM-L6-cos-v1-{lm_name}-{temperature}", response3)

response4 = ask(query="Which game would you suggest to bring to some friends' home who are not accustomed to board games?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"multi-qa-MiniLM-L6-cos-v1-{lm_name}-{temperature}", response4)

response5 = ask(query="My friends find Brass: Birmingham boring. Suggest a board game that they might like, break down the features of this game, and compare it with Brass: Birmingham",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"multi-qa-MiniLM-L6-cos-v1-{lm_name}-{temperature}", response5)

response6 = ask(query="Which game would you suggest to bring to some friends' home who are not accustomed to board games?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"multi-qa-MiniLM-L6-cos-v1-{lm_name}-{temperature}", response6)

#running questions on roberta embeddings with T=0.2
response0 = ask(query="How is Catan according to the reviews?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/nli-roberta-base-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"nli-roberta-base-v2-{lm_name}-{temperature}", response0)

response1 = ask(query="How is Brass: Birmingham according to the reviews",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/nli-roberta-base-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"nli-roberta-base-v2-{lm_name}-{temperature}", response1)

response2 = ask(query="Is Brass: Birmingham a good family game?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/nli-roberta-base-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"nli-roberta-base-v2-{lm_name}-{temperature}", response2)

response3 = ask(query="Break down the gameplay of Catan.",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/nli-roberta-base-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"nli-roberta-base-v2-{lm_name}-{temperature}", response3)

response4 = ask(query="Which game would you suggest to bring to some friends' home who are not accustomed to board games?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/nli-roberta-base-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"nli-roberta-base-v2-{lm_name}-{temperature}", response4)

response5 = ask(query="My friends find Brass: Birmingham boring. Suggest a board game that they might like, break down the features of this game, and compare it with Brass: Birmingham",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/nli-roberta-base-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"nli-roberta-base-v2-{lm_name}-{temperature}", response5)

response6 = ask(query="Which game would you suggest to bring to some friends' home who are not accustomed to board games?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/nli-roberta-base-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.2,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"nli-roberta-base-v2-{lm_name}-{temperature}", response6)

temperature=0.5

#running questions on all-minilm-l6 embeddings with T=0.5
response0 = ask(query="How is Catan according to the reviews?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.5,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"all-MiniLM-L6-v2-{lm_name}-{temperature}", response0)

response1 = ask(query="How is Brass: Birmingham according to the reviews",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.5,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"all-MiniLM-L6-v2-{lm_name}-{temperature}", response1)

response2 = ask(query="Is Brass: Birmingham a good family game?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.5,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"all-MiniLM-L6-v2-{lm_name}-{temperature}", response2)

response3 = ask(query="Break down the gameplay of Catan.",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.5,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"all-MiniLM-L6-v2-{lm_name}-{temperature}", response3)

response4 = ask(query="Which game would you suggest to bring to some friends' home who are not accustomed to board games?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.5,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"all-MiniLM-L6-v2-{lm_name}-{temperature}", response4)

response5 = ask(query="My friends find Brass: Birmingham boring. Suggest a board game that they might like, break down the features of this game, and compare it with Brass: Birmingham",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.5,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"all-MiniLM-L6-v2-{lm_name}-{temperature}", response5)

response6 = ask(query="Which game would you suggest to bring to some friends' home who are not accustomed to board games?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.5,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"all-MiniLM-L6-v2-{lm_name}-{temperature}", response6)

#running questions on multi-qa-minilm-l6 embeddings with T=0.5
response0 = ask(query="How is Catan according to the reviews?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.5,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"multi-qa-MiniLM-L6-cos-v1-{lm_name}-{temperature}", response0)

response1 = ask(query="How is Brass: Birmingham according to the reviews",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.5,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"multi-qa-MiniLM-L6-cos-v1-{lm_name}-{temperature}", response1)

response2 = ask(query="Is Brass: Birmingham a good family game?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.5,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"multi-qa-MiniLM-L6-cos-v1-{lm_name}-{temperature}", response2)

response3 = ask(query="Break down the gameplay of Catan.",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.5,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"multi-qa-MiniLM-L6-cos-v1-{lm_name}-{temperature}", response3)

response4 = ask(query="Which game would you suggest to bring to some friends' home who are not accustomed to board games?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.5,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"multi-qa-MiniLM-L6-cos-v1-{lm_name}-{temperature}", response4)

response5 = ask(query="My friends find Brass: Birmingham boring. Suggest a board game that they might like, break down the features of this game, and compare it with Brass: Birmingham",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.5,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"multi-qa-MiniLM-L6-cos-v1-{lm_name}-{temperature}", response5)

response6 = ask(query="Which game would you suggest to bring to some friends' home who are not accustomed to board games?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.5,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"multi-qa-MiniLM-L6-cos-v1-{lm_name}-{temperature}", response6)

#running questions on roberta embeddings with T=0.5
response0 = ask(query="How is Catan according to the reviews?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/nli-roberta-base-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.5,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"nli-roberta-base-v2-{lm_name}-{temperature}", response0)

response1 = ask(query="How is Brass: Birmingham according to the reviews",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/nli-roberta-base-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.5,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"nli-roberta-base-v2-{lm_name}-{temperature}", response1)

response2 = ask(query="Is Brass: Birmingham a good family game?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/nli-roberta-base-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.5,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"nli-roberta-base-v2-{lm_name}-{temperature}", response2)

response3 = ask(query="Break down the gameplay of Catan.",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/nli-roberta-base-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.5,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"nli-roberta-base-v2-{lm_name}-{temperature}", response3)

response4 = ask(query="Which game would you suggest to bring to some friends' home who are not accustomed to board games?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/nli-roberta-base-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.5,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"nli-roberta-base-v2-{lm_name}-{temperature}", response4)

response5 = ask(query="My friends find Brass: Birmingham boring. Suggest a board game that they might like, break down the features of this game, and compare it with Brass: Birmingham",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/nli-roberta-base-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.5,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"nli-roberta-base-v2-{lm_name}-{temperature}", response5)

response6 = ask(query="Which game would you suggest to bring to some friends' home who are not accustomed to board games?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/nli-roberta-base-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.5,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"nli-roberta-base-v2-{lm_name}-{temperature}", response6)

temperature=0.8

#running questions on all-minilm-l6 embeddings with T=0.8
response0 = ask(query="How is Catan according to the reviews?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.8,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"all-MiniLM-L6-v2-{lm_name}-{temperature}", response0)

response1 = ask(query="How is Brass: Birmingham according to the reviews",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.8,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"all-MiniLM-L6-v2-{lm_name}-{temperature}", response1)

response2 = ask(query="Is Brass: Birmingham a good family game?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.8,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"all-MiniLM-L6-v2-{lm_name}-{temperature}", response2)

response3 = ask(query="Break down the gameplay of Catan.",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.8,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"all-MiniLM-L6-v2-{lm_name}-{temperature}", response3)

response4 = ask(query="Which game would you suggest to bring to some friends' home who are not accustomed to board games?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.8,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"all-MiniLM-L6-v2-{lm_name}-{temperature}", response4)

response5 = ask(query="My friends find Brass: Birmingham boring. Suggest a board game that they might like, break down the features of this game, and compare it with Brass: Birmingham",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.8,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"all-MiniLM-L6-v2-{lm_name}-{temperature}", response5)

response6 = ask(query="Which game would you suggest to bring to some friends' home who are not accustomed to board games?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/all-MiniLM-L6-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.8,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"all-MiniLM-L6-v2-{lm_name}-{temperature}", response6)

#running questions on multi-qa-minilm-l6 embeddings with T=0.8
response0 = ask(query="How is Catan according to the reviews?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.8,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"multi-qa-MiniLM-L6-cos-v1-{lm_name}-{temperature}", response0)

response1 = ask(query="How is Brass: Birmingham according to the reviews",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.8,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"multi-qa-MiniLM-L6-cos-v1-{lm_name}-{temperature}", response1)

response2 = ask(query="Is Brass: Birmingham a good family game?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.8,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"multi-qa-MiniLM-L6-cos-v1-{lm_name}-{temperature}", response2)

response3 = ask(query="Break down the gameplay of Catan.",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.8,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"multi-qa-MiniLM-L6-cos-v1-{lm_name}-{temperature}", response3)

response4 = ask(query="Which game would you suggest to bring to some friends' home who are not accustomed to board games?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.8,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"multi-qa-MiniLM-L6-cos-v1-{lm_name}-{temperature}", response4)

response5 = ask(query="My friends find Brass: Birmingham boring. Suggest a board game that they might like, break down the features of this game, and compare it with Brass: Birmingham",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.8,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"multi-qa-MiniLM-L6-cos-v1-{lm_name}-{temperature}", response5)

response6 = ask(query="Which game would you suggest to bring to some friends' home who are not accustomed to board games?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.8,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"multi-qa-MiniLM-L6-cos-v1-{lm_name}-{temperature}", response6)

#running questions on Roberta embeddings with T=0.8
response0 = ask(query="How is Catan according to the reviews?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/nli-roberta-base-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.8,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"nli-roberta-base-v2-{lm_name}-{temperature}", response0)

response1 = ask(query="How is Brass: Birmingham according to the reviews",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/nli-roberta-base-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.8,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"nli-roberta-base-v2-{lm_name}-{temperature}", response1)

response2 = ask(query="Is Brass: Birmingham a good family game?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/nli-roberta-base-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.8,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"nli-roberta-base-v2-{lm_name}-{temperature}", response2)

response3 = ask(query="Break down the gameplay of Catan.",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/nli-roberta-base-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.8,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"nli-roberta-base-v2-{lm_name}-{temperature}", response3)

response4 = ask(query="Which game would you suggest to bring to some friends' home who are not accustomed to board games?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/nli-roberta-base-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.8,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"nli-roberta-base-v2-{lm_name}-{temperature}", response4)

response5 = ask(query="My friends find Brass: Birmingham boring. Suggest a board game that they might like, break down the features of this game, and compare it with Brass: Birmingham",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/nli-roberta-base-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.8,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"nli-roberta-base-v2-{lm_name}-{temperature}", response5)

response6 = ask(query="Which game would you suggest to bring to some friends' home who are not accustomed to board games?",
      	emb_model=create_emb_model(emb_model_name="sentence-transformers/nli-roberta-base-v2"),
      	chunk_size=1000,
      	overlap=100,
      	temperature=0.8,
      	model_dict=model_dict,
      	min_new_tokens=100)
save_response(f"nli-roberta-base-v2-{lm_name}-{temperature}", response6)



