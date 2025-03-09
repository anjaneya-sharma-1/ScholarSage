import re
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import config
import json
import time


model = SentenceTransformer(config.EMBEDDING_MODEL)

def expand_query(query, expansion_model=config.GENERATION_MODEL):
    """
    Convert a user query into a more effective search query by expanding it with related terms
    or reformulating it for better semantic search.
    """
    prompt = f"""
    Please reformulate the following search query to make it more effective for semantic search.
    Add relevant keywords, remove filler words, and structure it clearly.
    Keep your response brief and focused only on the reformulated query.
    
    Original query: {query}
    
    Reformulated query:
    """
    
    try:
     
        if config.HUGGINGFACE_API_KEY:
            API_URL = f"https://api-inference.huggingface.co/models/{expansion_model}"
            headers = {"Authorization": f"Bearer {config.HUGGINGFACE_API_KEY}"}
            
            simplified_params = {
                "max_new_tokens": 100,
                "temperature": 0.3,  
                "top_p": 0.9,
                "do_sample": False,  
                "return_full_text": False
            }
            
            payload = {
                "inputs": prompt,
                "parameters": simplified_params
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                expanded_query = result[0].get('generated_text', "").strip()
            elif isinstance(result, dict) and "generated_text" in result:
                expanded_query = result["generated_text"].strip()
            else:
                print("Query expansion returned unexpected format, using original query.")
                return query

            if not expanded_query or len(expanded_query) > 300:
                return query
                
            print(f"Query expansion: '{query}' → '{expanded_query}'")
            return expanded_query
            
    
        return query
        
    except Exception as e:
        print(f"Query expansion error: {str(e)}")
        return query 

def get_embedding(text, expand=False):
    """
    Get embedding using local Sentence Transformers model.
    If expand=True and text is a query, expand it before embedding.
    """
    if not text:
        return [0] * config.DIMENSION  
    
   
    if expand and len(text) < 200:  # Only expand short text (likely user queries)
        text = expand_query(text)
        
    return model.encode(text).tolist()

def search_query_embedding(query):
    """Get embedding specifically for search queries with expansion"""
    return get_embedding(query, expand=True)

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into chunks with some overlap"""
    if not text:
        return []
    
    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Simple character-based chunking
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
    
    return chunks

def batch_embed(texts, batch_size=20):
    """Generate embeddings for a batch of texts"""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch)
        all_embeddings.extend(batch_embeddings.tolist())
    
    return all_embeddings

def generate_answer(query, context, model_name=config.GENERATION_MODEL):
    """Generate an answer based on the query and context using Hugging Face"""
    
    # Format prompt based on the selected model
    if "llama" in model_name.lower():
        # Llama 2 Chat specific prompt format
        prompt = f"""<s>[INST] <<SYS>>
You are ScholarSage, a research assistant AI. Use the provided context to answer the question.
Be concise, accurate, and only use information from the context.
If you cannot answer the question based on the context, say so.
<</SYS>>

Context:
{context}

Question: {query} [/INST]
"""
    elif "gemma" in model_name.lower():
        # Gemma specific prompt format
        prompt = f"""<start_of_turn>user
You are ScholarSage, a research assistant AI. Use the provided context to answer the question.

Context:
{context}

Question: {query}<end_of_turn>

<start_of_turn>model
"""
    elif "mixtral" in model_name.lower() or "instruct" in model_name.lower():
        # Mistral Instruct/Mixtral specific format
        prompt = f"""<s>[INST] You are ScholarSage, a research assistant AI. Use the following context to answer the question.
        
Context:
{context}

Question: {query} [/INST]
"""
    else:
        # Default format for other models
        prompt = f"""
You are ScholarSage, a research assistant AI. Use the provided context to answer the question.
Be concise, accurate, and only use information from the context.
If you cannot answer the question based on the context, say so.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:
"""
    
    try:
   
        if config.HUGGINGFACE_API_KEY:
   
            restricted_models = ["google/gemma", "meta-llama", "mistralai/mixtral"]
            is_restricted = any(r in model_name.lower() for r in restricted_models)
            
            if is_restricted:
                print(f"Note: {model_name} may have access restrictions on Hugging Face API.")
                
            API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
            headers = {"Authorization": f"Bearer {config.HUGGINGFACE_API_KEY}"}
      
            payload = {
                "inputs": prompt, 
                "parameters": config.MODEL_CONFIG
            }
            

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                    
      
                    if response.status_code == 403:
                        print(f"Access denied to {model_name}. This model requires specific access permissions.")
                        raise Exception(f"Access denied (403) for model: {model_name}. This model may require special permissions or a Pro subscription.")
                    
                    # Check if the model is still loading
                    if response.status_code == 503 and "loading" in response.text.lower():
                        wait_time = min(2 ** attempt, 10)
                        print(f"Model is loading. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    
                    # Force the response to return proper JSON
                    response.raise_for_status()
                    
                    # Debug: Print raw response
                    print(f"Raw API response: {response.text[:100]}...")
                    
                    result = response.json()
                    
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get('generated_text', "")
                    elif isinstance(result, dict) and "generated_text" in result:
                        return result["generated_text"]
                    elif isinstance(result, dict) and "error" in result:
                        raise Exception(f"API Error: {result['error']}")
                    else:
                        print(f"Unexpected response format: {result}")
                        raise Exception("Unexpected response format from Hugging Face API")
                
                except requests.exceptions.RequestException as e:
                    print(f"Request error (attempt {attempt+1}/{max_retries}): {str(e)}")
                    if attempt == max_retries - 1:  # If this was the last attempt
                        raise
                    time.sleep(2)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {str(e)}. Response content: {response.text[:200]}")
                    raise Exception(f"Invalid JSON response from API: {str(e)}")
            
            raise Exception("Failed to get a valid response after multiple attempts")
        

        else:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            
            print(f"Loading {model_name}...")
            # Load tokenizer and model with appropriate trust_remote_code setting if needed
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True
            )
            

            generation_kwargs = {
                "max_new_tokens": config.MODEL_CONFIG["max_new_tokens"],
                "temperature": config.MODEL_CONFIG["temperature"],
                "top_p": config.MODEL_CONFIG["top_p"],
                "top_k": config.MODEL_CONFIG["top_k"],
                "repetition_penalty": config.MODEL_CONFIG["repetition_penalty"],
                "do_sample": config.MODEL_CONFIG["do_sample"],
            }
            
          
            if tokenizer.eos_token_id is not None:
                generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
            
       
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, **generation_kwargs)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
         
            if "llama" in model_name.lower():
                # Extract content after the prompt for llama
                answer = result.split("[/INST]")[-1].strip()
            elif "gemma" in model_name.lower():
                # Extract content after the start_of_turn model tag
                answer = result.split("<start_of_turn>model")[-1].strip()
            elif "mixtral" in model_name.lower() or "instruct" in model_name.lower():
                # Extract content after the [/INST] tag
                answer = result.split("[/INST]")[-1].strip()
            else:
                # Default extraction - get everything after ANSWER:
                parts = result.split("ANSWER:")
                answer = parts[-1].strip() if len(parts) > 1 else result
            
            return answer
            
    except Exception as e:
        print(f"Error in generate_answer: {str(e)}")
        try:
            # Fallback to a much simpler approach if both methods fail
            import re
            # Extract most relevant sentences from context based on keyword matching
            query_words = set(re.findall(r'\w+', query.lower()))
            context_sentences = re.split(r'(?<=[.!?])\s+', context)
            
            # Score sentences by number of query words they contain
            scored_sentences = []
            for sentence in context_sentences:
                sentence_words = set(re.findall(r'\w+', sentence.lower()))
                score = len(query_words.intersection(sentence_words))
                scored_sentences.append((score, sentence))
            
            # Get top 3 most relevant sentences
            scored_sentences.sort(reverse=True)
            answer = " ".join([s[1] for s in scored_sentences[:3]])
            
            return f"Based on the available information: {answer}\n\n(Note: Using fallback answer generation due to error: {str(e)})"
        except Exception as inner_e:
            print(f"Fallback error: {str(inner_e)}")
            return f"Error generating response. Please try a different query or research more information on this topic."
