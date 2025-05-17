from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.filterwarnings("ignore")

class SimpleChatbot:
    def __init__(self, model_name=r"facebook/blenderbot-400M-distill"):
        # Selecting the model.
        self.model_name=model_name
    
        # Load the model and tokenizer
        self.model=AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_name)

    def generate_response(self):
        while True:
            input_text=input("You : ")
            if input_text.lower() in ["bye", "quit", "exit"]:
                print("Chatbot : Goodbye!")
                break
            
            else:
                tokenized_text=self.tokenizer.encode(input_text, return_tensors="pt")
                output=self.model.generate(tokenized_text, max_new_tokens=150)
                response=self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
                print(f"Chatbot : {response}")
        return
        
if __name__=="__main__":
    obj=SimpleChatbot()
    obj.generate_response()
