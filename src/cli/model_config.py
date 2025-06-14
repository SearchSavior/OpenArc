from transformers import AutoTokenizer

class GetModelConfig:
    def __init__(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    def get_special_token_ids(self):
        """Get only BOS, EOS, and PAD token ids from the tokenizer"""
        return {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
    

    def get_all_special_token_ids(self):
        """Get all special token ids available in the tokenizer"""
        return {
            'all_special_ids': self.tokenizer.all_special_ids,
        }
    
#def main():
#    tokenizer = GetTokens()
#    print(tokenizer.get_special_token_ids())
#    print(tokenizer.get_all_special_token_ids())







