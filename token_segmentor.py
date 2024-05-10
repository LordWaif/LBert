import torch

class Segmentator:
    def __init__(self):
        self.tokens = None

    def splitEvenly(self,tokens, max_length, overlap=0.0, max_segment_qtd=5):
        cls = torch.tensor([101],dtype=torch.int64)
        sep = torch.tensor([102],dtype=torch.int64)
        max_length = max_length - 2
        self.tokens = tokens
        if overlap >= max_length or overlap < 0.0:
            raise ValueError(
                "Overlap must be lower than max_length and great than zero."
            )
        if overlap != 0.0:
            overlap = int(max_length * overlap)
        for k in self.tokens.keys():
            tamanho = list(range(0,self.tokens[k].size(1),max_length-overlap))
            if len(tamanho) > max_segment_qtd:
                tamanho = tamanho[:max_segment_qtd]
            _k = [self.tokens[k][:,i:i+max_length] for i in tamanho]
            for i in range(len(_k)-1,-1,-1):
                if _k[i].size(1) < max_length:
                    padding = torch.zeros(_k[i].size(0),max_length-_k[i].size(1),dtype=torch.int64)
                    _k[i] = torch.cat((_k[i],padding),dim=1)
                else:
                    break
                _k[i] = torch.cat((cls,_k[i],sep),dim=1)
            _k = torch.stack(_k).squeeze(1)
            # Add CLS and SEP
            self.tokens[k] = _k
        return self.tokens


if __name__ == "__main__":
    import torch
    from transformers import BertTokenizer

    # Carregar tokenizer
    tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

    input_text = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam eget.",
        "A dot not int mec fgtgk but",
        "Nothing",
        "Warning: This is a warning message!",
    ]
    # Instanciar segmentator
    segmentator = Segmentator()
    for inp in input_text:
        tokens = tokenizer.encode_plus(inp,add_special_tokens=True, return_tensors="pt", padding=False, truncation=False)
        # Dividir o texto em segmentos
        tokens = segmentator.splitEvenly(tokens,max_length=2, overlap=0.5, max_segment_qtd=6)
        print(tokens['input_ids'].shape)