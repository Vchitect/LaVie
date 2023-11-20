import numpy
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionUpscalePipeline

"""
Will encounter following warning:
- This IS expected if you are initializing CLIPTextModel from the checkpoint of a model trained on another task
or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing CLIPTextModel from the checkpoint of a model 
that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).

https://github.com/CompVis/stable-diffusion/issues/97 
according to this issue, this warning is safe.

This is expected since the vision backbone of the CLIP model is not needed to run Stable Diffusion. 
You can safely ignore the warning, it is not an error.

This clip usage is from U-ViT and same with Stable Diffusion.
"""

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, device="cuda", max_length=77):
        super().__init__()
        # self.tokenizer = CLIPTokenizer.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
        # self.text_encoder = CLIPTextModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
        # TBD: change to https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/config.json 
        # model_id = "stabilityai/stable-diffusion-x4-upscaler" # For VSR
        # upscale_pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id)
        upscale_pipeline = StableDiffusionUpscalePipeline.from_pretrained('./pretrained_models/upscaler4x')
        self.tokenizer = upscale_pipeline.tokenizer
        self.text_encoder = upscale_pipeline.text_encoder
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.text_encoder = self.text_encoder.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length,
                                        return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.text_encoder(input_ids=tokens)

        # return outputs.last_hidden_state
        return outputs[0]

    def encode(self, text):
        return self(text)
    

class TextEmbedder(nn.Module):
    """
    Embeds text prompt into vector representations. Also handles text dropout for classifier-free guidance.
    """
    def __init__(self, dropout_prob=0.1):
        super().__init__()
        self.text_encodder = FrozenCLIPEmbedder()
        self.dropout_prob = dropout_prob
    
    def token_drop(self, text_prompts, force_drop_ids=None):
        """
        Drops text to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = numpy.random.uniform(0, 1, len(text_prompts)) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = list(numpy.where(drop_ids, "None", text_prompts))
        # print(labels)
        return labels

    def forward(self, text_prompts, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            text_prompts = self.token_drop(text_prompts, force_drop_ids)
        embeddings = self.text_encodder(text_prompts)
        return embeddings
    

if __name__ == '__main__':

    r"""
    Returns:

    Examples from CLIPTextModel:

    ```python
    >>> from transformers import AutoTokenizer, CLIPTextModel

    >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> last_hidden_state = outputs.last_hidden_state
    >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
    ```"""

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_encoder = TextEmbedder(dropout_prob=0.00001).to(device)
    text_encoder1 = FrozenCLIPEmbedder().to(device)

    text_prompt = ["a photo of a cat", "a photo of a dog", 'a photo of a dog human']
    # text_prompt = ('None', 'None', 'None')
    output = text_encoder(text_prompts=text_prompt, train=True)
    output1 = text_encoder1(text_prompt)
    # print(output)
    print(output.shape)
    print(output1.shape)
    print((output==output1).all())