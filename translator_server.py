import torch
import litserve as ls
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
from mosestokenizer import MosesSentenceSplitter
from nltk import sent_tokenize
from indicnlp.tokenize.sentence_tokenize import sentence_split, DELIM_PAT_NO_DANDA
import warnings
import nltk

warnings.filterwarnings("ignore")

nltk.download('punkt_tab')

flores_codes = {
    "asm_Beng": "as",
    "awa_Deva": "hi",
    "ben_Beng": "bn",
    "bho_Deva": "hi",
    "brx_Deva": "hi",
    "doi_Deva": "hi",
    "eng_Latn": "en",
    "gom_Deva": "kK",
    "guj_Gujr": "gu",
    "hin_Deva": "hi",
    "hne_Deva": "hi",
    "kan_Knda": "kn",
    "kas_Arab": "ur",
    "kas_Deva": "hi",
    "kha_Latn": "en",
    "lus_Latn": "en",
    "mag_Deva": "hi",
    "mai_Deva": "hi",
    "mal_Mlym": "ml",
    "mar_Deva": "mr",
    "mni_Beng": "bn",
    "mni_Mtei": "hi",
    "npi_Deva": "ne",
    "ory_Orya": "or",
    "pan_Guru": "pa",
    "san_Deva": "hi",
    "sat_Olck": "or",
    "snd_Arab": "ur",
    "snd_Deva": "hi",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "urd_Arab": "ur",
}

class IndicTransLitAPI(ls.LitAPI):
    def setup(self, device):
        # Load the model, tokenizer, and Indic processor
        en_indic_model_name = "ai4bharat/indictrans2-en-indic-1B"
        indic_en_model_name = "ai4bharat/indictrans2-indic-en-1B"
        self.en_indic_tokenizer = AutoTokenizer.from_pretrained(en_indic_model_name, trust_remote_code=True)
        self.indic_en_tokenizer = AutoTokenizer.from_pretrained(indic_en_model_name, trust_remote_code=True)
        self.en_indic_model = AutoModelForSeq2SeqLM.from_pretrained(en_indic_model_name, trust_remote_code=True).to(device)
        self.indic_en_model = AutoModelForSeq2SeqLM.from_pretrained(indic_en_model_name, trust_remote_code=True).to(device)
        self.ip = IndicProcessor(inference=True)
        self.device = device
        self.model = None
        self.tokenizer = None

    def decode_request(self, request):
        """
        Extracts the 'text', 'input_lang', and 'target_lang' from the request.
        Splits the 'text' into individual sentences and returns them as a list.
        """
        text = request["text"]
        src_lang = request["input_lang"]
        tgt_lang = request["target_lang"]
        
        # Split the text into sentences
        if src_lang == "eng_Latn":
            input_sentences = sent_tokenize(text)
            with MosesSentenceSplitter(flores_codes[src_lang]) as splitter:
                sents_moses = splitter([text])
            sents_nltk = sent_tokenize(text)
            if len(sents_nltk) < len(sents_moses):
                input_sentences = sents_nltk
            else:
                input_sentences = sents_moses
            input_sentences = [sent.replace("\xad", "") for sent in input_sentences]
        else:
            input_sentences = sentence_split(
                text, lang=flores_codes[src_lang], delim_pat=DELIM_PAT_NO_DANDA
            )
        # print(input_sentences)
        return (input_sentences, src_lang, tgt_lang)

    def predict(self, x):
        """
        Preprocess the input sentences, generate translations, and return the translations.
        """
        input_sentences, src_lang, tgt_lang = x

        # Preprocess the batch of input sentences
        batch = self.ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)

        if src_lang == "eng_Latn":
            self.tokenizer = self.en_indic_tokenizer
            self.model = self.en_indic_model
        elif tgt_lang == "eng_Latn":
            self.tokenizer = self.indic_en_tokenizer
            self.model = self.indic_en_model
        else:
            raise ValueError("Either input_lang or target_lang must be 'eng_Latn' (english)")

        print(self.model)
        # Tokenize the sentences
        inputs = self.tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        with self.tokenizer.as_target_tokenizer():
            generated_tokens = self.tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        # Postprocess the translations
        translations = self.ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        return translations

    def encode_response(self, output):
        """
        Joins the list of translated sentences into a single string.
        """
        translated_text = ' '.join(output)
        print(translated_text)
        return {"translated_text": translated_text}

if __name__ == "__main__":
    # Initialize the API
    api = IndicTransLitAPI()
    
    # Configure the server with 4 GPUs and 2 workers per device, total 8 copies
    server = ls.LitServer(api, accelerator="auto", devices="auto", api_path="/translate")
    
    # Run the server on port 8000
    server.run(port=8000, reload=True)
