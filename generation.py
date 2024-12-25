import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, logging
)
import os
file_path = os.path.abspath(__file__)
PROJECT_PATH = os.path.dirname(file_path)
logging.set_verbosity(logging.CRITICAL)

class QAmodel:
    def __init__(self):
        path = PROJECT_PATH + '/model/llm'
        
        # No change params
        use_4bit, bnb_4bit_compute_dtype, bnb_4bit_quant_type, use_nested_quant = True, "float16", "nf4", False # To quantization
        device_map = {"": 0}

        print("Load tokenizer and generation model with QLoRA configuration...", end='') 

        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            path + "/llm.pt",
            quantization_config=bnb_config,
            device_map=device_map
        )

        model.config.use_cache = False
        model.config.pretraining_tp = 1

        tokenizer = AutoTokenizer.from_pretrained(path + "/tokenizer", trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" 

        print('Done')
        self.tokenizer = tokenizer
        self.pine = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
        
    def chat_template(self, input_text, context):
        # Định nghĩa các phần của prompt
        input_prefix = "<s>[INST] "
        input_suffix = "[/INST] "
        pre_prompt_prefix = "<<SYS>>\n"
        pre_prompt_suffix = "<</SYS>> \n\n"

        # Prompt chung định hình cách trả lời của mô hình
        pre_prompt = (
            "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, "
            "đồng thời giữ an toàn.\nCâu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, "
            "phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời "
            "của bạn không có thiên kiến xã hội và mang tính tích cực. Nếu một câu hỏi không có ý nghĩa hoặc không "
            "hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. "
            "Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trả lời rằng bạn không biết và vui lòng không "
            "chia sẻ thông tin sai lệch."
        )

        # Ghép các phần lại với nhau
        formatted_prompt = (
            f"{input_prefix}{pre_prompt_prefix}{pre_prompt}{pre_prompt_suffix}"
            f"Ngữ cảnh: {context}\n\n"
            f"Câu hỏi: {input_text}\n"
            f"{input_suffix}"
        )
        return formatted_prompt
    
    def generation(self, input_text, context):
        stop_strings = ['[/INST]', 'Câu hỏi:', '</s>']
        response = self.pine(f"{self.chat_template(input_text, context)}", stop_strings = stop_strings, tokenizer = self.tokenizer)[0]['generated_text']
        if '[/INST]  Câu trả lời: ' in response:
            response = response.split('[/INST]  Câu trả lời: ')[1]
        else:
            return 'Xin lỗi tôi không hiểu câu hỏi của bạn, bạn có thể nói rõ hơn được không?'
        
        for s in stop_strings:
            response = response.replace(s, '')
        while response[-1] == '\n':
            response = response[:-1]
        return response
