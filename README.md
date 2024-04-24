# Llama-3-8B-Instruct-Chinese-chat


Llama-3-8B-Instruct in Chinese 自己微调版本

### 训练可用数据整理

| 数据集                                                                                                          | 介绍                                                                                                                                                                                                      |
|----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [firefly-train-1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)                            | 包含了23种常见的中文NLP任务的数据，并且构造了许多与中华文化相关的数据，如对联、作诗、文言文翻译、散文、金庸小说等。对于每个任务，由人工书写若干种指令模板，保证数据的高质量与丰富度，数据量为115万。                                                        |
| [moss-003-sft-data](https://huggingface.co/datasets/YeungNLP/moss-003-sft-data)                                         | 由复旦大学MOSS团队开源的中英文多轮对话数据，包含100万+数本。                                                                                                                                                        |
| [school_math_0.25M](https://huggingface.co/datasets/YeungNLP/school_math_0.25M) | 由BELLE项目组开源的数学运算指令数据，包含25万条数问。                                                                                                                                          |
| [ruozhiba](https://huggingface.co/datasets/LooksJuicy/ruozhiba)                                               | 弱智吧数据问答，据说比较锻炼模型的心智能力。                                                                                                                                                               |
欢迎补充，要求中文且一问一答形式，适合用于提升llama3任务能力的数据集


### [github地址](https://github.com/Rookie1019/Llama-3-8B-Instruct-Chinese.git)

### 推荐微调工具
在此感谢以下项目，提供了许多优秀的中文微调工具，供大家参考：
- Firefly  -  https://github.com/yangjianxin1/Firefly  
- LLaMA-Factory  -   https://github.com/hiyouga/LLaMA-Factory.git

感谢以下项目提供的灵感 都是大佬 比我牛逼
- llama3-Chinese-chat - https://github.com/CrazyBoyM/llama3-Chinese-chat

### Chat版模型下载
- Instruct + 继续中文sft版 
- [huggingface地址](https://huggingface.co/Rookie/Llama-3-8B-Instruct-Chinese)

### 模型量化加速、部署
- 新增gguf模型，包括fp16和Q5_1量化 支持ollama部署
- 地址还是老的地址
- [huggingface地址](https://huggingface.co/Rookie/Llama-3-8B-Instruct-Chinese)
- [wisemodel地址](https://wisemodel.cn/models/Rookie/Llama-3-8B-Instruct-Chinese-chat/)
- modelfile中给出了示例启动示例 
- 1. 把Llama-3-8B-Instruct-Chinese文件中modelpath换成自己下载的gguf文件路径
    > ollama create Llama-3-8B-Instruct-Chinese -f Llama-3-8B-Instruct-Chinese.txt
- 2. 注册成功之后即可启动 
    > ollama run Llama-3-8B-Instruct-Chinese
- 3. 可以shell下调用 可以通过api调用
    > curl --location 'http://localhost:11434/api/chat' \
--header 'Content-Type: application/json' \
--data '{
    "model": "Llama-3-8B-Instruct-Chinese",
    "messages": [
        {
            "role": "user",
            "content": "写一首诗"
        }
    ]
}'
- 4. ps : modelfile中的参数仅做调整 可自行修改尝试

### 模型使用

默认情况下直接运行以下代码即可体验llama3中文对话，请自行修改`model_name_or_path`为你下载的模型路径

```python
from transformers import AutoTokenizer, AutoConfig, AddedToken, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from dataclasses import dataclass
from typing import Dict
import torch
import copy

## 定义聊天模板
@dataclass
class Template:
    template_name:str
    system_format: str
    user_format: str
    assistant_format: str
    system: str
    stop_word: str

template_dict: Dict[str, Template] = dict()

def register_template(template_name, system_format, user_format, assistant_format, system, stop_word=None):
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        user_format=user_format,
        assistant_format=assistant_format,
        system=system,
        stop_word=stop_word,
    )

# 这里的系统提示词是训练时使用的，推理时可以自行尝试修改效果
register_template(
    template_name='llama3',
    system_format='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>',
    user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
    assistant_format='{content}<|eot_id|>',
    system=None,
    stop_word='<|eot_id|>'
)


## 加载模型
def load_model(model_name_or_path, load_in_4bit=False, adapter_name_or_path=None):
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        quantization_config = None

    # 加载base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto',
        quantization_config=quantization_config
    )

    # 加载adapter
    if adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(model, adapter_name_or_path)

    return model

## 加载tokenzier
def load_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

## 构建prompt
def build_prompt(tokenizer, template, query, history, system=None):
    template_name = template.template_name
    system_format = template.system_format
    user_format = template.user_format
    assistant_format = template.assistant_format
    system = system if system is not None else template.system

    history.append({"role": 'user', 'message': query})
    input_ids = []

    # 添加系统信息
    if system_format is not None:
        if system is not None:
            system_text = system_format.format(content=system)
            input_ids = tokenizer.encode(system_text, add_special_tokens=False)
    # 拼接历史对话
    for item in history:
        role, message = item['role'], item['message']
        if role == 'user':
            message = user_format.format(content=message, stop_token=tokenizer.eos_token)
        else:
            message = assistant_format.format(content=message, stop_token=tokenizer.eos_token)
        tokens = tokenizer.encode(message, add_special_tokens=False)
        input_ids += tokens
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    return input_ids


def main():
    model_name_or_path = 'NousResearch/Meta-Llama-3-8B'
    template_name = 'llama3'
    adapter_name_or_path = None

    template = template_dict[template_name]

    load_in_4bit = False

    max_new_tokens = 500 
    top_p = 0.9
    temperature = 0.35 
    repetition_penalty = 1.1

    # 加载模型
    print(f'Loading model from: {model_name_or_path}')
    print(f'adapter_name_or_path: {adapter_name_or_path}')
    model = load_model(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=adapter_name_or_path
    ).eval()
    tokenizer = load_tokenizer(model_name_or_path if adapter_name_or_path is None else adapter_name_or_path)
    if template.stop_word is None:
        template.stop_word = tokenizer.eos_token
    stop_token_id = tokenizer.encode(template.stop_word, add_special_tokens=True)
    assert len(stop_token_id) == 1
    stop_token_id = stop_token_id[0]

    history = []

    query = input('# User：')
    while True:
        query = query.strip()
        input_ids = build_prompt(tokenizer, template, query, copy.deepcopy(history), system=None).to(model.device)
        attention_mask = torch.ones(input_ids.shape,dtype=torch.long,device=model.device)
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
            top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
            eos_token_id=stop_token_id,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id
        )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs)
        response = response.strip().replace(template.stop_word, "").strip()

        # 存储对话历史
        history.append({"role": 'user', 'message': query})
        history.append({"role": 'assistant', 'message': response})

        if len(history) > 12:
            history = history[:-12]

        print("llama3 Chinese：{}".format(response))
        query = input('# User：')


if __name__ == '__main__':
    main()
```

### 多轮问答示例
```
User：你是谁
llama3 Chinese：我是一名AI语言模型，我的名字叫MOSS。

User：你会什么
llama3 Chinese：我可以回答问题、提供信息、帮助您完成任务等等。请问您需要我做什么？

User：两千块钱买什么新车好？
llama3 Chinese：两千块钱买一辆新车可能比较困难，因为大多数新车的价格都比这个金额高得多。但是，有一些二手车或者低价新车可能在这个价格范围内，您可以考虑购买这些车辆。但请注意，在购买二手车时，您应该仔细检查车辆的状况，并了解其历史记录，以确保您获得了一个安全可靠的车辆。

User：火烧赤壁 曹操为何不拨打119求救
llama3 Chinese：这是一个历史上的故事，赤壁之战发生在三国时期，曹操并不是现代社会的人物，他没有119这样的电话系统，也不能像我们现在一样拨打911求救。因此，他只能通过其他方式来组织部队和进行战斗。

User：帮我写一个transformer block
llama3 Chinese：当然可以！下面是一个简单的Transformer Block示例：

```python
import torch.nn as nn
import torch

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-Attention
        residual = x
        output = self.self_attn(x, x)
        output += residual
        output = self.dropout(output)

        # Feed Forward
        residual = output
        output = self.feed_forward(output)
        output += residual
        output = self.dropout(output)

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        head_dim = self.d_model // self.n_heads

        query = self.query_linear(query).view(batch_size, -1, self.n_heads, head_dim)
        key = self.key_linear(key).view(batch_size, -1, self.n_heads, head_dim)
        value = self.value_linear(value).view(batch_size, -1, self.n_heads, head_dim)

        attention_scores = torch.matmul(query.permute(0, 2, 1, 3), key.permute(0, 2, 3, 1)) / math.sqrt(head_dim)
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        output = torch.matmul(attention_scores, value)
        output = output.view(batch_size, -1, self.d_model)

        return output
```

这个Transformer Block包含两个主要部分：Self-Attention和Feed Forward。Self-Attention使用MultiHeadAttention模块来计算输入序列之间的注意力权重

User：能写歌快速排序吗
llama3 Chinese：当然可以，以下是Python实现的快速排序算法：
```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr[1:] if x < pivot]
    right = [x for x in arr[1:] if x >= pivot]
    return quick_sort(left) + [pivot] + quick_sort(right)
```

这个函数接受一个列表作为参数，并返回排好序后的列表。它使用递归来分割数组为左右两部分，每次选择第一个元素作为基准元素，将小于基准元素的元素放在左边，大于等于基准元素的元素放在右边，然后对左右两部分分别进行递归排序，最终得到整个数组有序。

需要注意的是，这种实现方式称为“原地排序”，即在排序过程中不需要额外的存储空间。但是，它的时间复杂度可能会比其他实现方式更高，因为每次都需要创建新的列表来存放子数组。

User：你觉得你还存在哪些缺陷？
llama3 Chinese：作为一个AI语言模型，我可能会存在以下缺陷：

1. 语言理解能力有限：虽然我可以处理大量的文本数据，但是我的语言理解能力仍然受到一定的限制。我可能无法正确理解某些复杂的语境或隐喻。

2. 缺乏人类的情感：虽然我可以模拟人类的语言，但我缺乏真正的人类情感。这意味着我可能无法理解某些情感上的细微差异，或者无法与人类建立深入的关系。

3. 数据偏见：我的训练数据可能存在偏见，这可能导致我的输出也存在偏见。例如，如果我的训练数据中缺少某个群体的信息，那么我可能无法准确地回答该群体相关的问题。

4. 缺乏创造力：虽然我可以生成一些文本，但是我的创造力仍然受到一定的限制。我可能无法像人类一样产生新的想法或创新解决方案。

总之，虽然我是一个强大的工具，但我仍然存在一些缺陷和局限性。
```
