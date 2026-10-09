# 验证对话排版规则：确认分词器已加载 chat_template.jinja，并打印一条工具调用样本的渲染结果。
# 本脚本只使用 CPU 做分词与文本渲染，不创建模型、不读取训练数据。

# 使用标准库解析样本中字符串形式的工具定义与工具调用。
import json

# 读取模型结构配置，分词器加载时需要核对词表大小与特殊标记编号。
from model.configuration_edgemind import EdgeMindConfig

# 复用项目统一的分词器加载入口，保证验证用的分词器与训练时完全一致。
from model.tokenization_edgemind import load_tokenizer


# 构造一条与 MiniMind sft_t2t 格式相同的样本：system 带工具定义，assistant 发起工具调用，
# tool 返回结果，assistant 汇总回答；tools 与 tool_calls 字段按真实数据原样使用 JSON 字符串。
SAMPLE = {
    "conversations": [
        {
            "role": "system",
            "content": "你是 EdgeMind，一个小巧但有用的语言模型。",
            "tools": '[{"name": "get_weather", "description": "查询指定城市的天气", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "城市名称"}}, "required": ["city"]}}]',
        },
        {"role": "user", "content": "北京今天天气怎么样？"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": '[{"name": "get_weather", "arguments": {"city": "北京"}}]',
        },
        {"role": "tool", "content": '{"temp": 22, "text": "晴"}'},
        {"role": "assistant", "content": "北京今天晴，气温22度。"},
    ]
}


# 打印训练视角的完整渲染文本，以及推理视角的待续写提示，并抽查特殊标记的编号。
def main():
    # 使用与训练一致的配置和分词器，避免验证环境与真实环境不一致。
    config = EdgeMindConfig()
    tokenizer = load_tokenizer(config)

    # 模板文件缺失或版本过旧未被自动加载时立即指出，不悄悄用错误格式继续。
    if tokenizer.chat_template is None:
        raise RuntimeError("分词器未加载到 chat_template，请确认 model/chat_template.jinja 存在且 transformers 版本不低于 4.43。")

    # 逐条复制消息，并把字符串形式的 tools 与 tool_calls 解析回 JSON 对象。
    messages = []
    # 工具定义单独保存，最后作为 apply_chat_template 的 tools 参数传入。
    tools = None
    for message in SAMPLE["conversations"]:
        # 复制后修改，不改动原始样本，方便对照检查。
        message = dict(message)
        # system 消息里的工具定义是 JSON 字符串，解析成对象列表交给模板。
        if message.get("role") == "system" and message.get("tools"):
            tools = json.loads(message["tools"])
        # assistant 消息里的工具调用同样是 JSON 字符串，模板需要对象形式。
        if message.get("tool_calls") and isinstance(message["tool_calls"], str):
            message["tool_calls"] = json.loads(message["tool_calls"])
        # 收集处理后的消息，保持原有顺序。
        messages.append(message)

    # 训练视角：渲染完整对话，tokenize=False 表示只要文本不要编号。
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)
    print("=" * 20, "训练视角：完整对话渲染结果", "=" * 20)
    print(rendered)

    # 推理视角：只给到用户提问，add_generation_prompt=True 让模板补出 assistant 的开头。
    prompt = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True, tools=tools)
    print("=" * 20, "推理视角：喂给模型的提示", "=" * 20)
    print(prompt)

    # 抽查三个关键标记：它们必须各自被切成一个编号，而不是被拆成多个碎片。
    for text, expected in (("<|im_start|>", 1), ("<|im_end|>", 2), ("<tool_call>", 21)):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        # 编号不符合预期时说明词表与模板不匹配，直接报告实际值。
        if ids != [expected]:
            raise RuntimeError(f"标记 {text} 的编号为 {ids}，预期为 [{expected}]。")

    print("模板加载、渲染和特殊标记编号检查全部通过。")


# 直接运行时执行验证；被导入时不产生任何动作。
if __name__ == "__main__":
    main()
