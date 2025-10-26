import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


from project import (
    make_model,
    subsequent_mask,
    load_data,
    build_vocab,
    tokenize_en,
    tokenize_kn,
    sequential_transform,
    tensor_transform,
    special_symbols,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX
)


def translate(model, src_sentence, src_vocab, tgt_vocab, device, max_len=100):
    """
    使用训练好的模型将源句子翻译成目标语言。

    Args:
        model (nn.Module): 训练好的 Transformer 模型。
        src_sentence (str): 需要翻译的英文句子。
        src_vocab (torchtext.vocab.Vocab): 源语言（英语）词汇表。
        tgt_vocab (torchtext.vocab.Vocab): 目标语言（卡纳达语）词汇表。
        device (torch.device): 运行模型的设备 (CPU 或 CUDA)。
        max_len (int): 生成翻译句子的最大长度。

    Returns:
        str: 翻译后的句子。
    """
    # 1. 将模型设置为评估模式 (这会关闭 dropout 等训练特有的层)
    model.eval()

    # 2. 准备输入数据：分词 -> 索引化 -> 添加 SOS/EOS -> 转换成张量
    #    .unsqueeze(0) 是为了增加一个 batch 维度，因为模型期望批处理输入
    src_tensor = sequential_transform(tokenize_en, src_vocab, tensor_transform)(src_sentence).unsqueeze(0).to(device)
    src_mask = (src_tensor != PAD_IDX).unsqueeze(-2).to(device)

    # 3. 使用 Encoder 对输入句子进行编码
    #    使用 torch.no_grad() 可以禁用梯度计算，节省内存并加速推理
    with torch.no_grad():
        memory = model.encode(src_tensor, src_mask)

    # 4. 使用 Decoder 进行贪心解码 (Greedy Decoding)
    #    - 初始化目标序列，只包含一个 <sos> (Start of Sentence) 符号
    ys = torch.ones(1, 1).fill_(SOS_IDX).type(torch.long).to(device)

    for i in range(max_len - 1):
        with torch.no_grad():
            # a. 创建目标序列的掩码
            tgt_mask = subsequent_mask(ys.size(1)).type(torch.bool).to(device)

            # b. 使用 Decoder 进行一次前向传播
            out = model.decode(memory, src_mask, ys, tgt_mask)

            # c. 使用 Generator 从 Decoder 输出中预测下一个词的概率
            #    out[:, -1] 表示我们只关心序列中最后一个词的输出
            prob = model.generator(out[:, -1])

            # d. 从概率分布中选择概率最高的词作为预测结果
            _, next_word = torch.max(prob, dim=1)
            next_word_item = next_word.item()

        # e. 将预测出的词拼接到已生成的目标序列中
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src_tensor.data).fill_(next_word_item)], dim=1)

        # f. 如果预测出的词是 <eos> (End of Sentence)，则说明翻译已完成，跳出循环
        if next_word_item == EOS_IDX:
            break

    # 5. 将预测出的索引序列转换回文本
    #    使用 tgt_vocab.get_itos() (index-to-string) 来获取索引对应的词元列表
    tgt_tokens = [tgt_vocab.get_itos()[i] for i in ys.squeeze()]

    # 6. 组合词元并返回最终结果 (移除 <sos> 和 <eos> 符号)
    return " ".join(tgt_tokens[1:-1])


if __name__ == '__main__':
    # --- 配置参数 ---
    TRAIN_FILE_PATH = "data/train"
    # 测试集路径
    TEST_FILE_PATH = "data/test"
    MODEL_PATH = "transformer_en_kn.pth"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- 步骤 1: 重新加载词汇表  ---
    print("Loading vocabularies from training data...")
    train_src_texts, train_tgt_texts = load_data(TRAIN_FILE_PATH)

    src_vocab = build_vocab(train_src_texts, tokenize_en, special_symbols)
    tgt_vocab = build_vocab(train_tgt_texts, tokenize_kn, special_symbols)

    SRC_VOCAB_SIZE = len(src_vocab.get_itos())
    TGT_VOCAB_SIZE = len(tgt_vocab.get_itos())
    print("Vocabularies loaded successfully.")

    # --- 步骤 2: 加载训练好的模型  ---
    print(f"\nLoading trained model from '{MODEL_PATH}'...")
    try:
        model = make_model(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, N=6, d_model=256, d_ff=512, h=8, dropout=0.1)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        print("Model loaded successfully.")

    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'.")
        exit()

    # --- 步骤 3: 使用测试集进行最终评估 ---
    print("\n--- Starting Final Evaluation on Test Set ---")

    # 从测试集加载数据
    try:
        test_src_texts, test_tgt_texts = load_data(TEST_FILE_PATH)
    except ValueError as e:
        print(e)
        print("\nPlease ensure your 'data/test' directory is correctly populated.")
        exit()

    # 从测试集中随机选择几句话进行展示
    test_indices = [5,13,30,33,48,55,65]

    for i in test_indices:
        if i < len(test_src_texts):
            src_sentence = test_src_texts[i]
            true_translation = test_tgt_texts[i]

            start_time = time.time()
            model_translation = translate(model, src_sentence, src_vocab, tgt_vocab, DEVICE)
            end_time = time.time()

            print("-" * 40)
            print(f"Source (en):           {src_sentence}")
            print(f"True Target (kn):      {true_translation}")
            print(f"Model Translated (kn): {model_translation}")
            print(f"(Translation took {end_time - start_time:.2f} seconds)")

    print("\n--- Final Evaluation Complete ---")