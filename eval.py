from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 从训练脚本复用的基础组件 ──────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 50


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights



class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, teacher_forcing_ration=0.5):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None and random.random() < teacher_forcing_ration:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights


# ── 数据准备（仅构建词表，不需要 DataLoader）────────────────────────────────────

def readLangs(lang1, lang2):
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8') \
        .read().strip().split('\n')
    pairs = [[s for s in l.split('\t')] for l in lines]
    return pairs

def buildVocab(lang1, lang2):
    """读取语料，仅构建词表，返回 (input_lang, output_lang)。"""
    pairs = readLangs(lang1, lang2)
    pairs = [p for p in pairs if len(p[0].split()) < MAX_LENGTH and len(p[1].split()) < MAX_LENGTH]

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print(f"Vocab sizes — {input_lang.name}: {input_lang.n_words}, "
          f"{output_lang.name}: {output_lang.n_words}")
    return input_lang, output_lang


# ── Checkpoint 加载 ──────────────────────────────────────────────────────────

def load_checkpoint_for_eval(filepath, encoder, decoder):
    checkpoint = torch.load(filepath, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    epoch = checkpoint['epoch']
    loss  = checkpoint['loss']
    print(f"Loaded checkpoint '{filepath}'  (epoch={epoch}, loss={loss:.4f})")


# ── 推理 ─────────────────────────────────────────────────────────────────────

def tensorFromSentence(lang, sentence):
    try:
        indexes = [lang.word2index[w] for w in sentence.split(' ')]
    except KeyError as e:
        raise ValueError(f"OOV token: {e}") from e
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def evaluate(encoder, decoder, sentence, input_lang, output_lang,
             repetition_penalty=1.3):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, attn_weights = decoder(encoder_outputs, encoder_hidden)

        logits = decoder_outputs.squeeze(0)  # [MAX_LENGTH, vocab_size]
        decoded_words = []
        generated_ids = set()

        for step in range(logits.size(0)):
            step_logits = logits[step].clone()
            for prev_id in generated_ids:
                step_logits[prev_id] /= repetition_penalty
            token_id = step_logits.argmax().item()
            if token_id == EOS_token:
                break
            decoded_words.append(output_lang.index2word[token_id])
            generated_ids.add(token_id)

    return decoded_words, attn_weights


# ── 交互式 REPL ───────────────────────────────────────────────────────────────

def interactive_eval(encoder, decoder, input_lang, output_lang):
    print("\n=== 交互式推理模式 ===")
    print("输入句子（用空格分词），输入 'q' 退出\n")

    while True:
        try:
            sentence = input("输入 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if sentence.lower() in ('q', 'quit', 'exit'):
            print("退出。")
            break
        if not sentence:
            continue

        try:
            output_words, attn = evaluate(encoder, decoder, sentence, input_lang, output_lang)
            print("输出 <", ' '.join(output_words))
        except ValueError as e:
            print(f"  [错误] {e}")
        print()


# ── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Seq2Seq 推理脚本")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='checkpoint 文件路径，例如 checkpoint_final.pth')
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--lang1', type=str, default='pinyin')
    parser.add_argument('--lang2', type=str, default='zh')
    parser.add_argument('--sentence', type=str, default=None,
                        help='单句推理（不填则进入交互模式）')
    args = parser.parse_args()

    # 1. 构建词表
    input_lang, output_lang = buildVocab(args.lang1, args.lang2)

    # 2. 初始化模型
    encoder = EncoderRNN(input_lang.n_words, args.hidden_size).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, output_lang.n_words).to(device)

    # 3. 加载 checkpoint
    load_checkpoint_for_eval(args.checkpoint, encoder, decoder)

    # 4. 推理
    if args.sentence:
        # 单句模式
        try:
            words, _ = evaluate(encoder, decoder, args.sentence, input_lang, output_lang)
            print("输出:", ' '.join(words))
        except ValueError as e:
            print(f"[错误] {e}")
    else:
        # 交互模式
        interactive_eval(encoder, decoder, input_lang, output_lang)
