import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, BartModel


def compute_parameters(self):
    """统计模型参数量"""
    trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print(f"{self.__class__.__name__} 参数量: {trainable:,}")


def load_params(self, model_params_path):
    """加载模型参数"""
    try:
        self.load_state_dict(torch.load(model_params_path, map_location="cpu"))
        print(f"成功加载模型参数: {model_params_path}")
    except FileNotFoundError:
        print("模型参数文件不存在，使用默认参数")
    except RuntimeError as e:
        print(f"加载参数出错: {e}")


class SentimentModel(nn.Module):
    """情感分类模型（基于 BART Encoder）"""

    compute_parameters = compute_parameters
    load_params = load_params

    def __init__(self, model_name: str, label_list: list):
        super().__init__()
        self.label_list = label_list
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 只加载 BART encoder 部分
        self.encoder = BartModel.from_pretrained(model_name).encoder
        self.classifier = nn.Linear(
            self.encoder.config.hidden_size, len(label_list)
        )

        # 初始化分类器权重
        init_std = getattr(self.encoder.config, "init_std", 0.02)
        self.classifier.weight.data.normal_(mean=0.0, std=init_std)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # 取第一个 token 的隐藏状态作为句子表示
        cls_hidden = output.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_hidden)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

    @torch.inference_mode()
    def predict(self, text, device=torch.device("cpu"), batch_size=8):
        """批量预测情感"""
        self.eval()
        self.to(device)

        input_texts = text if isinstance(text, list) else [text]
        res = []
        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            outputs = self.forward(inputs["input_ids"], inputs["attention_mask"])
            logits = outputs["logits"]
            batch_res = [
                self.label_list[int(cat.item())]
                for cat in torch.argmax(logits, dim=1)
            ]
            res.extend(batch_res)

        return res if isinstance(text, list) else res[0]


class Seq2SeqModel(nn.Module):
    """序列到序列模型（支持摘要和问答，基于完整 BART）"""

    compute_parameters = compute_parameters
    load_params = load_params

    def __init__(self, model_name: str):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BartModel.from_pretrained(model_name)

        # 语言模型头
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size)
        # 共享嵌入层权重
        self.lm_head.weight = self.model.shared.weight

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask=None, labels=None):
        decoder_input_ids = None
        decoder_attention_mask = None

        if labels is not None:
            decoder_input_ids = labels.new_zeros(labels.shape)
            decoder_input_ids[:, 1:] = labels[:, :-1].clone()
            decoder_input_ids[:, 0] = self.config.decoder_start_token_id
            decoder_input_ids.masked_fill_(
                decoder_input_ids == -100, self.config.pad_token_id
            )
            decoder_attention_mask = (
                decoder_input_ids != self.config.pad_token_id
            ).long()

        # 编码器
        encoder_outputs = self.model.encoder(input_ids, attention_mask)
        # 解码器
        outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            use_cache=False,
        )

        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.loss_fn(
                logits.view(-1, self.config.vocab_size), labels.view(-1)
            )
        return {"loss": loss, "logits": logits}

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=128,
        num_beams=4,
    ):
        """束搜索生成"""
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        vocab_size = self.config.vocab_size

        # 编码器前向传播
        encoder_outputs = self.model.encoder(input_ids, attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state.repeat_interleave(
            num_beams, dim=0
        )
        encoder_attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)

        # 初始化解码输入
        decoder_input_ids = torch.full(
            (batch_size * num_beams, 1),
            self.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )

        beam_offset = torch.arange(batch_size, device=device) * num_beams
        beam_scores = torch.full((batch_size * num_beams,), -1e9, device=device)
        beam_scores[beam_offset] = 0
        done = torch.zeros(batch_size * num_beams, dtype=torch.bool, device=device)

        for step in range(max_length):
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=False,
            )
            logits = self.lm_head(decoder_outputs.last_hidden_state[:, -1, :])
            log_probs = nn.functional.log_softmax(logits, dim=-1)

            log_probs[done] = -float("inf")
            log_probs[done, self.config.eos_token_id] = 0

            log_probs += beam_scores.view(-1).unsqueeze(1)
            log_probs = log_probs.view(batch_size, num_beams * vocab_size)

            beam_scores, indices = torch.topk(log_probs, num_beams, dim=1)
            indices = indices.view(-1)

            beam_indices = indices // vocab_size
            beam_indices += beam_offset.repeat_interleave(num_beams, dim=0)
            token_ids = indices % vocab_size

            decoder_input_ids = torch.cat(
                [decoder_input_ids[beam_indices], token_ids.view(-1, 1)], dim=1
            )

            done = token_ids.eq(self.config.eos_token_id) | done[beam_indices]
            if done.all():
                break

        best_indices = beam_scores.argmax(dim=-1) + beam_offset
        return decoder_input_ids[best_indices]

    def predict(self, text, device=torch.device("cpu"), batch_size=8):
        """批量预测"""
        self.eval()
        self.to(device)

        input_texts = text if isinstance(text, list) else [text]
        res = []
        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            outputs = self.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            batch_res = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            batch_res = [r.replace(" ", "") for r in batch_res]
            res.extend(batch_res)

        return res if isinstance(text, list) else res[0]
