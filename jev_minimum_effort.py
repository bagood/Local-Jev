import numpy as np
from llama_cpp import Llama

model = Llama.from_pretrained(
    repo_id="Qwen/Qwen3-0.6B-GGUF",
    filename="Qwen3-0.6B-Q8_0.gguf",
    n_ctx=512,
    logits_all=True,
    verbose=False,
)

labels = ["A", "B", "C"]
choices = ["Legitimate", "Spam", "Phishing"]
email = "Payroll asks for your password on a non-company sign-in page."
options = "\n".join(
    f"{label}. {choice}" for label, choice in zip(labels, choices, strict=True)
)
prompt = f"""<|im_start|>system
Choose one option.<|im_end|>
<|im_start|>user
Email: {email}\n\n{options}<|im_end|>
<|im_start|>assistant
<think>\n\n</think>\n\n"""

model.eval(model.tokenize(prompt.encode(), add_bos=False, special=True))

logits = model.scores[model.n_tokens - 1]

token_ids = [model.tokenize(label.encode(), add_bos=False)[0] for label in labels]

choice_logits = np.asarray([logits[token_id] for token_id in token_ids])

probabilities = np.exp(choice_logits - np.logaddexp.reduce(choice_logits))

for choice, probability in zip(choices, probabilities, strict=True):
    print(f"{choice:10} {probability:.1%}")
