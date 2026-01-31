import torch
import torch.nn.functional as F
from src.model import setup_model_and_tokenizer
from src.rewards import reward_function

def train_step(model, tokenizer, prompt, group_size=8):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
    prompt_len = inputs["input_ids"].shape[1]

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.9,
            num_return_sequences=group_size,
            pad_token_id=tokenizer.eos_token_id
        )
    model.train()

    # rewards
    rewards = []
    completions = []
    for i in range(group_size):
        content = tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True)
        completions.append(content)
        rewards.append(reward_function(content))
    
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device="cuda")
    
    # Standardize rewards within the group
    mean_r = reward_tensor.mean()
    std_r = reward_tensor.std() + 1e-8
    advantages = (reward_tensor - mean_r) / std_r

    # POLICY GRADIENT LOSS
    # Re-run the model on the generated sequences to get log-probs for backprop
    logits = model(outputs).logits  # Shape: (G, SeqLen, Vocab)
    
    # Shift to align logits with the generated tokens
    log_probs = F.log_softmax(logits[:, prompt_len-1:-1, :], dim=-1)
    target_ids = outputs[:, prompt_len:]
    
    # Get log-probs of the specific tokens the model actually picked
    per_token_log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
    
    # GRPO Objective: Mean across group (Advantage * log_prob)
    # We sum across tokens and mean across the group
    loss = -(per_token_log_probs.sum(dim=1) * advantages).mean()

    loss.backward()
    return loss.item(), mean_r.item()

if __name__ == "__main__":
    m, t = setup_model_and_tokenizer()
    optimizer = torch.optim.AdamW(m.parameters(), lr=5e-6)
    
    # demo set
    prompts = [
        "Write a python function to add two numbers.",
        "Write a python script that prints 'hello'.",
        "Write a python function to calculate the area of a square."
    ]

    print("Starting Training...")
    for epoch in range(5):
        for p in prompts:
            optimizer.zero_grad()
            loss, avg_reward = train_step(m, t, p)
            optimizer.step()
            print(f"Loss: {loss:.4f} | Avg Reward: {avg_reward:.2f}")