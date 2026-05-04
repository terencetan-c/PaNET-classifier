# --- Performing multi-run experiments with HazyResearch's Hyperbolic Graph Convolutional Networks (HGCN) for hierarchical multi-label classification of scientific text ---

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoModel, Trainer, TrainingArguments, AutoTokenizer, DataCollatorWithPadding, set_seed
import numpy as np
from sklearn.metrics import f1_score
from pathlib import Path
import torch
import torch.nn as nn
import pickle
import geoopt
import scipy.sparse as sp



# A list of seeds for multi-run experiments to assess variability and robustness of results
seeds = [42,1,2,3,4]

# Enable TF32 for A100 (optional but recommended)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- Setup ---
MODEL_NAME = 'allenai/scibert_scivocab_uncased'  # SciBERT uncased model 
TRAIN_PATH = './data/train_set.parquet' # Path to your training data
EVAL_PATH = './data/eval_set.parquet' # Path to your evaluation data
TEST_PATH = './data/test_set.parquet' # Path to your test data
ANCESTER_INDICES_PATH = './data/ancestor_indices.pkl' # Path to pre-computed ancestor indices for hierarchical metrics
ADJACENCY_MATRIX_PATH = './data/hierarchy_adj.npz' # Path to pre-computed adjacency matrix for hierarchical metrics
OUTPUT_DIR = "./hgcn_results" # Directory to save model checkpoints and results

# 1. Load Data
def load_and_format(path):
    df = pd.read_parquet(path, engine='fastparquet')

    # Convert labels into a single list/array for the Dataset object
    label_columns = [col for col in df.columns if col not in ['DOI', 'Title', 'Abstract']]
    # Create a single column containing the list/vector
    df['labels'] = df[label_columns].astype(float).to_numpy().tolist()

    return Dataset.from_pandas(df[['DOI','Title', 'Abstract', 'labels']]), len(label_columns)

# Load and format the datasets
train_dataset, num_labels_train = load_and_format(TRAIN_PATH)
eval_dataset, num_labels_eval = load_and_format(EVAL_PATH)
test_dataset, num_labels_test = load_and_format(TEST_PATH)

assert num_labels_train == num_labels_eval == num_labels_test, "Number of labels must be consistent across datasets"


raw_datasets = DatasetDict({'train': train_dataset, 'test': test_dataset, 'eval': eval_dataset})


# Loading CSR adjacency matrix
adj_csr = sp.load_npz(ADJACENCY_MATRIX_PATH)


# 2. Tokenization Function
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(data):
    # Note: SciBERT's tokenizer can handle two separate text inputs (title and abstract) and will concatenate them with a special token in between.
    return tokenizer(data['Title'], data['Abstract'], truncation=True, max_length=512)

tokenized_ds = raw_datasets.map(
    preprocess_function, 
    batched=True, 
    remove_columns=['Title', 'Abstract', 'DOI']  # Keep only labels and tokenized data
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# 3. Model Definition with Mean Pooling and hyperbolic distance layer
class HyperbolicDistanceLayer(nn.Module):
    def __init__(self, input_dim, num_classes, curvature=1.0):
        super().__init__()
        self.manifold = geoopt.PoincareBall(c=curvature)
        self.centroids = geoopt.ManifoldParameter(
            torch.randn(num_classes, input_dim) * 0.1,  # Small random initialization
            manifold=self.manifold,
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))  # Learnable temperature for scaling distances

    def forward(self, x):        
        # Clip norm before projecting to prevent NaNs from large BERT embeddings
        x = x / (x.norm(dim=-1, keepdim=True).clamp(min=1.0) + 1e-6) * 0.9
        x_hyp = self.manifold.expmap0(x)
        distances = self.manifold.dist(x_hyp.unsqueeze(1), self.centroids.unsqueeze(0))
        return -distances / self.temperature.abs().clamp(min=0.01)


class SciBERTWithMeanPooling(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)

        self.classifier = HyperbolicDistanceLayer(input_dim=self.bert.config.hidden_size, num_classes=num_labels)   # Swapped linear classifier for hyperbolic distance layer
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [Batch, Seq_Len, Hidden_Dim]

        # Expand mask to match hidden state dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        
        # Perform Mean Pooling: sum vectors and divide by number of non-padded tokens
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        # Pass through classifier
        mean_embeddings = self.dropout(mean_embeddings)
        logits = self.classifier(mean_embeddings)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels.float())

        # Trainer expects a tuple or specialized object
        return (loss, logits) if loss is not None else logits


# Custom Trainer to use RiemannianAdam optimizer for the hyperbolic distance layer,
# otherwise gradient updates will move centroids off the manifold
class RiemannianTrainer(Trainer):
    def create_optimizer(self):
        self.optimizer = geoopt.optim.RiemannianAdam(
            self.model.parameters(),
            lr=self.args.learning_rate
        )


# 4. Metrics Functions (for training, evaluation, and testing)

# Load the ancestor map for computing hierarchical metrics
with open(ANCESTER_INDICES_PATH, 'rb') as f:
    ancestor_indices = pickle.load(f)

# hF1 calculation
def augment_with_ancestors(binary_matrix):
    # Given an (N, L) binary matrix, return augmented version including ancestors.
    augmented = binary_matrix.copy()
    for i, ancestors in enumerate(ancestor_indices):
        for anc_idx in ancestors:
            # If label i is active, activate its ancestors too
            augmented[:, anc_idx] |= binary_matrix[:, i]
    return augmented

def hierarchical_f1_micro(labels, predictions):
    labels_aug = augment_with_ancestors(labels.astype(bool))
    preds_aug = augment_with_ancestors(predictions.astype(bool))

    # Per-sample intersections
    tp = (labels_aug & preds_aug).sum(axis=1)
    pred_count = preds_aug.sum(axis=1)
    true_count = labels_aug.sum(axis=1)

    # Micro-averaged precision and recall
    precision = np.sum(tp) / np.sum(pred_count) if np.sum(pred_count) > 0 else 0.0
    recall = np.sum(tp) / np.sum(true_count) if np.sum(true_count) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# Bootstrap sampling
def bootstrap_metrics(labels, predictions, n_bootstrap=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n_samples = labels.shape[0]
    scores = {'f1_micro': [], 'f1_macro': [], 'hf1': []}

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_samples, size=n_samples)
        l, p = labels[indices], predictions[indices]
        scores['f1_micro'].append(f1_score(l, p, average='micro'))
        scores['f1_macro'].append(f1_score(l, p, average='macro'))
        scores['hf1'].append(hierarchical_f1_micro(l, p))

    alpha = (1 - ci) / 2
    results = {}
    for metric, vals in scores.items():
        vals = np.array(vals)
        results[f'{metric}_ci_lower'] = np.percentile(vals, 100 * alpha)
        results[f'{metric}_ci_upper'] = np.percentile(vals, 100 * (1 - alpha))
    return results


# This function will be passed to the Trainer to compute all metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Apply sigmoid to get probabilities for each label, then threshold at 0.5 to get binary predictions
    probs = 1 / (1 + np.exp(-logits))
    predictions = (probs >= 0.5).astype(int)

    ci = bootstrap_metrics(labels, predictions)

    return {'f1_micro': f1_score(labels, predictions, average='micro'), 
            'f1_macro': f1_score(labels, predictions, average='macro'),
            'hF1': hierarchical_f1_micro(labels, predictions),
            **ci,
            }

# --- The main training function ---
def run_experiment(seed):
    print(f"\n=== Running seed {seed} ===")
    set_seed(seed) # Set the seed for reproducibility
    
    # Fresh model instance for each seed
    model = SciBERTWithMeanPooling(MODEL_NAME, num_labels=num_labels_train)



    # --- PHASE 1: WARM-UP (FROZEN BACKBONE) ---
    print(f"--- Starting Phase 1: Training Classification Head ---")

    # 5. Freeze all parameters in the BERT backbone
    for param in model.bert.parameters():
        param.requires_grad = False

    # 6. Training Arguments Phase 1
    training_args_p1 = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/seed{seed}/phase1",
        eval_strategy='epoch',
        save_strategy='epoch',
        group_by_length=True,
        learning_rate=2e-4, # Higher LR for the classification head
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='hF1',
        fp16=False,
        bf16=True, # A100's native mixed precision format; Hopper Cluster Testbed has A100 GPUs
        logging_steps=100,
        save_total_limit=2,
        dataloader_num_workers=4, # Use multiple CPU cores for data loading
        gradient_accumulation_steps=1, # Can increase if needed for larger effective batch
    )

    # 7. The Trainer for Phase 1
    trainer = RiemannianTrainer(
        model=model,
        args=training_args_p1,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['eval'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Start/resume Phase 1 training
    trainer.train(resume_from_checkpoint=None)
    

    # After Phase 1
    Path(f"{OUTPUT_DIR}/seed{seed}").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/seed{seed}/phase1_final")

    # --- PHASE 2: FINE-TUNING (UNFROZEN) ---
    print(f"--- Starting Phase 2: Full Model Fine-Tuning ---")

    # Unfreeze all parameters in the BERT backbone
    for param in model.parameters():
        param.requires_grad = True

    # Update training arguments for Phase 2
    training_args_p2 = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/seed{seed}/phase2",
        eval_strategy='epoch',
        save_strategy='epoch',
        group_by_length=True,
        learning_rate=2e-5, # Lower LR for fine-tuning
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='hF1',
        fp16=False,
        bf16=True, # A100's native mixed precision format; Hopper Cluster Testbed has A100 GPUs
        logging_steps=100,
        save_total_limit=2,
        dataloader_num_workers=4, # Use multiple CPU cores for data loading
        gradient_accumulation_steps=1, # Can increase if needed for larger effective batch
    )

    # Re-initialize the trainer with updated arguments and unfrozen model
    trainer = RiemannianTrainer(
        model=model,
        args=training_args_p2,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['eval'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Start/resume Phase 2 training
    trainer.train(resume_from_checkpoint=None)

    # Save the final results
    Path(f"{OUTPUT_DIR}/seed{seed}").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/seed{seed}/final_panet_model")


    # --- EVALUATION ON TEST SET ---
    print("--- Evaluating on Test Set ---")

    test_results = trainer.predict(tokenized_ds['test'])
    metrics = test_results.metrics

    return metrics


# Run experiments for all seeds and collect results
all_results = {}
for seed in seeds:
    all_results[seed] = run_experiment(seed)

# Save raw per-seed results
results_df = pd.DataFrame(all_results).T
results_df.to_csv(f"{OUTPUT_DIR}/per_seed_results.csv")
print("\nPer-seed results:")
print(results_df)

# Aggregate mean ± std across seeds for the three main metrics
print("\n=== Final Results (mean ± std across seeds) ===")
for metric in ['test_f1_micro', 'test_f1_macro', 'test_hF1']:
    vals = results_df[metric].values.astype(float)
    print(f"{metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
