import sys

from tqdm import tqdm
from second_phase.rl_trainer import RLTrainer
from second_phase.predictor import Predictor
from coparator import Comparator
from datasets import load_dataset
import second_phase.config as configs

class CerevexSentinet():

    def train():
        trainer = RLTrainer()
        trainer.train(from_gdrive=False)

    def predict(sentence):
        predictor = Predictor()
        readed_sentence, pred_label, prediction_time = predictor.predict(sentence=sentence)
        print(f"Readed sentence: {readed_sentence}")
        print(f"Prediction label: {pred_label}")
        print(f"Prediction time: {prediction_time}")
        
        _, cls_prediction_time = predictor.predict_full_text(sentence=sentence)
        print(f"Prediction time with CLS full text: {cls_prediction_time}")

    def evaluate_rl():
        config = configs.load_config()
        dataset = load_dataset(config['data']['ag_news'])['test'].to_list()[:1000]

        predictor = Predictor()
        total_pred_time_rl = 0
        correct_pred_rl = 0
        readed_token_rl=0
        total_pred_time_sft = 0
        correct_pred_sft = 0
        readed_token_sft=0
        for row in tqdm(dataset, desc="Evaluating"):
            readed_tokens_1, pred_label_1, prediction_time_1 = predictor.predict(sentence=row['text'], k=16)
            total_pred_time_rl += prediction_time_1
            readed_token_rl += len(readed_tokens_1)
            if (pred_label_1 == row['label']): correct_pred_rl += 1

            readed_tokens_2, pred_label_2, prediction_time_2 = predictor.predict_full_text(sentence=row['text'])
            total_pred_time_sft += prediction_time_2
            readed_token_sft += len(readed_tokens_2)
            if (pred_label_2 == row['label']): correct_pred_sft += 1

        num_samples = len(dataset)

        avg_accu_rl = correct_pred_rl / num_samples
        avg_readed_token_rl = readed_token_rl / num_samples
        avg_pred_time_rl = total_pred_time_rl / num_samples
        avg_accu_sft = correct_pred_sft / num_samples
        avg_readed_token_sft = readed_token_sft / num_samples
        avg_pred_time_sft = total_pred_time_sft / num_samples

        print("\n=== Evaluation Results ===")
        print(f"SFT + RL:")
        print(f"- Accuracy:            {avg_accu_rl:.4f}")
        print(f"- Avg Readed Tokens:   {avg_readed_token_rl:.4f}")
        print(f"- Avg Prediction Time: {avg_pred_time_rl:.4f} seconds")

        print(f"\nSFT only:")
        print(f"- Accuracy:            {avg_accu_sft:.4f}")
        print(f"- Avg Readed Tokens:   {avg_readed_token_sft:.4f}")
        print(f"- Avg Prediction Time: {avg_pred_time_sft:.4f} seconds")

        Comparator.plot_avg_metrics(
            avg_accu_rl,
            avg_pred_time_rl,
            avg_readed_token_rl,
            avg_accu_sft,
            avg_pred_time_sft,
            avg_readed_token_sft
        )

    def evaluate_sft():
        config = configs.load_config()
        dataset = load_dataset(config['data']['ag_news'])['test'].to_list()

        predictor = Predictor()
        true_labels = []
        pred_labels = []

        for row in tqdm(dataset, desc="Evaluating SFT model"):
            true_labels.append(row['label'])
            _, pred_label, _ = predictor.predict_full_text(sentrl_avg_metrics_comparisonence=row['text'])
            pred_labels.append(pred_label)

        target_names = ['World', 'Sports', 'Business', 'Sci/Tech']  # ví dụ 4 class của AG News
        print("\n=== Classification Report for SFT Model ===")
        Comparator.save_classification_report_as_image(true_labels, pred_labels, target_names=target_names)  


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 src/main.py [train|predict|evaluate_rl|evaluate_sft]")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "train":
        CerevexSentinet.train()
    elif mode == "predict":
        sentence = (
            sys.argv[2]
            if len(sys.argv) > 2
            else "Some aspects of the partnership, which the carriers announced Thursday, will begin as early as the fall, though the airlines didn’t provide exact timing. They also did not provide financial details of the deal."
        )
        CerevexSentinet.predict(sentence=sentence)
    elif mode == "evaluate_rl":
        CerevexSentinet.evaluate_rl()
    elif mode == "evaluate_sft":
        CerevexSentinet.evaluate_sft()
    else:
        print("Invalid mode. Use 'train' or 'predict'.")