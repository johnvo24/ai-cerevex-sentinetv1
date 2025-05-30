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
        trainer.train(from_gdrive=True)

    def predict(sentence):
        predictor = Predictor()
        readed_sentence, pred_label, prediction_time = predictor.predict(sentence=sentence)
        print(f"Readed sentence: {readed_sentence}")
        print(f"Prediction label: {pred_label}")
        print(f"Prediction time: {prediction_time}")
        
        _, cls_prediction_time = predictor.predict_full_text(sentence=sentence)
        print(f"Prediction time with CLS full text: {cls_prediction_time}")

    def evaluate():
        config = configs.load_config()
        dataset = load_dataset(config['data']['ag_news'])['test'].to_list()[:1000]

        predictor = Predictor()
        total_pred_time_rl = 0
        correct_pred_rl = 0
        total_pred_time_sft = 0
        correct_pred_sft = 0
        for row in tqdm(dataset, desc="Evaluating"):
            readed_sentence_1, pred_label_1, prediction_time_1 = predictor.predict(sentence=row['text'])
            total_pred_time_rl += prediction_time_1
            if (pred_label_1 == row['label']): correct_pred_rl += 1
            readed_sentence_2, pred_label_2, prediction_time_2 = predictor.predict_full_text(sentence=row['text'])
            total_pred_time_sft += prediction_time_2
            if (pred_label_2 == row['label']): count_accuracy_2 += 1

        num_samples = len(dataset)

        print("\n=== Evaluation Results ===")
        print(f"SFT + RL:")
        print(f"- Accuracy:           {correct_pred_rl / num_samples:.4f}")
        print(f"- Avg Prediction Time: {total_pred_time_rl / num_samples:.4f} seconds")

        print(f"\nSFT only:")
        print(f"- Accuracy:           {correct_pred_sft / num_samples:.4f}")
        print(f"- Avg Prediction Time: {total_pred_time_sft / num_samples:.4f} seconds")



        


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 src/main.py [train|predict|evaluate]")
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
    elif mode == "evaluate":
        CerevexSentinet.evaluate()
    else:
        print("Invalid mode. Use 'train' or 'predict'.")