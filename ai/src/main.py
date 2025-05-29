import sys
from second_phase.rl_trainer import RLTrainer
from second_phase.predictor import Predictor

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



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 src/main.py [train|predict]")
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
    else:
        print("Invalid mode. Use 'train' or 'predict'.")