import sys
from second_phase.rl_trainer import RLTrainer
from second_phase.predictor import Predictor

class CerevexSentinet():

  def train():
    trainer = RLTrainer()
    trainer.train(from_gdrive=True)

  def predict(
    sentence="He stormed to win in the group round and quarter-finals before being stopped in the last-four round by world number one Dick Jaspers of the Netherlands."
  ):
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
            else "'He stormed to win in the group round and quarter-finals before being stopped in the last-four round by world number one Dick Jaspers of the Netherlands.'"
        )
        CerevexSentinet.predict(sentence=sentence*5)
    else:
        print("Invalid mode. Use 'train' or 'predict'.")