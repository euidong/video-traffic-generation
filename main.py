from train.gan import main as gan_trainer
from train.predictor import main as predictor_trainer

def main():
    # gan_trainer()
    predictor_trainer()
    
if __name__ == "__main__":
    main()
