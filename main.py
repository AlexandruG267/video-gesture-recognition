import argparse
import baseline




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true') # This argument determines if the code will train or test
    parser.add_argument('--loaded', action='store_true') # This argument is used to load a model from a checkpoint and keep training
    parser.add_argument('--checkpoint', default='model.ckpt')  # This argument specifies the checkpoint. Used when either test or loaded are true
    parser.add_argument('--model', default='baseline') # This argument determines the model used

    args = parser.parse_args()
    print(args)
    baseline.main(args) # Yes. I know this is lazy. It would be better if the main was the same for the baseline and transformer models

"""
TODO LIST:
- Make good baseline model
- Make transformer model
- Find happiness in life
"""