from argparse import ArgumentParser

class TrainArgs:
    with_tracking: bool = False
    

def parse_args():
    
    parser = ArgumentParser()
    
    # Logging params
    parser.add_argument("--with_trackng", default=False, action='store_true', help="Wether to track with w&b or not.")
    parser.add_argument("--log_every", default=10, help="Logs to w&b (and terminal) every log_every epochs.")
    
    # Training params
    parser.add_argument("-b", "--batch_size", default=256, help="Select Batch Size.")
    parser.add_argument("-e", "--epochs", default=1000, help="Number of epochs to train for.")
    parser.add_argument("--lr", default=3e-5, help="Select Learning Rate.")
    parser.add_argument("--weight_decay", default=0.0, help="Weight Decay for all parameters.")
    
    # Model params
    parser.add_argument("--layer_num", default=4, help="Decides which block of STGCN is to be used.")
    parser.add_argument("--ensemble", default=False, action='store_true', help="Will do an ensemble of 5 heads when True.")
    
    args = parser.parse_args()