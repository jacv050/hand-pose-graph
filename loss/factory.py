import loss.cross_entropy_2d
import loss.mean_square_error

LOSSES = {
    "cross_entropy_2d" : loss.cross_entropy_2d.CrossEntropyLoss2d,
    "mean_square_error" : loss.mean_square_error,
}

def get_loss(name):
    return LOSSES[name]()