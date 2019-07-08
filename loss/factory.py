import loss.cross_entropy_2d
import loss.mean_square_error
import loss.root_mean_square_error
import loss.mean_absolute_error

LOSSES = {
    "cross_entropy_2d" : loss.cross_entropy_2d.CrossEntropyLoss2d,
    "mean_square_error" : loss.mean_square_error.MeanSquareErrorLoss,
    "root_mean_square_error" :  loss.root_mean_square_error.RootMeanSquareErrorLoss,
    "mean_absolute_error" :  loss.mean_absolute_error.MeanAbsoluteErrorLoss,
}

def get_loss(name):
    return LOSSES[name]()
