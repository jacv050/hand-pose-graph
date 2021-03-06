import loss.cross_entropy_2d
import loss.mean_square_error
import loss.root_mean_square_error
import loss.mean_absolute_error
import loss.geodesic_distance
import loss.geodesic_distance_weighted

LOSSES = {
    "cross_entropy_2d" : loss.cross_entropy_2d.CrossEntropyLoss2d,
    "mean_square_error" : loss.mean_square_error.MeanSquareErrorLoss,
    "root_mean_square_error" :  loss.root_mean_square_error.RootMeanSquareErrorLoss,
    "mean_absolute_error" :  loss.mean_absolute_error.MeanAbsoluteErrorLoss,
    "geodesic_distance" :  loss.geodesic_distance.GeodesicDistanceLoss,
    "geodesic_distance_weighted" :  loss.geodesic_distance_weighted.GeodesicDistanceWeightedLoss,
}

def get_loss(name):
    return LOSSES[name]()
