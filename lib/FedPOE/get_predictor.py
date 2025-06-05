from lib.FedPOE.predictor_classification import predictor_classification

def get_predictor(args):
    if args.task=="classification":
        predictors = []
        for i in range(args.num_clients):
            predictors.append(predictor_classification(args.eta_c, args.num_models-2, args.period))
        return predictors