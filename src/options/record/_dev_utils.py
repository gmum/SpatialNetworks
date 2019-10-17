import torch
import torchfunc
import tqdm


def get_model(args):
    model = torch.load(args.model)
    model.eval()
    torchfunc.module.freeze(model)
    if args.cuda:
        return model.cuda()
    return model


def record_data(model, dataset, args):
    for element, _ in tqdm.tqdm(dataset, total=len(dataset)):
        if args.cuda is not None:
            element = element.cuda()

        model(element)
