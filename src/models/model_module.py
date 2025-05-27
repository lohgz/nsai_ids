import os
import torch
from .cnn import CNN_Classifier

def build_model(
    num_features: int,
    split_tag:    str,
    run:          int,
    base_path:    str = None,
    device:       torch.device = torch.device("cpu")
) -> torch.nn.Module:
    """
    Loads the checkpoint for the given split/run from:
       data/output/model/{split_tag}_run_{run}/ckpts/{ModelClass}{split_tag}_run_{run}.pth
    """
    if base_path is None:
        base_path = os.getcwd()

    # dynamically get the model class name
    model_name = CNN_Classifier.__name__

    # create checkpoint dir
    ckpt_dir = os.path.join(
        base_path,
        "data", "output", "model",
        f"{split_tag}_run_{run}",
        "ckpts"
    )
    ckpt_path = os.path.join(
        ckpt_dir,
        f"{model_name}{split_tag}_run_{run}.pth"
    )

    # instantiate and load
    model = CNN_Classifier(num_features).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
