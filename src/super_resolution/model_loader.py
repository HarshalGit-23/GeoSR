from huggingface_hub import snapshot_download
import sys


def load_model():
    model_dir = snapshot_download("WEO-SAS/sen2sr")

    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    from model import Model

    model = Model(local_dir=model_dir)

    print(model.description)

    return model


if __name__ == "__main__":
    load_model()