import importlib.util
import segmentation_models_pytorch as smp

MODEL_MAPPER = {
    'Unet': smp.Unet,
    'FPN': smp.FPN,
    'PSPNet': smp.PSPNet
}


def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module
