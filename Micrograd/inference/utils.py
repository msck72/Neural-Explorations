import torchvision.models as models

from inference_tensor import InferenceTensor
import conv_cpp
import pool_layers
from resnet import ResNet18

def build_resnet18_from_torch():
    print("Loading pretrained resnet's torch weights into the InferenceEngine...", flush=True)
    torch_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    torch_model.eval()

    model = ResNet18()
   
    for k, v in torch_model.state_dict().items():

        if('num_batches_tracked' in k):
            continue
        ls = k.split('.')
        if(len(ls) == 2):
            if('conv' in k):
                model.stem.conv.set_values(v.detach().numpy().tolist())
            elif('bn' in k):
                getattr(model.stem.bn_layer, f"{ls[-1]}").set_values(v.detach().numpy().tolist())
            elif('fc' in k):
                try:
                    flat_values = v.detach().numpy().ravel().tolist()
                    getattr(model.classifier_head, f"FC_{ls[-1]}").set_values(flat_values)
                except Exception as e:
                    print(f"Error setting the FC layer for {k}: {e}")
                    print('Expected torch shape:', v.shape)
                    print('Custom tensor shape:', getattr(model.classifier_head, f"FC_{ls[-1]}").shape)
                    exit()
                
            continue
        
        ln = (int(ls[0][-1]) - 1) * 2 + int(ls[1])
        if('conv' in k):
            getattr(model.blocks[ln], f"conv{1 if 'conv1' in k else 2}").set_values(v.detach().numpy().tolist())
        elif('bn' in k):
            try:
                getattr(model.blocks[ln].bn_layers[0 if '1' in ls[2] else 1], f"{ls[-1]}").set_values(v.detach().numpy().tolist())
            except Exception as e:
                print(f"Error setting values for {k}: {e}", flush=True)
                print(f'expected shape: {getattr(model.blocks[ln].bn_layers[0 if "1" in ls[2] else 1], f"{ls[-1]}").shape}, actual shape: {v.shape}', flush=True)
                break
        elif('downsample' in k):
            if(ls[3] == '0'):
                getattr(model.blocks[ln], "conv3").set_values(v.detach().numpy().tolist())
            else:
                getattr(model.blocks[ln].bn_layers[2], f"{ls[-1]}").set_values(v.detach().numpy().tolist())

    print('Loaded pretrained resnet weights into InferenceEngine')
    return model