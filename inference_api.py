import cv2
import dnnlib
import numpy as np
import PIL.Image as Image
import torch

import legacy

import random
from torchvision.transforms import ToTensor, Resize
from flask import Flask, jsonify, request
from service_streamer import ThreadedStreamer

def setup_network(network_pkl, device):
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"]

    return G.eval().to(device)


def generate_images(
    input_images,
    input_masks,
    G,
    device,
    truncation_psi=0.1,
):
    # Labels.
    label = torch.zeros([1, 0], device=device)
    input_sizes = []
    scaled_images = []
    scaled_masks = []
    with torch.no_grad():
        ## data is a tuple of (rgbs, rgbs_erased, amodal_tensors, visible_mask_tensors, erased_modal_tensors) ####
        for i in input_images:
            input_sizes.append(tuple(reversed(i.size)))
            image = cv2.resize(np.array(i), (512, 512), interpolation=cv2.INTER_AREA)
            image = image / 127.5 - 1
            scaled_images.append(image)

        for i in input_masks:
            mask = np.array(i) / 255
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, 2)
            scaled_masks.append(mask)

        # rgb_erased = scaled_images * (1 - scaled_masks) # erase rgb

        # Convert images and masks to tensors
        tensor_transform = ToTensor()
        images_tensor = torch.stack([tensor_transform(img) for img in scaled_images])
        invisible_masks_tensor = torch.stack(
            [tensor_transform(mask) for mask in scaled_masks]
        )
        # erased_img_tensor = torch.stack([tensor_transform(rgb_erased) for rgb_erased in rgb_erased])
        erased_img_tensor = images_tensor * (1 - invisible_masks_tensor)

        # Transfer tensors to the device
        erased_img = erased_img_tensor.to(device)
        mask = invisible_masks_tensor.to(device)

        # size_transform = Resize()
        pred_img = G(
            img=torch.cat([0.5 - mask, erased_img], dim=1),
            c=label,
            truncation_psi=truncation_psi,
            noise_mode="const",
        )

        out_images = []
        for i, img in enumerate(pred_img):
            resized_img = Resize(input_sizes[i])(img)
            # resized_mask = Resize(input_sizes[i])(mask[i])
            input_mask = np.array(input_masks[i]) / 255
            input_mask = tensor_transform(np.expand_dims(input_mask, 2)).to(device)
            comp_img = input_mask * resized_img \
                    + (1 - input_mask).to(device) * tensor_transform(np.array(input_images[i])/ 127.5 - 1).to(device)
            lo, hi = [-1, 1]
            comp_img = np.asarray(comp_img.cpu(), dtype=np.float32).transpose(
                1, 2, 0
            )
            comp_img = (comp_img - lo) * (255 / (hi - lo))
            comp_img = np.rint(comp_img).clip(0, 255).astype(np.uint8)

            out_images.append(comp_img)
        return out_images


app = Flask(__name__)
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
in_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = setup_network(network_pkl="./ckpt/10_15_best.pkl", device=in_device)

streamer = ThreadedStreamer(generate_images, batch_size=64)


@app.route('/stream_predict', methods=['POST'])
def stream_predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = streamer.predict([img_bytes])[0]
        return jsonify({'class_id': class_id, 'class_name': class_name})
    

if __name__ == "__main__":
    app.run()
    
    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # in_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # G = setup_network(network_pkl="./ckpt/10_15_best.pkl", device=in_device)
    # in_imgs = []
    # in_imgs.append(
    #     Image.open(
    #         "datasets/PG_full/airemove_f9b52a4e5e6787b27d73a0d11ba35b40_1696751112328_1696751114208.png"
    #     ).convert("RGB")
    # )
    # in_imgs.append(
    #     Image.open(
    #         "datasets/PG_full/airemove_f9b52a4e5e6787b27d73a0d11ba35b40_1696748550927_1696748550945.png"
    #     ).convert("RGB")
    # )


    # in_masks = []
    # in_masks.append(
    #     Image.open(
    #         "datasets/PG_full/airemove_f9b52a4e5e6787b27d73a0d11ba35b40_1696751112328_1696751114208_mask.png"
    #     )
    #     .convert("RGB")
    #     .convert("L")
    # )
    # in_masks.append(
    #     Image.open(
    #         "datasets/PG_full/airemove_f9b52a4e5e6787b27d73a0d11ba35b40_1696748550927_1696748550945_mask.png"
    #     )
    #     .convert("RGB")
    #     .convert("L")
    # )


    # generate_images(
    #     input_images=in_imgs, input_masks=in_masks, G=G, device=in_device
    # )
