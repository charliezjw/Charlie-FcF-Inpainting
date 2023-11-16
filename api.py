from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import torch
import cv2
import os
import dnnlib
import legacy
from torchvision.transforms import ToTensor, Resize



app = Flask(__name__)

def setup_network(network_pkl, device):
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"]

    return G.eval().to(device)

def generate_images(input_images, input_masks,output_dir, G, device, truncation_psi=0.1):
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

        #out_images = []
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

            comp_img_pil = Image.fromarray(comp_img)
            # 保存图像
            comp_img_pil.save(os.path.join(output_dir,str(i) + "_out.png")) 


        return []

 # 加载模型
in_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Global = setup_network("/data/app/stable-diffusion/work/fcf/ckpt/10_15_best.pkl", in_device)
    
def file_ext(fname):
    return os.path.splitext(fname)[1].lower()

@app.route('/inpaint', methods=['POST'])
def inpaint_image():

    request_data = request.get_json()  # 解析 JSON 请求体
    input_dir  = request_data['input_dir']
    output_dir = request_data['output_dir']
    in_imgs   = []
    mask_imgs = []
    if not input_dir.endswith('/'):
                input_dir += '/'

    if not output_dir.endswith('/'):
                output_dir += '/'

    for filename in os.listdir(input_dir):
        if filename.find("_mask")>0 : 
            tmp_image = Image.open(input_dir+filename).convert("RGB").convert("L")
            image_array = np.array(tmp_image)
            image_array[image_array != 0] = 255
            tmp_image = Image.fromarray(image_array)
            mask_imgs.append(tmp_image)
        else :
            in_imgs.append(Image.open(input_dir+filename).convert("RGB"))


    os.makedirs(output_dir, exist_ok=True)

    in_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 调用 generate_images 函数进行修复
    generate_images(in_imgs,mask_imgs,output_dir, Global, in_device)

    return jsonify({'result': 200})

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=9000)
