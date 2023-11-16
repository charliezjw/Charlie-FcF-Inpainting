from PIL import Image
import numpy as np
import os

def apply_union_and_stitch(mask_img, src_img, processed_img, output_path):
    '''
    Convert Pillow images to NumPy arrays for bitwise operations
    '''
    mask_arr = np.array(mask_img)[:,:,0]  # Using the blue channel of the mask image
    processed_arr = np.array(processed_img)
    src_arr = np.array(src_img)

    # Compute the union of the mask and the processed image
    union_arr = np.bitwise_and(mask_arr, processed_arr[:,:,0])

    # Create a 3-channel version of the union array
    union_arr_3channel = np.stack([union_arr]*3, axis=-1)

    # Use the union as a mask to copy the processed image over the src image
    np.copyto(src_arr, processed_arr, where=union_arr_3channel>0)

    # Convert the result back to a Pillow image
    result_img = Image.fromarray(src_arr)

    # Save the result to the output path
    result_img.save(output_path)

src_directory = '/root/charlie/fcf_out_v1'
dst_directory = '/root/charlie/fcf_out_v3'

for root, dirs, files in os.walk(src_directory):
    for file in files:
        if file.endswith("_hd.png"):
            img_file_path = os.path.join(root, file)
            # img_file_path = mask_file_path.replace("_mask.png", ".png")
            mask_file_path = img_file_path.replace("_hd.png", "_mask.png")
            pro_file_path = mask_file_path.replace("_mask.png", "_out.png")
            out_file_path = os.path.join(dst_directory, file.replace("_mask.png", "_final.png"))
            in_image = Image.open(img_file_path).convert('RGB')
            mask_image = Image.open(mask_file_path)
            pro_image = Image.open(pro_file_path).convert('RGB')

            original_size = mask_image.size
            in_image = in_image.resize(original_size)
            pro_image = pro_image.resize(original_size)
            apply_union_and_stitch(
                mask_img=mask_image,
                src_img=in_image,
                processed_img=pro_image,
                output_path=out_file_path,
            )
