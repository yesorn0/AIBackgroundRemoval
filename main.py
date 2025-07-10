
from huggingface_hub import hf_hub_download
from pathlib import Path
import requests
import openvino as ov


#다운로드 파일로 분류
repo_id = "briaai/RMBG-1.4"

download_files = ["utilities.py", "example_input.jpg"]

for file_for_downloading in download_files:
    if not Path(file_for_downloading).exists():
        hf_hub_download(repo_id=repo_id, filename=file_for_downloading, local_dir=".")




from transformers import AutoModelForImageSegmentation

net = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)


import torch
from PIL import Image
from utilities import preprocess_image, postprocess_image
import numpy as np
from matplotlib import pyplot as plt


def visualize_result(orig_img: Image, mask: Image, result_img: Image):
    """
    Helper for results visualization

    parameters:
       orig_img (Image): input image
       mask (Image): background mask
       result_img (Image) output image
    returns:
      plt.Figure: plot with 3 images for visualization
    """
    titles = ["Original", "Background Mask", "Without background"]
    im_w, im_h = orig_img.size
    is_horizontal = im_h <= im_w
    figsize = (20, 20)
    num_images = 3
    fig, axs = plt.subplots(
        num_images if is_horizontal else 1,
        1 if is_horizontal else num_images,
        figsize=figsize,
        sharex="all",
        sharey="all",
    )
    fig.patch.set_facecolor("white")
    list_axes = list(axs.flat)
    for a in list_axes:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
        a.grid(False)
    list_axes[0].imshow(np.array(orig_img))
    list_axes[1].imshow(np.array(mask), cmap="gray")
    list_axes[0].set_title(titles[0], fontsize=15)
    list_axes[1].set_title(titles[1], fontsize=15)
    list_axes[2].imshow(np.array(result_img))
    list_axes[2].set_title(titles[2], fontsize=15)

    fig.subplots_adjust(wspace=0.01 if is_horizontal else 0.00, hspace=0.01 if is_horizontal else 0.1)
    fig.tight_layout()
    return fig


im_path = "./example_input.jpg"

# prepare input
model_input_size = [1024, 1024]
orig_im = np.array(Image.open(im_path))
orig_im_size = orig_im.shape[0:2]
image = preprocess_image(orig_im, model_input_size)

# inference
result = net(image)

# post process
result_image = postprocess_image(result[0][0], orig_im_size)

# save result
pil_im = Image.fromarray(result_image)
no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
orig_image = Image.open(im_path)
no_bg_image.paste(orig_image, mask=pil_im)
no_bg_image.save("example_image_no_bg.png")

visualize_result(orig_image, pil_im, no_bg_image);


ov_model_path = Path("rmbg-1.4.xml")

if not ov_model_path.exists():
    ov_model = ov.convert_model(net, example_input=image, input=[1, 3, *model_input_size])
    ov.save_model(ov_model, ov_model_path)


device = "CPU"
core = ov.Core()

ov_compiled_model = core.compile_model(ov_model_path, device)

result = ov_compiled_model(image)[0]

# post process
result_image = postprocess_image(torch.from_numpy(result), orig_im_size)

# save result
pil_im = Image.fromarray(result_image)
no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
orig_image = Image.open(im_path)
no_bg_image.paste(orig_image, mask=pil_im)
no_bg_image.save("example_image_no_bg.png")

visualize_result(orig_image, pil_im, no_bg_image);


def get_background_mask(model, image):
    return model(image)[0]


def on_submit(image):
    original_image = image.copy()

    h, w = image.shape[:2]
    image = preprocess_image(original_image, model_input_size)

    mask = get_background_mask(ov_compiled_model, image)
    result_image = postprocess_image(torch.from_numpy(mask), (h, w))
    pil_im = Image.fromarray(result_image)
    orig_img = Image.fromarray(original_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    no_bg_image.paste(orig_img, mask=pil_im)

    return no_bg_image

import requests

if not Path("gradio_helper.py").exists():
    r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/rmbg-background-removal/gradio_helper.py")
    open("gradio_helper.py", "w").write(r.text)

from gradio_helper import make_demo

demo = make_demo(fn=on_submit)

try:
    demo.launch(debug=True)
except Exception:
    demo.launch(share=True, debug=True)
# If you are launching remotely, specify server_name and server_port
# EXAMPLE: `demo.launch(server_name='your server name', server_port='server port in int')`
# To learn more please refer to the Gradio docs: https://gradio.app/docs/



