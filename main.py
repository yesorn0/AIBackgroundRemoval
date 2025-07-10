
#1.필요한 모듈과 모델 관련 파일 다운로드
from huggingface_hub import hf_hub_download
from pathlib import Path
import requests
import openvino as ov


#Hugging Face 저장소 ID
repo_id = "briaai/RMBG-1.4"

#다운로드할 파일 목록
download_files = ["utilities.py", "example_input.jpg"]

#파일이 존재하지 않을 경우 Hugging Face에서 다운로드
for file_for_downloading in download_files:
    if not Path(file_for_downloading).exists():
        hf_hub_download(repo_id=repo_id, filename=file_for_downloading, local_dir=".")

        
        
 #2. 배경 제거 모델 불러오기       
from transformers import AutoModelForImageSegmentation

#Transformers를 통해 RMBG모델 로드
net = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)


#3.이미지 전처리 및 후처리 유틸 불러오기
import torch
from PIL import Image
from utilities import preprocess_image, postprocess_image
import numpy as np
from matplotlib import pyplot as plt


#4.결과 시각화 함수 정의
def visualize_result(orig_img: Image, mask: Image, result_img: Image):
   
   #입력 이미지, 생성된 마스크, 배경 제거 이미지를 한 화면에 시각화
    """
    Helper for results visualization

    parameters:
       orig_img (Image): input image
       mask (Image): background mask
       result_img (Image) output image
    returns:
      plt.Figure: plot with 3 images for visualization
    """
    #기본 설정
    titles = ["Original", "Background Mask", "Without background"]
    im_w, im_h = orig_img.size
    is_horizontal = im_h <= im_w
    figsize = (20, 20)
    num_images = 3
    
    #서브플롯 구성
    fig, axs = plt.subplots(
        num_images if is_horizontal else 1,
        1 if is_horizontal else num_images,
        figsize=figsize,
        sharex="all",
        sharey="all",
    )
    
    #시각화 설정
    fig.patch.set_facecolor("white")
    list_axes = list(axs.flat)
    for a in list_axes:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
        a.grid(False)
        
    #이미지 그리기
    list_axes[0].imshow(np.array(orig_img))
    list_axes[1].imshow(np.array(mask), cmap="gray")
    list_axes[0].set_title(titles[0], fontsize=15)
    list_axes[1].set_title(titles[1], fontsize=15)
    list_axes[2].imshow(np.array(result_img))
    list_axes[2].set_title(titles[2], fontsize=15)

    #레이아웃 정리
    fig.subplots_adjust(wspace=0.01 if is_horizontal else 0.00, hspace=0.01 if is_horizontal else 0.1)
    fig.tight_layout()
    return fig


#5.모델로 배경 제거 수행
im_path = "./example_input.jpg"

# prepare input
model_input_size = [1024, 1024]
orig_im = np.array(Image.open(im_path))
orig_im_size = orig_im.shape[0:2]
image = preprocess_image(orig_im, model_input_size)

#모델 추론
# inference
result = net(image)


#후처리로 마스크 생성
# post process
result_image = postprocess_image(result[0][0], orig_im_size)


#원본 이미지에 마스크 적용해 배경 제거 결과 생성
# save result
pil_im = Image.fromarray(result_image)
no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
orig_image = Image.open(im_path)
no_bg_image.paste(orig_image, mask=pil_im)
no_bg_image.save("example_image_no_bg.png")

#결과 시각화
visualize_result(orig_image, pil_im, no_bg_image);


#6.모데레을 OpenVINO 형식으로 변환하고 추론

#OpenVINO IR 형식으로 저장된 모델이 없으면 변환 수행
ov_model_path = Path("rmbg-1.4.xml")

if not ov_model_path.exists():
    ov_model = ov.convert_model(net, example_input=image, input=[1, 3, *model_input_size])
    ov.save_model(ov_model, ov_model_path)

#OpenVINO 모델 로딩 및 추론 실행
device = "CPU"
core = ov.Core()

ov_compiled_model = core.compile_model(ov_model_path, device)


#추론 및 후처리
result = ov_compiled_model(image)[0]

# post process
result_image = postprocess_image(torch.from_numpy(result), orig_im_size)

#결과 이미지 저장
# save result
pil_im = Image.fromarray(result_image)
no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
orig_image = Image.open(im_path)
no_bg_image.paste(orig_image, mask=pil_im)
no_bg_image.save("example_image_no_bg.png")

visualize_result(orig_image, pil_im, no_bg_image);


#7.Gradio 웹 인터페이스 함수 정의
def get_background_mask(model, image):
    return model(image)[0]


def on_submit(image):
    #Gradio에서 사용자가 이미지를 업로드했을 때 호출되는 함수
    original_image = image.copy()

    h, w = image.shape[:2]
    
    #전처리
    image = preprocess_image(original_image, model_input_size)


    #후처리 후 RGBA 이미지 생성
    mask = get_background_mask(ov_compiled_model, image)
    result_image = postprocess_image(torch.from_numpy(mask), (h, w))
    pil_im = Image.fromarray(result_image)
    orig_img = Image.fromarray(original_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    no_bg_image.paste(orig_img, mask=pil_im)

    return no_bg_image


#Gradio 웹 앱 실행
import requests

#Gradio용 헬퍼 파일 다운로드
if not Path("gradio_helper.py").exists():
    r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/rmbg-background-removal/gradio_helper.py")
    open("gradio_helper.py", "w").write(r.text)


#Gradio 인터페이스 생성 및 실행
from gradio_helper import make_demo

demo = make_demo(fn=on_submit)

#로컬 또는 외부 공유 URL로 실행
try:
    demo.launch(debug=True)
except Exception:
    demo.launch(share=True, debug=True)
# If you are launching remotely, specify server_name and server_port
# EXAMPLE: `demo.launch(server_name='your server name', server_port='server port in int')`
# To learn more please refer to the Gradio docs: https://gradio.app/docs/



