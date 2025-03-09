# Kytra's MatAnyone implementation for ComfyUI

This is a ComfyUI node for [MatAnyone](https://github.com/pq-yang/MatAnyone), a state-of-the-art video matting model that can remove backgrounds from videos using just a single mask for the first frame for enhanced/guided video matting. 

## Installation

1. Clone this repository into your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/KytraScript/ComfyUI_MatAnyone_Kytra.git
```

2. Install pip requirements
```bash
cd ComfyUI/custom_nodes/ComfyUI_MatAnyone_Kytra
pip install -r requirements.txt
```

3. The model weights are automatically downloaded during your first run from:
https://huggingface.co/Mothersuperior/ComfyUI_MatAnyone_Kytra and they will 
be placed in the ComfyUI_MatAnyone_Kytra/model folder

### Parameters

- **video_frames**: Video image batch (example workflow uses VHS Video Loader)
- **mask**: First frame mask (subject rgb(255,255,255))
- **warmup_frames**: Number of warmup iterations (default: 10)
- **erode_kernel**: Erosion kernel size (default: 10)
- **dilate_kernel**: Dilation kernel size (default: 10)
- **bg_red** & **bg_green** & **bg_blue**: Set values for background color for composite video output

## Example Workflow

- Provided in the repository 'workflow' directory

## Credits

- Original MatAnyone implementation: [https://github.com/pq-yang/MatAnyone](https://github.com/pq-yang/MatAnyone)
- Paper: [MatAnyone: Prompting Any Object for Open-world Matting](https://arxiv.org/abs/2401.05228)

## License

This project is licensed under the same license as the original MatAnyone project. 