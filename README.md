# Deep_learning_application_considering_fairness

Stitch it in Time: GAN-Based Facial Editing of Real Videos </br> 
https://arxiv.org/abs/2201.08361 </br>
[GitHub - rotemtzaban/STIT](https://github.com/rotemtzaban/STIT?tab=readme-ov-file) </br>

<img width="674" alt="111" src="https://github.com/user-attachments/assets/03199025-44b1-45e4-b0e7-55b91a56de68">

1. Video Preprocessing
    1. ffmpeg
    동영상을 프레임 단위로 나눠 이미지 파일로 저장
    
2. Training
    
    동영상을 프레임 단위로 나눈 이미지를 사용해 Inversion과 Pivot 이미지를 생성한다. 그 후 Edit에 사용하기 위해 이미지를 저장한다. 
    
    train.py
    
    ```python
    # Crop & Align
    crops, orig_images, quads = crop_faces(image_size, files, scale,
                            center_sigma=center_sigma, xy_sigma=xy_sigma, use_fa=use_fa)
                            
    ds = ImageListDataset(crops, transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
        
        
    # Pivot Tuning
    coach = Coach(ds, use_wandb)
    
    ws = coach.train()
    ```
    

1. Crop & Align

코드에서는 face_alignment 패키지를 사용하지 않고 dlib 패키지를 사용하여 crop을 진행한다. Landmark Detection을 통해 구한 얼굴의 위치에 Gaussian filter를 적용해 표정이나 얼굴의 방향에 영향을 적게 받는 일관성을 유지하는 얼굴의 위치를 계산한다. 

```python
def crop_faces(IMAGE_SIZE, files, scale, center_sigma=0.0, xy_sigma=0.0, use_fa=False):
    if use_fa:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)
        predictor = Non
        detector = None
    else:
        fa = None
        predictor = dlib.shape_predictor(paths_config.shape_predictor_path)
        detector = dlib.get_frontal_face_detector()

		# Crop
    cs, xs, ys = [], [], []
    for _, path in tqdm(files):
        c, x, y = compute_transform(path, predictor, detector=detector,
                                    scale=scale, fa=fa)
        cs.append(c)
        xs.append(x)
        ys.append(y)

    cs = np.stack(cs)
    xs = np.stack(xs)
    ys = np.stack(ys)
    
    if center_sigma != 0:
        cs = gaussian_filter1d(cs, sigma=center_sigma, axis=0)

    if xy_sigma != 0:
        xs = gaussian_filter1d(xs, sigma=xy_sigma, axis=0)
        ys = gaussian_filter1d(ys, sigma=xy_sigma, axis=0)

    quads = np.stack([cs - xs - ys, cs - xs + ys, cs + xs + ys, cs + xs - ys], axis=1)
    quads = list(quads)

    crops, orig_images = crop_faces_by_quads(IMAGE_SIZE, files, quads)

    return crops, orig_images, quads
```

predictor와 detector를 통해 landmark Detection을 진행해 얼굴의 위치를 찾아 얼굴의 중심 c, x축 거리 x, y 축 거리 y 값을 구한다. predicator와 detector는 pretrain된 모델 사용함.

```python
def compute_transform(filepath, predictor, detector=None, scale=1.0, fa=None):
    lm = get_landmark(filepath, predictor, detector, fa)
    if lm is None:
        raise Exception(f'Did not detect any faces in image: {filepath}')
    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise
    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)

    x *= scale
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    return c, x, y
```

1. Pivot Tuning

Coach.train

e4e Encoder를 사용해 이미지로부터 Pivot 생성. e4e는 ffhq 모델로 pretrain된 모델 사용 

```python
for fname, image in tqdm(self.dataset):
    image_name = fname
    w_pivot = self.get_inversion(image_name, image)
    w_pivots.append(w_pivot)
    images.append((image_name, image))
```

생성된 Pivot으로 Generator를 사용해 이미지를 생성한 후 real image와의 l2_loss, lpips loss를 계산해 이 loss의 합으로 Generator가 pivot으로부터 real image를 생성하도록 학습이 진행된다. 

```python
for step in trange(hyperparameters.max_pti_steps):
      step_loss_dict = defaultdict(list)
      t = (step + 1) / hyperparameters.max_pti_steps

      for data, w_pivot in zip(images, w_pivots):
          image_name, image = data
          image = image.unsqueeze(0)

          real_images_batch = image.to(global_config.device)

          generated_images = self.forward(w_pivot)
          loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,self.G, use_ball_holder, w_pivot)

          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

          use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

          global_config.training_step += 1
```

1. Editing

생성된 Pivot을 사용해 이미지를 수정한다.

edit_video_stitching_tuning.py

```python
# 학습된 generator, pivot, 이미지의 얼굴 위치 quad를 laod 
gen, orig_gen, pivots, quads = load_generators(run_name)

# 이미지에서 얼굴을 crop
crops, orig_images = crop_faces_by_quads(image_size, orig_files, quads)

latent_editor = LatentEditor()

# 변경하고자 하는 edit name에 맞는 w_direction, edit한 이미지를 edits에 호출
edits, is_style_input = latent_editor.get_interfacegan_edits(pivots, edit_name, edit_range)

# 프레임별로 이미지를 수정
for edits_list, direction, factor in edits:
		for i, (orig_image, crop, quad, inverse_transform) in \
		                tqdm(enumerate(zip(orig_images, crops, quads, inverse_transforms)), total=len(orig_images)):
		                
			   
				w_edit_interp = edits_list[i][None]
				
				# edit된 w_latent를 사용해 이미지를 생성 
				edited_tensor = gen.synthesis.forward(w_edit_interp, style_input=is_style_input, noise_mode='const', force_fp32=True)
				
				# 얼굴, 얼굴 주변, 전체 mask 계산
				crop_tensor = to_tensor(crop)[None].mul(2).sub(1).cuda()
        content_mask, border_mask, full_mask = calc_masks(crop_tensor, segmentation_model, border_pixels,inner_mask_dilation, outer_mask_dilation, whole_image_border)
        
        # 수정된 이미지, mask를 통해 stitching tuning을 수행후 video frame에 저장
        optimized_tensor = optimize_border(gen, crop_tensor, edited_tensor,
                                               w_edit_interp, border_mask=border_mask, content_mask=content_mask,optimize_generator=True, num_steps=num_steps, loss_l2=l2,is_style_input=is_style_input, content_loss_lambda=content_loss_lambda, border_loss_threshold=border_loss_threshold)

        video_frames[f'optimized_edits/{direction}/{factor}'].append(
            tensor2pil(optimized_tensor)
        )

# stitching tuning이 완료된 이미지들을 연결해 동영상을 생성       
for folder_name, frames in video_frames.items():
    folder_path = os.path.join(output_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    imageio.mimwrite(os.path.join(folder_path, 'out.mp4'), frames, fps=25, output_params=['-vf', 'fps=25'])
```

1. Editing
    
    1. edit
    original Image Latent = original Image Latent + w_direction * factor
    
    ```python
    def _apply_interfacegan(latent, direction, factor=1, factor_range=None):
            edit_latents = []
            if factor_range is not None:  # Apply a range of editing factors. for example, (-5, 5)
                for f in range(*factor_range):
                    edit_latent = latent + f * direction
                    edit_latents.append(edit_latent)
                edit_latents = torch.cat(edit_latents)
            else:
                edit_latents = latent + factor * direction
            return edit_latents
    ```
    

1. w_direction
기본 제공되는 w_direction은 아래와 같이 16개가 있다. Gender, Age는 기본 제공되는 w_direction으로도 사용 가능하지만 Ethnicity는 존재하지 않아 새롭게 학습이 필요함.
    1. age, gender , Ethnicity(Caucasian,  Asian)

```python
age.npy
eye_distance.npy
eye_eyebrow_distance.npy
eye_ratio.npy
eyes_open.npy
gender.npy
lip_ratio.npy
mouth_open.npy
mouth_ratio.npy
nose_mouth_distance.npy
nose_ratio.npy
nose_tip.npy
pitch.npy
roll.npy
smile.npy
yaw.npy
```

새로운 w_direction 제작 코드
e4e encoder를 사용해서 latent를 생성하고 이 latent로 LogisticRegression을 학습하고 LogisticRegression의 parameter를 w_direction로 설정한다.

Logistic Regression 훈련 (얼굴 속성 변화를 위한 방향 벡터를 학습)

```python
from models.e4e.encoders.psp_encoders import Encoder4Editing
from configs import paths_config
from argparse import Namespace
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
import torchvision
from tqdm import tqdm

device = torch.device('cpu')

batch_size = 8
epochs = 5

trans = transforms.Compose([transforms.Resize((256, 256)), 
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

trainset = torchvision.datasets.ImageFolder(root = "../input_data", 
					transform = trans
                    )

dataLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

ckpt = torch.load(paths_config.e4e, map_location='cpu')
opts = ckpt['opts']
opts['checkpoint_path'] = paths_config.e4e
opts = Namespace(**opts)

encoder = Encoder4Editing(num_layers=50, mode='ir_se', opts=opts).to(device)

model = LogisticRegression(class_weight='balanced')

for epoch in range(epochs):
		for src, tgt in tqdm(dataLoader):
		    out = encoder(src).reshape(batch_size, -1).detach().numpy()
        tgt = tgt.detach().numpy()
    
        model.fit(out, tgt)

direction = model.coef_.reshape(18, 512)
np.save('Asian_African-American.npy', direction)
```

1. Stiching Tuning
    
    <img width="445" alt="1" src="https://github.com/user-attachments/assets/4d2fb1b3-3fbe-4b15-abe0-661cc2cfb7bc">
    
    generator로 생성된 이미지가 w_direction으로 edit된 이미지, 얼굴 주변의 이미지와의 Loss가 최소가 되도록 학습하여 이미지를 생성한다. 
    
    ```python
    def optimize_border(G: torch.nn.Module, border_image, content_image, w: torch.Tensor, border_mask, content_mask, optimize_generator=False, optimize_wplus=False, num_steps=100, loss_l2=True, is_style_input=False, content_loss_lambda=0.01, border_loss_threshold=0.0):
        assert optimize_generator or optimize_wplus
    
        G = copy.deepcopy(G).train(optimize_generator).requires_grad_(optimize_generator).float()
    
        latent = w
        assert not optimize_wplus
        
        parameters = []
    
        parameters += list(G.parameters())
        
        optimizer = torch.optim.Adam(parameters, hyperparameters.stitching_tuning_lr)
        for step in trange(num_steps, leave=False):
            generated_image = G.synthesis(latent, style_input=is_style_input, noise_mode='const', force_fp32=True)
    
    				# generator로 생성된 이미지와 얼굴 주변 이미지의 Loss
            border_loss = masked_l2(generated_image, border_image, border_mask, loss_l2)
            # border_loss + (generator로 생성된 이미지와 w_direction으로 edit된 이미지간의 Loss) 
            loss = border_loss + content_loss_lambda * masked_l2(generated_image, content_image, content_mask, loss_l2)
            if border_loss < border_loss_threshold:
                break
            optimizer.zero_grad()
    
            loss.backward()
            optimizer.step()
    
        generated_image = G.synthesis(latent, style_input=is_style_input, noise_mode='const', force_fp32=True)
        del G
        return generated_image.detach()
    ```
    

1. Result

Asian - African-American

[[out (19).mp4](https://prod-files-secure.s3.us-west-2.amazonaws.com/e6f6d854-0306-4b25-b13d-e5848e491185/f6317f3c-6fb1-40c8-8ad5-fedd3f32f9f0/out_(19).mp4)](https://github.com/user-attachments/assets/60858e10-5bd9-4460-accf-85a9204f17e5)

Caucasian - African-American

[out (21).mp4](https://prod-files-secure.s3.us-west-2.amazonaws.com/e6f6d854-0306-4b25-b13d-e5848e491185/67b0995e-72aa-4b7b-97c3-4ff74a3cb20e/out_(21).mp4)

Caucasian - African-American
