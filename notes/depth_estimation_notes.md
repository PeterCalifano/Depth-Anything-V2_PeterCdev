# Notes on depth estimation task with ML models

## Depth estimation task definition

- Monocular depth estimation (MDE) consists in predicting pixel-wise depth maps from single static images.

- The architecture for this image-to-image task is usually an encoder-decoder model, like a U-Net, with various modifications. Formally, this is a **pixel-wise regression** problem similarly to semantic segmentation.

- **NOTE**: focal length plays an extremely important role in MDE. This is because different optics can alter the perception of depth (by varying focal length) thus producing ambiguity in the task.

### Absolute vs relative depths

Depth estimation models are typically trained to predict **relative depths**. That is:

- The pixel values indicate which points are closer or further away without referencing real-world units of measurement. Often the relative depth is inverted, i.e. the smaller the number, the farther the point is (**inverse depth model**).

This is because absolute (**metric**) depth cannot be generalized to any camera and any dataset due to scale ambiguity of monocular images. In practice, metric depth models can only be application and scenario specific and do **not** generalize.

## Depth-Anything V1-V2

> Some resources:
>
> 1. YT video: <https://www.youtube.com/watch?v=sz30TDttIBA&t=474s>
> 2. Tutorial on depth estimation: <https://huggingface.co/blog/Isayoften/monocular-depth-estimation-guide>
> 3. About D.A. model: <https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything_v2>

### Training and loss functions

- Teacher-student strategy:
  - A small labelled dataset was used to train the teacher model, which in turn generated a much larger pseudo-labelled dataset (from unlabelled images)
  - Loss function for the teacher:
    1. **Scale-Shift Invariant** loss (centred normalized depth maps) --> pixel-wise MSE on the whole map
    2. **Multi-scale Gradient Matching** loss --> learns sharp edges at several resolution levels (like pyramids from the original size)
  - The student gets trained on the combination of the datasets, with heavy augmentations and adding a **Semantic Preservation Loss**. This is necessary to make the student better than the teacher (which is a bound here)
    1. Cut-Mix augmentation
    2. Auxiliary task: semantic assisted perception, using DinoV2 self-supervised model. The **cosine similarity** loss between student and pre-trained **frozen** DinoV2 encoder embeddings is minimized.

- Key limitation highlighted by authors is not the architecture but the training dataset, especially for translucent objects (due to real worl sensor datasets limitations). This is why synthetic data only are used in DepthAnythingV2 for the teacher.

#### Scale and Shift Invariant Loss

>**The idea**: To enable multi-dataset joint training, we adopt the affine-invariant loss to ignore the unknown scale and shift of each sample.

The loss is defined as the sum of affine-invariant mean absolute error loss of the scaled and shifted depths (prediction vs GT):

$$

$$

This metric is obtained as difference of shifted-scaled depths:

$$

$$

with shift (median) and scale (sort of variance):

$$

t(d) = median(d) \hspace{1cm} s(d) = \dfrac{1}{HW} \sum^{HW}_{i=1} |d_i - t(d)|

$$

Reference that introduced these loss functions (MiDaS): <https://arxiv.org/abs/1907.01341>

### Some notes from experiments

- DepthAnythingV2 requires fine tuning to work well with typical space images where the body is partially or entirely surrounded by black pixels. This is because it is able to correctly distinguish depth only near the most illuminated limb whereas artifacts are always present near the (black) corners of the image. In other words, the model is not capable of identifying and consistently classifying the black pixels as very distant background.
- Features on the surface are not distinguished in terms of depth neither. However, the general shape is somewhat captured, with limbs predicted to be further than the regions near the centre.
- For asteroids in close proximity, the prediction fails in really interesting ways which highlight the inadeguancy of the training data in letting the model to generalize to the space domain. Example here:
![alt text](<Screenshot from 2025-10-19 14-53-42.png>)
![alt text](image-1.png)
