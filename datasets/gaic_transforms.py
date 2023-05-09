import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from util.misc import interpolate
from util.box_ops import box_xyxy_to_cxcywh

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
         augmentations.Compose([transforms.CenterCrop(10), transforms.ToTensor(),])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target=None):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class ConvertFromInts(object):
    def __call__(self, image, target):
        return image.astype(np.float32), target


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, target=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), target


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, target=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, target


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, target):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, target


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, target=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, target


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, target=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, target


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, target=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, target


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, target=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, target


class ToCV2Image(object):
    def __call__(self, tensor, target=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), target


class ToTensor(object):
    def __call__(self, cvimage, target=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), target


# class RandomMirror(object):
#     def __call__(self, image, annotations, classes):
#         _, width, _ = image.shape
#         if random.randint(2):
#             image = image[:, ::-1]
#             for i in range (len(annotations)):
#                 annotations[i][1] = width - annotations[i][1]
#                 annotations[i][3] = width - annotations[i][3]
#         return image, annotations, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, target):
        im = image.copy()
        im, target = self.rand_brightness(im, target)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        return distort(im, target)

        #return self.rand_light_noise(im, boxes, labels)



def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        h, w = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        # image_size: h,w
        # size: w,h
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    ori_shape = image.shape[:2] # h, w
    size = get_size(ori_shape, size, max_size)
    # the returned size is (h, w)
    # cv2.resize requires (w, h)
    # torchvision.functional.resize requires (h, w)
    rescaled_image = cv2.resize(image, size[::-1])

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.shape[:2], image.shape[:2]))
    ratio_height, ratio_width = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "good_boxes" in target:
        good_boxes = target["good_boxes"]
        scaled_good_boxes = good_boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["good_boxes"] = scaled_good_boxes

    if "best_boxes" in target:
        best_boxes = target["best_boxes"]
        scaled_best_boxes = best_boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["best_boxes"] = scaled_best_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.as_tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target



class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class Normalize(object):
    def __init__(self, mean, std, score_mean, score_std):
        self.mean = mean
        self.std = std
        self.score_mean = score_mean
        self.score_std = score_std

    def __call__(self, image, target=None):
        # input image is rgb, 0-255

        rgb_mean = np.array(self.mean, dtype=np.float32)
        rgb_std = np.array(self.std, dtype=np.float32)
        image = image.astype(np.float32)
        image = image / 255.0
        image -= rgb_mean
        image = image / rgb_std

        if target is None:
            return image, None

        target = target.copy()
        h, w = image.shape[:2]

        # minscale = torch.as_tensor([0.35, 0.35, 0.55, 0.55])
        # maxscale = torch.as_tensor([0.65, 0.65, 0.95, 0.95])
        minscale = torch.as_tensor([0., 0., 0., 0.])
        maxscale = torch.as_tensor([1., 1., 1., 1.])

        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            boxes = (boxes - minscale) / (maxscale - minscale)
            target["boxes"] = boxes

        if "good_boxes" in target:
            good_boxes = target["good_boxes"]
            good_boxes = box_xyxy_to_cxcywh(good_boxes)
            good_boxes = good_boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            good_boxes = (good_boxes - minscale) / (maxscale - minscale)
            target["good_boxes"] = good_boxes

        if 'scores' in target:
            scores = target["scores"]
            # scores = (scores - self.score_mean) / self.score_std
            target["scores"] = scores

        if 'good_scores' in target:
            good_scores = target["good_scores"]
            # good_scores = (good_scores - self.score_mean) / self.score_std
            target["good_scores"] = good_scores

        return image, target



class CropAugmentation(object):
    def __init__(self):
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            RandomResize(scales, max_size=1333)
        ])

    def __call__(self, img, target):
        image, target = self.augment(img, target)
        return image, target


class croptransform(object):
    def __init__(self, set='train', imgsize_test=640):
        scales_aug = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        scale_test = [imgsize_test]
        self.set = set
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            RandomResize(scales_aug, max_size=1333)
        ])
        self.normalize = Normalize(mean=(0.485, 0.456, 0.406),
                                   std=(0.229, 0.224, 0.225),
                                   score_mean=2.95,
                                   score_std=0.8)
        self.resize = RandomResize(scale_test, max_size=1333)


    def __call__(self, image, target):

        if self.set == 'train':
            image, target = self.augment(image, target)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.set == 'test' or self.set == 'val':
            image, target = self.resize(image, target)

        image, target = self.normalize(image, target)

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)

        return image, target


def reverse(img):

    img = img.numpy()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).numpy()
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).numpy()
    img = ((img * std + mean) * 255).astype(np.uint8)
    img = img.transpose((1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
