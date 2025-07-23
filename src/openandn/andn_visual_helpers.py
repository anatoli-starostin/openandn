def grayscale_to_red_and_blue(img):
    if len(img.shape) == 4:
        img = img.repeat(1, 3, 1, 1)
        img[:, 0, :, :] = img[:, 0, :, :].clip(0.0, 1.0)
        img[:, 1, :, :] = 0.0
        img[:, 2, :, :] = -img[:, 2, :, :].clip(-1.0, -0.0)
    else:
        img = img.repeat(3, 1, 1)
        img[0, :, :] = img[0, :, :].clip(0.0, 1.0)
        img[1, :, :] = 0.0
        img[2, :, :] = -img[2, :, :].clip(-1.0, -0.0)

    return img
