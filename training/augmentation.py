import imgaug.augmenters as iaa

augmenters = iaa.Sequential([
            iaa.SomeOf((0, 2), [
                iaa.GaussianBlur((0, 0.3)),
                iaa.Add((-30, 30)),
                iaa.MultiplyElementwise((0.8, 1.2)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.1)),
                iaa.MotionBlur(k=3),
                iaa.Grayscale(alpha=(0.7, 1.0)),
                iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30))
            ]),
            iaa.SomeOf((0, 1), [
                iaa.Affine(scale=(1.0, 2.0)),
                iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                iaa.Affine(rotate=(-30, 30)),
                iaa.Affine(shear=(-32, 32)),
                iaa.PerspectiveTransform(scale=(0.05, 0.25)),
                iaa.Fliplr()
            ])
        ])