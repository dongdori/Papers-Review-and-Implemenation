model = VGG19(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=(224, 224, 3),
    classes=1000,
    classifier_activation="softmax",
)
