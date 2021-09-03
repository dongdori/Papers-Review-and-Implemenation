def load_image(path, target_size=(224, 224)):
    x = image.load_img(path, target_size=target_size)
    x = image.img_to_array(x)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis = 0)
    return x
  
def load_valid_image(path, target_size=(224, 224)):
    x = image.load_img(path, target_size=target_size)
    x = image.img_to_array(x)
    x = x / 255.0
    x = np.expand_dims(x, axis = 0)
    return x
