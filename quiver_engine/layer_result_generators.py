from keras.models import Model


def get_outputs_generator(model, layer_name):
    get_outputs = Model(
        input=model.input,
        output=model.get_layer(layer_name).output
    ).predict

    def get_image_outputs(image):
        outputs = get_outputs(image)[0]
        return [
            outputs[..., idx]
            for idx in range(outputs.shape[-1])
        ]

    return get_image_outputs


def get_saliency_generator(model, layer_name):
    layer = model.get_layer(layer_name)
    saliency_fns = [
        get_saliency_fn(model, layer, idx)
        for idx in range(layer.output.shape[1])
    ]

    return lambda image: [
        get_saliency([image])
        for idx, get_saliency in enumerate(saliency_fns)
    ]

def get_saliency_fn(model, layer, idx):
    return K.function(

        [model.layers[0].input],

        K.gradients(
            K.sum(
                layer.output[:idx],
            ),
            model.layers[0].input
        )
    )
