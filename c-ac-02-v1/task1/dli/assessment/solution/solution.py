from math import exp

@vectorize(['float32(float32)'], target='cuda')
def normalize(grayscales):
    return grayscales / 255

@vectorize(['float32(float32, float32)'], target='cuda')
def weigh(values, weights):
    return values * weights

@vectorize(['float32(float32)'], target='cuda')
def activate(values):
    return ( exp(values) - exp(-values) ) / ( exp(values) + exp(-values) )

def create_hidden_layer(n, greyscales, weights, exp, normalize, weigh, activate):
    
    normalized = cuda.device_array_like(greyscales)
    weighted = cuda.device_array_like(greyscales)
    activated = cuda.device_array_like(greyscales)
    
    normalize(greyscales, out=normalized)
    weigh(normalized, weights, out=weighted)
    activate(weighted, out=activated)

    return activated.copy_to_host()


assessment_values = {
    "n":n,
    "greyscales": greyscales,
    "weights": weights,
    "exp": exp,
    "normalize": normalize,
    "weigh": weigh,
    "activate": activate
}
