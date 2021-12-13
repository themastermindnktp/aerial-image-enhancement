import cv2
import numpy as np

STRUCTURE_ELEMENT_SIZE = 3
STRUCTURE_ELEMENT_DELTA = STRUCTURE_ELEMENT_SIZE // 2

THICK_LINE_STRAIGHT_PATTERN = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
])

THICK_LINE_CROSS_PATTERN = np.array([
    [0, 1, 1],
    [-1, 0, 1],
    [-1, -1, 0]
])

THIN_LINE_STRAIGHT_PATTERN = np.array([
    [1, 1, 1],
    [-1, -1, -1],
    [1, 1, 1]
])

THIN_LINE_CROSS_PATTERN = np.array([
    [-1, 1, 1],
    [1, -1, 1],
    [1, 1, -1]
])


def shift_values(img, delta):
    shifted = np.zeros([height, width], dtype=int)
    for i in range(height):
        for j in range(width):
            shifted[i][j] = img[i][j] + delta
    return shifted


def to_binary(img, threshold=1):
    bin = np.zeros([height, width], dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            bin[i][j] = int(img[i][j] >= threshold)
    return bin


def threshold_decomposition(img, min_value=-128, max_value=128):
    layers = {}
    for threshold in range(min_value + 1, max_value + 1):
        layer = np.zeros([height, width], dtype=int)
        for i in range(height):
            for j in range(width):
                layer[i][j] = -1 if img[i][j] < threshold else 1
        layers[threshold] = layer
        if (threshold - min_value) % 10 == 0:
            print(f"Decomposed {threshold - min_value} layers")
    return layers


def match_pattern(arr, x, y, pattern):
    for dx in range(3):
        for dy in range(3):
            if pattern[dx][dy] and pattern[dx][dy] != arr[x + dx][y + dy]:
                return False
    return True


def is_edge(arr, x, y):
    x, y = x - 1, y - 1

    pattern = THICK_LINE_STRAIGHT_PATTERN
    for _ in range(4):
        if match_pattern(arr, x, y, pattern):
            return 1
        pattern = np.rot90(pattern)

    pattern = THICK_LINE_CROSS_PATTERN
    for _ in range(4):
        if match_pattern(arr, x, y, pattern):
            return 1
        pattern = np.rot90(pattern)

    pattern = THIN_LINE_STRAIGHT_PATTERN
    for _ in range(2):
        if match_pattern(arr, x, y, pattern) or match_pattern(img, x, y, pattern * -1):
            return 1
        pattern = np.rot90(pattern)

    pattern = THIN_LINE_CROSS_PATTERN
    for _ in range(2):
        if match_pattern(arr, x, y, pattern) or match_pattern(img, x, y, pattern * -1):
            return 1
        pattern = np.rot90(pattern)

    return 0


def detect_edge(img):
    edge = np.zeros([height, width], dtype=int)
    for i in range(1, height-1):
        for j in range(1, width-1):
            edge[i][j] = is_edge(img, i, j)

    return edge


def initiate_structure_element():
    res = np.zeros([STRUCTURE_ELEMENT_SIZE, STRUCTURE_ELEMENT_SIZE])
    center = STRUCTURE_ELEMENT_DELTA
    for i in range(STRUCTURE_ELEMENT_SIZE):
        for j in range(STRUCTURE_ELEMENT_SIZE):
            res[i][j] = 1 if abs(i - center) + abs(j - center) <= STRUCTURE_ELEMENT_DELTA else -1
    return res


def dilation(arr, x, y):
    x, y = x - STRUCTURE_ELEMENT_DELTA, y - STRUCTURE_ELEMENT_DELTA
    for dx in range(STRUCTURE_ELEMENT_SIZE):
        for dy in range(STRUCTURE_ELEMENT_SIZE):
            if arr[x+dx][y+dy] == 1 and structure_element[dx][dy] == 1:
                return 1
    return -1


def erosion(arr, x, y):
    x, y = x - STRUCTURE_ELEMENT_DELTA, y - STRUCTURE_ELEMENT_DELTA
    for dx in range(STRUCTURE_ELEMENT_SIZE):
        for dy in range(STRUCTURE_ELEMENT_SIZE):
            if arr[x+dx][y+dy] == -1 and structure_element[dx][dy] == 1:
                return -1
    return 1


def enhance_layer(layer):
    edge = detect_edge(layer)
    for i in range(STRUCTURE_ELEMENT_DELTA, height - STRUCTURE_ELEMENT_DELTA):
        for j in range(STRUCTURE_ELEMENT_DELTA, width - STRUCTURE_ELEMENT_DELTA):
            if edge[i][j] == 1:
                d = dilation(layer, i, j)
                e = erosion(layer, i, j)
                layer[i][j] = d if layer[i][j] >= (d + e) / 2 else e
    return layer


def enhance_image(img):
    shifted_img = shift_values(img, -128)
    layers = threshold_decomposition(shifted_img)

    enhanced_img = np.zeros([height, width], dtype=int)

    cv2.imshow("Layer", to_binary(layers[0]) * 255)
    cv2.waitKey()
    edge = detect_edge(layers[0])
    cv2.imshow("Edge", edge.astype(np.uint8) * 255)
    enhanced_layer = enhance_layer(layers[0])
    cv2.imshow("Result", to_binary(enhanced_layer) * 255)
    cv2.waitKey()

    # for threshold in range(-127, 129):
    #     enhanced_layer = enhance_layer(layers[threshold])
    #     enhanced_img = np.add(enhanced_img, enhanced_layer)
    #     print(f"Enhanced {threshold + 128} layers")

    # enhanced_img //= 2
    #
    # final_result = shift_values(enhanced_img, 128)
    # cv2.imwrite("result.jpg", final_result.astype(np.uint8))


if __name__ == "__main__":
    img = cv2.imread("small_input.jpg", cv2.IMREAD_GRAYSCALE)
    height = len(img)
    width = len(img[0])
    structure_element = initiate_structure_element()
    enhance_image(img)
